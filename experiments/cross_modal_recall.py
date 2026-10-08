# -*- coding: utf-8 -*-
"""
experiments.cross_modal_recall - 跨模态 Recall@5 评测脚本

评测口径（与 README 一致）：
- 跨模态检索：文本/图片/视频统一嵌入同一向量空间，对图文混合查询检索 Top-5，
  判定 Top-5 的模态集合是否覆盖期望模态（⊇ expected_modalities 视为命中）
- 纯文本基线：仅在 type in ("text","diverse_question") 的结果中取 Top-5，
  基线无法召回 image/video，用于对比证明跨模态统一嵌入的价值

输出文件：data/stats/cross_modal_recall_report.json

使用方法：
    export AGICTO_API_KEY=xxx
    export DASHSCOPE_API_KEY=xxx
    python experiments/build_eval_dataset.py   # 先生成评测集
    python experiments/cross_modal_recall.py     # 再跑跨模态评测
"""
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

# 把工程根加入 path，使 app.* 包可被导入
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app.config import QDRANT_COLLECTION_NAME, STATS_DIR
from app.core.retrieval import hybrid_retrieve, DEFAULT_IMAGE_QUOTA, DEFAULT_VIDEO_QUOTA
from app.index.builder import get_qdrant_client

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ========== 评测参数 ==========
TOP_K = 5  # Recall@K 的 K 值，与 README 口径一致

# ========== 输入/输出文件 ==========
EVAL_DATASET_FILE = os.path.join(STATS_DIR, "eval_dataset.json")
REPORT_FILE = os.path.join(STATS_DIR, "cross_modal_recall_report.json")

# 纯文本基线认可的文本类模态（与 metadata.type 对齐）
TEXT_TYPES = {"text", "diverse_question"}


def load_cross_modal_queries() -> List[Dict[str, Any]]:
    """加载评测集中的图文混合查询"""
    if not os.path.exists(EVAL_DATASET_FILE):
        raise FileNotFoundError(
            "评测集不存在: %s，请先运行 python experiments/build_eval_dataset.py" % EVAL_DATASET_FILE
        )
    with open(EVAL_DATASET_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("cross_modal_queries", [])


def _build_modality_quota(expected_modalities: List[str]) -> Dict[str, int]:
    """根据期望模态构建模态配额

    期望模态含 image 时配额 DEFAULT_IMAGE_QUOTA 条，含 video 时配额 DEFAULT_VIDEO_QUOTA 条。
    配额强制 top-k 中目标模态至少占 N 条，防止被多数文本条目挤出。
    """
    quota = {}
    if "image" in expected_modalities:
        quota["image"] = DEFAULT_IMAGE_QUOTA
    if "video" in expected_modalities:
        quota["video"] = DEFAULT_VIDEO_QUOTA
    return quota if quota else None


def is_hit(expected_modalities: List[str], retrieved_types: List[str]) -> bool:
    """判定是否命中（模态集合口径）：Top-K 的模态集合 ⊇ 期望模态集合

    宽松口径：检索结果只要覆盖所有期望模态就算命中，不校验检回的条目是否为
    expected_media_ref 指向的目标条目。该口径会被模态配额机制"保送"，无法
    真实反映检索排序质量，仅作为对比基线保留。

    例如 expected=["image","text"]，则 Top-5 必须同时含 image 和 text 类型。
    """
    retrieved_set = set(retrieved_types)
    expected_set = set(expected_modalities)
    return expected_set.issubset(retrieved_set)


def _resolve_media_ref(qdrant_client, m: Dict[str, Any]) -> Optional[str]:
    """解析单条检索结果的媒体引用（图片 path / 视频 url）

    用于精确匹配口径：判定检回的条目是否为期望的目标媒体条目。

    - type=image：返回 metadata.path（图片本地路径）
    - type=video：返回 metadata.url（视频 URL）
    - type=diverse_question：按 original_chunk_id 回查 Qdrant 原条目，
      递归解析原条目的 path/url；若原条目是文本则返回 None
    - 其他类型（text 等）：返回 None

    Args:
        qdrant_client: QdrantClient 实例（用于回查 diverse_question 的原条目）
        m: 检索结果条目（内部 metadata 结构）

    Returns:
        媒体引用字符串（path 或 url）；非媒体条目返回 None
    """
    entry_type = m.get("type")
    if entry_type == "image":
        return m.get("path")
    if entry_type == "video":
        return m.get("url")
    if entry_type == "diverse_question":
        # diverse_question 存的是 LLM 生成的问题，需按 original_chunk_id 回源到原条目
        original_chunk_id = m.get("original_chunk_id")
        if original_chunk_id is None:
            return None
        try:
            points = qdrant_client.retrieve(
                collection_name=QDRANT_COLLECTION_NAME,
                ids=[original_chunk_id],
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                return None
            # 复用 payload_to_metadata 转换，保持与主链路一致
            from app.index.builder import payload_to_metadata
            original_m = payload_to_metadata(points[0])
            # 递归解析原条目；原条目不会是 diverse_question，最多一层回源
            if original_m.get("type") == "image":
                return original_m.get("path")
            if original_m.get("type") == "video":
                return original_m.get("url")
        except Exception:
            logger.warning("回查 original_chunk_id=%s 失败", original_chunk_id, exc_info=True)
        return None
    return None


def is_precise_hit(qdrant_client, results: List[Dict[str, Any]], expected_media_ref: str) -> bool:
    """判定是否命中（精确匹配口径）：Top-K 中存在条目的媒体引用 == expected_media_ref

    严格口径：模态集合口径会被配额机制保送（只要候选池有任意图片/视频即命中），
    本口径额外校验检回的条目确实是评测集标注的目标媒体条目（path 或 url 相等），
    挤掉配额水分。diverse_question 按 original_chunk_id 回源后参与判定，
    使媒体的多样化问题也能为目标媒体贡献命中。

    Args:
        qdrant_client: QdrantClient 实例
        results: Top-K 检索结果列表
        expected_media_ref: 期望的目标媒体引用（图片 path 或视频 url）

    Returns:
        命中返回 True；expected_media_ref 为空或未命中返回 False
    """
    if not expected_media_ref:
        return False
    for m in results:
        ref = _resolve_media_ref(qdrant_client, m)
        if ref is not None and ref == expected_media_ref:
            return True
    return False


def evaluate_single(query_item: Dict[str, Any], qdrant_client) -> Dict[str, Any]:
    """对单条图文混合查询做跨模态 + 纯文本基线双路检索，返回命中情况

    跨模态路径使用完整 hybrid_retrieve 管线（查询改写 + 向量+BM25 RRF融合 + rerank 精排 + 模态配额），
    与 Agent 生产链路一致，确保评测结果反映真实检索能力。

    同时输出两个口径的命中结果：
    - 模态集合口径（宽松）：Top-K 模态集合 ⊇ expected_modalities，会被配额保送
    - 精确匹配口径（严格）：Top-K 中存在 path/url == expected_media_ref 的条目
    """
    query_text = query_item["query"]
    expected_modalities = query_item.get("expected_modalities", [])
    expected_media_ref = query_item.get("expected_media_ref", "")

    # 构建模态配额：根据期望模态强制 top-k 中目标模态至少占 N 条
    modality_quota = _build_modality_quota(expected_modalities)

    # 跨模态检索：完整 hybrid_retrieve 管线（含 BM25 + RRF + rerank + 模态配额）
    cross_modal_results = hybrid_retrieve(
        query_text,
        qdrant_client,
        k=TOP_K,
        modality_quota=modality_quota,
    )
    cross_modal_types = [m.get("type", "unknown") for m in cross_modal_results]
    # 模态集合口径（宽松）
    cross_modal_hit = is_hit(expected_modalities, cross_modal_types)
    # 精确匹配口径（严格）：检回条目的媒体引用 == expected_media_ref
    cross_modal_precise_hit = is_precise_hit(qdrant_client, cross_modal_results, expected_media_ref)

    # 纯文本基线：只在文本类结果中取 Top-K（使用 hybrid_retrieve 但限定文本类型）
    text_results = hybrid_retrieve(
        query_text,
        qdrant_client,
        k=TOP_K,
        allowed_types=TEXT_TYPES,
    )
    text_types = [m.get("type", "unknown") for m in text_results]
    # 纯文本基线只能命中 expected 中只含 text 的场景；若 expected 含 image/video，基线必 miss
    text_hit = is_hit([t for t in expected_modalities if t == "text"], text_types) if "text" in expected_modalities else False
    # 纯文本基线不含媒体条目，精确匹配口径必 miss（expected_media_ref 为图片/视频时）
    text_precise_hit = is_precise_hit(qdrant_client, text_results, expected_media_ref) if expected_media_ref else False

    return {
        "id": query_item.get("id", ""),
        "query": query_text,
        "expected_modalities": expected_modalities,
        "expected_media_ref": expected_media_ref,
        "modality_quota": modality_quota,
        "cross_modal": {
            "retrieved_types": cross_modal_types,
            "hit": cross_modal_hit,
            "precise_hit": cross_modal_precise_hit,
            "pipeline": "hybrid_retrieve (rewrite + vector + BM25 + RRF + rerank + quota)",
        },
        "text_only_baseline": {
            "retrieved_types": text_types,
            "hit": text_hit,
            "precise_hit": text_precise_hit,
            "pipeline": "hybrid_retrieve (text-only filter)",
        },
    }


def main():
    print("=" * 60)
    print("  跨模态 Recall@5 评测")
    print("=" * 60)

    queries = load_cross_modal_queries()
    logger.info("加载图文混合查询 %s 条", len(queries))
    if not queries:
        print("  评测集为空，退出")
        return

    qdrant_client = get_qdrant_client()
    total_vectors = qdrant_client.count(collection_name=QDRANT_COLLECTION_NAME).count
    logger.info("Qdrant collection '%s' 当前共 %s 条记录", QDRANT_COLLECTION_NAME, total_vectors)

    # 加载 BM25 索引（hybrid_retrieve 依赖 BM25 做关键词检索）
    # 直接从 Qdrant 拉取 metadata 构建 BM25，避免 state.load_resources() 重复创建 Qdrant 客户端导致锁冲突
    from app.index.builder import scroll_all_metadata
    from app.core.retrieval import get_bm25_index, build_bm25_from_metadata
    if get_bm25_index() is None:
        metadata = scroll_all_metadata(qdrant_client)
        build_bm25_from_metadata(metadata)
        logger.info("BM25 索引已加载: %s 条", len(metadata))

    per_query = []
    cross_modal_hits = 0
    cross_modal_precise_hits = 0
    text_baseline_hits = 0
    text_baseline_precise_hits = 0
    for i, q_item in enumerate(queries, 1):
        result = evaluate_single(q_item, qdrant_client)
        per_query.append(result)
        if result["cross_modal"]["hit"]:
            cross_modal_hits += 1
        if result["cross_modal"]["precise_hit"]:
            cross_modal_precise_hits += 1
        if result["text_only_baseline"]["hit"]:
            text_baseline_hits += 1
        if result["text_only_baseline"]["precise_hit"]:
            text_baseline_precise_hits += 1
        if i % 3 == 0 or i == len(queries):
            logger.info("  进度 %s/%s，跨模态命中 %s（精确 %s），文本基线命中 %s（精确 %s）",
                        i, len(queries),
                        cross_modal_hits, cross_modal_precise_hits,
                        text_baseline_hits, text_baseline_precise_hits)

    n = len(queries)
    cross_modal_recall = round(cross_modal_hits / n, 4) if n else 0.0
    cross_modal_precise_recall = round(cross_modal_precise_hits / n, 4) if n else 0.0
    text_baseline_recall = round(text_baseline_hits / n, 4) if n else 0.0
    text_baseline_precise_recall = round(text_baseline_precise_hits / n, 4) if n else 0.0

    report = {
        "metadata": {
            "total_queries": n,
            "top_k": TOP_K,
            "total_vectors": total_vectors,
            "hit_criteria_loose": "Top-K 模态集合 ⊇ expected_modalities（会被配额保送）",
            "hit_criteria_strict": "Top-K 中存在 path/url == expected_media_ref 的条目（diverse_question 按 original_chunk_id 回源）",
            "timestamp": datetime.now().isoformat(),
        },
        "cross_modal": {
            "recall@5_loose": cross_modal_recall,
            "recall@5_strict": cross_modal_precise_recall,
            "hit_count_loose": cross_modal_hits,
            "hit_count_strict": cross_modal_precise_hits,
            "per_query": per_query,
        },
        "text_only_baseline": {
            "recall@5_loose": text_baseline_recall,
            "recall@5_strict": text_baseline_precise_recall,
            "hit_count_loose": text_baseline_hits,
            "hit_count_strict": text_baseline_precise_hits,
        },
    }

    os.makedirs(STATS_DIR, exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("已保存跨模态评测报告 -> %s", REPORT_FILE)

    print("\n" + "=" * 60)
    print("  跨模态 Recall@5 评测完成")
    print("=" * 60)
    print("  查询数量: %s" % n)
    print("  Top-K: %s" % TOP_K)
    print("  知识库向量总数: %s" % total_vectors)
    print("  跨模态 Recall@5（模态集合口径）: %.4f（命中 %s/%s）" % (cross_modal_recall, cross_modal_hits, n))
    print("  跨模态 Recall@5（精确匹配口径）: %.4f（命中 %s/%s）" % (cross_modal_precise_recall, cross_modal_precise_hits, n))
    print("  纯文本基线 Recall@5（模态集合口径）: %.4f（命中 %s/%s）" % (text_baseline_recall, text_baseline_hits, n))
    print("  纯文本基线 Recall@5（精确匹配口径）: %.4f（命中 %s/%s）" % (text_baseline_precise_recall, text_baseline_precise_hits, n))
    print("  报告文件: %s" % REPORT_FILE)
    print("=" * 60)


if __name__ == "__main__":
    main()
