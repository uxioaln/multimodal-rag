# -*- coding: utf-8 -*-
"""
experiments.build_eval_dataset - 评测集生成脚本

为 README 声明的三组指标生成评测集：
1. 复杂查询评测集（50 条）：用于首答准确率 + 幻觉率评测
   - 从知识库 docx 切分 chunk，调 LLM 生成多实体/跨条件的复杂查询
   - 每条带 expected_answer（基于 chunk 原文提炼的一句话答案）+ source_chunk_id + source_content
2. 图文混合查询评测集（15 条）：用于跨模态 Recall@5 评测
   - 基于 Qdrant 中图片/视频条目 + 文本上下文，生成跨模态查询
   - 每条带 expected_modalities（如 ["image"] / ["image","text"]）+ expected_media_ref

输出文件：data/stats/eval_dataset.json

使用方法：
    export AGICTO_API_KEY=xxx
    export DASHSCOPE_API_KEY=xxx
    python experiments/build_eval_dataset.py

缓存：eval_dataset.json 已存在则跳过生成（与现有 benchmark 一致），如需重生成请删除该文件。
"""
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

# 把工程根加入 path，使 app.* 包可被导入
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app.config import (
    CHAT_MODEL,
    DOCS_DIR,
    QDRANT_COLLECTION_NAME,
    STATS_DIR,
)
from app.core import query
from app.core.cost_tracker import CostTracker, tracked_chat_completion
from app.index.builder import (
    get_qdrant_client,
    parse_docx,
    preprocess_json_response,
    scroll_all_metadata,
    split_text,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ========== 评测集参数 ==========
RANDOM_SEED = 42
CHUNK_MIN_LEN = 200          # chunk 最小长度（字）
CHUNK_MAX_LEN = 500          # chunk 最大长度（字）
TARGET_COMPLEX_QUERIES = 20  # 复杂查询目标条数
TARGET_CROSS_MODAL_QUERIES = 10  # 图文混合查询目标条数
# docx 切分后 chunk 数量有限（当前知识库约 8 个），每个 chunk 需生成较多查询
# 才能在去重后凑够目标条数
COMPLEX_PER_CHUNK = 5        # 每个 chunk 生成的复杂查询条数
LLM_CALL_INTERVAL = 0.2      # LLM 调用间隔（秒），防限流

# ========== 输出文件 ==========
EVAL_DATASET_FILE = os.path.join(STATS_DIR, "eval_dataset.json")

# 模块级 Qdrant 客户端单例：避免脚本内多次 get_qdrant_client() 创建多实例
# （本地持久化模式下多实例会触发 storage 锁冲突；服务端模式虽不锁也建议复用）
_qdrant_client_singleton = None


def get_qdrant_client_singleton():
    """获取脚本内单例 Qdrant 客户端（首次调用时创建，后续复用）"""
    global _qdrant_client_singleton
    if _qdrant_client_singleton is None:
        _qdrant_client_singleton = get_qdrant_client()
    return _qdrant_client_singleton


# ========== LLM 调用（复用 AGICTO client，走成本追踪） ==========

def chat_completion(prompt: str, temperature: float = 0.7, tracker: CostTracker = None) -> str:
    """调用 AGICTO chat LLM 生成文本（复用 app.core.query 的 client）

    项目约定：所有 chat LLM 调用使用 AGICTO 平台。
    """
    response = tracked_chat_completion(
        client=query.client,
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        tracker=tracker,
        source="评测集生成",
        temperature=temperature,
    )
    return response.choices[0].message.content


# ========== Step 1: 复杂查询生成 ==========

def extract_docx_chunks() -> List[Dict[str, str]]:
    """从 data/knowledge_base/ 下所有 docx 文件提取文本 chunk

    返回 [{"content": chunk_text, "source": filename, "chunk_index": idx}, ...]
    使用 builder.parse_docx / split_text 保持与索引构建一致的切分逻辑。
    """
    chunks = []
    for filename in sorted(os.listdir(DOCS_DIR)):
        if not filename.endswith(".docx"):
            continue
        file_path = os.path.join(DOCS_DIR, filename)
        if os.path.isdir(file_path):
            continue
        full_text = parse_docx(file_path)
        file_chunks = split_text(full_text)
        for idx, ch in enumerate(file_chunks):
            if CHUNK_MIN_LEN <= len(ch) <= CHUNK_MAX_LEN:
                chunks.append({
                    "content": ch,
                    "source": filename,
                    "chunk_index": idx,
                    "source_chunk_id": f"{filename}#{idx}",
                })
    return chunks


def generate_complex_queries_for_chunk(chunk: Dict[str, str], tracker: CostTracker) -> List[Dict[str, str]]:
    """为单个文本 chunk 生成复杂查询 + 期望答案

    复杂查询要求：多实体、跨条件、口语化、含媒体意图，确保非平凡。
    每条查询附带基于原文提炼的 expected_answer，用于 LLM-as-judge 裁判。
    """
    prompt = f"""请基于以下知识内容，生成 {COMPLEX_PER_CHUNK} 条复杂的用户提问。

复杂查询要求（务必满足）：
1. 多实体或多条件：问题涉及 2 个以上知识点或约束条件
2. 表达方式多样化：口语化、省略主语、含指代、同义改写均可
3. 可包含媒体意图：如"图片/海报/视频"等关键词
4. 问题必须能从给定知识内容中找到答案

每条查询需附带 expected_answer：基于原文提炼的一句话答案（要点与原文一致，不编造）。

请返回 JSON 格式：
{{
    "queries": [
        {{
            "query": "复杂查询文本",
            "expected_answer": "基于原文的一句话答案"
        }}
    ]
}}

知识内容：
{chunk["content"]}
"""
    response = chat_completion(prompt, tracker=tracker)
    response = preprocess_json_response(response)
    try:
        result = json.loads(response)
        queries = result.get("queries", [])
        out = []
        for q_data in queries:
            q = (q_data.get("query") or "").strip()
            a = (q_data.get("expected_answer") or "").strip()
            if q and a:
                out.append({
                    "query": q,
                    "expected_answer": a,
                    "source_chunk_id": chunk["source_chunk_id"],
                    "source": chunk["source"],
                    "source_content": chunk["content"],
                })
        return out
    except json.JSONDecodeError:
        logger.warning("复杂查询 JSON 解析失败，跳过该 chunk")
        return []


def build_complex_queries(tracker: CostTracker) -> List[Dict[str, str]]:
    """生成 50 条复杂查询（从 docx chunk 改写，合并去重）"""
    all_chunks = extract_docx_chunks()
    logger.info("从 docx 提取到 %s 个 chunk（长度 %s~%s 字）",
                len(all_chunks), CHUNK_MIN_LEN, CHUNK_MAX_LEN)

    random.seed(RANDOM_SEED)
    # 抽样至足够覆盖 50 条的 chunk 数量（每个 chunk 生成 3 条）
    needed_chunks = (TARGET_COMPLEX_QUERIES + COMPLEX_PER_CHUNK - 1) // COMPLEX_PER_CHUNK
    if len(all_chunks) > needed_chunks:
        sampled = random.sample(all_chunks, needed_chunks)
    else:
        sampled = all_chunks
    logger.info("抽样 %s 个 chunk 用于生成复杂查询", len(sampled))

    all_queries = []
    for i, chunk in enumerate(sampled, 1):
        queries = generate_complex_queries_for_chunk(chunk, tracker)
        all_queries.extend(queries)
        if i % 5 == 0 or i == len(sampled):
            logger.info("  已处理 %s/%s 个 chunk，累计生成 %s 条复杂查询",
                        i, len(sampled), len(all_queries))
        time.sleep(LLM_CALL_INTERVAL)

    # 合并去重（按 query 文本小写去重）
    seen = set()
    unique = []
    for q in all_queries:
        key = q["query"].strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(q)
        if len(unique) >= TARGET_COMPLEX_QUERIES:
            break

    # 若去重后不足目标数量，则保留现有（不强行凑数）
    if len(unique) < TARGET_COMPLEX_QUERIES:
        logger.warning("复杂查询去重后仅 %s 条，不足目标 %s 条，按实际数量继续",
                        len(unique), TARGET_COMPLEX_QUERIES)

    # 编号
    for i, q in enumerate(unique, 1):
        q["id"] = "C%03d" % i
    logger.info("复杂查询生成完成：共 %s 条", len(unique))
    return unique


# ========== Step 2: 图文混合查询生成 ==========

def build_cross_modal_queries(tracker: CostTracker) -> List[Dict[str, Any]]:
    """生成 15 条图文混合查询，每条带期望模态 + 期望媒体引用

    基于 Qdrant 中实际存在的图片/视频条目，让 LLM 生成对应查询。
    """
    qdrant_client = get_qdrant_client_singleton()
    all_metadata = scroll_all_metadata(qdrant_client)

    image_items = [m for m in all_metadata if m.get("type") == "image" and not m.get("deleted", False)]
    video_items = [m for m in all_metadata if m.get("type") == "video" and not m.get("deleted", False)]
    text_items = [m for m in all_metadata if m.get("type") == "text" and not m.get("deleted", False)]
    logger.info("Qdrant 中图片 %s 条，视频 %s 条，文本 %s 条",
                len(image_items), len(video_items), len(text_items))

    # 取若干文本条目作为上下文，帮助 LLM 生成更贴合知识库的跨模态查询
    context_text = "\n".join(m.get("content", "")[:120] for m in text_items[:8])

    # 每个媒体条目生成若干跨模态查询，凑满 15 条
    media_pool = []
    # 图片：每张生成 5 条，期望模态含 image
    for img in image_items:
        media_pool.append({
            "type": "image",
            "ref": img.get("path", ""),
            "description": img.get("content", "") or img.get("source", ""),
        })
    # 视频：每条生成 5 条，期望模态含 video
    for vid in video_items:
        media_pool.append({
            "type": "video",
            "ref": vid.get("url", ""),
            "description": vid.get("description", "") or vid.get("content", ""),
        })

    if not media_pool:
        logger.warning("Qdrant 中无图片/视频条目，无法生成图文混合查询")
        return []

    # 计算每个媒体条目生成多少条查询以凑满 15
    per_media = max(1, (TARGET_CROSS_MODAL_QUERIES + len(media_pool) - 1) // len(media_pool))

    all_queries = []
    for media in media_pool:
        if len(all_queries) >= TARGET_CROSS_MODAL_QUERIES:
            break
        need = min(per_media, TARGET_CROSS_MODAL_QUERIES - len(all_queries))

        expected_modalities = [media["type"]]
        # 一部分查询同时期望文本+媒体（更贴合"图文跨模态"宣传）
        if media["type"] == "image":
            expected_modalities_with_text = ["image", "text"]
        else:
            expected_modalities_with_text = ["video", "text"]

        prompt = f"""请基于以下信息，生成 {need} 条跨模态用户查询。

要求：
1. 查询应能同时关联到文本知识和媒体内容
2. 表达自然，符合真实用户习惯
3. 查询本身不要直接出现媒体文件名或路径
4. 每条查询标注 expected_modalities：期望检索结果包含哪些模态
   - 仅媒体：["{media['type']}"]
   - 图文混合：{expected_modalities_with_text}

参考媒体描述：{media['description']}
参考文本知识（节选）：
{context_text}

请返回 JSON 格式：
{{
    "queries": [
        {{
            "query": "查询文本",
            "expected_modalities": ["{media['type']}"] 或 {expected_modalities_with_text}
        }}
    ]
}}
"""
        response = chat_completion(prompt, tracker=tracker)
        response = preprocess_json_response(response)
        try:
            result = json.loads(response)
            for q_data in result.get("queries", []):
                q = (q_data.get("query") or "").strip()
                mods = q_data.get("expected_modalities") or expected_modalities
                if not isinstance(mods, list):
                    mods = expected_modalities
                mods = [str(m) for m in mods if m]
                if q and mods:
                    all_queries.append({
                        "query": q,
                        "expected_modalities": mods,
                        "expected_media_ref": media["ref"],
                        "expected_media_type": media["type"],
                    })
        except json.JSONDecodeError:
            logger.warning("跨模态查询 JSON 解析失败，跳过该媒体条目")
        time.sleep(LLM_CALL_INTERVAL)

    # 去重 + 截断到目标数量
    seen = set()
    unique = []
    for q in all_queries:
        key = q["query"].strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(q)
        if len(unique) >= TARGET_CROSS_MODAL_QUERIES:
            break

    for i, q in enumerate(unique, 1):
        q["id"] = "M%03d" % i
    logger.info("图文混合查询生成完成：共 %s 条", len(unique))
    return unique


# ========== 主流程 ==========

def main():
    print("=" * 60)
    print("  评测集生成（复杂查询 + 图文混合查询）")
    print("=" * 60)

    os.makedirs(STATS_DIR, exist_ok=True)

    # 缓存：已存在则跳过
    if os.path.exists(EVAL_DATASET_FILE):
        with open(EVAL_DATASET_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)
        n_complex = len(existing.get("complex_queries", []))
        n_modal = len(existing.get("cross_modal_queries", []))
        logger.info("评测集已存在，跳过生成：%s 条复杂查询 + %s 条图文混合查询 -> %s",
                    n_complex, n_modal, EVAL_DATASET_FILE)
        print("  评测集已存在，跳过生成")
        print("  报告文件: %s" % EVAL_DATASET_FILE)
        return

    tracker = CostTracker()
    tracker.begin()

    # Step 1: 生成复杂查询
    print("\n[Step 1] 生成 %s 条复杂查询..." % TARGET_COMPLEX_QUERIES)
    complex_queries = build_complex_queries(tracker)

    # Step 2: 生成图文混合查询
    print("\n[Step 2] 生成 %s 条图文混合查询..." % TARGET_CROSS_MODAL_QUERIES)
    cross_modal_queries = build_cross_modal_queries(tracker)

    # 统计知识库向量总数
    qdrant_client = get_qdrant_client_singleton()
    total_vectors = qdrant_client.count(collection_name=QDRANT_COLLECTION_NAME).count

    # 保存
    dataset = {
        "complex_queries": complex_queries,
        "cross_modal_queries": cross_modal_queries,
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_complex": len(complex_queries),
            "total_cross_modal": len(cross_modal_queries),
            "kb_vector_count": total_vectors,
            "chat_model": CHAT_MODEL,
            "random_seed": RANDOM_SEED,
        },
    }
    with open(EVAL_DATASET_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    tracker.finish()
    cost = tracker.get_summary()

    print("\n" + "=" * 60)
    print("  评测集生成完成")
    print("=" * 60)
    print("  复杂查询: %s 条" % len(complex_queries))
    print("  图文混合查询: %s 条" % len(cross_modal_queries))
    print("  知识库向量总数: %s" % total_vectors)
    print("  生成耗时: %s 秒" % cost["duration_seconds"])
    print("  LLM 调用次数: %s" % cost["call_count"])
    print("  估算成本: %s 元" % cost["estimated_cost"])
    print("  报告文件: %s" % EVAL_DATASET_FILE)
    print("=" * 60)


if __name__ == "__main__":
    main()
