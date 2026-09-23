# -*- coding: utf-8 -*-
"""
app.core.query - 多模态检索

迪士尼RAG助手 V2 - 查询处理：
- 加载 Qdrant collection，处理用户 query，打印相似度排名
- 支持图片/视频关键词检测
- search_vectors 返回与原 FAISS 兼容的 (distance, business_id, metadata) 格式
"""
import logging
from http import HTTPStatus
from typing import Dict, List, Optional, Tuple

import dashscope
from openai import OpenAI

from app.config import (
    AGICTO_API_KEY,
    CHAT_BASE_URL,
    CHAT_MODEL,
    DASHSCOPE_API_KEY,
    IMAGE_KEYWORDS,
    MEDIA_DISTANCE_THRESHOLD,
    MULTIMODAL_EMBEDDING_MODEL,
    QDRANT_COLLECTION_NAME,
    QDRANT_SEARCH_EF,
    VIDEO_KEYWORDS,
)
from app.core.cost_tracker import tracked_chat_completion
from app.index.builder import get_qdrant_client, payload_to_metadata

logger = logging.getLogger(__name__)

dashscope.api_key = DASHSCOPE_API_KEY

# chat LLM 统一走 agicto 平台，使用 deepseek-v4-flash（基于中文训练、性价比高）
# embedding 仍走 DASHSCOPE 多模态（tongyi-embedding-vision-plus，AGICTO 兼容接口不支持该模型）
client = OpenAI(
    api_key=AGICTO_API_KEY,
    base_url=CHAT_BASE_URL
)


def load_index():
    """加载 Qdrant 客户端与 collection（独立运行入口，生产环境由 state.py 管理）"""
    qdrant_client = get_qdrant_client()
    from app.index.builder import collection_exists
    if not collection_exists(qdrant_client):
        raise FileNotFoundError(
            f"Qdrant collection '{QDRANT_COLLECTION_NAME}' 不存在！"
            f"\n请先运行 python scripts/build_index.py 构建索引，"
            f"或运行 python scripts/migrate_faiss_to_qdrant.py 从旧版 FAISS 迁移。"
        )
    count = qdrant_client.count(collection_name=QDRANT_COLLECTION_NAME)
    logger.info("已加载 Qdrant collection '%s': %s 条记录", QDRANT_COLLECTION_NAME, count.count)
    return qdrant_client


def get_text_embedding(text):
    """文本embedding"""
    resp = dashscope.MultiModalEmbedding.call(
        model=MULTIMODAL_EMBEDDING_MODEL,
        input=[{'text': text}]
    )
    if resp.status_code != HTTPStatus.OK:
        raise Exception(f"Embedding失败: {resp.message}")
    return resp.output['embeddings'][0]['embedding']


def distance_to_similarity(distance):
    """L2距离转相似度 (0-1之间，越大越相似)"""
    return 1 / (1 + distance)


def detect_media_intent(query):
    """检测query中是否包含图片/视频意图"""
    query_lower = query.lower()
    want_image = any(kw in query_lower for kw in IMAGE_KEYWORDS)
    want_video = any(kw in query_lower for kw in VIDEO_KEYWORDS)
    return want_image, want_video


def search_vectors(
    query_vec: List[float],
    qdrant_client,
    top_k: Optional[int] = None,
) -> List[Tuple[float, str, Dict]]:
    """使用 Qdrant 向量检索，返回与原 FAISS 兼容的格式

    Args:
        query_vec: 查询向量
        qdrant_client: QdrantClient 实例
        top_k: 返回前 top_k 条结果，None 表示返回全部

    Returns:
        List[Tuple[float, str, dict]]: [(distance, business_id, metadata), ...]
        distance: L2 距离（越小越相似）
        business_id: 业务文档唯一标识
        metadata: 从 payload 转换的内部 metadata 结构
    """
    if top_k is None:
        count = qdrant_client.count(collection_name=QDRANT_COLLECTION_NAME)
        top_k = count.count
    if top_k == 0:
        return []

    # qdrant-client >=1.10 移除了 search()，改用 query_points()
    # ef 统一从 config.QDRANT_SEARCH_EF 读取；本地持久化模式下 ef 不生效（退化为精确搜索）
    from qdrant_client import models as qdrant_models
    response = qdrant_client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        query=query_vec,
        limit=top_k,
        with_payload=True,
        search_params=qdrant_models.SearchParams(hnsw_ef=QDRANT_SEARCH_EF, exact=False),
    )

    results = []
    for point in response.points:
        m = payload_to_metadata(point)
        business_id = (point.payload or {}).get("business_id", "")
        results.append((point.score, business_id, m))
    return results


def search_with_details(query, qdrant_client, k=20):
    """使用 Qdrant 检索并打印相似度详情"""
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print('='*60)

    query_vec = get_text_embedding(query)
    results = search_vectors(query_vec, qdrant_client, top_k=k)

    print(f"\n语义检索相似度排名 (越大越相似):")
    print("-" * 80)
    print(f"{'排名':4s} {'ID':4s} {'类型':6s} {'相似度':8s} {'距离':8s} 内容")
    print("-" * 80)

    formatted = []
    for rank, (dist, business_id, m) in enumerate(results, 1):
        sim = distance_to_similarity(dist)
        content_preview = m.get('content', '')[:45].replace('\n', ' ')
        type_tag = m.get('type', 'unknown')

        marker = ""
        if type_tag == "image":
            marker = " <-- 图片"
        elif type_tag == "video":
            marker = " <-- 视频"
        elif type_tag == "diverse_question":
            marker = " <-- 多样化问题"

        print(f"{rank:4d} {m.get('id', '?'):4} [{type_tag:5s}] {sim:6.4f}  {dist:8.4f}  {content_preview}...{marker}")
        formatted.append({
            "idx": m.get("id"),
            "distance": dist,
            "similarity": sim,
            "metadata": m,
        })

    return formatted


def rag_ask(query: str, qdrant_client, k: int = 3) -> str:
    """RAG问答，支持图片/视频关键词检测"""
    results = search_with_details(query, qdrant_client, k=20)

    # 检测媒体意图
    want_image, want_video = detect_media_intent(query)
    logger.info("意图检测: 需要图片=%s, 需要视频=%s", want_image, want_video)

    # 取 top-k 文本结果，diverse_question 映射回原始 chunk 内容
    metadata_map = {m.get("id"): m for m in [r["metadata"] for r in results]}
    top_results = []
    for r in results:
        m = r["metadata"]
        if m["type"] in ("text", "diverse_question"):
            # 若是多样化问题，使用原始 chunk 作为上下文
            if m["type"] == "diverse_question":
                original_id = m.get("original_chunk_id")
                if original_id is not None and original_id in metadata_map:
                    r = r.copy()
                    r["metadata"] = metadata_map[original_id]
            top_results.append(r)
        if len(top_results) >= k:
            break

    # 如果需要图片，找距离<3的图片中距离最小的Top1
    matched_image = None
    if want_image:
        image_results = [r for r in results if r["metadata"]["type"] == "image" and r["distance"] < MEDIA_DISTANCE_THRESHOLD]
        if image_results:
            image_results.sort(key=lambda x: x["distance"])
            matched_image = image_results[0]
            logger.info("  -> 匹配到图片: %s (距离: %.4f, 相似度: %.4f)",
                        matched_image['metadata'].get('path', ''), matched_image['distance'], matched_image['similarity'])

    # 如果需要视频，找距离<3的视频中距离最小的Top1
    matched_video = None
    if want_video:
        video_results = [r for r in results if r["metadata"]["type"] == "video" and r["distance"] < MEDIA_DISTANCE_THRESHOLD]
        if video_results:
            video_results.sort(key=lambda x: x["distance"])
            matched_video = video_results[0]
            logger.info("  -> 匹配到视频: %s (距离: %.4f, 相似度: %.4f)",
                        matched_video['metadata'].get('url', ''), matched_video['distance'], matched_video['similarity'])

    logger.info("选取Top-%s文本构建Prompt:", k)
    for r in top_results:
        logger.info("  - %s... (相似度: %.4f)", r['metadata'].get('content', '')[:50], r['similarity'])

    # 构建context
    context_str = ""
    for i, r in enumerate(top_results):
        m = r["metadata"]
        context_str += f"背景知识 {i+1} (来源: {m.get('source', '')}, 相似度: {r['similarity']:.4f}):\n{m.get('content', '')}\n\n"

    prompt = f"""你是一个迪士尼客服助手。请根据以下背景知识回答用户问题。

[背景知识]
{context_str}
[用户问题]
{query}
"""

    # 调用LLM（接入成本追踪）
    logger.info("调用LLM生成答案...")
    completion = tracked_chat_completion(
        client=client,
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "你是一个迪士尼客服助手。"},
            {"role": "user", "content": prompt}
        ],
        source="问答"
    )
    answer = completion.choices[0].message.content

    # 附加匹配到的媒体
    if matched_image:
        answer += f"\n\n[相关图片]: {matched_image['metadata'].get('path', '')}"
    if matched_video:
        answer += f"\n\n[相关视频]: {matched_video['metadata'].get('url', '')}"

    logger.info("最终答案:\n%s", answer)
    return answer


if __name__ == "__main__":
    qdrant_client = load_index()

    print("\n" + "="*60)
    rag_ask("我想了解一下迪士尼门票的退款流程", qdrant_client, k=3)

    print("\n" + "="*60)
    rag_ask("最近万圣节的活动海报是什么", qdrant_client, k=3)

    print("\n" + "="*60)
    rag_ask("我的汽车被剐蹭了，你能看到视频么？", qdrant_client, k=3)

    print("\n" + "="*60)
    rag_ask("聚在一起说奇妙的海报", qdrant_client, k=3)
