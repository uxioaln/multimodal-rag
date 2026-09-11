# -*- coding: utf-8 -*-
"""
app.index.builder - 知识库构建（索引入库）

功能：
- 解析文档/图片/视频，生成 embedding，写入 Qdrant 本地持久化存储
- 提供 clean_index_files() 删除旧索引（Qdrant 存储目录 + doc_hashes）
- build_and_save() 支持 progress_callback 回调（用于 SSE 流式推送进度）
- 既可独立运行，也可被后端通过 import 调用

payload 与内部 metadata 的双向转换：
- metadata_to_payload：内部 metadata 字典 -> Qdrant payload（spec 字段命名）
- payload_to_metadata：Qdrant point -> 内部 metadata 字典（兼容旧业务逻辑）
"""
import base64
import json
import logging
import os
import shutil
from datetime import datetime
from http import HTTPStatus
from typing import Callable, Dict, List, Optional, Tuple

import dashscope
import numpy as np
from docx import Document as DocxDocument
from openai import OpenAI

from app.config import (
    AGICTO_API_KEY,
    CHAT_BASE_URL,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DASHSCOPE_API_KEY,
    DIVERSE_REWRITE_MODEL,
    DOCS_DIR,
    HASHES_FILE,
    IMG_DIR,
    MULTIMODAL_EMBEDDING_MODEL,
    QDRANT_COLLECTION_NAME,
    QDRANT_HNSW_M,
    QDRANT_PATH,
    QDRANT_VECTOR_SIZE,
)
from app.core.cost_tracker import tracked_chat_completion

logger = logging.getLogger(__name__)

dashscope.api_key = DASHSCOPE_API_KEY

# Qdrant 客户端，用于大模型生成多样化问题
# 仅在启用多样化改写时才需要有效配置
agicto_client = None
if AGICTO_API_KEY:
    agicto_client = OpenAI(
        api_key=AGICTO_API_KEY,
        base_url=CHAT_BASE_URL
    )

# 重建索引时需要清理的所有旧文件（仅 doc_hashes，Qdrant 存储目录单独处理）
INDEX_FILES_TO_CLEAN = [HASHES_FILE]

# 是否使用大模型为每个文本chunk生成多样化问题并加入向量索引
USE_DIVERSE_REWRITE = True

# 视频知识库
VIDEO_KNOWLEDGE = [
    {
        "url": "https://dataset-1255932437.cos.ap-nanjing.myqcloud.com/mp4/car.mp4",
        "description": "汽车剐蹭视频"
    }
]


# ========== Payload 与 Metadata 双向转换 ==========

def metadata_to_payload(m: Dict) -> Dict:
    """将内部 metadata 结构转换为 Qdrant payload（spec 字段命名）

    内部 metadata 使用 type/path/url/doc_id 等字段，
    Qdrant payload 使用 media_type/file_path/business_id 等字段。
    """
    media_type = m.get("type", "text")
    payload = {
        "business_id": m.get("doc_id", "") or str(m.get("id", "")),
        "content": m.get("content", ""),
        "source": m.get("source", ""),
        "chunk_index": m.get("chunk_index", 0),
        "media_type": media_type,
        "diverse_questions": m.get("diverse_questions", []),
        "doc_hash": m.get("hash", ""),
        "created_at": m.get("created_at", ""),
        "last_updated": m.get("last_updated", ""),
        "deleted": m.get("deleted", False),
    }
    if media_type == "image":
        payload["file_path"] = m.get("path", "")
    elif media_type == "video":
        payload["file_path"] = m.get("url", "")
        payload["description"] = m.get("description", "")
    elif media_type == "diverse_question":
        payload["original_chunk_id"] = m.get("original_chunk_id")
        payload["question_type"] = m.get("question_type", "")
        payload["perspective"] = m.get("perspective", "")
    return payload


def payload_to_metadata(point) -> Dict:
    """将 Qdrant point 转换为内部 metadata 结构（兼容旧业务逻辑）

    保留 type/path/url/doc_id/id 等旧字段名，使 knowledge/health_check 等模块无需改动。
    """
    p = point.payload or {}
    media_type = p.get("media_type", "text")
    m: Dict = {
        "id": point.id,
        "doc_id": p.get("business_id", ""),
        "source": p.get("source", ""),
        "type": media_type,
        "content": p.get("content", ""),
        "diverse_questions": p.get("diverse_questions", []),
        "last_updated": p.get("last_updated", ""),
        "deleted": p.get("deleted", False),
        "hash": p.get("doc_hash", ""),
        "created_at": p.get("created_at", ""),
        "chunk_index": p.get("chunk_index", 0),
    }
    if media_type == "image":
        m["path"] = p.get("file_path", "")
    elif media_type == "video":
        m["url"] = p.get("file_path", "")
        m["description"] = p.get("description", "")
    elif media_type == "diverse_question":
        m["original_chunk_id"] = p.get("original_chunk_id")
        m["question_type"] = p.get("question_type", "")
        m["perspective"] = p.get("perspective", "")
    return m


# ========== Qdrant Collection 管理 ==========

def get_qdrant_client():
    """获取 Qdrant 客户端

    连接模式由环境变量 QDRANT_URL 控制：
    - 设置了 QDRANT_URL（如 docker-compose 编排时为 http://qdrant:6333）：
      连接 Qdrant 服务端，向量数据由独立的 qdrant 容器持久化。
    - 未设置 QDRANT_URL：使用本地持久化存储（原有逻辑，QdrantClient(path=...)）。
    """
    from qdrant_client import QdrantClient
    qdrant_url = os.getenv("QDRANT_URL")
    if qdrant_url:
        # 服务端模式：连接 docker-compose 编排的 qdrant 服务
        return QdrantClient(url=qdrant_url)
    # 本地持久化模式（原有逻辑）
    return QdrantClient(path=QDRANT_PATH)


def collection_exists(client) -> bool:
    """检查 collection 是否已存在"""
    collections = client.get_collections()
    existing_names = [c.name for c in collections.collections]
    return QDRANT_COLLECTION_NAME in existing_names


def detect_vector_dimension() -> int:
    """通过生成一个样本 embedding 检测实际向量维度"""
    logger.info("检测向量维度...")
    sample_vector = get_text_embedding("sample")
    actual_dim = len(sample_vector)
    if actual_dim != QDRANT_VECTOR_SIZE:
        logger.info("实际向量维度为 %s（配置默认值为 %s）", actual_dim, QDRANT_VECTOR_SIZE)
    else:
        logger.info("向量维度: %s", actual_dim)
    return actual_dim


def suggest_hnsw_m(vector_count: int) -> int:
    """根据向量规模建议 HNSW 图连接数 m

    规模越小图越稀疏，过大的 m 会增加构建成本而收益有限；
    规模越大需要更大的 m 保证图连通性与召回率。

    - <500：建议 8（小规模精确搜索已足够，m 小降低开销）
    - 500~10000：建议 16（中等规模默认值）
    - >=10000：建议 32（大规模需更强连通性）
    """
    if vector_count < 500:
        return 8
    if vector_count < 10000:
        return 16
    return 32


def get_or_create_collection(client, vector_size: int) -> None:
    """确保 collection 存在：若已存在则先删除后重建（与"重建索引"语义一致）"""
    from qdrant_client import models

    if collection_exists(client):
        logger.info("已存在 collection '%s'，先删除...", QDRANT_COLLECTION_NAME)
        client.delete_collection(QDRANT_COLLECTION_NAME)

    client.create_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.EUCLID,
        ),
        hnsw_config=models.HnswConfigDiff(m=QDRANT_HNSW_M),
    )
    logger.info("已创建 collection '%s'（向量维度: %s，距离: EUCLID，HNSW m=%s）",
                QDRANT_COLLECTION_NAME, vector_size, QDRANT_HNSW_M)


def ensure_collection(client) -> None:
    """确保 collection 存在但不删数据（用于 state.load_resources 启动时）"""
    from qdrant_client import models
    if not collection_exists(client):
        client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=QDRANT_VECTOR_SIZE,
                distance=models.Distance.EUCLID,
            ),
            hnsw_config=models.HnswConfigDiff(m=QDRANT_HNSW_M),
        )
        logger.info("已创建 collection '%s'（向量维度: %s，距离: EUCLID，HNSW m=%s）",
                    QDRANT_COLLECTION_NAME, QDRANT_VECTOR_SIZE, QDRANT_HNSW_M)


def scroll_all_metadata(client) -> List[Dict]:
    """从 Qdrant 滚动读取所有 point，返回内部 metadata 列表（兼容旧格式）"""
    all_metadata = []
    offset = None
    while True:
        result = client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points, next_offset = result
        for point in points:
            all_metadata.append(payload_to_metadata(point))
        if next_offset is None:
            break
        offset = next_offset
    return all_metadata


def upsert_points(client, points: List, batch_size: int = 100) -> None:
    """批量 upsert points 到 Qdrant"""
    from qdrant_client import models
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=batch,
        )
        logger.info("已写入 %s/%s 条", min(i + batch_size, len(points)), len(points))


# ========== 文档解析与文本切分 ==========

def parse_docx(file_path):
    """解析 DOCX 文件，提取全部文本"""
    doc = DocxDocument(file_path)
    all_text = []

    for element in doc.element.body:
        if element.tag.endswith('p'):
            paragraph_text = ""
            for run in element.findall('.//w:t', {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}):
                paragraph_text += run.text if run.text else ""
            if paragraph_text.strip():
                all_text.append(paragraph_text.strip())

        elif element.tag.endswith('tbl'):
            table = [t for t in doc.tables if t._element is element][0]
            if table.rows:
                md_table = []
                header = [cell.text.strip() for cell in table.rows[0].cells]
                md_table.append("| " + " | ".join(header) + " |")
                md_table.append("|" + "---|"*len(header))
                for row in table.rows[1:]:
                    row_data = [cell.text.strip() for cell in row.cells]
                    md_table.append("| " + " | ".join(row_data) + " |")
                all_text.append("\n".join(md_table))

    return "\n".join(all_text)


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """按固定长度切分文本"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks


def preprocess_json_response(response):
    """预处理AI响应，移除markdown代码块格式"""
    if not response:
        return ""

    if response.startswith('```json'):
        response = response[7:]
    elif response.startswith('```'):
        response = response[3:]

    if response.endswith('```'):
        response = response[:-3]

    return response.strip()


def get_llm_completion(prompt, model=DIVERSE_REWRITE_MODEL):
    """调用AGICTO大模型生成文本"""
    if agicto_client is None:
        raise ValueError("错误：启用多样化改写需要设置 'AGICTO_API_KEY' 环境变量。")

    response = tracked_chat_completion(
        client=agicto_client,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        source="索引构建"
    )
    return response.choices[0].message.content


def generate_diverse_questions(knowledge_chunk, num_questions=5):
    """为单个知识切片生成多样化问题"""
    instruction = """
你是一个专业的问答系统专家。请为给定的知识内容生成高度多样化的问题，确保：
1. 问题类型多样化：直接问、间接问、对比问、条件问、假设问、推理问等
2. 表达方式多样化：使用不同的句式、词汇、语气
3. 角度多样化：从不同角度和维度提问
4. 确保问题不超出知识内容范围

请返回JSON格式：
{
    "questions": [
        {
            "question": "问题内容",
            "question_type": "问题类型",
            "perspective": "提问角度"
        }
    ]
}
"""

    prompt = f"""
### 指令 ###
{instruction}

### 知识内容 ###
{knowledge_chunk}

### 生成问题数量 ###
{num_questions}

### 生成结果 ###
"""

    response = get_llm_completion(prompt)
    response = preprocess_json_response(response)

    try:
        result = json.loads(response)
        return result.get('questions', [])
    except json.JSONDecodeError as e:
        logger.warning("多样化问题生成JSON解析失败: %s", e)
        logger.debug("AI返回内容: %s", response[:200])
        return []


# ========== Embedding 函数 ==========

def get_text_embedding(text: str) -> List[float]:
    """文本 embedding"""
    resp = dashscope.MultiModalEmbedding.call(
        model=MULTIMODAL_EMBEDDING_MODEL,
        input=[{'text': text}]
    )
    if resp.status_code != HTTPStatus.OK:
        raise Exception(f"文本Embedding失败: {resp.message}")
    return resp.output['embeddings'][0]['embedding']


def get_image_embedding(image_path):
    """图片embedding"""
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode('utf-8')

    ext = os.path.splitext(image_path)[1].lower().lstrip('.')
    if ext == 'jpg':
        ext = 'jpeg'
    image_data = f"data:image/{ext};base64,{base64_image}"

    resp = dashscope.MultiModalEmbedding.call(
        model=MULTIMODAL_EMBEDDING_MODEL,
        input=[{'image': image_data}]
    )
    if resp.status_code != HTTPStatus.OK:
        raise Exception(f"图片Embedding失败: {resp.message}")
    return resp.output['embeddings'][0]['embedding']


def get_video_embedding(video_url: str) -> List[float]:
    """视频 embedding（多帧取平均）"""
    resp = dashscope.MultiModalEmbedding.call(
        model=MULTIMODAL_EMBEDDING_MODEL,
        input=[{'video': video_url}]
    )
    if resp.status_code != HTTPStatus.OK:
        raise Exception(f"视频Embedding失败: {resp.message}")

    embeddings = resp.output['embeddings']
    if len(embeddings) > 1:
        vectors = [np.array(e['embedding']) for e in embeddings]
        return np.mean(vectors, axis=0).tolist()
    return embeddings[0]['embedding']


# ========== 文本 chunk 条目构建 ==========

# 重建索引进度回调签名：stage 阶段名 / current 当前进度 / total 总数 / message 描述
ProgressCallback = Callable[[str, int, int, str], None]


def build_text_chunk_entries(
    chunk: str,
    source: str,
    start_id: int,
    today_str: Optional[str] = None,
    doc_id: Optional[str] = None,
    extra_metadata: Optional[dict] = None,
) -> Tuple[List, List[Dict], int]:
    """为单个文本 chunk 生成向量与元数据（与 build_and_save 内文档 chunk 处理逻辑一致）

    Args:
        chunk: 文本内容
        source: 来源描述
        start_id: 起始内部 id（Qdrant point id）
        today_str: 更新时间字符串，默认当天
        doc_id: 业务文档唯一标识（可选）
        extra_metadata: 写入主文本条目的额外字段

    Returns:
        (vectors, metadata_entries, next_id)
        vectors: list[float]，每个向量对应一条 metadata_entries
        metadata_entries: list[dict]，内部 metadata 结构
        next_id: 下一个可用的 point id
    """
    if not chunk or not chunk.strip():
        return [], [], start_id

    today_str = today_str or datetime.now().strftime('%Y-%m-%d')
    extra_metadata = extra_metadata or {}

    vectors = []
    entries = []
    doc_id = doc_id or str(start_id)

    chunk_id = start_id
    metadata = {
        "id": chunk_id,
        "source": source,
        "type": "text",
        "content": chunk,
        "diverse_questions": [],
        "last_updated": today_str,
        **extra_metadata,
    }
    if doc_id is not None:
        metadata["doc_id"] = doc_id

    vector = get_text_embedding(chunk)
    vectors.append(vector)
    entries.append(metadata)
    next_id = chunk_id + 1

    if USE_DIVERSE_REWRITE:
        diverse_questions = generate_diverse_questions(chunk)
        metadata["diverse_questions"] = diverse_questions

        for q_data in diverse_questions:
            question = q_data.get('question', '')
            if not question.strip():
                continue
            q_metadata = {
                "id": next_id,
                "source": source,
                "type": "diverse_question",
                "content": question,
                "original_chunk_id": chunk_id,
                "question_type": q_data.get('question_type', ''),
                "perspective": q_data.get('perspective', ''),
                "last_updated": today_str,
                "doc_id": doc_id,
            }
            combined_text = f"内容：{chunk} 问题：{question}"
            q_vector = get_text_embedding(combined_text)
            vectors.append(q_vector)
            entries.append(q_metadata)
            next_id += 1

    return vectors, entries, next_id


# ========== 全量构建 ==========

def build_and_save(progress_callback: Optional[ProgressCallback] = None) -> None:
    """构建多模态知识库并保存到 Qdrant：多样化改写 + 纯向量语义检索

    Args:
        progress_callback: 可选进度回调，签名 (stage, current, total, message)
                          为 None 时走 logger.info 输出（独立运行模式）
    """
    from qdrant_client import models

    def emit(stage: str, current: int, total: int, message: str) -> None:
        """内部进度发射器：有 callback 就调 callback，否则用 logger 输出"""
        if progress_callback:
            progress_callback(stage, current, total, message)
        else:
            logger.info("[%s %s/%s] %s", stage, current, total, message)

    logger.info("--- 构建多模态知识库 (Qdrant) ---")
    logger.info("切分参数: chunk_size=%s, overlap=%s", CHUNK_SIZE, CHUNK_OVERLAP)
    logger.info("多样化改写: %s", USE_DIVERSE_REWRITE)
    emit("init", 0, 1, "开始构建知识库索引")

    # 本次构建时间，作为所有条目的 last_updated 字段值
    today_str = datetime.now().strftime('%Y-%m-%d')
    created_at = datetime.utcnow().isoformat() + "Z"

    # 初始化 Qdrant 客户端，检测实际向量维度后创建 collection
    client = get_qdrant_client()
    actual_vector_dim = detect_vector_dimension()
    get_or_create_collection(client, actual_vector_dim)

    # 收集所有待写入的 points
    all_points = []
    point_id = 1

    # 处理Word文档
    for filename in os.listdir(DOCS_DIR):
        if filename.startswith('.') or os.path.isdir(os.path.join(DOCS_DIR, filename)):
            continue

        file_path = os.path.join(DOCS_DIR, filename)
        if filename.endswith(".docx"):
            logger.info("  处理文档: %s", filename)
            full_text = parse_docx(file_path)
            chunks = split_text(full_text)
            logger.info("    文档长度: %s 字符, 切分为 %s 个chunk", len(full_text), len(chunks))
            emit("doc", len(all_points), 0, f"开始处理文档: {filename} ({len(chunks)} chunks)")

            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = point_id
                metadata = {
                    "id": chunk_id,
                    "doc_id": f"docx_{filename}_{chunk_idx}",
                    "source": filename,
                    "type": "text",
                    "content": chunk,
                    "diverse_questions": [],
                    "last_updated": today_str,
                    "created_at": created_at,
                    "chunk_index": chunk_idx,
                }

                # 原文加入向量索引
                vector = get_text_embedding(chunk)
                all_points.append(models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=metadata_to_payload(metadata),
                ))
                point_id += 1

                # 多样化改写：为当前chunk生成多样化问题并加入向量索引
                if USE_DIVERSE_REWRITE:
                    logger.info("      正在为 chunk %s 生成多样化问题...", chunk_id)
                    diverse_questions = generate_diverse_questions(chunk)
                    metadata["diverse_questions"] = diverse_questions
                    logger.info("      生成 %s 个问题", len(diverse_questions))
                    # 更新已加入的原文 point 的 payload（包含 diverse_questions）
                    all_points[-1].payload = metadata_to_payload(metadata)

                    for j, q_data in enumerate(diverse_questions):
                        question = q_data.get('question', '')
                        if not question.strip():
                            continue
                        q_metadata = {
                            "id": point_id,
                            "doc_id": f"docx_{filename}_{chunk_idx}",
                            "source": filename,
                            "type": "diverse_question",
                            "content": question,
                            "original_chunk_id": chunk_id,
                            "question_type": q_data.get('question_type', ''),
                            "perspective": q_data.get('perspective', ''),
                            "last_updated": today_str,
                            "created_at": created_at,
                            "chunk_index": chunk_idx,
                        }
                        combined_text = f"内容：{chunk} 问题：{question}"
                        q_vector = get_text_embedding(combined_text)
                        all_points.append(models.PointStruct(
                            id=point_id,
                            vector=q_vector,
                            payload=metadata_to_payload(q_metadata),
                        ))
                        point_id += 1

            emit("doc", len(all_points), 0, f"文档处理完成: {filename}，累计 {len(all_points)} 条")

    # 处理图片（不再需要OCR，多模态embedding已包含图片语义）
    logger.info("  处理图片...")
    image_list = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    emit("image", 0, len(image_list), f"开始处理 {len(image_list)} 张图片")
    for idx, img_filename in enumerate(image_list, 1):
        img_path = os.path.join(IMG_DIR, img_filename)
        logger.info("    - %s", img_filename)

        metadata = {
            "id": point_id,
            "doc_id": f"image_{img_filename}",
            "source": f"图片: {img_filename}",
            "type": "image",
            "path": img_path,
            "content": f"[图片] {img_filename}",
            "last_updated": today_str,
            "created_at": created_at,
            "chunk_index": 0,
        }

        vector = get_image_embedding(img_path)
        all_points.append(models.PointStruct(
            id=point_id,
            vector=vector,
            payload=metadata_to_payload(metadata),
        ))
        point_id += 1
        emit("image", idx, len(image_list), f"已处理图片 {idx}/{len(image_list)}: {img_filename}")

    # 处理视频
    logger.info("  处理视频...")
    emit("video", 0, len(VIDEO_KNOWLEDGE), f"开始处理 {len(VIDEO_KNOWLEDGE)} 个视频")
    for idx, video_info in enumerate(VIDEO_KNOWLEDGE, 1):
        logger.info("    - %s", video_info['description'])

        metadata = {
            "id": point_id,
            "doc_id": f"video_{video_info['description']}",
            "source": f"视频: {video_info['description']}",
            "type": "video",
            "url": video_info["url"],
            "description": video_info["description"],
            "content": f"[视频] {video_info['description']}",
            "last_updated": today_str,
            "created_at": created_at,
            "chunk_index": 0,
        }

        vector = get_video_embedding(video_info["url"])
        all_points.append(models.PointStruct(
            id=point_id,
            vector=vector,
            payload=metadata_to_payload(metadata),
        ))
        point_id += 1
        emit("video", idx, len(VIDEO_KNOWLEDGE), f"已处理视频 {idx}/{len(VIDEO_KNOWLEDGE)}: {video_info['description']}")

    # 批量写入 Qdrant
    if all_points:
        logger.info("向量维度: %s", len(all_points[0].vector))
        emit("save", 0, len(all_points), f"开始批量写入 Qdrant，共 {len(all_points)} 条")
        upsert_points(client, all_points, batch_size=100)
        emit("save", len(all_points), len(all_points), f"已写入 Qdrant 本地存储: {QDRANT_PATH}")

    # 统计
    count_result = client.count(collection_name=QDRANT_COLLECTION_NAME)
    total_count = count_result.count
    # 从 payload 统计类型
    type_counts = {"text": 0, "image": 0, "video": 0, "diverse_question": 0}
    for p in all_points:
        mt = p.payload.get("media_type", "text")
        type_counts[mt] = type_counts.get(mt, 0) + 1

    # 根据 vector 规模校验 HNSW m 是否合理
    suggested_m = suggest_hnsw_m(total_count)
    logger.info("HNSW m 校验: 实际 m=%s，规模 %s 条向量建议 m=%s",
                QDRANT_HNSW_M, total_count, suggested_m)
    if suggested_m != QDRANT_HNSW_M:
        logger.warning(
            "HNSW m=%s 与规模建议值 m=%s（向量数 %s）不一致，"
            "可调整 config.QDRANT_HNSW_M 后重建索引以优化召回与开销",
            QDRANT_HNSW_M, suggested_m, total_count,
        )

    logger.info("完成! 文本:%s, 图片:%s, 视频:%s, 多样化问题:%s（共 %s 条）",
                type_counts["text"], type_counts["image"],
                type_counts["video"], type_counts["diverse_question"], total_count)

    emit("done", total_count, 0, f"构建完成: 文本{type_counts['text']}, 图片{type_counts['image']}, "
        f"视频{type_counts['video']}, 多样化问题{type_counts['diverse_question']}")


def clean_index_files():
    """删除所有旧索引数据：清空 Qdrant 本地存储目录 + 删除 doc_hashes 文件

    不会触碰 disney_knowledge_base/ 源文档和 users.db。
    """
    deleted = []
    # 1. 删除 Qdrant 本地存储目录
    if os.path.exists(QDRANT_PATH):
        shutil.rmtree(QDRANT_PATH)
        deleted.append(QDRANT_PATH)
    # 2. 删除 doc_hashes 文件（增量检测哈希表）
    for f in INDEX_FILES_TO_CLEAN:
        if os.path.exists(f):
            os.remove(f)
            deleted.append(f)
    return deleted


if __name__ == "__main__":
    build_and_save()
