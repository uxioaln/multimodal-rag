# -*- coding: utf-8 -*-
"""
app.agent.tools - 工具注册表（简化版 MCP）

把现有检索能力 + OCR 能力封装为 LLM 可自主调用的独立函数。
Agent 可根据检索计划（planner 输出）选择调用合适的工具。

工具清单：
1. knowledge_search  - 语义检索知识库文本（等同 search_documents）
2. image_search      - 检索与查询相关的图片（等同 search_images）
3. ocr_image         - 对指定图片做 OCR 文字识别（AGICTO qwen3-vl-8b-instruct）
4. video_search      - 检索与查询相关的视频
5. get_conversation_history - 读取当前会话的近期对话（短期记忆）

设计原则：
- 每个工具是窄职责、job-specific 的函数，不做"万能查询"
- 输入/输出均有明确结构，返回 dict 供 LLM 理解
- 复用 app.core.query 的 embedding / search_vectors，不重复实现检索逻辑
"""
import logging
import os
from typing import Any, Dict, List

from app.config import MEDIA_DISTANCE_THRESHOLD, KB_DIR
from app.core import query

logger = logging.getLogger(__name__)

# OCR 使用的视觉语言模型（AGICTO 平台，qwen3-vl-8b-instruct 性价比高）
OCR_MODEL = "qwen3-vl-8b-instruct"


# ========== 工具实现（纯函数，依赖通过参数注入） ==========

def _knowledge_search(query_str: str, k: int = 3) -> Dict[str, Any]:
    """语义检索知识库文本片段

    复用 query.get_text_embedding + query.search_vectors，
    返回 top-k 文本类型的结果（排除已删除/图片/视频）。

    Args:
        query_str: 用户问题
        k: 返回条数，默认 3

    Returns:
        {"results": [{"source", "content", "similarity"}, ...]}
    """
    qdrant_client = _get_qdrant_client()
    if qdrant_client is None:
        return {"error": "Qdrant 客户端未初始化", "results": []}

    query_vec = query.get_text_embedding(query_str)
    raw_results = query.search_vectors(query_vec, qdrant_client)

    # 转换为统一格式，过滤非文本和已删除条目
    text_results = []
    for dist, _, m in raw_results:
        if m.get("deleted", False):
            continue
        if m.get("type") not in ("text", "diverse_question"):
            continue
        sim = query.distance_to_similarity(dist)
        text_results.append({
            "source": m.get("source", ""),
            "content": m.get("content", ""),
            "similarity": round(float(sim), 4),
        })
        if len(text_results) >= k:
            break

    logger.info("knowledge_search('%s') -> %s 条文本结果", query_str, len(text_results))
    return {"results": text_results}


def _image_search(query_str: str) -> Dict[str, Any]:
    """检索与查询语义最接近的图片

    复用 query.get_text_embedding + query.search_vectors，
    筛选 type=image 且 distance < MEDIA_DISTANCE_THRESHOLD 的结果。

    Returns:
        {"found": bool, "image_path": str | null, "similarity": float}
    """
    qdrant_client = _get_qdrant_client()
    if qdrant_client is None:
        return {"error": "Qdrant 客户端未初始化", "found": False}

    query_vec = query.get_text_embedding(query_str)
    raw_results = query.search_vectors(query_vec, qdrant_client)

    image_results = []
    for dist, _, m in raw_results:
        if m.get("deleted", False):
            continue
        if m.get("type") != "image":
            continue
        if dist < MEDIA_DISTANCE_THRESHOLD:
            sim = query.distance_to_similarity(dist)
            image_results.append({
                "image_path": m.get("path", ""),
                "similarity": round(float(sim), 4),
                "distance": round(float(dist), 4),
            })

    if not image_results:
        logger.info("image_search('%s') -> 无匹配图片", query_str)
        return {"found": False, "image_path": None}

    image_results.sort(key=lambda x: x["distance"])
    best = image_results[0]
    logger.info("image_search('%s') -> 匹配图片: %s", query_str, best["image_path"])
    return {"found": True, **best}


def _video_search(query_str: str) -> Dict[str, Any]:
    """检索与查询语义最接近的视频

    逻辑同 _image_search，筛选 type=video。
    """
    qdrant_client = _get_qdrant_client()
    if qdrant_client is None:
        return {"error": "Qdrant 客户端未初始化", "found": False}

    query_vec = query.get_text_embedding(query_str)
    raw_results = query.search_vectors(query_vec, qdrant_client)

    video_results = []
    for dist, _, m in raw_results:
        if m.get("deleted", False):
            continue
        if m.get("type") != "video":
            continue
        if dist < MEDIA_DISTANCE_THRESHOLD:
            sim = query.distance_to_similarity(dist)
            video_results.append({
                "video_url": m.get("url", ""),
                "similarity": round(float(sim), 4),
                "distance": round(float(dist), 4),
            })

    if not video_results:
        logger.info("video_search('%s') -> 无匹配视频", query_str)
        return {"found": False, "video_url": None}

    video_results.sort(key=lambda x: x["distance"])
    best = video_results[0]
    logger.info("video_search('%s') -> 匹配视频: %s", query_str, best["video_url"])
    return {"found": True, **best}


def _get_conversation_history(limit: int = 5) -> Dict[str, Any]:
    """读取当前会话的近期对话记录（短期记忆）

    延迟到 memory.py 中实现，此处由 memory 模块提供函数。
    """
    from app.agent.memory import get_recent_messages
    messages = get_recent_messages(limit=limit)
    return {"messages": messages}


def _ocr_image(image_path: str) -> Dict[str, Any]:
    """对指定图片做 OCR 文字识别

    使用 AGICTO qwen3-vl-8b-instruct 视觉语言模型，提取图片中的文字内容。
    支持绝对路径和相对路径（相对路径基于知识库目录 KB_DIR）。

    Args:
        image_path: 图片路径（绝对或相对路径）

    Returns:
        {"found_text": bool, "text": str, "image_path": str}
    """
    import base64

    # 处理相对路径：基于知识库目录解析
    if not os.path.isabs(image_path):
        image_path = os.path.join(KB_DIR, image_path)

    if not os.path.exists(image_path):
        logger.warning("ocr_image: 图片不存在: %s", image_path)
        return {"found_text": False, "text": "", "image_path": image_path, "error": "图片不存在"}

    # 将本地图片转为 base64 data URI（AGICTO OpenAI 兼容接口不支持 file:// 本地路径）
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    if ext == "jpg":
        ext = "jpeg"
    image_data = f"data:image/{ext};base64,{base64_image}"

    try:
        response = query.client.chat.completions.create(
            model=OCR_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data}},
                    {"type": "text", "text": "请识别并提取这张图片中的所有文字内容，按原文输出。如果没有文字，回复'无文字'。"},
                ],
            }],
        )
        text = response.choices[0].message.content.strip()
        logger.info("ocr_image('%s') -> 提取文字 %s 字", image_path, len(text))
        return {"found_text": len(text) > 0 and text != "无文字", "text": text, "image_path": image_path}

    except Exception as e:
        logger.exception("ocr_image 异常")
        return {"found_text": False, "text": "", "image_path": image_path, "error": str(e)}


# ========== 依赖获取（从 state 全局状态） ==========

def _get_qdrant_client():
    """从 app.state 获取 Qdrant 客户端（懒导入避免循环依赖）"""
    from app.state import get_qdrant_client
    return get_qdrant_client()


# ========== 工具注册表：OpenAI function calling 格式 ==========

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "knowledge_search",
            "description": "搜索迪士尼知识库，返回与用户问题最相关的文本知识片段。适用于回答关于门票、酒店、游玩攻略、退款规则等问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_str": {
                        "type": "string",
                        "description": "搜索关键词或问题，例如'门票退款流程'",
                    },
                    "k": {
                        "type": "integer",
                        "description": "返回结果条数，默认 3",
                        "default": 3,
                    },
                },
                "required": ["query_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "image_search",
            "description": "搜索与用户问题语义最接近的图片（如活动海报、照片）。当用户想看图片/海报时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_str": {
                        "type": "string",
                        "description": "描述想看的图片内容，例如'万圣节活动海报'",
                    },
                },
                "required": ["query_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "video_search",
            "description": "搜索与用户问题语义最接近的视频。当用户想看视频/录像时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_str": {
                        "type": "string",
                        "description": "描述想看的视频内容，例如'汽车剐蹭监控视频'",
                    },
                },
                "required": ["query_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ocr_image",
            "description": "对指定图片做 OCR 文字识别，提取图片中的文字内容。当需要从图片/海报中读取文字时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "图片路径，可以是绝对路径或相对路径（相对知识库目录），例如'images/1-聚在一起说奇妙.jpg'",
                    },
                },
                "required": ["image_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_conversation_history",
            "description": "读取当前会话中近期的对话记录，用于理解上下文或追指代关系。当需要回顾之前聊了什么时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "读取最近几条对话，默认 5",
                        "default": 5,
                    },
                },
                "required": [],
            },
        },
    },
]

# 工具名 -> 可调用函数的映射
TOOL_HANDLERS = {
    "knowledge_search": _knowledge_search,
    "image_search": _image_search,
    "video_search": _video_search,
    "ocr_image": _ocr_image,
    "get_conversation_history": _get_conversation_history,
}


def get_tool_schemas() -> List[Dict[str, Any]]:
    """返回 OpenAI function calling 格式的工具 schema 列表"""
    return TOOL_SCHEMAS


def execute_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """执行指定工具，返回结果 dict

    Args:
        name: 工具名称（需在 TOOL_HANDLERS 中注册）
        arguments: 工具参数（已从 JSON 解析）

    Returns:
        工具返回的 dict；若工具不存在或执行异常，返回 error 字段
    """
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return {"error": f"未知工具: {name}"}
    try:
        result = handler(**arguments)
        return result
    except Exception as e:
        logger.exception("工具 %s 执行失败", name)
        return {"error": f"工具 {name} 执行失败: {str(e)}"}
