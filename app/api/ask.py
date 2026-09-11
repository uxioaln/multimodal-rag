# -*- coding: utf-8 -*-
"""
app.api.ask - 问答路由（Flask Blueprint）

包含：
- POST /api/ask -> 调用 RAG 检索并返回答案
            同步把本次问答写入 conversations 表（user + assistant 两条记录），
            并 upsert chat_sessions 表（更新 last_active_at，若 ended_at 已有则清空以支持重新开始对话）
"""
import json
import sqlite3
import logging

from flask import Blueprint, jsonify, request

from app.config import DB_FILE
from app.core.rag_service import rag_ask_api
from app.state import get_qdrant_client, get_metadata

ask_bp = Blueprint("ask", __name__)

logger = logging.getLogger(__name__)


def _save_conversation(session_id: str, role: str, content: str, meta: dict = None):
    """把单条对话消息写入 conversations 表（失败仅打印，不影响主流程）"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        # meta_json 存 JSON 字符串，便于后续按字符串检索
        meta_json = json.dumps(meta, ensure_ascii=False) if meta else None
        cur.execute(
            "INSERT INTO conversations (session_id, role, content, meta_json) VALUES (?, ?, ?, ?)",
            (session_id, role, content, meta_json)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        # 落库失败不能影响用户拿到答案，仅打印日志
        logger.warning("保存对话记录失败: %s", e)


def _upsert_chat_session(session_id: str, visitor_id: str = ""):
    """upsert 会话生命周期记录：
    - 新会话：插入一行，started_at/last_active_at = NOW，ended_at = NULL
    - 已有会话：更新 last_active_at，若 ended_at 非空则清空（用户重新进入对话会重开会话）
    失败仅打印日志，不影响主流程
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO chat_sessions (session_id, visitor_id, started_at, last_active_at, ended_at)
            VALUES (?, ?, datetime('now', 'localtime'), datetime('now', 'localtime'), NULL)
            ON CONFLICT(session_id) DO UPDATE SET
                last_active_at = datetime('now', 'localtime'),
                ended_at = NULL
        """, (session_id, visitor_id))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning("更新会话生命周期失败: %s", e)


@ask_bp.route("/api/ask", methods=["POST"])
def api_ask():
    """问答接口"""
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"code": 1, "message": "问题不能为空"}), 400
    # 优先使用前端传来的 session_id（每次浏览器开新页 = 一次会话），
    # 缺省时回退到 visitor_id（兼容旧版前端：sessionStorage 未启用时）
    session_id = (data.get("session_id") or data.get("visitor_id") or "").strip() or "anonymous"
    visitor_id = (data.get("visitor_id") or "").strip() or "anonymous"
    try:
        result = rag_ask_api(query, get_qdrant_client(), get_metadata())
        # 1) upsert 会话生命周期（让 knowledge_distill 知道这个 session 是进行中的）
        _upsert_chat_session(session_id, visitor_id)
        # 2) 落库：user 提问（无 meta） + assistant 回答（带 meta）
        _save_conversation(session_id, "user", query, None)
        _save_conversation(session_id, "assistant", result.get("answer", ""), {
            "image_path": result.get("image_path"),
            "video_url": result.get("video_url"),
            "references": result.get("references", [])
        })
        return jsonify({"code": 0, "data": result})
    except Exception as e:
        return jsonify({"code": 1, "message": f"查询失败: {str(e)}"}), 500
