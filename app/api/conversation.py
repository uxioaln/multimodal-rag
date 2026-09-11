# -*- coding: utf-8 -*-
"""
app.api.conversation - 对话会话生命周期路由（Flask Blueprint）

包含：
- POST /api/conversation/end -> 标记当前会话为已结束
            （前端在 beforeunload / pagehide 时调用，settle 到 DB 供 distillation 处理）

约束：
- 不做权限校验（任何访客都能结束自己的会话）
- 同一 session_id 重复调用是幂等的（只更新 ended_at 为 NULL 的行）
"""
import logging
import sqlite3

from flask import Blueprint, jsonify, request

from app.config import DB_FILE

conversation_bp = Blueprint("conversation", __name__)
logger = logging.getLogger(__name__)


@conversation_bp.route("/api/conversation/end", methods=["POST"])
def api_conversation_end():
    """结束当前对话会话

    请求体：{"session_id": "xxx"}
    行为：把 chat_sessions.ended_at 置为当前时间（仅当 ended_at IS NULL 时更新，幂等）
    返回：{"code": 0/1, "message": "..."}
    """
    data = request.get_json(silent=True) or {}
    session_id = (data.get("session_id") or "").strip()
    if not session_id:
        return jsonify({"code": 1, "message": "缺少 session_id"}), 400
    try:
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        # 只更新 ended_at IS NULL 的行：已结束的会话不会再次被改写结束时间
        cur.execute(
            "UPDATE chat_sessions SET ended_at = datetime('now', 'localtime') "
            "WHERE session_id = ? AND ended_at IS NULL",
            (session_id,),
        )
        conn.commit()
        affected = cur.rowcount
        conn.close()
        return jsonify({
            "code": 0,
            "message": "会话已结束" if affected else "会话不存在或已结束",
            "affected": affected,
        })
    except Exception as e:
        logger.warning("结束会话失败: %s", e)
        return jsonify({"code": 1, "message": f"结束会话失败: {e}"}), 500
