# -*- coding: utf-8 -*-
"""
app.agent.memory - 会话记忆读写

复用现有 SQLite conversations 表（由 app.api.ask 中 _save_conversation 写入），
为 Agent 推理循环提供短期记忆注入能力。

记忆分层：
- 短期记忆：当前会话的近期对话，从 conversations 表读取，注入 Agent prompt
- 长期记忆：沉淀的知识（knowledge_distill 产出），通过 knowledge_search 工具按需检索

本模块只做"读取"和"写入"，不做记忆压缩或遗忘策略——保持简单。
"""
import json
import logging
import sqlite3
from typing import Dict, List, Optional

from app.config import DB_FILE

logger = logging.getLogger(__name__)


def get_recent_messages(session_id: str = "", limit: int = 5) -> List[Dict[str, str]]:
    """读取指定会话的近期对话记录

    从 conversations 表按时间正序读取最近 limit 条记录，
    转换为 {"role": "user"/"assistant", "content": str} 格式。

    Args:
        session_id: 会话 ID；空字符串时读取全局最近的对话
        limit: 最多读取几条，默认 5

    Returns:
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        if session_id:
            cur.execute(
                "SELECT role, content, meta_json FROM conversations "
                "WHERE session_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            )
        else:
            cur.execute(
                "SELECT role, content, meta_json FROM conversations "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            )
        rows = cur.fetchall()
        conn.close()
    except Exception as e:
        logger.warning("读取对话记录失败: %s", e)
        return []

    # 反转为时间正序（SQL 取的是 DESC，需要 reverse）
    rows.reverse()
    messages = []
    for role, content, meta_json in rows:
        msg = {"role": role, "content": content}
        # 如果 assistant 消息有 meta（含图片/视频路径），附加上
        if meta_json and role == "assistant":
            try:
                meta = json.loads(meta_json)
                if meta.get("image_path"):
                    msg["content"] += f"\n[已展示图片: {meta['image_path']}]"
                if meta.get("video_url"):
                    msg["content"] += f"\n[已展示视频: {meta['video_url']}]"
            except (json.JSONDecodeError, TypeError):
                pass
        messages.append(msg)

    logger.info("get_recent_messages(session=%s, limit=%s) -> %s 条",
                session_id or "global", limit, len(messages))
    return messages


def save_message(session_id: str, role: str, content: str, meta: Optional[dict] = None):
    """把单条对话消息写入 conversations 表

    与 app.api.ask._save_conversation 逻辑一致，
    供 Agent API 路由调用，保证 Agent 问答也落库可追溯。

    Args:
        session_id: 会话 ID
        role: "user" 或 "assistant"
        content: 消息文本
        meta: 可选元数据（图片路径/视频URL/引用等），存为 JSON 字符串
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        meta_json = json.dumps(meta, ensure_ascii=False) if meta else None
        cur.execute(
            "INSERT INTO conversations (session_id, role, content, meta_json) "
            "VALUES (?, ?, ?, ?)",
            (session_id, role, content, meta_json),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning("保存对话记录失败: %s", e)


def upsert_session(session_id: str, visitor_id: str = ""):
    """upsert 会话生命周期记录

    与 app.api.ask._upsert_chat_session 逻辑一致，
    新会话插入、已有会话更新 last_active_at 并清空 ended_at。

    Args:
        session_id: 会话 ID
        visitor_id: 访客 ID
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO chat_sessions (session_id, visitor_id, started_at, last_active_at, ended_at) "
            "VALUES (?, ?, datetime('now', 'localtime'), datetime('now', 'localtime'), NULL) "
            "ON CONFLICT(session_id) DO UPDATE SET "
            "last_active_at = datetime('now', 'localtime'), "
            "ended_at = NULL",
            (session_id, visitor_id),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning("更新会话生命周期失败: %s", e)
