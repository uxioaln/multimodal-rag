# -*- coding: utf-8 -*-
"""
app.api.agent_chat - Agent 问答路由（Flask Blueprint）

提供两个端点：
- POST /api/agent/chat   同步调用 Agent，返回最终结果
- POST /api/agent/stream SSE 流式输出 Agent 推理过程

与 /api/ask（普通 RAG）并存，支持面试演示对比。
"""
import logging

from flask import Blueprint, Response, jsonify, request

from app.agent.loop import agent_chat, agent_chat_sse
from app.agent.memory import save_message, upsert_session

agent_bp = Blueprint("agent", __name__)

logger = logging.getLogger(__name__)


@agent_bp.route("/api/agent/chat", methods=["POST"])
def api_agent_chat():
    """Agent 同步问答接口

    请求体：{"query": "门票退款流程", "session_id": "xxx", "visitor_id": "yyy"}
    返回：{"code": 0, "data": {answer, image_path, video_url, references, steps, cost}}
    """
    data = request.get_json(silent=True) or {}
    user_query = (data.get("query") or "").strip()
    if not user_query:
        return jsonify({"code": 1, "message": "问题不能为空"}), 400

    session_id = (data.get("session_id") or data.get("visitor_id") or "").strip() or "anonymous"
    visitor_id = (data.get("visitor_id") or "").strip() or "anonymous"

    try:
        # 1) 更新会话生命周期
        upsert_session(session_id, visitor_id)
        # 2) 落库用户提问
        save_message(session_id, "user", user_query)
        # 3) 执行 Agent 推理循环
        result = agent_chat(user_query, session_id=session_id, visitor_id=visitor_id)
        # 4) 落库 Agent 回答
        save_message(session_id, "assistant", result.get("answer", ""), {
            "image_path": result.get("image_path"),
            "video_url": result.get("video_url"),
            "references": result.get("references", []),
            "steps": result.get("steps", []),
            "cost": result.get("cost", {}),
        })

        return jsonify({"code": 0, "data": result})
    except Exception as e:
        logger.exception("Agent 问答失败")
        return jsonify({"code": 1, "message": f"Agent 问答失败: {str(e)}"}), 500


@agent_bp.route("/api/agent/stream", methods=["POST"])
def api_agent_stream():
    """Agent SSE 流式问答接口

    请求体：{"query": "...", "session_id": "xxx", "visitor_id": "yyy"}
    返回：SSE 流，事件类型见 agent_chat_sse 文档
    """
    data = request.get_json(silent=True) or {}
    user_query = (data.get("query") or "").strip()
    if not user_query:
        return jsonify({"code": 1, "message": "问题不能为空"}), 400

    session_id = (data.get("session_id") or data.get("visitor_id") or "").strip() or "anonymous"
    visitor_id = (data.get("visitor_id") or "").strip() or "anonymous"

    # 更新会话 + 落库用户提问（在生成器外部执行，确保立即写入）
    upsert_session(session_id, visitor_id)
    save_message(session_id, "user", user_query)

    def generate():
        """SSE 生成器：逐事件 yield，完成后落库 Agent 回答"""
        final_answer = ""
        final_image = None
        final_video = None
        final_refs = []
        try:
            for sse_chunk in agent_chat_sse(user_query, session_id=session_id, visitor_id=visitor_id):
                yield sse_chunk
                # 从 SSE 数据中提取最终结果用于落库
                try:
                    import json
                    line = sse_chunk.strip()
                    if line.startswith("data: "):
                        payload = json.loads(line[6:])
                        if payload.get("type") == "final_answer":
                            final_answer = payload.get("answer", "")
                            final_image = payload.get("image_path")
                            final_video = payload.get("video_url")
                            final_refs = payload.get("references", [])
                except (json.JSONDecodeError, IndexError):
                    pass
        except Exception as e:
            logger.exception("Agent SSE 流式失败")
            import json
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
        finally:
            # 落库 Agent 回答
            if final_answer:
                save_message(session_id, "assistant", final_answer, {
                    "image_path": final_image,
                    "video_url": final_video,
                    "references": final_refs,
                })

    return Response(generate(), mimetype="text/event-stream")
