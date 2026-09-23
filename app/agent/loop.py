# -*- coding: utf-8 -*-
"""
app.agent.loop - ReAct 推理循环（Planner + Executor）

Agent 核心编排逻辑：
1. 构建系统 prompt，注入工具描述和当前对话上下文
2. 调用 LLM（带 function calling），获取回复或工具调用请求
3. 若 LLM 请求调用工具 -> 执行工具 -> 将结果注入对话 -> 回到步骤 2
4. 若 LLM 返回最终回复 -> 结束循环，返回答案

安全护栏：
- max_steps 上限（默认 8 步），防止无限循环
- 每步累计 token 预算（默认 8000 tokens），超限强制终止
- 循环检测：同一工具+相同参数调用 2 次 -> 告警并跳过
- 所有 LLM 调用走 cost_tracker，自动纳入成本统计

提供两种入口：
- agent_chat()      同步调用，返回最终结果 dict
- agent_chat_sse()  生成器，逐步 yield SSE 事件，供前端流式展示
"""
import json
import logging
from typing import Any, Dict, Iterator, List

from app.config import CHAT_MODEL
from app.core import query
from app.core.cost_tracker import CostTracker, tracked_chat_completion
from app.agent.tools import get_tool_schemas, execute_tool
from app.agent.memory import get_recent_messages
from app.agent.planner import plan_retrieval, format_plan_summary
from app.agent.validator import validate_answer, format_validation_summary

logger = logging.getLogger(__name__)

# ========== 可调参数 ==========

MAX_STEPS = 8          # 单次 Agent 对话最多循环步数
MAX_TOKENS = 8000      # 累计 token 预算上限
MAX_REPEAT = 2         # 同一工具+相同参数最多调用次数
VALIDATION_THRESHOLD = 0.8  # 准确性分数低于此值触发重新检索

SYSTEM_PROMPT = """你是一个迪士尼乐园智能客服 Agent。

你可以调用以下工具来帮助用户：
1. knowledge_search - 搜索知识库文本（门票、酒店、攻略、规则等）
2. image_search - 搜索相关图片（海报、照片等）
3. video_search - 搜索相关视频
4. get_conversation_history - 读取近期对话记录（理解上下文）

工作原则：
- 先用 knowledge_search 检索知识，再基于检索结果回答，不要凭空编造
- 当用户提到"图片/海报/照片"时，调用 image_search
- 当用户提到"视频/录像"时，调用 video_search
- 当需要回顾之前聊了什么时，调用 get_conversation_history
- 回答要准确、简洁，引用知识来源
- 完成回答后直接回复用户，不要再调用工具"""


# ========== 核心循环 ==========

def _build_messages(
    user_query: str,
    session_id: str = "",
    history_limit: int = 4,
) -> List[Dict[str, str]]:
    """构建初始 messages 数组

    包含：system prompt + 近期对话历史（短期记忆） + 当前用户问题

    Args:
        user_query: 当前用户问题
        session_id: 会话 ID，用于读取历史对话
        history_limit: 读取几条历史，默认 4 条（2 轮对话）

    Returns:
        OpenAI messages 格式的列表
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # 读取短期记忆：当前会话的近期对话
    if session_id:
        history = get_recent_messages(session_id=session_id, limit=history_limit)
        messages.extend(history)

    messages.append({"role": "user", "content": user_query})
    return messages


def _detect_loop(
    tool_calls_history: List[Dict[str, str]],
    name: str,
    arguments_str: str,
) -> bool:
    """循环检测：同一工具+相同参数是否已调用过 MAX_REPEAT 次

    Args:
        tool_calls_history: 历史调用记录 [{"name", "arguments"}]
        name: 当前工具名
        arguments_str: 当前参数 JSON 字符串

    Returns:
        True 表示检测到循环（应跳过本次调用）
    """
    count = sum(
        1 for tc in tool_calls_history
        if tc["name"] == name and tc["arguments"] == arguments_str
    )
    return count >= MAX_REPEAT


def agent_chat(
    user_query: str,
    session_id: str = "",
    visitor_id: str = "",
) -> Dict[str, Any]:
    """Agent 对话入口（同步）

    执行完整的 ReAct 推理循环，返回最终结果。

    Args:
        user_query: 用户问题
        session_id: 会话 ID（用于短期记忆）
        visitor_id: 访客 ID

    Returns:
        {
            "answer": str,           # 最终回答
            "image_path": str|None,  # 匹配的图片路径
            "video_url": str|None,   # 匹配的视频 URL
            "references": list,      # 引用的知识片段
            "steps": list,           # 推理步骤记录
            "cost": dict,            # 成本统计
            "plan": dict,            # 检索计划
        }
    """
    tracker = CostTracker()
    tracker.begin()

    # 0) 检索规划：在进入 ReAct 循环前，先用 LLM 分析问题并生成检索计划
    plan = plan_retrieval(user_query, tracker=tracker)
    plan_steps = plan.get("steps", [])
    if len(plan_steps) == 1:
        print("单步检索")
    else:
        print("多步迭代检索")
    logger.info("检索规划:\n%s", format_plan_summary(plan))

    messages = _build_messages(user_query, session_id)
    tool_schemas = get_tool_schemas()
    steps: List[Dict[str, Any]] = []
    tool_calls_history: List[Dict[str, str]] = []

    # 最终结果字段，从工具返回中提取
    final_image_path = None
    final_video_url = None
    references: List[Dict[str, Any]] = []

    # 验证与重试状态
    retry_attempted = False
    validation_result = None

    for step_idx in range(MAX_STEPS):
        step_info: Dict[str, Any] = {"step": step_idx + 1}

        # 调用 LLM（带工具定义）
        response = tracked_chat_completion(
            client=query.client,
            model=CHAT_MODEL,
            messages=messages,
            tools=tool_schemas,
            tracker=tracker,
            source="Agent问答",
        )
        choice = response.choices[0]
        msg = choice.message

        # 检查 token 预算
        total_tokens = _get_total_tokens(tracker)
        if total_tokens > MAX_TOKENS:
            step_info["action"] = "budget_exceeded"
            step_info["total_tokens"] = total_tokens
            steps.append(step_info)
            logger.warning("Agent 因 token 预算超限终止 (已用 %s tokens)", total_tokens)
            break

        # 情况 A：LLM 请求调用工具
        if msg.tool_calls:
            # 先把 assistant 的 tool_calls 消息加入对话
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}
                arguments_str = tc.function.arguments

                # 循环检测
                if _detect_loop(tool_calls_history, tool_name, arguments_str):
                    logger.warning("检测到循环调用: %s(%s)，跳过", tool_name, arguments_str)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"error": "检测到循环调用，已跳过"}, ensure_ascii=False),
                    })
                    step_info["action"] = "loop_detected"
                    steps.append(step_info)
                    continue

                tool_calls_history.append({"name": tool_name, "arguments": arguments_str})

                # 执行工具
                logger.info("Step %s -> 调用工具: %s(%s)", step_idx + 1, tool_name, arguments_str)
                result = execute_tool(tool_name, arguments)

                # 提取媒体结果
                if tool_name == "image_search" and result.get("found"):
                    final_image_path = result.get("image_path")
                if tool_name == "video_search" and result.get("found"):
                    final_video_url = result.get("video_url")
                if tool_name == "knowledge_search":
                    references = result.get("results", [])

                # 工具结果注入对话
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })

                step_info.setdefault("tool_calls", []).append({
                    "name": tool_name,
                    "arguments": arguments,
                    "result_preview": json.dumps(result, ensure_ascii=False)[:200],
                })

            step_info.setdefault("action", "tool_executed")
            steps.append(step_info)
            # 继续下一轮循环，让 LLM 根据工具结果决定下一步

        # 情况 B：LLM 返回最终回复（无工具调用）
        elif msg.content:
            step_info["action"] = "final_answer"
            steps.append(step_info)

            # 1) 验证答案质量
            validation_result = validate_answer(references, msg.content, tracker=tracker)
            logger.info("答案验证:\n%s", format_validation_summary(validation_result))

            # 2) 准确性低于阈值且未重试过 -> 触发重新检索
            acc_score = validation_result["accuracy"]["score"]
            if acc_score < VALIDATION_THRESHOLD and not retry_attempted:
                print("验证失败，触发重新检索...")
                retry_attempted = True
                logger.warning("准确性 %.1f < %.1f，触发重试", acc_score, VALIDATION_THRESHOLD)

                # 用 plan 中第一个 step 的 target 作为替代关键词重新检索
                retry_keyword = user_query
                if plan_steps:
                    retry_keyword = plan_steps[0].get("target", user_query)
                retry_result = execute_tool("knowledge_search", {"query_str": retry_keyword, "k": 5})
                new_refs = retry_result.get("results", [])
                if new_refs:
                    references = new_refs
                    # 注入新的检索结果，要求 LLM 基于新结果重新回答
                    retry_context = "请基于以下重新检索到的知识回答问题，确保答案忠实于原文：\n"
                    for i, ref in enumerate(new_refs, 1):
                        retry_context += f"背景知识{i} (来源: {ref.get('source','')}):\n{ref.get('content','')}\n\n"
                    messages.append({"role": "user", "content": retry_context + user_query})
                    # 继续循环让 LLM 重新生成
                    continue
                # 无新检索结果则直接返回（兜底）

            # 3) 验证通过或已重试过 -> 返回最终结果
            tracker.finish()
            cost = tracker.get_summary()
            logger.info("Agent 完成，共 %s 步，token: %s，成本: %s 元",
                        len(steps), cost["total_tokens"], cost["estimated_cost"])

            return {
                "answer": msg.content,
                "image_path": final_image_path,
                "video_url": final_video_url,
                "references": references,
                "steps": steps,
                "plan": plan,
                "validation": validation_result,
                "retried": retry_attempted,
                "cost": {
                    "call_count": cost["call_count"],
                    "total_tokens": cost["total_tokens"],
                    "estimated_cost": cost["estimated_cost"],
                    "duration_seconds": cost["duration_seconds"],
                },
            }

        else:
            # 兜底：既无 tool_calls 也无 content，终止
            step_info["action"] = "empty_response"
            steps.append(step_info)
            logger.warning("Agent 收到空回复，终止")
            break

    # 循环用尽或预算超限，生成兜底回复
    tracker.finish()
    cost = tracker.get_summary()
    fallback = "抱歉，我在处理您的问题时遇到了一些限制，请尝试简化问题后重新提问。"

    return {
        "answer": fallback,
        "image_path": final_image_path,
        "video_url": final_video_url,
        "references": references,
        "steps": steps,
        "plan": plan,
        "validation": validation_result,
        "retried": retry_attempted,
        "cost": {
            "call_count": cost["call_count"],
            "total_tokens": cost["total_tokens"],
            "estimated_cost": cost["estimated_cost"],
            "duration_seconds": cost["duration_seconds"],
        },
    }


def agent_chat_sse(
    user_query: str,
    session_id: str = "",
    visitor_id: str = "",
) -> Iterator[str]:
    """Agent 对话入口（SSE 流式）

    与 agent_chat 逻辑一致，但逐步 yield SSE 事件字符串，
    供前端通过 EventSource 实时展示推理过程。

    Yield 格式：data: {"type": "...", "content": "..."}\\n\\n
    事件类型：
    - step_start    每步开始
    - tool_call     工具调用信息
    - tool_result   工具执行结果
    - final_answer  最终回答
    - done          结束（含成本统计）

    Yields:
        SSE 格式的字符串
    """
    tracker = CostTracker()
    tracker.begin()

    # 0) 检索规划：在进入 ReAct 循环前，先用 LLM 分析问题并生成检索计划
    plan = plan_retrieval(user_query, tracker=tracker)
    plan_steps = plan.get("steps", [])
    if len(plan_steps) == 1:
        print("单步检索")
    else:
        print("多步迭代检索")
    logger.info("检索规划:\n%s", format_plan_summary(plan))
    yield _sse("plan", {"plan": plan})

    messages = _build_messages(user_query, session_id)
    tool_schemas = get_tool_schemas()
    steps: List[Dict[str, Any]] = []
    tool_calls_history: List[Dict[str, str]] = []

    final_image_path = None
    final_video_url = None
    references: List[Dict[str, Any]] = []

    for step_idx in range(MAX_STEPS):
        yield _sse("step_start", {"step": step_idx + 1, "max_steps": MAX_STEPS})

        response = tracked_chat_completion(
            client=query.client,
            model=CHAT_MODEL,
            messages=messages,
            tools=tool_schemas,
            tracker=tracker,
            source="Agent问答",
        )
        choice = response.choices[0]
        msg = choice.message

        # token 预算检查
        total_tokens = _get_total_tokens(tracker)
        if total_tokens > MAX_TOKENS:
            yield _sse("budget_exceeded", {"total_tokens": total_tokens, "limit": MAX_TOKENS})
            logger.warning("Agent SSE: token 预算超限 (%s)", total_tokens)
            break

        if msg.tool_calls:
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}
                arguments_str = tc.function.arguments

                if _detect_loop(tool_calls_history, tool_name, arguments_str):
                    yield _sse("loop_detected", {"tool": tool_name, "arguments": arguments})
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"error": "循环调用已跳过"}, ensure_ascii=False),
                    })
                    continue

                tool_calls_history.append({"name": tool_name, "arguments": arguments_str})

                yield _sse("tool_call", {"tool": tool_name, "arguments": arguments})

                result = execute_tool(tool_name, arguments)

                if tool_name == "image_search" and result.get("found"):
                    final_image_path = result.get("image_path")
                if tool_name == "video_search" and result.get("found"):
                    final_video_url = result.get("video_url")
                if tool_name == "knowledge_search":
                    references = result.get("results", [])

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })

                yield _sse("tool_result", {
                    "tool": tool_name,
                    "result": result,
                })

        elif msg.content:
            tracker.finish()
            cost = tracker.get_summary()
            yield _sse("final_answer", {
                "answer": msg.content,
                "image_path": final_image_path,
                "video_url": final_video_url,
                "references": references,
                "plan": plan,
            })
            yield _sse("done", {
                "steps": len(steps) + 1,
                "total_tokens": cost["total_tokens"],
                "estimated_cost": cost["estimated_cost"],
                "duration_seconds": cost["duration_seconds"],
            })
            return

        else:
            yield _sse("empty_response", {"step": step_idx + 1})
            break

    # 兜底
    tracker.finish()
    cost = tracker.get_summary()
    yield _sse("final_answer", {
        "answer": "抱歉，我在处理您的问题时遇到了一些限制，请尝试简化问题后重新提问。",
        "image_path": final_image_path,
        "video_url": final_video_url,
        "references": references,
        "plan": plan,
    })
    yield _sse("done", {
        "steps": len(steps),
        "total_tokens": cost["total_tokens"],
        "estimated_cost": cost["estimated_cost"],
        "duration_seconds": cost["duration_seconds"],
    })


# ========== 辅助函数 ==========

def _get_total_tokens(tracker: CostTracker) -> int:
    """从 CostTracker 获取累计 total_tokens"""
    summary = tracker.get_summary()
    return summary.get("total_tokens", 0)


def _sse(event_type: str, data: Dict[str, Any]) -> str:
    """格式化为 SSE 事件字符串

    Args:
        event_type: 事件类型
        data: 事件数据

    Returns:
        "data: {...}\\n\\n" 格式的字符串
    """
    payload = {"type": event_type, **data}
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
