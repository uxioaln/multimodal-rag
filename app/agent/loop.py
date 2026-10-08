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
import math
import re
from typing import Any, Dict, Iterator, List, Optional

from app.config import CHAT_MODEL, MIN_RELEVANCE_SCORE, MIN_RELEVANCE_COUNT
from app.core import query
from app.core.cost_tracker import CostTracker, tracked_chat_completion
from app.agent.tools import get_tool_schemas, execute_tool
from app.agent.memory import get_recent_messages
from app.agent.planner import plan_retrieval, format_plan_summary
from app.agent.validator import validate_answer, format_validation_summary

logger = logging.getLogger(__name__)

# ========== 可调参数 ==========

MAX_STEPS = 8          # 单次 Agent 对话最多循环步数
MAX_TOKENS = 40000     # 累计 token 预算上限（复杂查询多步推理+重检索+验证需更大预算）
MAX_REPEAT = 2         # 同一工具+相同参数最多调用次数
MAX_TOOL_CALLS = 6     # 单次 Agent 对话中工具调用总次数上限
QUERY_SIM_THRESHOLD = 0.72  # 语义级循环检测阈值（经 DashScope embedding 实测校准）
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
- 回答要准确、简洁，必须引用知识来源
- 完成回答后直接回复用户，不要再调用工具
- 重要限制：你最多只能调用 3 次 knowledge_search。如果检索后仍无法回答，请直接告知用户"未找到相关信息"，不要反复换关键词搜索

引用规范（必须遵守）：
- 检索结果会以编号形式呈现：背景知识[1]、背景知识[2] 等
- 回答中每个事实陈述的句末必须标注引用编号，如：门票退款需在7天内申请[1]
- 引用编号必须真实对应检索结果中的编号，不得编造不存在的编号
- 如检索结果不足以支撑回答，直接说明"未找到相关信息"，不要编造无引用的内容"""


# ========== 核心循环 ==========

def _build_messages(
    user_query: str,
    session_id: str = "",
    history_limit: int = 4,
    plan: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    """构建初始 messages 数组

    包含：system prompt + 检索计划指引（可选） + 近期对话历史（短期记忆） + 当前用户问题

    Args:
        user_query: 当前用户问题
        session_id: 会话 ID，用于读取历史对话
        history_limit: 读取几条历史，默认 4 条（2 轮对话）
        plan: 检索计划（plan_retrieval 输出）；传入时将步骤指引注入 system 消息，
              引导 Agent 按计划调用对应模态工具（image_search/video_search 定向检索）

    Returns:
        OpenAI messages 格式的列表
    """
    system_content = SYSTEM_PROMPT
    # 注入检索计划指引：引导 Agent 按计划步骤调用对应模态工具
    if plan and plan.get("steps"):
        plan_lines = ["", "## 检索计划指引（请按以下步骤调用对应工具，不要跳过媒体步骤）"]
        for i, step in enumerate(plan["steps"], 1):
            modality = step.get("modality", "text")
            target = step.get("target", "")
            plan_lines.append(
                f"  Step {i}: modality={modality} | target={target} | reason={step.get('reason', '')}"
            )
        plan_lines.append(
            "注意：modality=image 的步骤必须调用 image_search，modality=video 的步骤必须调用 video_search，"
            "modality=text 的步骤调用 knowledge_search，不要用 knowledge_search 替代 image/video 步骤。"
        )
        system_content = system_content + "\n".join(plan_lines)

    messages = [{"role": "system", "content": system_content}]

    # 读取短期记忆：当前会话的近期对话
    if session_id:
        history = get_recent_messages(session_id=session_id, limit=history_limit)
        messages.extend(history)

    messages.append({"role": "user", "content": user_query})
    return messages


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """计算两个 embedding 向量的余弦相似度（越大越相似）"""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _get_search_embedding(tool_name: str, arguments: Dict[str, Any]) -> Optional[List[float]]:
    """为搜索类工具提取 query_str 并计算 embedding，供语义级循环检测使用

    非搜索类工具或缺少 query_str 时返回 None。
    embedding 接口异常时返回 None，检测自动退化为精确匹配，不阻断主流程。
    """
    if tool_name not in ("knowledge_search", "image_search", "video_search"):
        return None
    query_str = arguments.get("query_str", "")
    if not query_str:
        return None
    try:
        return query.get_text_embedding(query_str)
    except Exception:
        logger.warning("语义循环检测: query embedding 计算失败，退化为精确匹配", exc_info=True)
        return None


def _detect_loop(
    tool_calls_history: List[Dict[str, Any]],
    name: str,
    arguments_str: str,
    total_tool_calls: int,
    query_embedding: Optional[List[float]] = None,
) -> bool:
    """循环检测：精确重复 + 语义重复 + 次数上限，任一命中即视为循环

    Args:
        tool_calls_history: 历史调用记录 [{"name", "arguments", "embedding"}]
        name: 当前工具名
        arguments_str: 当前参数 JSON 字符串
        total_tool_calls: 当前已执行的工具调用总数
        query_embedding: 当前搜索关键词的 embedding（仅搜索类工具传入）

    Returns:
        True 表示检测到循环或超限（应跳过本次调用）
    """
    # 检测同一工具+相同参数的重复调用
    same_param_count = sum(
        1 for tc in tool_calls_history
        if tc["name"] == name and tc["arguments"] == arguments_str
    )
    if same_param_count >= MAX_REPEAT:
        return True

    # 语义级检测：同一工具的搜索关键词与历史关键词高度相似
    # 解决“退款流程”与“门票退款”这类换词重搜绕过精确匹配的问题
    if query_embedding:
        for tc in tool_calls_history:
            if tc["name"] != name:
                continue
            hist_embedding = tc.get("embedding")
            if not hist_embedding:
                continue
            sim = _cosine_similarity(query_embedding, hist_embedding)
            if sim >= QUERY_SIM_THRESHOLD:
                logger.warning(
                    "语义级循环: %s 当前搜索词与历史搜索词相似度 %.3f >= %.2f，判定为换词重搜",
                    name, sim, QUERY_SIM_THRESHOLD,
                )
                return True

    # 检测同一工具（不限参数）是否被调用过多次
    same_tool_count = sum(
        1 for tc in tool_calls_history
        if tc["name"] == name
    )
    if name == "knowledge_search" and same_tool_count >= 3:
        return True
    if same_tool_count >= 4:
        return True

    # 检测总工具调用次数是否超限
    if total_tool_calls >= MAX_TOOL_CALLS:
        return True

    return False


def _check_relevance(references: List[Dict[str, Any]]) -> bool:
    """检查检索结果相关性是否达标

    判定标准：结果数 >= MIN_RELEVANCE_COUNT 且最高 similarity >= MIN_RELEVANCE_SCORE
    不达标时返回 False，引导 Agent 拒答而非编造。

    Args:
        references: knowledge_search 返回的结果列表

    Returns:
        True 表示相关性达标，False 表示应拒答
    """
    if not references or len(references) < MIN_RELEVANCE_COUNT:
        return False
    max_sim = max((r.get("similarity", 0.0) for r in references), default=0.0)
    return max_sim >= MIN_RELEVANCE_SCORE


def _check_citations(answer: str, references: List[Dict[str, Any]]) -> Dict[str, Any]:
    """程序化引用校验：检查答案中的引用编号是否合法

    规则：
    1. 从答案中提取所有 [n] 格式的引用编号
    2. 检查引用编号是否在合法范围内（1 ~ len(references)）
    3. 返回校验结果，含非法引用列表

    Args:
        answer: LLM 生成的答案文本
        references: 检索到的知识片段列表

    Returns:
        {"passed": bool, "valid_citations": [1,2], "invalid_citations": [3], "has_citations": bool}
    """
    # 提取答案中所有 [n] 格式的引用编号
    cited_nums = set()
    for m in re.finditer(r'\[(\d+)\]', answer):
        cited_nums.add(int(m.group(1)))

    valid_range = set(range(1, len(references) + 1)) if references else set()
    valid_citations = cited_nums & valid_range
    invalid_citations = cited_nums - valid_range

    return {
        "passed": len(invalid_citations) == 0,
        "valid_citations": sorted(valid_citations),
        "invalid_citations": sorted(invalid_citations),
        "has_citations": len(cited_nums) > 0,
    }


def _build_abstention_answer(references: List[Dict[str, Any]]) -> str:
    """构建拒答模板，附带已检索到的线索

    Args:
        references: 已检索到的知识片段（即使相关性低也会附上）

    Returns:
        拒答文本，包含线索列表
    """
    answer = "根据现有资料无法确认您的问题，建议咨询人工客服。"
    if references:
        clue_lines = ["以下是已检索到的相关信息供参考："]
        for i, ref in enumerate(references[:3], 1):
            source = ref.get("source", "")
            content = ref.get("content", "")[:80]
            clue_lines.append(f"  [{i}] (来源: {source}) {content}...")
        answer += "\n" + "\n".join(clue_lines)
    return answer


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

    messages = _build_messages(user_query, session_id, plan=plan)
    tool_schemas = get_tool_schemas()
    steps: List[Dict[str, Any]] = []
    tool_calls_history: List[Dict[str, str]] = []
    total_tool_calls: int = 0  # 已执行的工具调用总次数

    # 最终结果字段，从工具返回中提取
    final_image_path = None
    final_video_url = None
    references: List[Dict[str, Any]] = []

    # 验证与重试状态
    retry_attempted = False
    last_answer = ""  # 追踪最后一次 LLM 生成的文本答案（预算超限时可复用）
    validation_result = None

    for step_idx in range(MAX_STEPS):
        step_info: Dict[str, Any] = {"step": step_idx + 1}

        # 调用 LLM（带工具定义），低温采样减少幻觉
        response = tracked_chat_completion(
            client=query.client,
            model=CHAT_MODEL,
            messages=messages,
            tools=tool_schemas,
            tracker=tracker,
            source="Agent问答",
            temperature=0.3,
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

                # 为搜索类工具预计算关键词 embedding（用于语义级循环检测）
                query_embedding = _get_search_embedding(tool_name, arguments)

                # 循环检测（精确匹配 + 语义相似 + 次数上限）
                if _detect_loop(tool_calls_history, tool_name, arguments_str, total_tool_calls, query_embedding):
                    logger.warning("检测到循环调用或超限: %s(%s)，跳过", tool_name, arguments_str)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"error": "该搜索与之前的搜索语义重复或调用已达上限，请基于已有检索结果直接回答，不要换用同义关键词重新搜索"}, ensure_ascii=False),
                    })
                    step_info["action"] = "loop_detected"
                    steps.append(step_info)
                    continue

                tool_calls_history.append({"name": tool_name, "arguments": arguments_str, "embedding": query_embedding})
                total_tool_calls += 1

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
                    # 相关性检查：检索结果相关性低时引导拒答
                    if references and not _check_relevance(references):
                        logger.warning("检索结果相关性不足，引导 Agent 拒答")
                        result["relevance_warning"] = "检索结果相关性较低，如无法从已有结果中找到依据，请直接告知用户'未找到相关信息'，不要编造内容"
                    elif not references:
                        result["relevance_warning"] = "未检索到任何结果，请直接告知用户'未找到相关信息'"

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
            last_answer = msg.content  # 追踪已有答案，预算超限时可复用
            step_info["action"] = "final_answer"
            steps.append(step_info)

            # 1) 程序化引用校验（确定性检查，零噪声）
            citation_check = None
            if references:
                citation_check = _check_citations(msg.content, references)
                if not citation_check["passed"]:
                    logger.warning("引用校验失败: 非法引用 %s", citation_check["invalid_citations"])
                if not citation_check["has_citations"]:
                    logger.warning("答案中未发现任何引用编号 [n]，可能存在未引用原文的回答")

            # 2) 验证答案质量（维度打分+claim级判定）
            # references 为空时（LLM 仅调用了 image_search/ocr_image/video_search，
            # 或 knowledge_search 无结果），跳过验证——没有文本依据可供比对，
            # 强行验证只会判 0 分并触发无意义的重试
            if not references:
                logger.info("references 为空，跳过验证（图片/视频/OCR 类查询）")
                validation_result = None
            else:
                validation_result = validate_answer(references, msg.content, tracker=tracker)
                logger.info("答案验证:\n%s", format_validation_summary(validation_result))

            # 3) 准确性低于阈值且未重试过 -> 触发重新检索
            acc_score = validation_result["accuracy"]["score"] if validation_result else 1.0
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

            # 4) 重试后验证仍不通过 -> 返回拒答模板+已检索线索，不返回低质量答案
            if validation_result and not validation_result.get("passed") and retry_attempted:
                logger.warning("重试后验证仍不通过，返回拒答模板")
                abstention_answer = _build_abstention_answer(references)
                tracker.finish()
                cost = tracker.get_summary()
                return {
                    "answer": abstention_answer,
                    "image_path": final_image_path,
                    "video_url": final_video_url,
                    "references": references,
                    "steps": steps,
                    "plan": plan,
                    "validation": validation_result,
                    "citation_check": citation_check,
                    "retried": retry_attempted,
                    "abstained": True,
                    "cost": {
                        "call_count": cost["call_count"],
                        "total_tokens": cost["total_tokens"],
                        "estimated_cost": cost["estimated_cost"],
                        "duration_seconds": cost["duration_seconds"],
                    },
                }

            # 5) 验证通过或未经验证 -> 返回最终结果
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
                "citation_check": citation_check,
                "retried": retry_attempted,
                "abstained": False,
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
    # 预算超限时优先使用已生成的答案，其次用带线索的拒答模板
    if last_answer:
        fallback = last_answer
        logger.info("预算超限但已有生成答案，返回已有答案")
    elif references:
        fallback = _build_abstention_answer(references)
        logger.info("预算超限且无生成答案，返回带线索的拒答模板")
    else:
        fallback = "抱歉，我在处理您的问题时遇到了一些限制，请尝试简化问题后重新提问。"

    return {
        "answer": fallback,
        "image_path": final_image_path,
        "video_url": final_video_url,
        "references": references,
        "steps": steps,
        "plan": plan,
        "validation": validation_result,
        "citation_check": None,
        "retried": retry_attempted,
        "abstained": True,
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

    messages = _build_messages(user_query, session_id, plan=plan)
    tool_schemas = get_tool_schemas()
    steps: List[Dict[str, Any]] = []
    tool_calls_history: List[Dict[str, str]] = []
    total_tool_calls: int = 0

    final_image_path = None
    final_video_url = None
    references: List[Dict[str, Any]] = []
    last_answer = ""  # 追踪最后一次 LLM 生成的文本答案（预算超限时可复用）

    for step_idx in range(MAX_STEPS):
        yield _sse("step_start", {"step": step_idx + 1, "max_steps": MAX_STEPS})

        response = tracked_chat_completion(
            client=query.client,
            model=CHAT_MODEL,
            messages=messages,
            tools=tool_schemas,
            tracker=tracker,
            source="Agent问答",
            temperature=0.3,
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

                query_embedding = _get_search_embedding(tool_name, arguments)

                if _detect_loop(tool_calls_history, tool_name, arguments_str, total_tool_calls, query_embedding):
                    yield _sse("loop_detected", {"tool": tool_name, "arguments": arguments})
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"error": "该搜索与之前的搜索语义重复或调用已达上限，请基于已有检索结果直接回答，不要换用同义关键词重新搜索"}, ensure_ascii=False),
                    })
                    continue

                tool_calls_history.append({"name": tool_name, "arguments": arguments_str, "embedding": query_embedding})
                total_tool_calls += 1

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
            last_answer = msg.content
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
    # 预算超限时优先使用已生成的答案，其次用带线索的拒答模板
    if last_answer:
        sse_fallback = last_answer
    elif references:
        sse_fallback = _build_abstention_answer(references)
    else:
        sse_fallback = "抱歉，我在处理您的问题时遇到了一些限制，请尝试简化问题后重新提问。"
    yield _sse("final_answer", {
        "answer": sse_fallback,
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
