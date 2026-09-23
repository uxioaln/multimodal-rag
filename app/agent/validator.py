# -*- coding: utf-8 -*-
"""
app.agent.validator - 答案验证器（三维打分）

在 Agent 生成答案后，用 LLM 扮演"阅卷老师"，
对答案进行三维质量评分，检测幻觉和引用造假。

三个维度（0-1 分）：
1. 准确性 (accuracy)    - 答案是否忠实于检索到的原文？
2. 引用质量 (citation)  - 答案中的引用是否真实存在于原文中？
3. 推理链 (reasoning)  - 每一步推理是否有证据支撑？

输出格式：
{
  "accuracy": {"score": 0.0, "reason": "..."},
  "citation": {"score": 0.0, "reason": "..."},
  "reasoning": {"score": 0.0, "reason": "..."},
  "overall": 0.0,
  "passed": true/false
}

设计原则：
- Prompt 中明确评分标准和 few-shot 示例（含错误答案的反例）
- LLM 调用走 cost_tracker，纳入成本统计
- 解析失败时返回默认零分，不阻断 Agent 主流程
"""
import json
import logging
import re
from typing import Any, Dict, List, Optional

from app.config import CHAT_MODEL
from app.core import query
from app.core.cost_tracker import CostTracker, tracked_chat_completion

logger = logging.getLogger(__name__)

# 通过阈值：三维平均分 >= 0.6 视为通过
PASS_THRESHOLD = 0.6


# ========== 验证 Prompt ==========

VALIDATOR_SYSTEM_PROMPT = """你是一位严格的阅卷老师。请根据检索到的原文，对生成的答案进行三维打分。

## 输入说明

你会收到两个部分：
- [检索原文]：知识库中检索到的原始文本片段，是事实依据。
- [生成答案]：AI 基于检索原文生成的回答，是需要被打分评估的对象。

## 评分维度（每个维度 0-1 分，保留一位小数）

1. 准确性 (accuracy)：答案的内容是否忠实于检索原文？
   - 1.0：完全忠于原文，无任何编造或歪曲。
   - 0.5：部分准确，存在少量不严谨的表述但无重大错误。
   - 0.0：答案与原文矛盾，或包含原文中不存在的信息（幻觉）。

2. 引用质量 (citation)：答案中引用的来源、数据、规则是否真实存在于原文中？
   - 1.0：所有引用均可在原文中找到对应内容。
   - 0.5：部分引用可以找到，部分无法核实。
   - 0.0：引用了原文中完全不存在的内容（伪造引用）。

3. 推理链 (reasoning)：答案的每一步推理是否有原文证据支撑？
   - 1.0：每个结论都有对应的原文证据，逻辑链完整。
   - 0.5：部分推理有支撑，部分跳跃或缺乏依据。
   - 0.0：推理过程与原文无关，或存在明显逻辑错误。

## 输出格式（必须是合法 JSON，不要输出其他内容）

{
  "accuracy": {"score": 0.0, "reason": "打分理由，一句话"},
  "citation": {"score": 0.0, "reason": "打分理由，一句话"},
  "reasoning": {"score": 0.0, "reason": "打分理由，一句话"},
  "overall": 0.0,
  "passed": true
}

其中 overall 是三维分数的算术平均值，passed 是 overall >= 0.6 时为 true。

## 规则

1. 严格基于[检索原文]评分，不要使用原文以外的知识。
2. 如果答案中出现了原文没有的事实、数据或规则，准确性应给低分。
3. 如果答案声称"根据xxx规定"但原文中找不到该规定，引用质量应给 0 分。
4. 输出必须是合法 JSON，不要包裹在 markdown 代码块中。

## 示例

[检索原文]
背景知识1: 门票退款需在购票后7天内申请，需提供购票凭证。特殊票种（年卡）不支持退款。

[生成答案]
门票可以在30天内随时退款，无需任何凭证。所有票种均支持退款。

输出:
{"accuracy": {"score": 0.0, "reason": "答案称30天内可退款且无需凭证，与原文7天内需凭证矛盾"}, "citation": {"score": 0.0, "reason": "答案声称所有票种支持退款，但原文明确年卡不支持"}, "reasoning": {"score": 0.0, "reason": "推理与原文事实完全相悖，无证据支撑"}, "overall": 0.0, "passed": false}

[检索原文]
背景知识1: 上海迪士尼乐园酒店提供免费班车接送，每30分钟一班，运营时间为早7点至晚10点。

[生成答案]
上海迪士尼乐园酒店提供免费班车服务，每30分钟一班，运营时间为早7点至晚10点。

输出:
{"accuracy": {"score": 1.0, "reason": "答案完全忠实于原文，未添加任何额外信息"}, "citation": {"score": 1.0, "reason": "班车频率、运营时间均在原文中可找到"}, "reasoning": {"score": 1.0, "reason": "陈述直接来自原文，无推理跳跃"}, "overall": 1.0, "passed": true}
"""


# ========== 核心函数 ==========

def validate_answer(
    references: List[Dict[str, Any]],
    answer: str,
    tracker: Optional[CostTracker] = None,
) -> Dict[str, Any]:
    """对 Agent 生成的答案进行三维质量打分

    Args:
        references: 检索到的原文片段列表，每项应含 content/source 字段
        answer: Agent 生成的答案文本
        tracker: 可选的 CostTracker 实例

    Returns:
        打分结果 dict，格式见模块文档。
        解析失败时返回默认零分结果。
    """
    # 构建检索原文文本
    context_str = ""
    for i, ref in enumerate(references, 1):
        source = ref.get("source", "未知来源")
        content = ref.get("content", "")
        context_str += f"背景知识{i} (来源: {source}):\n{content}\n\n"

    if not context_str.strip():
        context_str = "(无检索结果)"

    user_content = f"[检索原文]\n{context_str}\n[生成答案]\n{answer}"

    messages = [
        {"role": "system", "content": VALIDATOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    response = tracked_chat_completion(
        client=query.client,
        model=CHAT_MODEL,
        messages=messages,
        tracker=tracker,
        source="答案验证",
        temperature=0.1,
    )

    raw_output = response.choices[0].message.content.strip()
    logger.info("答案验证 LLM 原始输出:\n%s", raw_output)

    # 解析 JSON
    result = _parse_json_safely(raw_output)
    if result is None:
        logger.warning("答案验证 JSON 解析失败，返回默认零分。原始输出: %s", raw_output[:200])
        return _default_failed_result("JSON 解析失败")

    # 校验并补全结构
    return _normalize_result(result)


def _normalize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """校验并补全打分结果结构，确保所有字段存在

    Args:
        result: LLM 返回的解析后 dict

    Returns:
        补全后的标准结构
    """
    for dim in ("accuracy", "citation", "reasoning"):
        if dim not in result or not isinstance(result[dim], dict):
            result[dim] = {"score": 0.0, "reason": "维度缺失"}
        else:
            score = result[dim].get("score", 0.0)
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = 0.0
            result[dim]["score"] = max(0.0, min(1.0, score))
            result[dim].setdefault("reason", "无说明")

    # 计算 overall
    scores = [
        result["accuracy"]["score"],
        result["citation"]["score"],
        result["reasoning"]["score"],
    ]
    overall = round(sum(scores) / len(scores), 2)
    result["overall"] = overall
    result["passed"] = overall >= PASS_THRESHOLD

    return result


def _default_failed_result(reason: str) -> Dict[str, Any]:
    """生成默认的零分结果（解析失败时使用）"""
    return {
        "accuracy": {"score": 0.0, "reason": f"验证失败: {reason}"},
        "citation": {"score": 0.0, "reason": f"验证失败: {reason}"},
        "reasoning": {"score": 0.0, "reason": f"验证失败: {reason}"},
        "overall": 0.0,
        "passed": False,
    }


def _parse_json_safely(raw: str) -> Any:
    """安全解析 JSON，兼容 LLM 可能包裹 markdown 代码块的情况

    与 planner.py 中的 _parse_json_safely 逻辑一致，避免跨模块依赖。
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    code_block = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    if code_block:
        try:
            return json.loads(code_block.group(1))
        except json.JSONDecodeError:
            pass

    first_brace = raw.find("{")
    last_brace = raw.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(raw[first_brace:last_brace + 1])
        except json.JSONDecodeError:
            pass

    return None


def format_validation_summary(result: Dict[str, Any]) -> str:
    """把打分结果格式化为可读字符串（用于日志和前端展示）"""
    lines = [
        f"准确性: {result['accuracy']['score']} - {result['accuracy']['reason']}",
        f"引用质量: {result['citation']['score']} - {result['citation']['reason']}",
        f"推理链: {result['reasoning']['score']} - {result['reasoning']['reason']}",
        f"综合: {result['overall']} | {'通过' if result['passed'] else '不通过'}",
    ]
    return "\n".join(lines)
