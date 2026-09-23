# -*- coding: utf-8 -*-
"""
app.agent.planner - 检索规划模块

在 Agent 执行工具调用之前，先用 LLM 分析用户问题，
生成结构化的 JSON 检索计划，指导后续工具调用的顺序和参数。

输出格式：
{
  "intent": "用户意图的一句话概括",
  "steps": [
    {
      "target": "检索目标关键词",
      "modality": "text | image",
      "granularity": "paragraph | chunk",
      "reason": "为什么需要这一步"
    }
  ]
}

设计原则：
- Prompt 中明确 JSON schema 和 few-shot 示例，减少格式歧义
- LLM 调用走 cost_tracker，纳入成本统计
- 解析失败时返回空计划，不阻断 Agent 主流程
"""
import json
import logging
import re
from typing import Any, Dict, List

from app.config import CHAT_MODEL
from app.core import query
from app.core.cost_tracker import CostTracker, tracked_chat_completion

logger = logging.getLogger(__name__)


# ========== 检索规划 Prompt ==========

PLANNER_SYSTEM_PROMPT = """你是一个检索规划专家。请分析用户问题，并生成一个 JSON 格式的检索计划。

## 可用工具

你可以调用以下工具来规划检索：
- search_documents(query): 语义检索知识库文本（门票、酒店、攻略、规则等）
- search_images(query): 检索与查询相关的图片（海报、照片等）
- ocr_image(image_path): 对指定图片做 OCR 文字识别，提取图片中的文字

## 输出格式（必须是合法 JSON，不要输出其他内容）

{
  "intent": "用一句话概括用户的检索意图",
  "steps": [
    {
      "target": "检索目标关键词",
      "modality": "text 或 image",
      "granularity": "paragraph 或 chunk",
      "reason": "为什么需要这一步检索"
    }
  ]
}

## 字段说明

- intent: 用户问题的核心意图，一句话。
- steps: 检索步骤列表，按执行顺序排列。
  - target: 这一步要检索的具体关键词或主题（中文）。
  - modality: 检索模态。文本知识用 "text"，图片/海报用 "image"。
  - granularity: 检索粒度。"paragraph" 适合需要完整段落理解的问题；"chunk" 适合需要片段精确匹配的问题。
  - reason: 这一步检索的原因，一句话。

## 规则

1. 当用户问题涉及多个实体的对比或分别检索时，应拆分为多个步骤，每个实体一步。
2. 当用户明确提到"图片/海报/照片/看看"时，增加 image 检索步骤。
3. 当用户需要从图片中提取文字信息时，规划 ocr_image 步骤。
4. 不要编造不存在的检索目标，只基于用户问题中提到的内容。
5. 输出必须是合法 JSON，不要包裹在 markdown 代码块中，不要输出任何解释文字。

## 示例

输入: "门票退款流程是什么？"
输出:
{"intent": "查询门票退款的具体流程", "steps": [{"target": "门票退款流程", "modality": "text", "granularity": "paragraph", "reason": "用户需要了解退款流程的完整步骤"}]}

输入: "米奇和米妮的服装有什么区别？"
输出:
{"intent": "对比米奇和米妮的服装差异", "steps": [{"target": "米奇服装", "modality": "text", "granularity": "paragraph", "reason": "需要检索米奇服装的描述信息"}, {"target": "米妮服装", "modality": "text", "granularity": "paragraph", "reason": "需要检索米妮服装的描述信息"}]}

输入: "万圣节活动海报长什么样？"
输出:
{"intent": "查看万圣节活动海报图片", "steps": [{"target": "万圣节活动海报", "modality": "image", "granularity": "chunk", "reason": "用户想看万圣节海报的图片"}]}

输入: "活动海报上写了什么内容？"
输出:
{"intent": "提取活动海报中的文字信息", "steps": [{"target": "活动海报", "modality": "image", "granularity": "chunk", "reason": "需要先检索到海报图片"}, {"target": "活动海报", "modality": "image", "granularity": "chunk", "reason": "对海报图片做OCR文字识别，提取海报上的文字"}]}
"""


# ========== 核心函数 ==========

def plan_retrieval(user_query: str, tracker: CostTracker = None) -> Dict[str, Any]:
    """分析用户问题，生成结构化检索计划

    Args:
        user_query: 用户问题文本
        tracker: 可选的 CostTracker 实例；不传则不追踪本次调用成本

    Returns:
        解析后的检索计划 dict，格式见模块文档。
        解析失败时返回 {"intent": "", "steps": []}。
    """
    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

    response = tracked_chat_completion(
        client=query.client,
        model=CHAT_MODEL,
        messages=messages,
        tracker=tracker,
        source="检索规划",
        temperature=0.1,
    )

    raw_output = response.choices[0].message.content.strip()
    logger.info("检索规划 LLM 原始输出:\n%s", raw_output)

    # 尝试解析 JSON
    plan = _parse_json_safely(raw_output)
    if plan is None:
        logger.warning("检索规划 JSON 解析失败，返回空计划。原始输出: %s", raw_output[:200])
        return {"intent": "", "steps": []}

    # 校验结构
    if not isinstance(plan, dict) or "steps" not in plan:
        logger.warning("检索计划结构不合法，返回空计划")
        return {"intent": "", "steps": []}

    if not isinstance(plan["steps"], list):
        plan["steps"] = []

    return plan


def _parse_json_safely(raw: str) -> Any:
    """安全解析 JSON，兼容 LLM 可能包裹 markdown 代码块的情况

    Args:
        raw: LLM 原始输出文本

    Returns:
        解析后的 Python 对象；失败返回 None
    """
    # 直接尝试解析
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 尝试去除 markdown 代码块后再解析
    code_block = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    if code_block:
        try:
            return json.loads(code_block.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试提取第一个 { 到最后一个 } 之间的内容
    first_brace = raw.find("{")
    last_brace = raw.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(raw[first_brace:last_brace + 1])
        except json.JSONDecodeError:
            pass

    return None


def format_plan_summary(plan: Dict[str, Any]) -> str:
    """把检索计划格式化为可读字符串（用于日志和前端展示）

    Args:
        plan: plan_retrieval 返回的计划 dict

    Returns:
        可读的文本摘要
    """
    intent = plan.get("intent", "")
    steps: List[Dict] = plan.get("steps", [])

    lines = [f"检索意图: {intent}", f"步骤数: {len(steps)}"]
    for i, step in enumerate(steps, 1):
        target = step.get("target", "")
        modality = step.get("modality", "")
        granularity = step.get("granularity", "")
        reason = step.get("reason", "")
        lines.append(
            f"  Step {i}: target={target} | modality={modality} | "
            f"granularity={granularity} | reason={reason}"
        )
    return "\n".join(lines)
