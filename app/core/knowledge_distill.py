# -*- coding: utf-8 -*-
"""
app.core.knowledge_distill - 对话知识沉淀模块

把 conversations 表里的历史对话聚合成若干结构化知识点（事实/流程/注意等），
供管理员审阅后决定是否纳入知识库。

实现思路：
1. 从对话中按条提取候选知识点（含 type/content/confidence/keywords/category）
2. 过滤掉"需求/问题"等临时性条目
3. 按 type 分组，使用 LLM 合并相似条目

与原版的差异：
- LLM 调用统一走 app.core.cost_tracker.tracked_chat_completion，自动汇总到全局成本统计
- 提供 run_distill_pipeline() 顶层流程函数，外部只需传入对话文本列表
"""
import json
import logging
import os
from collections import Counter
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from openai import OpenAI

from app.config import AGICTO_API_KEY, CHAT_BASE_URL
from app.core.cost_tracker import CostTracker, tracked_chat_completion

logger = logging.getLogger(__name__)

# 进度回调签名：stage 阶段名 / current 当前进度 / total 总数 / message 描述
ProgressCallback = Callable[[str, int, int, str], None]

# 初始化 AGICTO 兼容的 OpenAI 客户端（与工程内其他模块保持一致）
_client = OpenAI(
    api_key=AGICTO_API_KEY,
    base_url=CHAT_BASE_URL
)


def _get_completion(prompt: str, model: str, tracker: CostTracker, source: str):
    """包装 LLM 调用，自动统计成本；网络/认证错误时返回 None"""
    try:
        response = tracked_chat_completion(
            client=_client,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            tracker=tracker,
            source=source,
            temperature=0.3,
        )
    except Exception as e:
        logger.warning("LLM 调用失败: %s: %s", type(e).__name__, e)
        return None
    # 防御：choices/message/content 任意环节为 None 都返回 None
    try:
        return response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as e:
        logger.warning("LLM 响应结构异常: %s", e)
        return None


def _preprocess_json_response(response: str) -> str:
    """预处理 AI 响应，移除 markdown 代码块格式"""
    if not response:
        return ""
    if response.startswith("```json"):
        response = response[7:]
    elif response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]
    return response.strip()


class ConversationKnowledgeExtractor:
    """从对话中提取并合并知识点（工程版，接入 cost_tracker）"""

    def __init__(self, model="deepseek-v4-flash", source="对话知识沉淀"):
        self.model = model
        # 标记本实例的 LLM 调用归属哪个业务模块（用于 cost_stats.json 分类）
        self.source = source
        self.extracted_knowledge = []
        self.knowledge_frequency = Counter()

    def _call_llm(self, prompt: str, tracker: CostTracker) -> str:
        return _get_completion(prompt, self.model, tracker, self.source)

    def extract_knowledge_from_conversation(self, conversation: str, tracker: CostTracker) -> dict:
        """从单次对话中提取知识（每次 LLM 调用都计入 tracker）"""
        instruction = """
你是一个专业的知识提取专家。请从给定的对话中提取有价值的知识点，包括：
1. 事实性信息（地点、时间、价格、规则等）
2. 用户需求和偏好
3. 常见问题和解答
4. 操作流程和步骤
5. 注意事项和提醒

请返回JSON格式：
{
    "extracted_knowledge": [
        {
            "knowledge_type": "知识类型（事实/需求/问题/流程/注意）",
            "content": "知识内容",
            "confidence": "置信度(0-1)",
            "source": "来源（用户/AI/对话）",
            "keywords": ["关键词1", "关键词2"],
            "category": "分类"
        }
    ],
    "conversation_summary": "对话摘要",
    "user_intent": "用户意图"
}
"""
        prompt = f"""
### 指令 ###
{instruction}

### 对话内容 ###
{conversation}

### 提取结果 ###
"""
        response = self._call_llm(prompt, tracker)
        response = _preprocess_json_response(response)
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning("对话知识提取JSON解析失败: %s", e)
            logger.debug("AI返回内容: %s", response[:200])
            return {
                "extracted_knowledge": [],
                "conversation_summary": "无法解析对话",
                "user_intent": "未知",
            }
        except Exception as e:
            logger.warning("对话知识提取异常: %s: %s", type(e).__name__, e)
            logger.debug("AI返回内容: %s", response[:200])
            return {
                "extracted_knowledge": [],
                "conversation_summary": "异常",
                "user_intent": "未知",
            }

    def batch_extract_knowledge(
        self,
        conversations: List[str],
        tracker: CostTracker,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> List[Dict[str, Any]]:
        """批量提取知识，每完成一条回调一次进度"""
        all_knowledge = []
        total = len(conversations)
        for i, conv in enumerate(conversations, 1):
            if progress_callback:
                progress_callback(i, total, f"正在提取第 {i}/{total} 段对话的知识点...")
            result = self.extract_knowledge_from_conversation(conv, tracker)
            all_knowledge.extend(result.get("extracted_knowledge", []))
            for k in result.get("extracted_knowledge", []):
                key = f"{k.get('knowledge_type','')}:{k.get('content','')[:50]}"
                self.knowledge_frequency[key] += 1
        return all_knowledge

    def merge_similar_knowledge(self, knowledge_list, tracker: CostTracker, progress_callback=None):
        """使用 LLM 合并相似知识点；过滤掉"需求/问题"等临时性条目"""
        filtered = [k for k in knowledge_list if k.get("knowledge_type") not in ("需求", "问题")]
        # 按 type 分组
        grouped = {}
        for k in filtered:
            grouped.setdefault(k.get("knowledge_type", "其他"), []).append(k)

        merged = []
        group_items = list(grouped.items())
        for idx, (ktype, group) in enumerate(group_items, 1):
            if progress_callback:
                progress_callback(idx, len(group_items), f"正在合并 {ktype} 类型的 {len(group)} 条知识点...")
            if len(group) == 1:
                merged.append(group[0])
            else:
                merged.append(self._merge_one_group(group, ktype, tracker))
        return {
            "filtered_count": len(knowledge_list) - len(filtered),
            "raw_count": len(knowledge_list),
            "merged": merged,
        }

    def _merge_one_group(self, group, ktype, tracker):
        """合并同类型的一组知识点"""
        lines = []
        all_keywords = set()
        all_sources = []
        for i, k in enumerate(group, 1):
            content = (k.get("content") or "")
            confidence = (k.get("confidence") if k.get("confidence") is not None else 0.5)
            keywords = k.get("keywords") or []
            source = (k.get("source") or "")
            category = (k.get("category") or "")
            lines.append(f"{i}. 内容: {content}")
            lines.append(f"   置信度: {confidence}")
            lines.append(f"   分类: {category}")
            lines.append(f"   来源: {source}")
            lines.append(f"   关键词: {', '.join(keywords)}")
            lines.append("")
            all_keywords.update(keywords)
            if source and source not in all_sources:
                all_sources.append(source)

        prompt = f"""
你是一个专业的知识整理专家。请将以下{ktype}类型的知识点进行智能合并，生成一个更完整、准确的知识点。

### 合并要求：
1. 保留所有重要信息，避免信息丢失
2. 消除重复内容，整合相似表述
3. 提高内容的准确性和完整性
4. 保持逻辑清晰，结构合理
5. 合并后的置信度取所有知识点中的最高值

### 待合并的知识点：
{chr(10).join(lines)}

### 请返回JSON格式：
{{
    "knowledge_type": "{ktype}",
    "content": "合并后的知识内容",
    "confidence": 最高置信度值,
    "keywords": ["合并后的关键词列表"],
    "category": "合并后的分类",
    "sources": ["所有来源"],
    "frequency": {len(group)}
}}

### 合并结果：
"""
        response = self._call_llm(prompt, tracker)
        if response is None:
            # LLM 调用失败：兜底返回置信度最高的那条
            best = max(group, key=lambda x: x.get("confidence") or 0) if group else {}
            return {
                "knowledge_type": ktype,
                "content": best.get("content") or "",
                "confidence": best.get("confidence") or 0.5,
                "frequency": len(group),
                "keywords": list(all_keywords),
                "category": best.get("category") or "",
                "sources": all_sources,
            }
        response = _preprocess_json_response(response)
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning("知识合并JSON解析失败: %s", e)
            logger.debug("AI返回内容: %s", response[:200])
            best = max(group, key=lambda x: x.get("confidence") or 0)
            return {
                "knowledge_type": ktype,
                "content": best.get("content") or "",
                "confidence": best.get("confidence") or 0.5,
                "frequency": len(group),
                "keywords": list(all_keywords),
                "category": best.get("category") or "",
                "sources": all_sources,
            }


# ========== 顶层流程函数 ==========
def run_distill_pipeline(
    conversations_text_list: List[str],
    model: str = "deepseek-v4-flash",
    source: str = "对话知识沉淀",
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """
    执行完整的对话知识沉淀流程：
    1) 对每段对话调用 LLM 提取候选知识点
    2) 过滤临时条目并按 type 分组合并
    3) 汇总成本统计

    Args:
        conversations_text_list: list[str]，每元素是一段对话的纯文本
        model: 使用的 chat 模型
        source: 成本统计的来源标签
        progress_callback: 可选，签名 fn(stage, current, total, message)，
            stage 取值 "extract" / "merge" / "save"

    Returns:
        dict: {
            "extracted_count": int,
            "merged_count": int,
            "filtered_count": int,
            "merged_knowledge": list[dict],
            "cost": dict  # 来自 CostTracker.get_summary()
        }
    """
    tracker = CostTracker()
    tracker.begin()
    extractor = ConversationKnowledgeExtractor(model=model, source=source)

    def extract_cb(current, total, message):
        if progress_callback:
            progress_callback("extract", current, total, message)

    def merge_cb(current, total, message):
        if progress_callback:
            progress_callback("merge", current, total, message)

    try:
        all_knowledge = extractor.batch_extract_knowledge(
            conversations_text_list, tracker, progress_callback=extract_cb
        )
        merge_result = extractor.merge_similar_knowledge(
            all_knowledge, tracker, progress_callback=merge_cb
        )
        merged = merge_result["merged"]
    finally:
        tracker.finish()

    return {
        "extracted_count": len(all_knowledge),
        "merged_count": len(merged),
        "filtered_count": merge_result["filtered_count"],
        "raw_count": merge_result["raw_count"],
        "merged_knowledge": merged,
        "cost": tracker.get_summary(),
    }


# 缓存文件：最新一次沉淀结果，便于详情页/导出接口直接读取，避免重复 LLM 调用
DISTILL_CACHE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "stats", "disney_distilled.json"
)


def save_distill_cache(result: dict):
    """把沉淀结果写入磁盘缓存"""
    payload = dict(result)
    # cost 里的 datetime 无法直接 json 化，转字符串
    cost = payload.get("cost", {})
    if cost:
        for k in ("start_time", "end_time"):
            v = cost.get(k)
            if hasattr(v, "isoformat"):
                cost[k] = v.isoformat()
    payload["saved_at"] = datetime.now().isoformat()
    os.makedirs(os.path.dirname(DISTILL_CACHE_FILE), exist_ok=True)
    with open(DISTILL_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_distill_cache():
    """读取最近一次沉淀结果；不存在则返回 None"""
    if not os.path.exists(DISTILL_CACHE_FILE):
        return None
    try:
        with open(DISTILL_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
