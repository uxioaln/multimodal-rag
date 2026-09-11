# -*- coding: utf-8 -*-
"""
app.core.health_check - 知识库健康度检查

提供 KnowledgeBaseHealthChecker，对知识库进行三维度检查：
- 完整性（missing）：查不到/查不准的查询
- 时效性（outdated）：价格/时间/政策等可能过期的条目
- 一致性（conflicting）：同一主题在不同条目间存在的冲突

特点：
- 检查结果按 10 分制分级 + 中文等级描述
- 所有 LLM 调用统一走 app.core.cost_tracker，自动汇总到全局成本统计
- 可指定 conflict_model 单独升级冲突检测模型（默认与主模型一致）
"""
import json
import logging
from datetime import datetime

from openai import OpenAI

from app.config import AGICTO_API_KEY, CHAT_BASE_URL
from app.core.cost_tracker import get_default_tracker, tracked_chat_completion

logger = logging.getLogger(__name__)

# 初始化 AGICTO 兼容的 OpenAI 客户端
client = OpenAI(
    api_key=AGICTO_API_KEY,
    base_url=CHAT_BASE_URL
)


def get_completion(prompt, model="deepseek-v4-flash"):
    """基于 prompt 生成文本（兼容旧调用方式）"""
    messages = [{"role": "user", "content": prompt}]
    response = tracked_chat_completion(client=client, model=model, messages=messages, temperature=0.3)
    return response.choices[0].message.content


# 0-1 分数转换为 0-10 分数，并给出中文等级描述
SCORE_LEVELS = {
    10: "没有问题",
    9: "极其好",
    8: "非常好",
    7: "好",
    6: "基本可以",
    5: "一般",
    4: "存在问题",
    3: "较差",
    2: "差",
    1: "极差",
    0: "极差"
}


def score_to_level(score_0_1):
    """把 0-1 分数映射为 0-10 整数与中文等级"""
    score_10 = int(round(score_0_1 * 10))
    score_10 = max(0, min(10, score_10))
    return score_10, SCORE_LEVELS.get(score_10, "未知")


def translate_item(item):
    """把 LLM 返回的英文/混合字段翻译成中文 key"""
    translated = {}
    mapping = {
        "query": "查询",
        "missing_aspect": "缺少的方面",
        "importance": "重要性",
        "suggested_content": "建议补充的内容",
        "category": "知识分类",
        "chunk_id": "条目编号",
        "content": "内容",
        "outdated_aspect": "过期方面",
        "severity": "严重程度",
        "suggested_update": "建议更新",
        "last_verified": "最后验证时间",
        "conflict_type": "冲突类型",
        "chunk_ids": "相关条目编号",
        "conflicting_content": "冲突内容",
        "resolution_suggestion": "解决建议",
        "question": "问题",
        "question_type": "问题类型",
        "perspective": "提问角度",
        "original_chunk_id": "原始条目编号",
    }
    for k, v in item.items():
        new_k = mapping.get(k, k)
        translated[new_k] = v
    return translated


def build_dimension_recommendations(missing_items, outdated_items, conflicting_items):
    """
    按维度（完整性/时效性/一致性）汇总改进建议。
    每个维度包含：维度名、建议条数、涉及知识点（条目ID + 简短原因）。
    返回结构：
    [
        {"维度": "完整性", "建议条数": N, "涉及知识点": [{"条目ID": "xx", "原因": "简短原因"}, ...]},
        ...
    ]
    """
    dimensions = []

    # 完整性：缺少的知识（无对应条目，用查询内容定位）
    dims_missing = []
    for item in missing_items:
        query = item.get('查询', '')
        aspect = item.get('缺少的方面', '')
        dims_missing.append({
            "条目ID": f"查询: {query}",
            "原因": f"缺少{aspect}" if aspect else "知识库中未找到相关内容",
        })
    dimensions.append({
        "维度": "完整性",
        "建议条数": len(dims_missing),
        "涉及知识点": dims_missing,
    })

    # 时效性：过期的知识（条目ID 与知识库列表页一致）
    dims_outdated = []
    for item in outdated_items:
        chunk_id = item.get('条目编号', 'unknown')
        aspect = item.get('过期方面', '')
        dims_outdated.append({
            "条目ID": str(chunk_id),
            "原因": aspect if aspect else "信息可能已过期",
        })
    dimensions.append({
        "维度": "时效性",
        "建议条数": len(dims_outdated),
        "涉及知识点": dims_outdated,
    })

    # 一致性：冲突的知识（涉及多个条目）
    dims_conflict = []
    for item in conflicting_items:
        chunk_ids = item.get('相关条目编号', [])
        conflict_type = item.get('冲突类型', '')
        dims_conflict.append({
            "条目ID": "、".join(str(cid) for cid in chunk_ids) if chunk_ids else "unknown",
            "原因": conflict_type if conflict_type else "存在内容冲突",
        })
    dimensions.append({
        "维度": "一致性",
        "建议条数": len(dims_conflict),
        "涉及知识点": dims_conflict,
    })

    return dimensions


class KnowledgeBaseHealthChecker:
    def __init__(self, model="deepseek-v4-flash", conflict_model=None):
        self.model = model
        # 一致性(冲突)检查对对比推理要求更高，可单独指定更强模型
        # 不传则默认与主模型一致；建议冲突检测漏检时升级为 deepseek-v4-pro
        self.conflict_model = conflict_model or model
        self.health_report = {}
        # 每个 checker 实例使用独立的成本追踪器
        self.tracker = get_default_tracker()
        self.tracker.reset()

    def _call_llm(self, prompt, model=None):
        """统一调用 LLM，自动走成本追踪"""
        model = model or self.model
        messages = [{"role": "user", "content": prompt}]
        response = tracked_chat_completion(client=client, model=model, messages=messages, temperature=0.3, source="健康检查")
        return response.choices[0].message.content

    def check_missing_knowledge(self, knowledge_base, test_queries):
        """使用LLM检查缺少的知识"""
        instruction = """
你是一个知识库完整性检查专家。请分析给定的测试查询和知识库内容，判断知识库中是否缺少相关的知识。

检查标准：
1. 查询是否能在知识库中找到相关答案
2. 知识是否完整、准确
3. 是否覆盖了用户的主要需求
4. 是否存在知识空白

请返回JSON格式：
{
    "missing_knowledge": [
        {
            "query": "测试查询",
            "missing_aspect": "缺少的知识方面",
            "importance": "重要性（高/中/低）",
            "suggested_content": "建议的知识内容",
            "category": "知识分类"
        }
    ],
    "coverage_score": "覆盖率评分(0-1)",
    "completeness_analysis": "完整性分析"
}
"""

        # 构建知识库内容摘要
        knowledge_summary = []
        for chunk in knowledge_base:
            knowledge_summary.append(f"ID: {chunk.get('id', 'unknown')} - {chunk.get('content', '')}")

        knowledge_text = "\n".join(knowledge_summary)

        # 构建测试查询列表
        queries_text = []
        for query_info in test_queries:
            query_text = query_info['query']
            expected = query_info.get('expected_answer', '')
            queries_text.append(f"查询: {query_text} | 期望答案: {expected}")

        queries_text = "\n".join(queries_text)

        prompt = f"""
### 指令 ###
{instruction}

### 知识库内容 ###
{knowledge_text}

### 测试查询 ###
{queries_text}

### 分析结果 ###
"""

        try:
            response = self._call_llm(prompt, self.model)

            # 预处理响应，移除markdown代码块格式
            if response.startswith('```json'):
                response = response[7:]
            elif response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]

            result = json.loads(response.strip())
            return result

        except Exception as e:
            logger.warning("LLM检查缺少知识失败: %s", e)
            return None

    def check_outdated_knowledge(self, knowledge_base):
        """使用LLM检查过期的知识"""
        instruction = """
你是一个知识时效性检查专家。请分析给定的知识内容，判断是否存在过期或需要更新的信息。

检查标准：
1. 时间相关信息是否过期（年份、日期、时间范围）
2. 价格信息是否最新（价格、费用、票价等）
3. 政策规则是否更新（政策、规定、规则等）
4. 活动信息是否有效（活动、节日、特殊安排等）
5. 联系方式是否准确（电话、地址、网址等）
6. 技术信息是否过时（版本、技术标准等）

请返回JSON格式：
{
    "outdated_knowledge": [
        {
            "chunk_id": "知识切片ID",
            "content": "知识内容",
            "outdated_aspect": "过期方面",
            "severity": "严重程度（高/中/低）",
            "suggested_update": "建议更新内容",
            "last_verified": "最后验证时间"
        }
    ],
    "freshness_score": "新鲜度评分(0-1)",
    "update_recommendations": "更新建议"
}
"""

        # 构建知识库内容
        knowledge_text = []
        for chunk in knowledge_base:
            content = chunk.get('content', '')
            chunk_id = chunk.get('id', 'unknown')
            last_updated = chunk.get('last_updated', 'unknown')
            knowledge_text.append(f"ID: {chunk_id} | 更新时间: {last_updated} | 内容: {content}")

        knowledge_text = "\n".join(knowledge_text)

        prompt = f"""
### 指令 ###
{instruction}

### 知识库内容 ###
{knowledge_text}

### 当前时间 ###
{datetime.now().strftime('%Y年%m月%d日')}

### 分析结果 ###
"""

        try:
            response = self._call_llm(prompt, self.model)

            # 预处理响应，移除markdown代码块格式
            if response.startswith('```json'):
                response = response[7:]
            elif response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]

            result = json.loads(response.strip())
            return result

        except Exception as e:
            logger.warning("LLM检查过期知识失败: %s", e)
            return None

    def check_conflicting_knowledge(self, knowledge_base):
        """使用LLM检查冲突的知识"""
        instruction = """
你是一个知识一致性检查专家。请分析给定的知识库，找出可能存在冲突或矛盾的信息。

检查标准：
1. 同一主题的不同说法（地点、名称、描述等）
2. 价格信息的差异（价格、费用、收费标准等）
3. 时间信息的不一致（营业时间、开放时间、活动时间等）
4. 规则政策的冲突（规定、政策、要求等）
5. 操作流程的差异（步骤、方法、流程等）
6. 联系方式的差异（地址、电话、网址等）

请返回JSON格式：
{
    "conflicting_knowledge": [
        {
            "conflict_type": "冲突类型",
            "chunk_ids": ["相关切片ID"],
            "conflicting_content": ["冲突内容"],
            "severity": "严重程度（高/中/低）",
            "resolution_suggestion": "解决建议"
        }
    ],
    "consistency_score": "一致性评分(0-1)",
    "conflict_analysis": "冲突分析"
}
"""

        # 构建知识库内容
        knowledge_text = []
        for chunk in knowledge_base:
            content = chunk.get('content', '')
            chunk_id = chunk.get('id', 'unknown')
            knowledge_text.append(f"ID: {chunk_id} | 内容: {content}")

        knowledge_text = "\n".join(knowledge_text)

        prompt = f"""
### 指令 ###
{instruction}

### 知识库内容 ###
{knowledge_text}

### 分析结果 ###
"""

        try:
            # 冲突检查对对比推理要求更高，使用单独指定的冲突检测模型
            response = self._call_llm(prompt, self.conflict_model)

            # 预处理响应，移除markdown代码块格式
            if response.startswith('```json'):
                response = response[7:]
            elif response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]

            result = json.loads(response.strip())
            return result

        except Exception as e:
            logger.warning("LLM检查冲突知识失败: %s", e)
            return None

    def calculate_overall_health_score(self, missing_result, outdated_result, conflicting_result):
        """计算整体健康度评分"""
        coverage_score = missing_result.get('coverage_score', 0)
        freshness_score = outdated_result.get('freshness_score', 0)
        consistency_score = conflicting_result.get('consistency_score', 0)

        # 加权计算
        overall_score = (
            coverage_score * 0.4 +      # 覆盖率权重40%
            freshness_score * 0.3 +     # 新鲜度权重30%
            consistency_score * 0.3      # 一致性权重30%
        )

        return overall_score

    def generate_health_report(self, knowledge_base, test_queries):
        """生成完整的健康度报告"""
        logger.info("正在检查知识库健康度...")
        self.tracker.begin()

        # 1. 检查缺少的知识
        logger.info("1. 检查缺少的知识...")
        missing_result = self.check_missing_knowledge(knowledge_base, test_queries)

        # 2. 检查过期的知识
        logger.info("2. 检查过期的知识...")
        outdated_result = self.check_outdated_knowledge(knowledge_base)

        # 3. 检查冲突的知识
        logger.info("3. 检查冲突的知识...")
        conflicting_result = self.check_conflicting_knowledge(knowledge_base)

        # 4. 计算整体健康度
        overall_score = self.calculate_overall_health_score(missing_result, outdated_result, conflicting_result)

        # 5. 结束成本追踪
        self.tracker.finish()
        usage_summary = self.tracker.get_summary()

        # 6. 生成报告：中文化 + 分数分级
        overall_score_10, overall_level = score_to_level(overall_score)
        coverage_score = missing_result.get('coverage_score', 0)
        freshness_score = outdated_result.get('freshness_score', 0)
        consistency_score = conflicting_result.get('consistency_score', 0)

        missing_items = [translate_item(it) for it in missing_result.get('missing_knowledge', [])]
        outdated_items = [translate_item(it) for it in outdated_result.get('outdated_knowledge', [])]
        conflicting_items = [translate_item(it) for it in conflicting_result.get('conflicting_knowledge', [])]

        report = {
            "总体健康分": {
                "分数_10分制": overall_score_10,
                "等级描述": overall_level,
                "原始得分_0到1": round(overall_score, 4),
            },
            "覆盖率": {
                "分数_10分制": score_to_level(coverage_score)[0],
                "等级描述": score_to_level(coverage_score)[1],
                "原始得分_0到1": coverage_score,
            },
            "新鲜度": {
                "分数_10分制": score_to_level(freshness_score)[0],
                "等级描述": score_to_level(freshness_score)[1],
                "原始得分_0到1": freshness_score,
            },
            "一致性": {
                "分数_10分制": score_to_level(consistency_score)[0],
                "等级描述": score_to_level(consistency_score)[1],
                "原始得分_0到1": consistency_score,
            },
            "缺少的知识": {
                "说明": missing_result.get('completeness_analysis', ''),
                "问题列表": missing_items,
            },
            "过期的知识": {
                "说明": outdated_result.get('update_recommendations', ''),
                "问题列表": outdated_items,
            },
            "冲突的知识": {
                "说明": conflicting_result.get('conflict_analysis', ''),
                "问题列表": conflicting_items,
            },
            "改进建议": self.generate_recommendations(missing_result, outdated_result, conflicting_result),
            "检查时间": datetime.now().isoformat(),
            "成本追踪": usage_summary,
        }

        return report

    def get_health_level(self, score):
        """根据评分确定健康等级（旧接口保留）"""
        return score_to_level(score)[1]

    def generate_recommendations(self, missing_result, outdated_result, conflicting_result):
        """
        生成按维度汇总的改进建议（完整性/时效性/一致性）。
        每个维度汇总条数，并列出对应知识点ID（与知识库列表页ID一致）及简短原因。
        """
        missing_items = [translate_item(it) for it in missing_result.get('missing_knowledge', [])]
        outdated_items = [translate_item(it) for it in outdated_result.get('outdated_knowledge', [])]
        conflicting_items = [translate_item(it) for it in conflicting_result.get('conflicting_knowledge', [])]
        return build_dimension_recommendations(missing_items, outdated_items, conflicting_items)


def main():
    # 初始化知识库健康度检查器
    # 默认使用 deepseek-v4-flash（完整性/时效性检查）；冲突检查默认同模型
    # 如冲突检测漏检，可单独升级为 conflict_model="deepseek-v4-pro"
    checker = KnowledgeBaseHealthChecker()

    print("=== 知识库健康度检查示例（迪士尼主题乐园） ===\n")

    # 示例知识库（包含一些故意的问题）
    knowledge_base = [
        {
            "id": "kb_001",
            "content": "上海迪士尼乐园位于上海市浦东新区，是中国大陆首座迪士尼主题乐园，于2016年6月16日开园。乐园占地面积390公顷，包含七大主题园区。",
            "last_updated": "2024-01-15"
        },
        {
            "id": "kb_002",
            "content": "上海迪士尼乐园的门票价格：平日成人票价为399元，周末和节假日为499元。儿童票平日为299元，周末为374元。",
            "last_updated": "2023-12-01"  # 故意设置为较旧的时间
        },
        {
            "id": "kb_003",
            "content": "上海迪士尼乐园门票价格：成人票平日350元，周末450元。儿童票平日250元，周末350元。",  # 故意设置冲突的价格
            "last_updated": "2024-02-01"
        },
        {
            "id": "kb_004",
            "content": "上海迪士尼乐园营业时间为上午8:00至晚上8:00，全年无休。",
            "last_updated": "2024-01-20"
        },
        {
            "id": "kb_005",
            "content": "从上海市区到迪士尼乐园可以乘坐地铁11号线到迪士尼站，或乘坐迪士尼专线巴士。",
            "last_updated": "2024-01-10"
        }
    ]

    # 测试查询
    test_queries = [
        {
            "query": "上海迪士尼乐园在哪里？",
            "expected_answer": "浦东新区"
        },
        {
            "query": "门票多少钱？",
            "expected_answer": "价格信息"
        },
        {
            "query": "营业时间是什么？",
            "expected_answer": "8:00-20:00"
        },
        {
            "query": "怎么去迪士尼？",
            "expected_answer": "地铁11号线"
        },
        {
            "query": "有什么特别活动？",  # 知识库中没有相关信息
            "expected_answer": "活动信息"
        },
        {
            "query": "停车费是多少？",  # 知识库中没有相关信息
            "expected_answer": "停车费信息"
        }
    ]

    # 生成健康度报告
    health_report = checker.generate_health_report(knowledge_base, test_queries)

    # 显示报告
    print("=== 知识库健康度报告 ===\n")
    print(json.dumps(health_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
