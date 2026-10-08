# -*- coding: utf-8 -*-
"""
experiments.run_eval - 一站式评测总入口

串联四个评测子模块，评测结束后汇总输出总报告（JSON + Markdown 文档）：
1. build_eval_dataset          生成评测集（50 条复杂查询 + 15 条图文混合查询）
2. cross_modal_recall          跨模态 Recall@5 评测（对比纯文本基线）
3. agent_accuracy_eval         首答准确率 + 幻觉率评测（基线 RAG vs Agent）
4. qdrant_faiss_recall_benchmark  Qdrant vs FAISS 召回质量离线对比（HNSW 调参）

汇总报告输出到：
- data/stats/eval_summary_report.json  （机器可读结构化数据）
- data/stats/eval_report.md            （人类可读 Markdown 文档）

使用方法：
    export AGICTO_API_KEY=xxx
    export DASHSCOPE_API_KEY=xxx
    export QDRANT_URL=http://localhost:6333   # 连 docker qdrant，避免与 Flask 本地存储锁冲突
    python experiments/run_eval.py

依赖：
- docker qdrant 上需已构建索引（python scripts/build_index.py，会调 DashScope embedding）
- 四个子模块脚本：build_eval_dataset / cross_modal_recall / agent_accuracy_eval / qdrant_faiss_recall_benchmark
- Step 4 额外依赖 faiss-cpu（未安装时自动跳过该步并标注）
"""
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional

# 把工程根加入 path，使 app.* 包与 experiments.* 包可被导入
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app.config import STATS_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ========== 子模块与报告文件 ==========
from experiments import (
    build_eval_dataset as step1,
    cross_modal_recall as step2,
    agent_accuracy_eval as step3,
)

# Step 4 延迟导入：faiss 未安装时 gracefully 跳过
step4 = None
try:
    from experiments import qdrant_faiss_recall_benchmark as step4_mod
    step4 = step4_mod
except Exception:
    step4 = None

SUMMARY_REPORT_FILE = os.path.join(STATS_DIR, "eval_summary_report.json")
MARKDOWN_REPORT_FILE = os.path.join(STATS_DIR, "eval_report.md")


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    """安全加载 JSON 报告文件，不存在或解析失败返回 None"""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("加载报告失败 %s: %s", path, e)
        return None


def _build_summary(
    dataset: Optional[Dict],
    cross_modal: Optional[Dict],
    accuracy: Optional[Dict],
    recall_benchmark: Optional[Dict],
    step_status: Dict[str, str],
    total_duration: float,
) -> Dict[str, Any]:
    """汇总四个子报告为总报告"""
    # 评测集摘要
    dataset_summary = {}
    if dataset:
        meta = dataset.get("metadata", {})
        dataset_summary = {
            "complex_queries": len(dataset.get("complex_queries", [])),
            "cross_modal_queries": len(dataset.get("cross_modal_queries", [])),
            "kb_vector_count": meta.get("kb_vector_count"),
            "generated_at": meta.get("generated_at"),
        }

    # 跨模态 Recall@5 摘要
    cross_modal_summary = {}
    if cross_modal:
        cm = cross_modal.get("cross_modal", {})
        tb = cross_modal.get("text_only_baseline", {})
        cross_modal_summary = {
            "cross_modal_recall@5": cm.get("recall@5"),
            "cross_modal_hits": cm.get("hit_count"),
            "text_only_baseline_recall@5": tb.get("recall@5"),
            "text_only_hits": tb.get("hit_count"),
            "total_queries": cross_modal.get("metadata", {}).get("total_queries"),
            "top_k": cross_modal.get("metadata", {}).get("top_k"),
        }

    # 准确率 + 幻觉率摘要
    accuracy_summary = {}
    if accuracy:
        baseline = accuracy.get("baseline", {})
        agent = accuracy.get("agent", {})
        meta = accuracy.get("metadata", {})
        accuracy_summary = {
            "baseline_first_answer_accuracy": baseline.get("first_answer_accuracy"),
            "agent_first_answer_accuracy": agent.get("first_answer_accuracy"),
            "baseline_hallucination_rate": baseline.get("hallucination_rate"),
            "agent_hallucination_rate": agent.get("hallucination_rate"),
            "agent_retry_triggered_count": agent.get("retry_triggered_count"),
            "total_queries": meta.get("total_queries"),
            "judge_model": meta.get("judge_model"),
            "hallucination_threshold": meta.get("hallucination_threshold"),
            "judge_cost": accuracy.get("judge_cost"),
        }

    # Qdrant vs FAISS 召回对比摘要
    recall_summary = {}
    if recall_benchmark:
        meta = recall_benchmark.get("metadata", {})
        recall_summary = {
            "total_queries": meta.get("total_queries"),
            "total_vectors": meta.get("total_vectors"),
            "search_top_k": meta.get("search_top_k"),
            "gt_top_k": meta.get("gt_top_k"),
            "faiss": recall_benchmark.get("faiss", {}),
            "qdrant": recall_benchmark.get("qdrant", {}),
            "recommendation": recall_benchmark.get("recommendation", {}),
            "scale_valid": meta.get("scale_valid"),
            "scale_warning": meta.get("scale_warning"),
        }

    return {
        "metadata": {
            "evaluated_at": datetime.now().isoformat(),
            "total_duration_seconds": round(total_duration, 2),
            "step_status": step_status,
        },
        "dataset": dataset_summary,
        "cross_modal_recall": cross_modal_summary,
        "accuracy_and_hallucination": accuracy_summary,
        "recall_benchmark": recall_summary,
    }


def _print_summary_table(summary: Dict[str, Any]) -> None:
    """打印可读的汇总表到控制台"""
    print("\n" + "=" * 70)
    print("  评测汇总报告")
    print("=" * 70)

    meta = summary.get("metadata", {})
    print("  评测时间: %s" % meta.get("evaluated_at"))
    print("  总耗时:   %s 秒" % meta.get("total_duration_seconds"))
    print("  各步状态:")
    for step, status in meta.get("step_status", {}).items():
        print("    - %s: %s" % (step, status))

    ds = summary.get("dataset", {})
    if ds:
        print("\n  [评测集]")
        print("    复杂查询:    %s 条" % ds.get("complex_queries", 0))
        print("    图文混合查询: %s 条" % ds.get("cross_modal_queries", 0))
        print("    知识库向量:   %s 条" % ds.get("kb_vector_count", "N/A"))

    cm = summary.get("cross_modal_recall", {})
    if cm and cm.get("cross_modal_recall@5") is not None:
        print("\n  [跨模态 Recall@5]")
        print("    跨模态检索:     %.2f%%（%s/%s）" % (
            (cm.get("cross_modal_recall@5") or 0) * 100,
            cm.get("cross_modal_hits", 0),
            cm.get("total_queries", 0)))
        print("    纯文本基线:     %.2f%%（%s/%s）" % (
            (cm.get("text_only_baseline_recall@5") or 0) * 100,
            cm.get("text_only_hits", 0),
            cm.get("total_queries", 0)))

    acc = summary.get("accuracy_and_hallucination", {})
    if acc and acc.get("baseline_first_answer_accuracy") is not None:
        print("\n  [首答准确率 + 幻觉率]")
        print("    基线 RAG:")
        print("      首答准确率: %.2f%%" % (acc.get("baseline_first_answer_accuracy", 0) * 100))
        print("      幻觉率:     %.2f%%" % (acc.get("baseline_hallucination_rate", 0) * 100))
        print("    Agent (ReAct + 验证自纠错):")
        print("      首答准确率: %.2f%%" % (acc.get("agent_first_answer_accuracy", 0) * 100))
        print("      幻觉率:     %.2f%%" % (acc.get("agent_hallucination_rate", 0) * 100))
        print("      触发重检索: %s 次" % acc.get("agent_retry_triggered_count", 0))
        jc = acc.get("judge_cost", {}) or {}
        print("    judge 成本: %s 元，%s 次调用，%s tokens" % (
            jc.get("estimated_cost", 0), jc.get("call_count", 0), jc.get("total_tokens", 0)))

    rb = summary.get("recall_benchmark", {})
    if rb and rb.get("faiss"):
        print("\n  [Qdrant vs FAISS 召回对比]")
        fm = rb.get("faiss", {})
        print("    FAISS: recall@5=%.4f, recall@10=%.4f, mrr=%.4f" % (
            fm.get("recall@5", 0), fm.get("recall@10", 0), fm.get("mrr", 0)))
        rec = rb.get("recommendation", {})
        print("    推荐 ef: %s" % rec.get("best_ef", "N/A"))

    print("\n  汇总报告(JSON): %s" % SUMMARY_REPORT_FILE)
    print("  汇总报告(MD):   %s" % MARKDOWN_REPORT_FILE)
    print("=" * 70)


def _build_markdown_report(summary: Dict[str, Any]) -> str:
    """生成人类可读的 Markdown 评测报告文档"""
    meta = summary.get("metadata", {})
    lines = []

    # ===== 文档头 =====
    lines.append("# 多模态 RAG 系统评测报告")
    lines.append("")
    lines.append("| 项目 | 值 |")
    lines.append("|------|-----|")
    lines.append("| 评测时间 | %s |" % meta.get("evaluated_at", "N/A"))
    lines.append("| 总耗时 | %s 秒 |" % meta.get("total_duration_seconds", "N/A"))

    step_status = meta.get("step_status", {})
    lines.append("| 步骤数 | %d 步 |" % len(step_status))
    for step, status in step_status.items():
        lines.append("| %s | %s |" % (step, status))
    lines.append("")

    # ===== 1. 评测集概览 =====
    ds = summary.get("dataset", {})
    if ds:
        lines.append("## 1. 评测集概览")
        lines.append("")
        lines.append("| 维度 | 数量 |")
        lines.append("|------|------|")
        lines.append("| 复杂查询（用于准确率/幻觉率评测） | %s 条 |" % ds.get("complex_queries", 0))
        lines.append("| 图文混合查询（用于跨模态召回评测） | %s 条 |" % ds.get("cross_modal_queries", 0))
        lines.append("| 知识库向量总数 | %s 条 |" % ds.get("kb_vector_count", "N/A"))
        lines.append("| 评测集生成时间 | %s |" % ds.get("generated_at", "N/A"))
        lines.append("")
        lines.append("> 评测集由 LLM 自动生成：从知识库 docx 切分 chunk 后，调 LLM 改写为多实体/跨条件的复杂查询；"
                     "图文混合查询基于 Qdrant 中实际存在的图片/视频条目生成。")
        lines.append("")

    # ===== 2. 跨模态 Recall@5 =====
    cm = summary.get("cross_modal_recall", {})
    if cm and cm.get("cross_modal_recall@5") is not None:
        lines.append("## 2. 跨模态 Recall@5")
        lines.append("")
        lines.append("**评测口径：** 文本/图片/视频统一嵌入同一向量空间，对图文混合查询检索 Top-%s，"
                     "判定 Top-K 的模态集合是否覆盖期望模态（含期望模态视为命中）。"
                     "纯文本基线仅在文本类结果中取 Top-K，用于对比证明跨模态统一嵌入的价值。" % cm.get("top_k", 5))
        lines.append("")
        lines.append("| 检索方式 | Recall@5 | 命中数 | 总查询数 |")
        lines.append("|----------|---------|--------|---------|")
        lines.append("| 跨模态检索（文本+图片+视频统一嵌入） | %.2f%% | %s | %s |" % (
            (cm.get("cross_modal_recall@5") or 0) * 100,
            cm.get("cross_modal_hits", 0),
            cm.get("total_queries", 0)))
        lines.append("| 纯文本基线（仅文本模态） | %.2f%% | %s | %s |" % (
            (cm.get("text_only_baseline_recall@5") or 0) * 100,
            cm.get("text_only_hits", 0),
            cm.get("total_queries", 0)))
        lines.append("")

    # ===== 3. 首答准确率 + 幻觉率 =====
    acc = summary.get("accuracy_and_hallucination", {})
    if acc and acc.get("baseline_first_answer_accuracy") is not None:
        lines.append("## 3. 首答准确率 + 幻觉率")
        lines.append("")
        lines.append("**评测口径：**")
        lines.append("- **首答准确率：** LLM-as-judge 判定 Agent/基线的首次回答是否与期望答案语义一致")
        lines.append("- **幻觉率：** 验证器三维打分（准确性/引用/推理）综合分 < %.1f 判定为幻觉" % (
            acc.get("hallucination_threshold", 0.6)))
        lines.append("- **judge 模型：** %s" % acc.get("judge_model", "N/A"))
        lines.append("")
        lines.append("| 系统 | 首答准确率 | 幻觉率 | 触发重检索 |")
        lines.append("|------|-----------|--------|-----------|")
        lines.append("| 基线 RAG | %.2f%% | %.2f%% | - |" % (
            (acc.get("baseline_first_answer_accuracy") or 0) * 100,
            (acc.get("baseline_hallucination_rate") or 0) * 100))
        lines.append("| Agent（ReAct + 验证自纠错） | %.2f%% | %.2f%% | %s 次 |" % (
            (acc.get("agent_first_answer_accuracy") or 0) * 100,
            (acc.get("agent_hallucination_rate") or 0) * 100,
            acc.get("agent_retry_triggered_count", 0)))
        lines.append("")

        jc = acc.get("judge_cost", {}) or {}
        if jc:
            lines.append("**judge 成本：** %.4f 元，%s 次调用，%s tokens" % (
                jc.get("estimated_cost", 0), jc.get("call_count", 0), jc.get("total_tokens", 0)))
            lines.append("")

    # ===== 4. Qdrant vs FAISS 召回对比 =====
    rb = summary.get("recall_benchmark", {})
    if rb and (rb.get("faiss") or rb.get("qdrant")):
        lines.append("## 4. Qdrant vs FAISS 召回质量离线对比")
        lines.append("")
        lines.append("**实验设计：**")
        lines.append("- 测试 query 数量：%s 条" % rb.get("total_queries", "N/A"))
        lines.append("- 向量总数：%s 条" % rb.get("total_vectors", "N/A"))
        lines.append("- Ground Truth：FAISS IndexFlatL2 暴力搜索精确 Top-%s" % rb.get("gt_top_k", 20))
        lines.append("- 检索 Top-K：%s" % rb.get("search_top_k", 10))
        lines.append("")

        # 规模警告
        if not rb.get("scale_valid", True):
            lines.append("> **规模警告：** %s" % rb.get("scale_warning", ""))
            lines.append("")

        # FAISS 指标
        fm = rb.get("faiss", {})
        if fm:
            lines.append("### 4.1 FAISS 基线指标")
            lines.append("")
            lines.append("| 指标 | 值 |")
            lines.append("|------|-----|")
            lines.append("| 索引类型 | %s |" % fm.get("index_type", "N/A"))
            lines.append("| Recall@1 | %.4f |" % fm.get("recall@1", 0))
            lines.append("| Recall@3 | %.4f |" % fm.get("recall@3", 0))
            lines.append("| Recall@5 | %.4f |" % fm.get("recall@5", 0))
            lines.append("| Recall@10 | %.4f |" % fm.get("recall@10", 0))
            lines.append("| MRR | %.4f |" % fm.get("mrr", 0))
            lat = fm.get("latency", {})
            lines.append("| 延迟 P50 | %.3f ms |" % lat.get("p50_ms", 0))
            lines.append("| 延迟 P95 | %.3f ms |" % lat.get("p95_ms", 0))
            lines.append("| 延迟 P99 | %.3f ms |" % lat.get("p99_ms", 0))
            lines.append("| 延迟均值 | %.3f ms |" % lat.get("mean_ms", 0))
            lines.append("")

        # Qdrant 各 ef 档位指标
        qdrant = rb.get("qdrant", {})
        if qdrant:
            lines.append("### 4.2 Qdrant HNSW 参数网格搜索")
            lines.append("")
            lines.append("| hnsw_ef | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | Jaccard@5 vs FAISS | P99延迟(ms) |")
            lines.append("|---------|---------|---------|---------|----------|-----|-------------------|------------|")
            for ef_key in sorted(qdrant.keys()):
                m = qdrant[ef_key]
                lines.append("| %s | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.3f |" % (
                    ef_key,
                    m.get("recall@1", 0),
                    m.get("recall@3", 0),
                    m.get("recall@5", 0),
                    m.get("recall@10", 0),
                    m.get("mrr", 0),
                    m.get("jaccard_top5_vs_faiss", 0),
                    m.get("latency", {}).get("p99_ms", 0)))
            lines.append("")

        # 推荐结论
        rec = rb.get("recommendation", {})
        if rec:
            lines.append("### 4.3 推荐配置")
            lines.append("")
            lines.append("**推荐 ef：** %s" % rec.get("best_ef", "N/A"))
            lines.append("")
            lines.append("**理由：** %s" % rec.get("reason", "N/A"))
            lines.append("")
            lines.append("> Qdrant HNSW 参数调优策略：在 recall@10 达到最高值 99%% 阈值的前提下选最小 ef，"
                         "兼顾召回质量与搜索效率。")
            lines.append("")

    # ===== 5. 评测方法说明 =====
    lines.append("## 5. 评测方法说明")
    lines.append("")
    lines.append("### 5.1 评测集生成")
    lines.append("- 从知识库 docx 切分 chunk（长度 200-500 字），调 LLM 生成多实体/跨条件的复杂查询")
    lines.append("- 基于 Qdrant 中图片/视频条目，调 LLM 生成图文混合查询，每条带期望模态")
    lines.append("- 评测集缓存到 data/stats/eval_dataset.json，已存在则跳过生成")
    lines.append("")
    lines.append("### 5.2 跨模态 Recall@5")
    lines.append("- 对每条图文混合查询计算多模态 embedding（DashScope multimodal-embedding）")
    lines.append("- 跨模态检索：全量 Top-K（含文本/图片/视频/多样化问题）")
    lines.append("- 纯文本基线：仅在文本类结果中取 Top-K")
    lines.append("- 命中条件：Top-K 的模态集合包含所有期望模态")
    lines.append("")
    lines.append("### 5.3 首答准确率 + 幻觉率")
    lines.append("- 基线 RAG：app.core.rag_service.rag_ask_api（单次检索 + 生成）")
    lines.append("- Agent：app.agent.loop.agent_chat（ReAct 多步推理 + 验证自纠错）")
    lines.append("- judge：LLM-as-judge 判定首答是否与期望答案语义一致")
    lines.append("- 幻觉判定：验证器三维打分综合分 < 阈值则判定为幻觉")
    lines.append("")
    lines.append("### 5.4 Qdrant vs FAISS 召回对比")
    lines.append("- 测试 query：docx 改写 + 历史对话，合并去重")
    lines.append("- Ground Truth：FAISS IndexFlatL2 暴力搜索精确 Top-20")
    lines.append("- 双路召回：FAISS backup 索引 vs Qdrant HNSW（多 ef 档位）")
    lines.append("- 指标：Recall@1/3/5/10、MRR、Jaccard 重叠度、P50/P95/P99 延迟")
    lines.append("")

    # ===== 6. 文件清单 =====
    lines.append("## 6. 评测产物文件清单")
    lines.append("")
    lines.append("| 文件 | 说明 |")
    lines.append("|------|------|")
    lines.append("| data/stats/eval_dataset.json | 评测集（复杂查询 + 图文混合查询） |")
    lines.append("| data/stats/cross_modal_recall_report.json | 跨模态 Recall@5 详细报告 |")
    lines.append("| data/stats/accuracy_eval_report.json | 首答准确率 + 幻觉率详细报告 |")
    lines.append("| data/stats/recall_benchmark_report.json | Qdrant vs FAISS 召回对比报告 |")
    lines.append("| data/stats/eval_summary_report.json | 汇总报告（JSON） |")
    lines.append("| data/stats/eval_report.md | 汇总报告（Markdown，即本文件） |")
    lines.append("")

    lines.append("---")
    lines.append("*本报告由 experiments/run_eval.py 自动生成*")
    lines.append("")

    return "\n".join(lines)


def main():
    print("=" * 70)
    print("  一站式评测（评测集生成 -> 跨模态 -> 准确率+幻觉率 -> 召回对比 -> 汇总报告）")
    print("=" * 70)

    os.makedirs(STATS_DIR, exist_ok=True)
    overall_start = time.time()
    step_status: Dict[str, str] = {}

    # ===== Step 1: 评测集生成 =====
    print("\n>>> Step 1/4: 生成评测集 (build_eval_dataset)...")
    t0 = time.time()
    try:
        step1.main()
        step_status["build_eval_dataset"] = "success (%.1fs)" % (time.time() - t0)
    except Exception as e:
        logger.exception("Step 1 评测集生成失败")
        step_status["build_eval_dataset"] = "failed: %s" % str(e)[:200]
        # 评测集是后续步骤的依赖，失败则直接汇总并退出
        summary = _build_summary(None, None, None, None, step_status, time.time() - overall_start)
        _save_and_print(summary)
        return

    # ===== Step 2: 跨模态 Recall@5 =====
    print("\n>>> Step 2/4: 跨模态 Recall@5 评测 (cross_modal_recall)...")
    t0 = time.time()
    if os.path.exists(step2.REPORT_FILE):
        print("  报告已存在，跳过：%s" % step2.REPORT_FILE)
        step_status["cross_modal_recall"] = "cached"
    else:
        try:
            step2.main()
            step_status["cross_modal_recall"] = "success (%.1fs)" % (time.time() - t0)
        except Exception as e:
            logger.exception("Step 2 跨模态评测失败")
            step_status["cross_modal_recall"] = "failed: %s" % str(e)[:200]

    # ===== Step 3: 准确率 + 幻觉率 =====
    print("\n>>> Step 3/4: 首答准确率 + 幻觉率评测 (agent_accuracy_eval)...")
    t0 = time.time()
    if os.path.exists(step3.REPORT_FILE):
        print("  报告已存在，跳过：%s" % step3.REPORT_FILE)
        step_status["agent_accuracy_eval"] = "cached"
    else:
        try:
            step3.main()
            step_status["agent_accuracy_eval"] = "success (%.1fs)" % (time.time() - t0)
        except Exception as e:
            logger.exception("Step 3 准确率评测失败")
            step_status["agent_accuracy_eval"] = "failed: %s" % str(e)[:200]

    # ===== Step 4: Qdrant vs FAISS 召回对比 =====
    print("\n>>> Step 4/4: Qdrant vs FAISS 召回对比 (qdrant_faiss_recall_benchmark)...")
    t0 = time.time()
    if step4 is None:
        print("  faiss 未安装或模块导入失败，跳过此步")
        step_status["recall_benchmark"] = "skipped (faiss not installed)"
    elif os.path.exists(step4.REPORT_FILE):
        print("  报告已存在，跳过：%s" % step4.REPORT_FILE)
        step_status["recall_benchmark"] = "cached"
    else:
        try:
            step4.main()
            step_status["recall_benchmark"] = "success (%.1fs)" % (time.time() - t0)
        except Exception as e:
            logger.exception("Step 4 召回对比失败")
            step_status["recall_benchmark"] = "failed: %s" % str(e)[:200]

    # ===== 汇总 =====
    total_duration = time.time() - overall_start
    dataset = _load_json(step1.EVAL_DATASET_FILE)
    cross_modal = _load_json(step2.REPORT_FILE)
    accuracy = _load_json(step3.REPORT_FILE)
    recall_benchmark = _load_json(step4.REPORT_FILE) if step4 else None
    summary = _build_summary(dataset, cross_modal, accuracy, recall_benchmark, step_status, total_duration)
    _save_and_print(summary)


def _save_and_print(summary: Dict[str, Any]) -> None:
    """保存汇总报告（JSON + Markdown）并打印汇总表"""
    # JSON 报告
    with open(SUMMARY_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("已保存汇总报告(JSON) -> %s", SUMMARY_REPORT_FILE)

    # Markdown 报告
    markdown_content = _build_markdown_report(summary)
    with open(MARKDOWN_REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    logger.info("已保存汇总报告(Markdown) -> %s", MARKDOWN_REPORT_FILE)

    # 控制台汇总
    _print_summary_table(summary)


if __name__ == "__main__":
    main()
