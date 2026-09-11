# -*- coding: utf-8 -*-
"""
app.api.knowledge - 知识库管理路由（Flask Blueprint）

包含：
- GET    /api/knowledge/list         -> 列出所有知识库条目
- POST   /api/knowledge/add          -> 新增知识条目（文档/图片/视频）
- DELETE /api/knowledge/delete       -> 删除指定条目
- GET    /api/knowledge/health       -> 流式知识库健康检查（SSE）
- GET    /api/knowledge/rebuild      -> 流式重建知识库索引（SSE，删除旧索引后调 builder）
- GET    /api/knowledge/distill      -> 流式对话知识沉淀（SSE，读 conversations 表 -> 提取 -> 合并）
- POST   /api/knowledge/distill/add  -> 将单条沉淀知识点添加进知识库（建立文本索引）
- GET    /api/knowledge/distill/export -> 导出最近一次沉淀结果为 jsonl
"""
import hashlib
import json
import logging
import os
import queue
import sqlite3
import threading
from datetime import datetime

from flask import Blueprint, jsonify, request, Response
from werkzeug.utils import secure_filename

from app.api.auth import require_login
from app.config import DB_FILE, DOCS_DIR, IMG_DIR, UPLOAD_DOC_EXT, UPLOAD_IMG_EXT
from app.core.health_check import KnowledgeBaseHealthChecker, score_to_level
from app.core.knowledge_distill import (
    load_distill_cache,
    run_distill_pipeline,
    save_distill_cache,
)
from app.index import builder
from app.index.builder import ProgressCallback
from app.state import get_kb_manager, get_metadata, load_resources, save_resources

logger = logging.getLogger(__name__)

knowledge_bp = Blueprint("knowledge", __name__)


# ========== 路由 ==========
@knowledge_bp.route("/api/knowledge/list", methods=["GET"])
def api_knowledge_list():
    """列出所有知识库条目"""
    guard = require_login()
    if guard:
        return guard
    metadata = get_metadata()
    items = []
    for m in metadata:
        content = m.get("content", "")
        items.append({
            "id": m["id"],
            "source": m.get("source", ""),
            "type": m.get("type", ""),
            "content_preview": content[:80].replace("\n", " ") + ("..." if len(content) > 80 else ""),
            "content_full": content,
            "path": m.get("path"),
            "url": m.get("url")
        })
    stats = {
        "text": sum(1 for m in metadata if m["type"] == "text"),
        "image": sum(1 for m in metadata if m["type"] == "image"),
        "video": sum(1 for m in metadata if m["type"] == "video"),
        "total": len(metadata)
    }
    return jsonify({"code": 0, "data": {"items": items, "stats": stats}})


@knowledge_bp.route("/api/knowledge/add", methods=["POST"])
def api_knowledge_add():
    """新增知识条目：支持 docx 文档 / 图片 / 视频URL"""
    guard = require_login()
    if guard:
        return guard
    kb_manager = get_kb_manager()
    try:
        entry_type = (request.form.get("type") or "").strip()
        if entry_type == "doc":
            f = request.files.get("file")
            if not f or not f.filename:
                return jsonify({"code": 1, "message": "请上传docx文档"}), 400
            ext = os.path.splitext(f.filename)[1].lower()
            if ext not in UPLOAD_DOC_EXT:
                return jsonify({"code": 1, "message": "仅支持docx文档"}), 400
            os.makedirs(DOCS_DIR, exist_ok=True)
            save_name = secure_filename(f.filename)
            if not save_name:
                save_name = "upload.docx"
            save_path = os.path.join(DOCS_DIR, save_name)
            f.save(save_path)
            full_text = builder.parse_docx(save_path)
            chunks = builder.split_text(full_text)
            # 每个 chunk 作为一个独立文档入库（doc_id 用文件名+序号区分）
            for i, chunk in enumerate(chunks):
                kb_manager.add_document(
                    doc_id=f"doc:{save_name}#{i}",
                    text=chunk,
                    source=save_name
                )
            save_resources()
            return jsonify({"code": 0, "message": f"文档已入库，切分为{len(chunks)}个片段"})

        elif entry_type == "image":
            f = request.files.get("file")
            if not f or not f.filename:
                return jsonify({"code": 1, "message": "请上传图片"}), 400
            ext = os.path.splitext(f.filename)[1].lower()
            if ext not in UPLOAD_IMG_EXT:
                return jsonify({"code": 1, "message": "仅支持png/jpg/jpeg/gif/bmp图片"}), 400
            os.makedirs(IMG_DIR, exist_ok=True)
            save_name = secure_filename(f.filename)
            if not save_name:
                save_name = "image" + ext
            save_path = os.path.join(IMG_DIR, save_name)
            f.save(save_path)
            kb_manager.add_image(save_path, filename=save_name)
            save_resources()
            return jsonify({"code": 0, "message": "图片已入库"})

        elif entry_type == "video":
            url = (request.form.get("url") or "").strip()
            description = (request.form.get("description") or "").strip()
            if not url:
                return jsonify({"code": 1, "message": "请填写视频URL"}), 400
            if not description:
                description = "自定义视频"
            kb_manager.add_video(url, description)
            save_resources()
            return jsonify({"code": 0, "message": "视频已入库"})

        else:
            return jsonify({"code": 1, "message": "type参数无效，应为 doc/image/video"}), 400
    except Exception as e:
        return jsonify({"code": 1, "message": f"入库失败: {str(e)}"}), 500


@knowledge_bp.route("/api/knowledge/delete", methods=["DELETE"])
def api_knowledge_delete():
    """删除指定id的知识条目（标记删除 + 立即重建索引）"""
    guard = require_login()
    if guard:
        return guard
    kb_manager = get_kb_manager()
    metadata = get_metadata()
    target_id = request.args.get("id", type=int)
    if target_id is None:
        return jsonify({"code": 1, "message": "缺少id参数"}), 400

    # 找到 metadata 中对应内部 id 的条目
    targets = [m for m in metadata if m.get("id") == target_id]
    if not targets:
        return jsonify({"code": 1, "message": "未找到该条目"}), 404

    # 按 doc_id 标记删除（一个 chunk 只有一条 doc_id 对应）
    for m in targets:
        doc_id = m.get("doc_id")
        if doc_id:
            kb_manager.delete_document(doc_id)

    # 重建索引并持久化
    save_resources()
    return jsonify({"code": 0, "message": f"已删除{len(targets)}条记录"})


# ========== SSE 工具函数 ==========
def _sse(data):
    """SSE 数据行格式化"""
    return "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"


# ========== 健康检查 SSE ==========
@knowledge_bp.route("/api/knowledge/health", methods=["GET"])
def api_knowledge_health():
    """流式知识库健康检查（SSE）"""
    guard = require_login()
    if guard:
        return guard
    metadata = get_metadata()

    def generate():
        # 1. 收集有效文本 chunks
        yield _sse({"type": "progress", "text": "正在收集文本知识片段..."})
        text_chunks = [
            {
                "id": str(m.get("id", "unknown")),
                "content": m.get("content", ""),
                "last_updated": m.get("last_updated", datetime.now().strftime('%Y-%m-%d'))
            }
            for m in metadata
            if m.get("type") == "text" and not m.get("deleted", False)
        ]
        yield _sse({"type": "progress", "text": f"共收集到 {len(text_chunks)} 个文本片段"})

        if not text_chunks:
            yield _sse({"type": "error", "text": "知识库中没有文本片段，无法执行健康检查"})
            return

        # 2. 自动生成测试查询（每个 chunk 取前 30 字作为 query）
        yield _sse({"type": "progress", "text": "正在生成测试查询..."})
        test_queries = []
        for chunk in text_chunks[:20]:
            content = chunk["content"]
            query = content[:30] + ("..." if len(content) > 30 else "")
            test_queries.append({"query": query, "expected_answer": content[:50]})
        yield _sse({"type": "progress", "text": f"已生成 {len(test_queries)} 个测试查询"})

        # 3. 分批执行健康检查
        checker = KnowledgeBaseHealthChecker()
        batch_size = 20
        batch_reports = []
        batch_count = (len(text_chunks) + batch_size - 1) // batch_size

        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i:i + batch_size]
            yield _sse({"type": "progress", "text": f"正在检查第 {i // batch_size + 1}/{batch_count} 批文本片段..."})
            try:
                report = checker.generate_health_report(batch, test_queries)
                batch_reports.append(report)
            except Exception as e:
                yield _sse({"type": "error", "text": f"第 {i // batch_size + 1} 批检查失败: {str(e)}"})
                return

        # 4. 聚合报告
        yield _sse({"type": "progress", "text": "正在聚合检查结果..."})
        if not batch_reports:
            yield _sse({"type": "error", "text": "没有生成任何检查报告"})
            return

        # 4. 聚合报告：以中文化报告为基础，合并多批次信息
        yield _sse({"type": "progress", "text": "正在聚合检查结果..."})

        base_report = batch_reports[0]
        # 累加所有批次的 token 使用量
        usage_sum = {
            "llm_call_count": sum(r.get("成本追踪", {}).get("call_count", 0) for r in batch_reports),
            "prompt_tokens": sum(r.get("成本追踪", {}).get("prompt_tokens", 0) for r in batch_reports),
            "completion_tokens": sum(r.get("成本追踪", {}).get("completion_tokens", 0) for r in batch_reports),
            "total_tokens": sum(r.get("成本追踪", {}).get("total_tokens", 0) for r in batch_reports),
            "estimated_cost": round(sum(r.get("成本追踪", {}).get("estimated_cost", 0) for r in batch_reports), 6),
            "cost_unit": "元",
            "cost_note": "按 deepseek-v4-flash 典型单价估算，仅供参考",
            "start_time": batch_reports[0].get("成本追踪", {}).get("start_time"),
            "end_time": batch_reports[-1].get("成本追踪", {}).get("end_time"),
            "duration_seconds": round(sum(r.get("成本追踪", {}).get("duration_seconds", 0) for r in batch_reports), 2),
        }

        # 合并问题列表（各批次追加）
        missing_items = []
        outdated_items = []
        conflicting_items = []
        for r in batch_reports:
            missing_items.extend(r.get("缺少的知识", {}).get("问题列表", []))
            outdated_items.extend(r.get("过期的知识", {}).get("问题列表", []))
            conflicting_items.extend(r.get("冲突的知识", {}).get("问题列表", []))

        # 重新计算总体均分（基于各批次总体健康分的原始得分）
        avg_score = sum(r.get("总体健康分", {}).get("原始得分_0到1", 0) for r in batch_reports) / len(batch_reports)
        overall_score_10, overall_level = score_to_level(avg_score)

        # 合并多批次的维度改进建议（按维度累加条数与知识点）
        merged_dimensions = [
            {"维度": "完整性", "建议条数": 0, "涉及知识点": []},
            {"维度": "时效性", "建议条数": 0, "涉及知识点": []},
            {"维度": "一致性", "建议条数": 0, "涉及知识点": []},
        ]
        for r in batch_reports:
            for dim in r.get("改进建议", []):
                for md in merged_dimensions:
                    if md["维度"] == dim.get("维度"):
                        md["建议条数"] += dim.get("建议条数", 0)
                        md["涉及知识点"].extend(dim.get("涉及知识点", []))
                        break

        aggregated = {
            "总体健康分": {
                "分数_10分制": overall_score_10,
                "等级描述": overall_level,
                "原始得分_0到1": round(avg_score, 4),
            },
            "覆盖率": base_report.get("覆盖率", {}),
            "新鲜度": base_report.get("新鲜度", {}),
            "一致性": base_report.get("一致性", {}),
            "缺少的知识": {
                "说明": base_report.get("缺少的知识", {}).get("说明", ""),
                "问题列表": missing_items,
            },
            "过期的知识": {
                "说明": base_report.get("过期的知识", {}).get("说明", ""),
                "问题列表": outdated_items,
            },
            "冲突的知识": {
                "说明": base_report.get("冲突的知识", {}).get("说明", ""),
                "问题列表": conflicting_items,
            },
            "改进建议": merged_dimensions,
            "检查时间": datetime.now().isoformat(),
            "成本追踪": usage_sum,
            "batch_count": len(batch_reports),
            "total_text_chunks": len(text_chunks),
        }
        yield _sse({"type": "result", "data": aggregated})

    return Response(generate(), mimetype="text/event-stream")


# ========== 重建索引 SSE ==========
@knowledge_bp.route("/api/knowledge/rebuild", methods=["GET"])
def api_knowledge_rebuild():
    """流式重建知识库索引（SSE）

    流程：clean_index_files() -> builder.build_and_save(progress_callback) -> load_resources() 刷新内存
    耗时较长（embedding + 多样化改写），通过 SSE 流式推送进度给前端
    """
    guard = require_login()
    if guard:
        return guard

    def generate():
        # 进度消息队列（后台线程写入，SSE generator 读取并 yield）
        progress_queue = queue.Queue()
        # 标志后台线程是否完成
        done_flag = {"finished": False, "error": None}

        def progress_cb(stage, current, total, message):
            """builder.build_and_save 的进度回调：把消息塞进队列"""
            progress_queue.put({"type": "progress", "stage": stage, "current": current, "total": total, "text": message})

        def run_rebuild():
            """后台线程：删除旧文件 + 重新构建索引"""
            try:
                # 1. 删除旧索引文件
                deleted = builder.clean_index_files()
                progress_queue.put({"type": "progress", "stage": "clean", "current": len(deleted), "total": 0, "text": f"已清理旧索引文件: {deleted}"})
                # 2. 重新构建索引
                builder.build_and_save(progress_callback=progress_cb)
                # 3. 刷新内存中的索引/元数据/向量/管理器
                load_resources()
            except Exception as e:
                done_flag["error"] = str(e)
            finally:
                done_flag["finished"] = True
                progress_queue.put(None)  # 哨兵值，通知 SSE generator 结束

        # 启动后台线程
        thread = threading.Thread(target=run_rebuild, daemon=True)
        thread.start()

        yield _sse({"type": "progress", "stage": "init", "current": 0, "total": 0, "text": "初始化任务已启动..."})

        # 流式读取队列消息
        while True:
            msg = progress_queue.get()
            if msg is None:  # 哨兵值，结束
                break
            yield _sse(msg)

        if done_flag["error"]:
            yield _sse({"type": "error", "text": f"重建失败: {done_flag['error']}"})
        else:
            yield _sse({"type": "result", "text": "索引重建完成，内存已刷新"})

    return Response(generate(), mimetype="text/event-stream")


# ========== 对话知识沉淀 SSE ==========
def _load_conversations_grouped_by_session():
    """从 conversations 表按 session_id 聚合，仅返回 chat_sessions.ended_at IS NOT NULL 的会话
    （即多轮对话在结束/页面关闭/浏览器卸载后才"沉淀"到 DB 可被 distillation 抽取）。
    返回 list[dict]: {session_id, messages: [(role, content)]}
    """
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    # INNER JOIN + WHERE ended_at IS NOT NULL：进行中的会话不会被抽取
    cur.execute("""
        SELECT c.session_id, c.role, c.content, c.created_at
        FROM conversations c
        INNER JOIN chat_sessions s ON s.session_id = c.session_id
        WHERE s.ended_at IS NOT NULL
        ORDER BY c.session_id, c.id ASC
    """)
    rows = cur.fetchall()
    conn.close()
    grouped = {}
    for r in rows:
        grouped.setdefault(r["session_id"], []).append(
            {"role": r["role"], "content": r["content"], "created_at": r["created_at"]}
        )
    return [
        {"session_id": sid, "messages": msgs}
        for sid, msgs in grouped.items()
    ]


def _format_conversations_for_llm(grouped_sessions):
    """把每个 session 的消息列表拼成 LLM 易读的对话文本"""
    blocks = []
    for sess in grouped_sessions:
        sid_short = sess["session_id"][:8]
        lines = [f"=== 会话 {sid_short} ==="]
        for m in sess["messages"]:
            role_label = "用户" if m["role"] == "user" else "助手"
            content = (m["content"] or "").strip()
            if not content:
                continue
            lines.append(f"{role_label}: {content}")
        blocks.append("\n".join(lines))
    return blocks


@knowledge_bp.route("/api/knowledge/distill", methods=["GET"])
def api_knowledge_distill():
    """流式对话知识沉淀（SSE）
    流程：读 conversations -> 按 session 聚合 -> LLM 提取 -> 按 type 合并 -> 缓存到 disney_distilled.json
    """
    guard = require_login()
    if guard:
        return guard

    def generate():
        progress_queue = queue.Queue()
        done_flag = {"finished": False, "error": None, "result": None}

        def progress_cb(stage, current, total, message):
            progress_queue.put({
                "type": "progress",
                "stage": stage,
                "current": current,
                "total": total,
                "text": message,
            })

        def run_distill():
            try:
                # 1. 加载对话
                progress_cb("load", 0, 0, "正在从 conversations 表加载历史对话...")
                sessions = _load_conversations_grouped_by_session()
                progress_cb("load", len(sessions), len(sessions),
                            f"已加载 {len(sessions)} 个会话，共 "
                            f"{sum(len(s['messages']) for s in sessions)} 条消息")
                if not sessions:
                    raise ValueError("conversations 表为空，无可沉淀的对话")

                # 2. 拼成 LLM 友好的对话文本
                conv_texts = _format_conversations_for_llm(sessions)
                progress_cb("load", len(conv_texts), len(conv_texts),
                            f"已构造 {len(conv_texts)} 段对话文本，开始提取知识点...")

                # 3. 调 LLM 跑完整流程
                result = run_distill_pipeline(conv_texts, progress_callback=progress_cb)
                # 4. 写缓存
                save_distill_cache(result)
                done_flag["result"] = result
            except Exception as e:
                done_flag["error"] = str(e)
            finally:
                done_flag["finished"] = True
                progress_queue.put(None)

        thread = threading.Thread(target=run_distill, daemon=True)
        thread.start()

        yield _sse({"type": "progress", "stage": "init", "current": 0, "total": 0, "text": "沉淀任务已启动..."})

        while True:
            msg = progress_queue.get()
            if msg is None:
                break
            yield _sse(msg)

        if done_flag["error"]:
            yield _sse({"type": "error", "text": f"沉淀失败: {done_flag['error']}"})
        else:
            result = done_flag["result"] or {}
            yield _sse({
                "type": "result",
                "data": {
                    "extracted_count": result.get("extracted_count", 0),
                    "merged_count": result.get("merged_count", 0),
                    "filtered_count": result.get("filtered_count", 0),
                    "raw_count": result.get("raw_count", 0),
                    "merged_knowledge": result.get("merged_knowledge", []),
                    "cost": result.get("cost", {}),
                },
            })

    return Response(generate(), mimetype="text/event-stream")


@knowledge_bp.route("/api/knowledge/distill/add", methods=["POST"])
def api_knowledge_distill_add():
    """将单条沉淀知识点添加进知识库（参考 builder 文本索引：embedding + 多样化改写）"""
    guard = require_login()
    if guard:
        return guard

    data = request.get_json(silent=True) or {}
    content = (data.get("content") or "").strip()
    if not content:
        return jsonify({"code": 1, "message": "知识内容不能为空"}), 400

    knowledge_type = (data.get("knowledge_type") or "沉淀").strip()
    category = (data.get("category") or "").strip()
    doc_id = f"distill:{hashlib.md5(content.encode('utf-8')).hexdigest()}"
    source = f"对话沉淀:{knowledge_type}"
    if category:
        source = f"{source}({category})"

    kb_manager = get_kb_manager()
    try:
        added_count = kb_manager.add_text_chunk(
            text=content,
            source=source,
            doc_id=doc_id,
            metadata={
                "knowledge_type": knowledge_type,
                "category": category,
                "from_distill": True,
            },
        )
        save_resources()
        if added_count == 0:
            return jsonify({"code": 0, "message": "该知识点已在知识库中，未重复添加", "data": {"added_count": 0}})
        return jsonify({
            "code": 0,
            "message": f"已添加进知识库（共 {added_count} 条索引）",
            "data": {"added_count": added_count, "doc_id": doc_id},
        })
    except Exception as e:
        logger.exception("沉淀知识入库失败")
        return jsonify({"code": 1, "message": f"添加失败: {str(e)}"}), 500


@knowledge_bp.route("/api/knowledge/distill/export", methods=["GET"])
def api_knowledge_distill_export():
    """导出最近一次沉淀结果为 jsonl（每条合并后的知识点一行）"""
    guard = require_login()
    if guard:
        return guard
    cache = load_distill_cache()
    if not cache:
        return jsonify({"code": 1, "message": "尚未生成沉淀结果，请先点击「开始沉淀」"}), 404

    def generate():
        # 头部元信息
        meta = {
            "_type": "meta",
            "saved_at": cache.get("saved_at"),
            "extracted_count": cache.get("extracted_count", 0),
            "merged_count": cache.get("merged_count", 0),
            "filtered_count": cache.get("filtered_count", 0),
            "raw_count": cache.get("raw_count", 0),
        }
        yield (json.dumps(meta, ensure_ascii=False) + "\n").encode("utf-8")
        # 每条合并后的知识点一行
        for k in cache.get("merged_knowledge", []):
            yield (json.dumps(k, ensure_ascii=False) + "\n").encode("utf-8")

    return Response(generate(), mimetype="application/x-ndjson", headers={
        "Content-Disposition": "attachment; filename=distilled_knowledge.jsonl"
    })
