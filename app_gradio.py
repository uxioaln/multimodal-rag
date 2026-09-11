# -*- coding: utf-8 -*-
"""
app_gradio.py - 迪士尼客服助手 RAG 项目的 Gradio 前端单页应用

说明：
- 后端 Flask API 保持不变，本文件仅作为前端，通过 requests / sseclient-py
  与现有 Flask 路由（/api/ask、/api/login、/api/knowledge/*、/api/stats 等）交互。
- 每个浏览器会话对应一个独立的 requests.Session（保存在 gr.State 中），
  以便携带登录 cookie 调用需要鉴权的接口与 SSE 流。
- 标签页一：客服助手（gr.Chatbot + gr.MultimodalTextbox，支持文本/图片/视频渲染）
- 标签页二：管理员后台（登录、重建索引、健康检查、知识沉淀、成本统计，均通过 SSE 流式展示）

运行：
    python app_gradio.py
    # 可选环境变量：
    #   BACKEND_URL   后端 Flask 地址，默认 http://127.0.0.1:5050
    #   GRADIO_PORT   Gradio 监听端口，默认 7860
"""
import json
import logging
import os
import uuid

import gradio as gr
import requests
from sseclient import SSEClient

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ========== 后端地址 ==========
BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:5050").rstrip("/")


# ========== 工具函数 ==========
def media_url(image_path):
    """把后端返回的 image_path 转换为可直接访问的 /media/ URL

    后端 image_path 可能形如：
      - data/knowledge_base/images/xxx.png（绝对或相对路径）
      - disney_knowledge_base/images/xxx.png（历史数据）
    Flask 的 /media/<path:filename> 路由从 DOCS_DIR（knowledge_base 目录）serve 文件，
    因此需要截取 "knowledge_base/" 之后的相对路径，再拼接成完整 URL。
    """
    if not image_path:
        return None
    marker = "knowledge_base/"
    if marker in image_path:
        rel = image_path.rsplit(marker, 1)[1]
    else:
        rel = os.path.basename(image_path)
    return f"{BACKEND}/media/{rel}"


def fmt_references(refs):
    """格式化引用来源：source(similarity)；source(similarity)"""
    if not refs:
        return ""
    parts = []
    for r in refs:
        src = r.get("source", "") or ""
        sim = r.get("similarity", "")
        parts.append(f"{src}({sim})")
    return "；".join(parts)


def build_assistant_message(data):
    """把 /api/ask 返回的 result 构造为 gr.Chatbot 的 assistant 消息

    content 采用列表形式：文本 + 图片元组 + 视频元组，Gradio 会分别渲染。
    """
    answer = (data.get("answer") or "无答案").strip()
    image_path = data.get("image_path")
    video_url = data.get("video_url")
    refs = data.get("references") or []

    text = answer
    ref_str = fmt_references(refs)
    if ref_str:
        text += f"\n\n引用来源：{ref_str}"

    parts = [text]
    img_url = media_url(image_path)
    if img_url:
        # 图片以元组形式追加，Gradio 在消息中内联渲染图片
        parts.append((img_url,))
    if video_url:
        # 视频以元组形式追加，Gradio 渲染为视频播放器
        parts.append((video_url,))

    # 仅文本时直接用字符串，带媒体时用列表
    content = parts if len(parts) > 1 else parts[0]
    return {"role": "assistant", "content": content}


# ========== 会话状态 ==========
def init_session_state():
    """初始化单个 Gradio 会话的状态：独立的 requests.Session + 会话 ID"""
    return {
        "http": requests.Session(),
        # chat_session_id：后端用它聚合同一会话的多轮对话，结束后供知识沉淀抽取
        "chat_session_id": uuid.uuid4().hex[:16],
        "logged_in": False,
        "username": "",
    }


def load_history(session_state):
    """页面加载时尝试从后端获取历史对话并初始化 Chatbot

    后端目前未提供独立的 history 查询接口，此处为"尝试"语义：
    若接口不可用（404 等）则返回空对话，不影响使用。
    """
    http = session_state["http"]
    sid = session_state["chat_session_id"]
    try:
        r = http.get(
            f"{BACKEND}/api/conversation/history",
            params={"session_id": sid},
            timeout=3,
        )
        if r.status_code == 200:
            payload = r.json()
            if payload.get("code") == 0:
                items = (payload.get("data") or {}).get("messages") or []
                history = []
                for m in items:
                    role = "user" if m.get("role") == "user" else "assistant"
                    history.append({"role": role, "content": m.get("content", "")})
                return history
    except Exception as e:
        logger.info("加载历史对话失败（可忽略）: %s", e)
    return []


def end_session(session_state):
    """页面关闭 / 会话结束时调用 POST /api/conversation/end

    后端将该会话标记为已结束（ended_at），知识沉淀时只抽取已结束的会话。
    幂等：重复调用不会改写已结束的会话。
    """
    http = session_state["http"]
    sid = session_state.get("chat_session_id", "")
    if not sid:
        return
    try:
        http.post(
            f"{BACKEND}/api/conversation/end",
            json={"session_id": sid},
            timeout=3,
        )
    except Exception as e:
        logger.info("结束会话请求失败（可忽略）: %s", e)


# Gradio 6 的 unload(fn) 不再接受 inputs/outputs，无法直接读取 gr.State。
# 故在 load 时按 request.session_hash 注册会话状态，unload 时取出并结束会话。
_SESSION_REGISTRY = {}


def end_session_on_unload(request: gr.Request):
    """页面关闭时（Gradio 6 unload 事件）按 session_hash 取出状态并结束会话"""
    sid_hash = getattr(request, "session_hash", None)
    state = _SESSION_REGISTRY.pop(sid_hash, None) if sid_hash else None
    if state:
        end_session(state)


# ========== 客服助手：问答 ==========
def chat_respond(message, history, session_state):
    """用户提交问题后调用后端 /api/ask，流式更新 Chatbot

    流程：追加用户消息 + "正在检索..."占位 -> 调 /api/ask -> 用真实回复替换占位。
    使用 generator 实现中间态更新，提升交互体验。
    """
    msg_dict = message or {}
    text = (msg_dict.get("text") or "").strip()
    if not text:
        yield history, gr.update(value=None), session_state
        return

    # 追加用户消息与助手占位消息
    history = history + [{"role": "user", "content": text},
                         {"role": "assistant", "content": "正在检索知识库并生成答案..."}]
    yield history, gr.update(value=None), session_state

    http = session_state["http"]
    sid = session_state.get("chat_session_id", "anonymous")
    try:
        resp = http.post(
            f"{BACKEND}/api/ask",
            json={"query": text, "session_id": sid, "visitor_id": sid},
            timeout=120,
        )
        payload = resp.json()
        if payload.get("code") == 0:
            history[-1] = build_assistant_message(payload.get("data") or {})
        else:
            history[-1] = {"role": "assistant",
                           "content": f"查询失败：{payload.get('message', '未知错误')}"}
    except Exception as e:
        history[-1] = {"role": "assistant", "content": f"请求异常：{e}"}

    yield history, gr.update(value=None), session_state


# ========== 管理员：登录 / 登出 ==========
def do_login(username, password, session_state):
    """调用 POST /api/login，成功后隐藏登录表单、显示后台功能区"""
    username = (username or "").strip()
    password = password or ""
    if not username or not password:
        return session_state, gr.update(), gr.update(), "账号或密码不能为空"

    http = session_state["http"]
    try:
        resp = http.post(
            f"{BACKEND}/api/login",
            json={"username": username, "password": password},
            timeout=15,
        )
        data = resp.json()
    except Exception as e:
        return session_state, gr.update(visible=True), gr.update(visible=False), f"请求异常：{e}"

    if data.get("code") == 0:
        session_state["logged_in"] = True
        session_state["username"] = (data.get("data") or {}).get("username", username)
        return session_state, gr.update(visible=False), gr.update(visible=True), "登录成功"
    return session_state, gr.update(visible=True), gr.update(visible=False), data.get("message", "登录失败")


def do_logout(session_state):
    """退出登录，恢复登录表单显示"""
    http = session_state["http"]
    try:
        http.post(f"{BACKEND}/api/logout", timeout=10)
    except Exception as e:
        logger.info("登出请求失败（可忽略）: %s", e)
    session_state["logged_in"] = False
    session_state["username"] = ""
    return session_state, gr.update(visible=True), gr.update(visible=False), "已退出登录"


# ========== 通用 SSE 流式消费 ==========
def stream_sse(endpoint, session_state, label):
    """连接后端 SSE 接口，流式把 progress/result/error 推送为累积日志字符串

    本函数仅返回"累积日志文本"（每次 yield 一个字符串），
    适用于只有单一日志输出框的重建索引场景。result 只有文本、无结构化数据。
    """
    if not session_state.get("logged_in"):
        yield f"请先登录后再执行{label}。"
        return

    http = session_state["http"]
    log = f"开始{label}...\n"
    yield log

    try:
        # SSEClient 的 kwargs 透传给 requests.get，这里带上登录 cookie 以通过鉴权
        events = SSEClient(f"{BACKEND}{endpoint}", cookies=http.cookies).events()
        for evt in events:
            try:
                msg = json.loads(evt.data)
            except Exception:
                # 非 JSON 行直接跳过
                continue

            msg_type = msg.get("type")
            if msg_type == "progress":
                text = msg.get("text", "")
                # 重建/沉淀进度还带 stage/current/total，补充显示
                stage = msg.get("stage")
                cur, total = msg.get("current"), msg.get("total")
                if stage and cur is not None and total is not None:
                    text = f"[{stage} {cur}/{total}] {text}"
                log += (text or "") + "\n"
                yield log
            elif msg_type == "result":
                log += "===== 完成 =====\n"
                yield log
                return
            elif msg_type == "error":
                log += "[错误] " + (msg.get("text", "") or "") + "\n"
                yield log
                return
    except Exception as e:
        log += f"[异常] SSE 连接失败：{e}\n"
        yield log


# ========== 知识库重建 ==========
def rebuild_index(session_state):
    """触发 GET /api/knowledge/rebuild（SSE），实时展示重建进度（单一日志输出）"""
    yield from stream_sse("/api/knowledge/rebuild", session_state, "重建索引")


# ========== 知识库体检 ==========
def health_check(session_state):
    """触发 GET /api/knowledge/health（SSE），展示进度并把最终报告写入 JSON 组件

    输出按位置映射：(health_log, health_result)。
    """
    if not session_state.get("logged_in"):
        yield "请先登录后再执行健康检查。", None
        return

    http = session_state["http"]
    log = "开始知识库体检...\n"
    yield log, None

    try:
        events = SSEClient(f"{BACKEND}/api/knowledge/health", cookies=http.cookies).events()
        for evt in events:
            try:
                msg = json.loads(evt.data)
            except Exception:
                continue
            msg_type = msg.get("type")
            if msg_type == "progress":
                log += (msg.get("text", "") or "") + "\n"
                yield log, None
            elif msg_type == "result":
                log += "===== 体检完成 =====\n"
                yield log, msg.get("data")
                return
            elif msg_type == "error":
                log += "[错误] " + (msg.get("text", "") or "") + "\n"
                yield log, None
                return
    except Exception as e:
        log += f"[异常] SSE 连接失败：{e}\n"
        yield log, None


# ========== 对话知识沉淀 ==========
def distill_knowledge(session_state):
    """触发 GET /api/knowledge/distill（SSE），展示进度并把沉淀知识点写入 Dataframe

    输出按位置映射：(distill_log, distill_df)。
    """
    if not session_state.get("logged_in"):
        yield "请先登录后再执行知识沉淀。", None
        return

    http = session_state["http"]
    log = "开始对话知识沉淀...\n"
    yield log, None

    try:
        events = SSEClient(f"{BACKEND}/api/knowledge/distill", cookies=http.cookies).events()
        for evt in events:
            try:
                msg = json.loads(evt.data)
            except Exception:
                continue
            msg_type = msg.get("type")
            if msg_type == "progress":
                log += (msg.get("text", "") or "") + "\n"
                yield log, None
            elif msg_type == "result":
                data = msg.get("data") or {}
                merged = data.get("merged_knowledge") or []
                log += (f"\n===== 沉淀完成 =====\n"
                        f"原始 {data.get('raw_count', 0)} 条 -> 过滤后 {data.get('filtered_count', 0)} 条 -> "
                        f"提取 {data.get('extracted_count', 0)} 条 -> 合并 {data.get('merged_count', 0)} 条\n")
                # 构造表格数据：类型 / 内容 / 分类 / 来源
                rows = []
                for k in merged:
                    sources = k.get("sources") or []
                    if isinstance(sources, list):
                        sources = "、".join(str(s) for s in sources)
                    rows.append([
                        k.get("knowledge_type", ""),
                        k.get("content", ""),
                        k.get("category", ""),
                        sources,
                    ])
                yield log, rows
                return
            elif msg_type == "error":
                log += "[错误] " + (msg.get("text", "") or "") + "\n"
                yield log, None
                return
    except Exception as e:
        log += f"[异常] SSE 连接失败：{e}\n"
        yield log, None


# ========== 成本统计 ==========
def get_stats(session_state):
    """调用 GET /api/stats，返回大模型调用累计统计（按模块汇总）"""
    if not session_state.get("logged_in"):
        return None, "请先登录后再查看统计。"
    http = session_state["http"]
    try:
        resp = http.get(f"{BACKEND}/api/stats", timeout=15)
        if resp.status_code == 401:
            return None, "未登录或登录已过期，请重新登录。"
        summary = resp.json()
        return summary, ""
    except Exception as e:
        return None, f"请求异常：{e}"


# ========== 界面构建 ==========
def build_ui():
    """构建 Gradio Blocks 单页界面"""
    with gr.Blocks(title="迪士尼客服助手") as demo:
        # 每个 Gradio 会话独立的状态：requests.Session + 会话 ID + 登录态
        session_state = gr.State(None)

        gr.Markdown("# 迪士尼客服助手 RAG\n多模态问答与知识库管理")

        # ---------------- 标签页一：客服助手 ----------------
        with gr.Tab("客服助手"):
            chatbot = gr.Chatbot(
                label="对话历史",
                height=520,
                avatar_images=(None, None),
            )
            chat_input = gr.MultimodalTextbox(
                label="输入问题（可上传文件）",
                placeholder="请输入您的问题，回车发送...",
                interactive=True,
                sources=["upload"],
            )
            gr.Markdown(
                "*说明：问答接口仅接收文本；如需新增知识文档/图片，请在管理员后台通过知识库管理功能添加。*"
            )

        # ---------------- 标签页二：管理员后台 ----------------
        with gr.Tab("管理员后台"):
            # 登录表单（未登录时可见）
            with gr.Group(visible=True) as login_group:
                gr.Markdown("### 管理员登录")
                login_user = gr.Textbox(label="用户名", placeholder="admin")
                login_pwd = gr.Textbox(label="密码", type="password", placeholder="admin123")
                login_btn = gr.Button("登录", variant="primary")
                login_msg = gr.Textbox(label="提示", interactive=False)

            # 后台功能区（登录后可见）
            with gr.Group(visible=False) as admin_group:
                gr.Markdown(f"### 知识库重建\n点击按钮触发索引重建（SSE 流式进度，会调用云端 embedding API，产生费用）")
                with gr.Row():
                    rebuild_btn = gr.Button("重建索引", variant="primary")
                rebuild_log = gr.Textbox(
                    label="重建进度", lines=10, max_lines=20, interactive=False,
                    placeholder="点击按钮后此处实时显示重建日志...",
                )

                gr.Markdown("### 知识库体检\n检查知识库的完整性 / 时效性 / 一致性")
                with gr.Row():
                    health_btn = gr.Button("健康检查", variant="primary")
                health_log = gr.Textbox(
                    label="体检进度", lines=6, max_lines=15, interactive=False,
                    placeholder="点击按钮后此处实时显示体检进度...",
                )
                health_result = gr.JSON(label="体检报告")

                gr.Markdown("### 对话知识沉淀\n从已结束的会话中提取并合并知识点")
                with gr.Row():
                    distill_btn = gr.Button("开始沉淀", variant="primary")
                distill_log = gr.Textbox(
                    label="沉淀进度", lines=8, max_lines=18, interactive=False,
                    placeholder="点击按钮后此处实时显示沉淀进度...",
                )
                distill_df = gr.Dataframe(
                    headers=["类型", "内容", "分类", "来源"],
                    label="沉淀出的知识点",
                    interactive=False,
                    wrap=True,
                )

                gr.Markdown("### 大模型使用统计")
                with gr.Row():
                    stats_btn = gr.Button("获取统计", variant="primary")
                stats_msg = gr.Textbox(label="提示", interactive=False, visible=True)
                stats_json = gr.JSON(label="累计调用统计")

                gr.Markdown("---")
                logout_btn = gr.Button("退出登录")

        # ---------------- 事件绑定 ----------------
        def on_load(s, request: gr.Request):
            """页面加载：初始化会话状态 + 尝试加载历史对话，并注册到全局表供 unload 使用"""
            if not s:
                s = init_session_state()
            _SESSION_REGISTRY[getattr(request, "session_hash", "")] = s
            history = load_history(s)
            return s, history

        # 页面加载：初始化会话状态 + 尝试加载历史对话
        demo.load(
            fn=on_load,
            inputs=[session_state],
            outputs=[session_state, chatbot],
        )

        # 页面关闭：结束当前对话会话（Gradio 6 unload 仅接收 fn，通过全局注册表取状态）
        demo.unload(fn=end_session_on_unload)

        # 客服助手：提交问题
        chat_input.submit(
            fn=chat_respond,
            inputs=[chat_input, chatbot, session_state],
            outputs=[chatbot, chat_input, session_state],
        )

        # 管理员：登录 / 登出
        login_btn.click(
            fn=do_login,
            inputs=[login_user, login_pwd, session_state],
            outputs=[session_state, login_group, admin_group, login_msg],
        )
        logout_btn.click(
            fn=do_logout,
            inputs=[session_state],
            outputs=[session_state, login_group, admin_group, login_msg],
        )

        # 知识库重建（SSE 流式）
        rebuild_btn.click(
            fn=rebuild_index,
            inputs=[session_state],
            outputs=[rebuild_log],
        )

        # 知识库体检（SSE 流式 + JSON 结果）
        health_btn.click(
            fn=health_check,
            inputs=[session_state],
            outputs=[health_log, health_result],
        )

        # 对话知识沉淀（SSE 流式 + Dataframe 结果）
        distill_btn.click(
            fn=distill_knowledge,
            inputs=[session_state],
            outputs=[distill_log, distill_df],
        )

        # 成本统计
        stats_btn.click(
            fn=get_stats,
            inputs=[session_state],
            outputs=[stats_json, stats_msg],
        )

    return demo


# ========== 启动入口 ==========
if __name__ == "__main__":
    demo = build_ui()
    port = int(os.getenv("GRADIO_PORT", "7860"))
    demo.launch(server_port=port, show_error=True, theme=gr.themes.Soft())
