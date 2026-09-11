# Gradio 前端

除默认的 HTML 单页前端（`web/templates/index.html`）外，项目还提供基于 [Gradio](https://gradio.app/) 的前端 `app_gradio.py`，与同一 Flask 后端交互，提供更现代的聊天与管理界面。

Gradio 前端**不替代后端**，需先按 [README](../README.md)「快速开始」启动 Flask 服务（`python run.py` 或 Docker Compose），再单独启动 Gradio。

## 1. 安装额外依赖

```bash
pip install gradio requests sseclient-py -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> 上述依赖已包含在 `requirements.txt` 中，若已执行 `pip install -r requirements.txt` 则可跳过本步。

## 2. 启动 Flask 后端

```bash
python run.py          # 监听 http://127.0.0.1:5050
```

## 3. 启动 Gradio 前端

```bash
python app_gradio.py   # 默认监听 http://127.0.0.1:7860
```

## 4. 可选环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `BACKEND_URL` | `http://127.0.0.1:5050` | Flask 后端地址 |
| `GRADIO_PORT` | `7860` | Gradio 监听端口 |

## 5. 功能说明

启动后访问 http://localhost:7860 ：

- **客服助手**：聊天界面（`gr.Chatbot` + `gr.MultimodalTextbox`，支持文本、图片、视频渲染），回车发送问题。
- **管理员后台**：登录后可触发知识库重建、健康检查、对话知识沉淀（均通过 SSE 流式展示进度）与大模型成本统计。

## 6. 实现说明

- Gradio 前端通过 `requests` / `sseclient-py` 调用 Flask 的 `/api/*` 路由；每个浏览器会话对应一个独立的 `requests.Session`（保存在 `gr.State` 中），用于携带登录 cookie 调用需要鉴权的接口与 SSE 流。
- 后端返回的 `image_path` 会被转换为 Flask `/media/<path:filename>` 路由可访问的完整 URL（该路由从 `data/knowledge_base/` 目录 serve 文件）。
- 问答接口仅接收文本；新增知识文档请使用默认 HTML 前端的管理页面。
