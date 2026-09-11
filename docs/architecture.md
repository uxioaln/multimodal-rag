# 架构设计与核心流程

本文档说明多模态数据处理 RAG（迪士尼客服助手）的分层架构、SSE 流式机制与五大核心流程。快速上手请先阅读 [README](../README.md)。

## 1. 分层架构

系统整体分为五层：

1. **客户端层**：默认前端为 `web/templates/index.html` 浏览器单页应用（聊天页 / 管理页 / 体检 / 沉淀 / 成本统计）；另提供可选的 Gradio 前端 `app_gradio.py`（默认 `:7860`），两个前端共用同一套 Flask API。
2. **Flask 路由层（`app/api/`）**：使用 Blueprint 拆分为 `auth`（登录鉴权）、`ask`（问答）、`knowledge`（知识库管理 / 体检 / 重建 / 沉淀）、`conversation`（会话生命周期）、`stats`（成本统计），仅做 HTTP 协议适配。
3. **业务逻辑层（`app/core/`）**：`rag_service.py`（RAG 问答纯函数，依赖通过参数注入）、`query.py`（embedding / Qdrant 检索 / LLM client）、`cost_tracker.py`（成本追踪）、`health_check.py`（健康检查）、`knowledge_distill.py`（对话知识沉淀）。
4. **索引构建与存储层（`app/index/`）**：`builder.py` 负责全量构建 / 清理 / 进度回调 / payload 转换，`add_index.py` 的 `KnowledgeBaseManager` 负责增量增删改；向量存入 Qdrant collection `disney_knowledge`，结构化数据存入 SQLite（`data/db/users.db`）。
5. **外部模型服务**：DashScope 多模态 embedding（`tongyi-embedding-vision-plus`，文本 / 图片 / 视频统一向量空间）；AGICTO 平台（OpenAI 兼容协议）`deepseek-v4-flash` 用于日常问答、`deepseek-v4-pro` 用于多样化改写与知识沉淀。

全局状态（Qdrant 客户端、元数据缓存、向量缓存、KB 管理器）集中在 `app/state.py` 管理，构建完成后通过 `state.load_resources()` 刷新内存索引，无需重启服务。

## 2. SSE 流式机制

重建索引、健康检查、知识沉淀三个任务耗时较长（分钟级，期间持续调用云端 API），若使用普通 HTTP 同步响应容易触发浏览器与 Flask 的长连接超时。项目采用 Server-Sent Events（SSE）解决：

- 接口响应类型为 `text/event-stream`，路由内返回 Flask 生成器（generator），任务在请求上下文中分步执行，每完成一个阶段即 `yield` 一条事件。
- 事件数据为 JSON，通过 `type` 字段区分：
  - `progress`：进度文本（如「正在检查第 1/3 批文本片段...」），前端逐条追加展示。
  - `result`：任务最终结果（聚合报告 / 重建统计 / 沉淀知识点列表），任务成功时发送一次。
  - `error`：失败信息，发送后流结束。
- 前端使用 `EventSource`（HTML 前端）或 `sseclient-py`（Gradio 前端）消费事件流；鉴权依赖 Flask session cookie，Gradio 侧为每个浏览器会话维护独立的 `requests.Session` 携带登录态。

涉及的 SSE 接口：`GET /api/knowledge/health`、`GET /api/knowledge/rebuild`、`GET /api/knowledge/distill`，字段细节见 [api.md](api.md)。

## 3. 核心流程

### 3.1 索引构建流程

1. 解析 docx 文档（段落与表格），按 `CHUNK_SIZE`（500）/ `CHUNK_OVERLAP`（50）切分文本 chunk；图片与视频条目单独整理。
2. 启用多样化问题改写时（`USE_DIVERSE_REWRITE=True`，定义于 `app/index/builder.py`），为每个文本 chunk 调用 `deepseek-v4-pro` 生成多样化问题，问题与原文拼接后一并向量化，提升用户问句的召回率。
3. 文本 / 图片 / 视频向量统一写入 Qdrant collection `disney_knowledge`，通过 payload（`media_type`、`business_id`、`question_type` 等）区分条目类型并映射回业务对象。
4. 构建完成后刷新内存索引，后续问答直接检索新 collection。

重建索引（`GET /api/knowledge/rebuild`）在构建前会执行 `clean_index_files()`：清空 Qdrant 本地存储目录（`data/qdrant/`）并删除 `disney_doc_hashes.json`，避免增量检测残留 metadata 脏数据；但**不会**删除 `data/knowledge_base/` 源文档与 `data/db/users.db`。

### 3.2 RAG 问答流程

1. 用户提问，后端获取 query 文本向量（DashScope `tongyi-embedding-vision-plus`）。
2. 在 Qdrant collection 中执行向量相似度检索（L2 距离 / EUCLID），按 payload 中的 `business_id` 映射回元数据。
3. 检测是否包含图片 / 视频意图（`IMAGE_KEYWORDS` / `VIDEO_KEYWORDS` 关键词命中），按 `MEDIA_DISTANCE_THRESHOLD`（3.0）距离阈值匹配最相关的媒体条目。
4. 取 top-k 文本片段拼接为背景知识，调用 Chat LLM（`deepseek-v4-flash`）生成答案。
5. 返回 `{answer, image_path, video_url, references}`，并把本轮问答写入 `conversations` 表（user + assistant 两条记录，assistant 记录的 `meta_json` 携带媒体与引用信息）。

### 3.3 知识库健康检查

- 从元数据中收集未删除的文本 chunk，自动生成测试查询（每个 chunk 取前 30 字作为 query，最多 20 条）。
- 调用 LLM 从**完整性、时效性、一致性**三个维度检查，分数按 10 分制分级并中文化。
- 按每批 20 个 chunk 分批执行，多批次结果聚合：累加 token 用量与估算成本、合并问题列表与改进建议、重新计算总体均分。
- 全程通过 SSE 推送进度，最终发送聚合报告（含总体健康分、覆盖率、新鲜度、一致性、问题列表、成本追踪）。

### 3.4 对话知识沉淀

1. 仅读取 `chat_sessions.ended_at IS NOT NULL` 的已结束会话（会话在前端页面关闭时通过 `POST /api/conversation/end` 标记结束）。
2. 按会话聚合成对话文本，调用 LLM 提取候选知识点。
3. 过滤「需求 / 问题」等临时性条目，按类型分组用 LLM 合并相似条目。
4. 结果缓存到 `data/stats/disney_distilled.json`，管理员可在沉淀详情页逐条点击「添加进知识库」（`POST /api/knowledge/distill/add`，走与 builder 一致的 embedding + 多样化改写索引链路），或通过 `GET /api/knowledge/distill/export` 导出 jsonl。

### 3.5 成本追踪

- 所有 LLM 调用统一通过 `app.core.cost_tracker.tracked_chat_completion` 包装，自动记录起止时间、Token 用量（prompt / completion / total）与估算成本。
- 按模块（问答 / 健康检查 / 索引构建 / 对话知识沉淀）分类汇总，持久化到 `data/stats/cost_stats.json`。
- 管理员通过「大模型使用统计」页面（`GET /api/stats`）查看累计统计，`POST /api/stats/reset` 可清空全局统计。

## 4. 相关文档

- 接口字段与 SSE 事件流：[api.md](api.md)
- 环境变量与运维约束：[configuration.md](configuration.md)
- 向量库选型决策：[adr-001-qdrant-migration.md](adr-001-qdrant-migration.md)
- 检索参数调优：[vector-db-tuning.md](vector-db-tuning.md)
