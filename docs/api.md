# API 接口文档

Flask 后端默认监听 `http://localhost:5050`，所有接口前缀为 `/api`。媒体文件通过 `/media/<path:filename>` 访问（从 `data/knowledge_base/` 目录 serve）。

## 1. 通用约定

- **请求 / 响应**：除文件上传为 `multipart/form-data`、SSE 接口为 `text/event-stream` 外，均使用 `application/json`。
- **统一响应包络**：`{"code": 0, "data": ..., "message": "..."}` 表示成功；`{"code": 1, "message": "错误原因"}` 表示失败，HTTP 状态码为 400（参数错误）/ 401（未登录）/ 404（不存在）/ 500（服务异常）。
- **鉴权**：标注「是」的接口需要管理员登录，登录态通过 Flask session cookie 维护；未登录访问返回 401 与 `{"code": 1, "message": "请先登录管理员账号"}`。curl 中使用 `-c cookies.txt` 保存 cookie、`-b cookies.txt` 携带 cookie。

## 2. 接口一览

| 方法 | 路径 | 说明 | 鉴权 |
| --- | --- | --- | --- |
| POST | `/api/login` | 管理员登录 | 否 |
| POST | `/api/logout` | 退出登录 | 否 |
| GET | `/api/login_status` | 查询登录状态 | 否 |
| POST | `/api/ask` | RAG 问答，同步落库对话 | 否 |
| POST | `/api/conversation/end` | 标记会话为已结束 | 否 |
| GET | `/api/knowledge/list` | 列出所有知识条目 | 是 |
| POST | `/api/knowledge/add` | 新增文档 / 图片 / 视频条目 | 是 |
| DELETE | `/api/knowledge/delete` | 删除指定条目（`?id=<内部id>`，标记删除后重建索引） | 是 |
| GET | `/api/knowledge/health` | 流式知识库健康检查（SSE） | 是 |
| GET | `/api/knowledge/rebuild` | 流式重建索引（SSE） | 是 |
| GET | `/api/knowledge/distill` | 流式对话知识沉淀（SSE） | 是 |
| POST | `/api/knowledge/distill/add` | 将单条沉淀知识点添加进知识库 | 是 |
| GET | `/api/knowledge/distill/export` | 导出最近一次沉淀结果为 jsonl | 是 |
| GET | `/api/stats` | 大模型使用累计统计 | 是 |
| POST | `/api/stats/reset` | 清空全局统计 | 是 |

## 3. 鉴权接口

### POST /api/login

请求体：

```json
{"username": "admin", "password": "admin123"}
```

成功响应（HTTP 200，同时下发 session cookie）：

```json
{"code": 0, "data": {"username": "admin"}, "message": "登录成功"}
```

失败响应：账号或密码为空返回 400；账号不存在或密码错误返回 401：

```json
{"code": 1, "message": "账号或密码错误，数据库中不存在该用户"}
```

密码校验方式为加盐 SHA256（`sha256(salt + password)`），默认管理员由 `scripts/init_db.py` 创建（`admin` / `admin123`）。

### POST /api/logout

清除 session，无请求体要求。

### GET /api/login_status

返回当前登录状态（供前端初始化时判断是否展示管理入口）。

## 4. 问答接口

### POST /api/ask

请求体：

```json
{"query": "上海迪士尼老人票怎么收费？", "session_id": "可选，浏览器会话ID", "visitor_id": "可选，访客ID"}
```

- `query` 必填，为空返回 400 `{"code": 1, "message": "问题不能为空"}`。
- `session_id` 缺省时回退 `visitor_id`，再缺省为 `"anonymous"`。

成功响应：

```json
{
  "code": 0,
  "data": {
    "answer": "……（LLM 生成的答案）",
    "image_path": "data/knowledge_base/images/xxx.jpg 或 null",
    "video_url": "视频 URL 或 null",
    "references": ["来源文件名(相似度)", "..."]
  }
}
```

副作用：upsert `chat_sessions` 会话生命周期记录，并向 `conversations` 表写入 user / assistant 两条消息（assistant 消息的 `meta_json` 携带媒体路径与引用来源）；落库失败仅记录日志，不影响返回答案。

curl 示例：

```bash
curl -X POST http://localhost:5050/api/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "上海迪士尼老人票怎么收费？"}'
```

### POST /api/conversation/end

请求体：`{"session_id": "xxx"}`，为空返回 400。将 `chat_sessions.ended_at` 置为当前时间（仅更新 `ended_at IS NULL` 的行，幂等），返回 `{"code": 0, "message": "会话已结束", "affected": 1}`。前端在 `beforeunload` / `pagehide` 时调用，已结束的会话才会进入知识沉淀流程。

## 5. 知识库管理接口

### GET /api/knowledge/list

返回全部知识条目元数据（含 `id`、`type`、`business_id`、`doc_id`、`deleted` 等字段）。

### POST /api/knowledge/add

`multipart/form-data` 表单，按 `type` 字段分三类：

| type | 表单字段 | 说明 |
| --- | --- | --- |
| `doc` | `file`（docx） | 保存到 `data/knowledge_base/`，解析切分后每个 chunk 独立入库 |
| `image` | `file`（png/jpg/jpeg/gif/bmp） | 保存到 `data/knowledge_base/images/` 并向量化入库 |
| `video` | `url`（必填）、`description`（可选，缺省「自定义视频」） | 视频 URL 条目入库 |

参数无效返回 400（如「仅支持docx文档」「type参数无效，应为 doc/image/video」）。

curl 示例（需先登录）：

```bash
# 上传 docx 文档
curl -b cookies.txt -X POST http://localhost:5050/api/knowledge/add \
  -F "type=doc" -F "file=@/path/to/攻略.docx"

# 上传图片
curl -b cookies.txt -X POST http://localhost:5050/api/knowledge/add \
  -F "type=image" -F "file=@/path/to/海报.jpg"

# 新增视频条目
curl -b cookies.txt -X POST http://localhost:5050/api/knowledge/add \
  -F "type=video" -F "url=https://example.com/play.mp4" -F "description=花车巡游视频"
```

### DELETE /api/knowledge/delete

查询参数 `id`（元数据内部整数 id），缺省返回 400「缺少id参数」，找不到返回 404「未找到该条目」。执行标记删除（按 `doc_id` 调用 `KnowledgeBaseManager.delete_document`）后立即重建索引并持久化。

```bash
curl -b cookies.txt -X DELETE "http://localhost:5050/api/knowledge/delete?id=12"
```

## 6. SSE 流式接口

`GET /api/knowledge/health`、`GET /api/knowledge/rebuild`、`GET /api/knowledge/distill` 均返回 `text/event-stream`，需登录后携带 session cookie 访问。每条消息格式为：

```text
data: {"type": "progress", "text": "正在收集文本知识片段..."}

data: {"type": "result", "data": { ... }}

data: {"type": "error", "text": "失败原因"}
```

| 事件 type | 负载字段 | 时机 |
| --- | --- | --- |
| `progress` | `text` | 每个执行阶段开始 / 完成时发送，前端逐条追加展示 |
| `result` | `data` | 任务成功完成时发送一次，携带聚合报告 / 重建统计 / 沉淀知识点列表 |
| `error` | `text` | 任一步骤失败时发送，发送后流结束 |

curl 查看 SSE 流：

```bash
curl -b cookies.txt -N http://localhost:5050/api/knowledge/health
```

- **health**：收集文本 chunk → 生成测试查询 → 每批 20 个 chunk 分批检查 → 聚合报告。`result.data` 含总体健康分（10 分制 + 等级）、覆盖率 / 新鲜度 / 一致性、缺少 / 过期 / 冲突知识列表、改进建议与成本追踪。
- **rebuild**：`clean_index_files()` → 全量构建（embedding + 多样化改写）→ `state.load_resources()` 刷新内存索引。
- **distill**：读取 `ended_at IS NOT NULL` 的会话 → LLM 提取候选知识点 → 过滤临时条目并合并相似项 → 结果缓存到 `data/stats/disney_distilled.json`。

### POST /api/knowledge/distill/add

将单条沉淀知识点添加进知识库（走与索引构建一致的 embedding + 多样化改写链路）。

### GET /api/knowledge/distill/export

导出最近一次沉淀结果为 jsonl 文件下载。

## 7. 统计接口

### GET /api/stats

返回所有模块（问答 / 健康检查 / 索引构建 / 对话知识沉淀）的 LLM 调用累计统计：调用次数、prompt / completion / total tokens、估算成本与起止时间。数据持久化在 `data/stats/cost_stats.json`。

### POST /api/stats/reset

清空全局统计，返回 `{"status": "ok", "message": "统计已重置"}`。

## 8. 完整调用示例

```bash
# 1. 管理员登录（保存 cookie）
curl -c cookies.txt -X POST http://localhost:5050/api/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'

# 2. 上传 docx 文档入库
curl -b cookies.txt -X POST http://localhost:5050/api/knowledge/add \
  -F "type=doc" -F "file=@/path/to/攻略.docx"

# 3. 匿名问答（无需登录）
curl -X POST http://localhost:5050/api/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "上海迪士尼老人票怎么收费？", "session_id": "demo-session-001"}'

# 4. 结束会话（页面关闭时）
curl -X POST http://localhost:5050/api/conversation/end \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo-session-001"}'

# 5. 查看大模型使用统计
curl -b cookies.txt http://localhost:5050/api/stats
```
