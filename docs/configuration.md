# 配置说明

本文档汇总环境变量、`app/config.py` 关键常量与运维注意事项。

## 1. 环境变量

复制 `.env.example` 为 `.env` 并填写（Docker Compose 通过 `env_file` 自动加载；本地运行可 `export` / `set`）：

| 变量 | 必填 | 默认 | 说明 |
| --- | --- | --- | --- |
| `DASHSCOPE_API_KEY` | 是 | - | DashScope 密钥，用于多模态 embedding；未设置时 `app.config` 在导入阶段 fail-fast 抛错 |
| `AGICTO_API_KEY` | 是 | - | AGICTO 密钥，用于 Chat LLM；未设置时同上 fail-fast |
| `FLASK_SECRET_KEY` | 建议 | 未设置时随机生成 | Flask session 签名密钥；生产环境必须设为强随机值，否则容器重启后登录态失效 |
| `QDRANT_URL` | 否 | 未设置 | 设置后连接 Qdrant 服务端（如 Docker Compose 内的 `http://qdrant:6333`）；不设置则使用 `data/qdrant` 本地持久化模式（`QdrantClient(path=...)`） |
| `QDRANT_VECTOR_SIZE` | 否 | 自动探测 | 向量维度覆盖值；未设置时依次尝试环境变量 → 旧版 `disney_vectors.npy` → 兜底 1024 |
| `BACKEND_URL` | 否 | `http://127.0.0.1:5050` | Gradio 前端专用，Flask 后端地址 |
| `GRADIO_PORT` | 否 | `7860` | Gradio 前端专用，监听端口 |

## 2. 关键常量（app/config.py）

| 常量 | 默认值 | 说明 |
| --- | --- | --- |
| `MULTIMODAL_EMBEDDING_MODEL` | `tongyi-embedding-vision-plus` | 多模态 embedding 模型 |
| `CHAT_BASE_URL` | `https://api.agicto.cn/v1` | AGICTO 平台 OpenAI 兼容接口 |
| `CHAT_MODEL` | `deepseek-v4-flash` | Chat LLM 模型（日常问答） |
| `DIVERSE_REWRITE_MODEL` | `deepseek-v4-pro` | 多样化改写 / 知识沉淀专用模型 |
| `QDRANT_PATH` | `data/qdrant` | Qdrant 本地持久化存储路径 |
| `QDRANT_COLLECTION_NAME` | `disney_knowledge` | Qdrant collection 名称 |
| `QDRANT_VECTOR_SIZE` | `1024` | 向量维度（优先环境变量，其次旧版 .npy，兜底 1024） |
| `QDRANT_SEARCH_EF` | `128` | HNSW 搜索扩展邻居数 ef；本地持久化模式下退化为精确搜索、ef 不生效，取值依据与复测时机见 [vector-db-tuning.md](vector-db-tuning.md) |
| `QDRANT_HNSW_M` | `16` | HNSW 图连接数 m，构建索引时按向量规模自动建议（见第 4 节） |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 500 / 50 | 文本切分参数 |
| `MEDIA_DISTANCE_THRESHOLD` | 3.0 | 图片 / 视频匹配的 L2 距离阈值 |
| `USE_DIVERSE_REWRITE` | `True`（定义于 `app/index/builder.py`） | 是否启用多样化问题改写 |
| `UPLOAD_DOC_EXT` | `.docx` | 允许上传的文档类型 |
| `UPLOAD_IMG_EXT` | `.png` `.jpg` `.jpeg` `.gif` `.bmp` | 允许上传的图片类型 |

路径常量：`data/db/users.db`（用户库）、`data/indexes/disney_doc_hashes.json`（增量检测）、`data/stats/cost_stats.json`（成本统计）、`data/knowledge_base/`（源文档，图片在其 `images/` 子目录）。

## 3. 重建索引的数据安全约束

- 重建索引时会清空 Qdrant 本地存储目录（`data/qdrant/`）并删除 `disney_doc_hashes.json`，避免增量检测残留 metadata 脏数据（该逻辑在 `clean_index_files()` 中实现）。
- **不会**删除 `data/knowledge_base/` 源文档与 `data/db/users.db`。
- 重建索引完成后会自动调用 `state.load_resources()` 刷新内存索引，无需重启服务。
- Docker Compose 部署时向量数据在 `./data/qdrant_storage`（qdrant 容器卷），应用容器启动入口 `scripts/docker-entrypoint.sh` 仅在 collection 不存在或为空时才触发构建。

## 4. HNSW 参数随规模自适应

`app/index/builder.py` 的 `suggest_hnsw_m()` 按向量规模自动建议 m 值：

- 向量数 < 500：建议 m = 8
- 500 ~ 10000：建议 m = 16
- >= 10000：建议 m = 32

构建索引时若实际 m（`QDRANT_HNSW_M`）与建议值不一致会打印 WARNING；基准实验报告 `metadata` 中含 `scale_valid` / `scale_warning` / `recommendation` 字段，规模不足时跳过 ef 网格差异详情并标注待复测。

## 5. 运行模式对照

| 维度 | 本地运行（`python run.py`） | Docker Compose |
| --- | --- | --- |
| Qdrant 形态 | 本地持久化 `QdrantClient(path=data/qdrant)` | 独立 qdrant 容器，`QDRANT_URL=http://qdrant:6333` |
| 向量数据位置 | `./data/qdrant/` | `./data/qdrant_storage/`（容器内 `/qdrant/storage`） |
| 索引初始化 | 手动 `python scripts/build_index.py` | entrypoint 检测 collection 为空时自动构建 |
| HNSW 近似检索 | 不生效（退化为暴力搜索，ef 无效） | 生效（服务端模式） |
| 访问地址 | http://localhost:5050 | http://localhost:5050 ，Qdrant 面板 http://localhost:6333/dashboard |
