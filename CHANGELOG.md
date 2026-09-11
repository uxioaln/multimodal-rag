# 更新日志

本项目遵循 [语义化版本（Semantic Versioning）](https://semver.org/lang/zh-CN/)。

## [Unreleased]

- README 重构为「项目门面」结构：新增徽章区、30 秒 Docker Compose 极速开始、Mermaid 架构图、功能特性分组、系统要求、Roadmap、贡献指南与许可证章节。
- 新增 `docs/` 文档目录：`architecture.md`（架构与核心流程）、`api.md`（接口详解）、`configuration.md`（配置与运维）、`vector-db-tuning.md`（hnsw_ef 调优）、`adr-001-qdrant-migration.md`（向量库选型决策）、`gradio-frontend.md`（Gradio 前端）。
- 明确许可证为 MIT（见 [LICENSE](LICENSE)）。

## [0.1.0] - 2026-09

### Added

- 多模态知识库：文本（docx）/ 图片 / 视频通过 DashScope `tongyi-embedding-vision-plus` 统一向量化，存入 Qdrant collection `disney_knowledge`。
- RAG 问答：向量检索 top-k 片段 + AGICTO `deepseek-v4-flash` 生成答案，媒体意图检测自动附带图片 / 视频。
- 索引期多样化问题改写（`deepseek-v4-pro`），提升问句召回率。
- 运营能力：知识库健康检查（完整性 / 时效性 / 一致性）、对话知识沉淀、LLM 成本追踪，长耗时任务均通过 SSE 流式推送进度。
- 会话生命周期管理与 SQLite + 加盐 SHA256 管理员鉴权。
- 部署：Docker Compose 一键编排 Flask 应用与 Qdrant 服务端（`scripts/docker-entrypoint.sh` 等待就绪、初始化数据库、缺失时自动构建索引）；支持 `QDRANT_URL` 环境变量在本地持久化模式与服务端模式间切换。
- 工具脚本：`scripts/init_db.py`、`scripts/build_index.py`、`scripts/migrate_faiss_to_qdrant.py`；离线实验 `experiments/qdrant_faiss_recall_benchmark.py`。
- 可选 Gradio 前端 `app_gradio.py`。
