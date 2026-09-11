# 参与贡献

感谢你对本项目的关注！欢迎提交 Issue 与 Pull Request。

## 提交 Issue

- 提问前请先阅读 [README](README.md) 与 [docs/](docs/) 目录下的相关文档。
- Bug 报告请附带：运行方式（本地 / Docker Compose）、复现步骤、期望与实际结果、日志片段。
- 安全漏洞请勿提交公开 Issue，请按 [SECURITY.md](SECURITY.md) 私下披露。

## 提交 Pull Request

1. Fork 仓库并基于最新主分支创建特性分支。
2. 代码约定：
   - 注释使用中文，文件统一 UTF-8 编码。
   - 分层结构：路由放在 `app/api/`，业务逻辑放在 `app/core/`，索引构建放在 `app/index/`；`app/config.py` 为叶子模块，不导入项目内其他模块。
   - Flask 路由使用 Blueprint 注册，业务逻辑优先实现为纯函数并通过参数注入依赖。
3. 长耗时任务请使用 SSE 流式推送进度，避免 HTTP 长连接超时。
4. 提交前请自测：本地运行 `python run.py` 或 `docker compose up --build` 验证主流程。
5. PR 描述中说明改动目的、涉及模块与测试方式。

## 文档约定

- README 只保留「是什么、为什么、怎么快速跑、核心能力概览、关键链接」，实现细节请放入 `docs/`。
- 命令、环境变量名、API 路径、默认值、文件路径必须与代码保持一致。
