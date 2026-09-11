# 安全策略

## 漏洞披露

如果你发现安全漏洞，**请勿提交公开 Issue 或 Pull Request**。请通过以下方式私下联系维护者：

- 在 GitHub 仓库使用 [Security Advisories](https://docs.github.com/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability) 私下报告（仓库启用后）；或
- 通过仓库主页提供的联系方式邮件说明。

报告请尽量包含：漏洞描述、影响范围、复现步骤、建议修复方式。维护者会在收到后尽快确认并跟进修复。

## 安全注意事项

- `.env` 中的 `DASHSCOPE_API_KEY`、`AGICTO_API_KEY`、`FLASK_SECRET_KEY` 属于敏感信息，请勿提交到版本库或分享给他人。
- 生产环境部署时，`FLASK_SECRET_KEY` 必须改为强随机值；默认管理员密码（`admin` / `admin123`）应在初始化后立即修改。
- 本项目调用云端模型服务会产生费用，请妥善保管 API Key 并关注用量统计（`/api/stats`）。
