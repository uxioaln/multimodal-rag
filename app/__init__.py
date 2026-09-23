# -*- coding: utf-8 -*-
"""
app 包入口：暴露 create_app() 工厂函数

业务模块拆分：
- app.config        常量 + 环境变量（叶子模块）
- app.state         索引/元数据/向量/KnowledgeBaseManager 全局状态
- app.core          业务逻辑层（rag_service / query / cost_tracker / health_check / knowledge_distill）
- app.index         索引构建层（builder / add_index）
- app.api           路由层（auth / ask / knowledge / stats）
- app.models        数据访问层（user）
"""
import logging
import os
import secrets

from flask import Flask, send_from_directory

from app.config import DOCS_DIR
from app.api import auth_bp, ask_bp, knowledge_bp, stats_bp, conversation_bp, agent_bp


# 工程根目录：app/ 的父目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_DIR = os.path.join(PROJECT_ROOT, "web", "templates")
STATIC_DIR = os.path.join(PROJECT_ROOT, "web", "static")


def create_app() -> Flask:
    """创建并配置 Flask 应用实例"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # 模板与静态资源使用绝对路径，避免工作目录差异导致的路径错误
    app = Flask(
        __name__,
        template_folder=TEMPLATES_DIR,
        static_folder=STATIC_DIR,
    )
    # session 签名密钥：优先使用环境变量 FLASK_SECRET_KEY（容器/生产环境重启后保持登录态），未设置时随机生成
    app.secret_key = os.getenv("FLASK_SECRET_KEY") or secrets.token_hex(16)

    # 注册蓝图
    app.register_blueprint(auth_bp)
    app.register_blueprint(ask_bp)
    app.register_blueprint(knowledge_bp)
    app.register_blueprint(stats_bp)
    app.register_blueprint(conversation_bp)
    app.register_blueprint(agent_bp)

    @app.route("/")
    def index():
        return send_from_directory(TEMPLATES_DIR, "index.html")

    @app.route("/media/<path:filename>")
    def media(filename):
        return send_from_directory(DOCS_DIR, filename)

    return app
