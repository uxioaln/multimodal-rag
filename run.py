# -*- coding: utf-8 -*-
"""
run.py - 项目启动入口

使用：
    export DASHSCOPE_API_KEY=xxx
    export AGICTO_API_KEY=xxx
    python run.py
"""
import os

from app import create_app
from app.config import DB_FILE, QDRANT_PATH
from app.state import load_resources


def check_env():
    """检查必要的文件资源是否存在（环境变量已由 app.config 校验）"""
    if not os.path.exists(DB_FILE):
        raise RuntimeError(f"错误：数据库 {DB_FILE} 不存在，请先运行 scripts/init_db.py 初始化。")
    # Qdrant 服务端模式（设置 QDRANT_URL 时）跳过本地存储路径检查，
    # 向量数据由独立的 qdrant 服务持久化；本地模式仍校验 QDRANT_PATH 是否存在
    if not os.getenv("QDRANT_URL") and not os.path.exists(QDRANT_PATH):
        raise RuntimeError(
            f"错误：Qdrant 本地存储 {QDRANT_PATH} 不存在，"
            f"请先运行 scripts/build_index.py 构建索引，"
            f"或运行 scripts/migrate_faiss_to_qdrant.py 从旧版 FAISS 迁移。"
        )


if __name__ == "__main__":
    check_env()
    load_resources()
    app = create_app()
    app.run(host="0.0.0.0", port=5050, debug=False)
