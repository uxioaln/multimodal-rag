#!/bin/sh
# -*- coding: utf-8 -*-
# scripts/docker-entrypoint.sh - docker-compose 应用容器启动入口
#
# 启动顺序：
#   1. 等待 Qdrant 服务端就绪（带重试，避免 Flask 启动时报连接错误）
#   2. 初始化 SQLite 数据库（幂等：建表 IF NOT EXISTS + 默认管理员）
#   3. 检查 Qdrant collection，仅在为空时构建向量索引
#      （索引构建会调用云端 embedding API，产生费用，故只在缺失时执行）
#   4. 启动 Flask 服务（python run.py）
#
# 本脚本由 docker-compose 的 app 服务通过
#   command: ["sh", "scripts/docker-entrypoint.sh"]
# 调用，无需可执行权限。
set -e

echo "[entrypoint] 启动应用容器..."
echo "[entrypoint] QDRANT_URL=${QDRANT_URL:-（未设置，将使用本地持久化模式）}"

# ========== 1. 等待 Qdrant 服务就绪 ==========
# 仅在服务端模式（QDRANT_URL 已设置）下等待；本地模式无需等待外部服务
if [ -n "$QDRANT_URL" ]; then
    echo "[entrypoint] 等待 Qdrant 服务就绪: ${QDRANT_URL}"
    ATTEMPTS=0
    MAX_ATTEMPTS=60
    until python -c "
import os, sys
sys.path.insert(0, '/app')
from qdrant_client import QdrantClient
QdrantClient(url=os.getenv('QDRANT_URL')).get_collections()
" 2>/dev/null; do
        ATTEMPTS=$((ATTEMPTS + 1))
        if [ "$ATTEMPTS" -ge "$MAX_ATTEMPTS" ]; then
            echo "[entrypoint] 等待 Qdrant 超时（${MAX_ATTEMPTS} 次重试均失败），退出。"
            echo "[entrypoint] 请检查 qdrant 服务状态：docker compose logs qdrant"
            exit 1
        fi
        echo "[entrypoint] Qdrant 未就绪，5 秒后重试（${ATTEMPTS}/${MAX_ATTEMPTS}）..."
        sleep 5
    done
    echo "[entrypoint] Qdrant 已就绪。"
fi

# ========== 2. 初始化数据库 ==========
echo "[entrypoint] 初始化数据库..."
python scripts/init_db.py

# ========== 3. 检查并构建向量索引（仅当 collection 为空时） ==========
NEED_BUILD=$(python -c "
import os, sys
sys.path.insert(0, '/app')
from app.index.builder import get_qdrant_client, collection_exists
from app.config import QDRANT_COLLECTION_NAME
client = get_qdrant_client()
if not collection_exists(client):
    print('1')
else:
    count = client.count(collection_name=QDRANT_COLLECTION_NAME).count
    print('1' if count == 0 else '0')
" 2>/dev/null || echo "1")

if [ "$NEED_BUILD" = "1" ]; then
    echo "[entrypoint] Qdrant collection 为空，开始构建向量索引..."
    echo "[entrypoint] 注意：索引构建会调用 DashScope 多模态 embedding 与 AGICTO 改写 API，会产生费用。"
    python scripts/build_index.py
    echo "[entrypoint] 索引构建完成。"
else
    echo "[entrypoint] 索引已存在，跳过构建。"
fi

# ========== 4. 启动 Flask 服务 ==========
echo "[entrypoint] 启动 Flask 服务..."
exec python run.py
