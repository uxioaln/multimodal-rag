# -*- coding: utf-8 -*-
"""
app.state - Qdrant 客户端 / 元数据 / 管理器的全局状态单点

设计要点：
- 避免在 app/ 多个 Blueprint 里散落 module-level globals
- 所有路由通过 get_qdrant_client() / get_metadata() / get_kb_manager() 等 getter 访问
- 加载和保存逻辑只在这里实现一次
"""
import logging
import os

from app.config import INDEX_DIR, QDRANT_COLLECTION_NAME, QDRANT_PATH
from app.index.add_index import KnowledgeBaseManager
from app.index.builder import (
    ensure_collection,
    get_qdrant_client as _create_qdrant_client,
    scroll_all_metadata,
)

logger = logging.getLogger(__name__)

# 运行期常驻内存的 Qdrant 客户端 / 元数据缓存 / 管理器
_qdrant_client = None
_metadata = None
_kb_manager = None  # KnowledgeBaseManager 实例


# ========== Getter 访问接口 ==========
def get_qdrant_client():
    """获取 QdrantClient 实例"""
    return _qdrant_client


def get_metadata():
    """获取元数据列表（注意：返回内部引用，调用方不应直接修改）"""
    return _metadata


def get_kb_manager():
    """获取 KnowledgeBaseManager 实例"""
    return _kb_manager


# ========== 加载 / 保存 ==========
def _check_legacy_faiss_files():
    """检查是否存在旧版 FAISS 索引文件（用于提示用户运行迁移脚本）"""
    legacy_files = [
        os.path.join(INDEX_DIR, "disney_index.faiss"),
        os.path.join(INDEX_DIR, "disney_metadata.json"),
        os.path.join(INDEX_DIR, "disney_vectors.npy"),
    ]
    return [f for f in legacy_files if os.path.exists(f)]


def load_resources():
    """加载 Qdrant 客户端与元数据缓存，并初始化 KnowledgeBaseManager"""
    global _qdrant_client, _metadata, _kb_manager

    # 1. 初始化 Qdrant 客户端
    _qdrant_client = _create_qdrant_client()

    # 2. 确保 collection 存在（不存在则创建空 collection）
    ensure_collection(_qdrant_client)

    # 3. 检查 collection 是否为空 + 是否存在旧版 FAISS 文件
    count = _qdrant_client.count(collection_name=QDRANT_COLLECTION_NAME)
    if count.count == 0:
        legacy = _check_legacy_faiss_files()
        if legacy:
            logger.warning(
                "Qdrant collection '%s' 为空，且检测到旧版 FAISS 索引文件：\n  %s\n"
                "请先运行迁移脚本：python scripts/migrate_faiss_to_qdrant.py",
                QDRANT_COLLECTION_NAME,
                "\n  ".join(legacy),
            )
        else:
            logger.warning(
                "Qdrant collection '%s' 为空，请运行 python scripts/build_index.py 构建索引",
                QDRANT_COLLECTION_NAME,
            )

    # 4. 滚动读取所有 point 到内存 metadata 缓存
    _metadata = scroll_all_metadata(_qdrant_client)

    # 5. 初始化 KnowledgeBaseManager
    _kb_manager = KnowledgeBaseManager(_qdrant_client, _metadata)
    logger.info("已加载资源: collection '%s' 共 %s 条, metadata 缓存 %s 条",
                QDRANT_COLLECTION_NAME, count.count, len(_metadata))


def save_resources():
    """刷新内存 metadata 缓存（Qdrant 中的数据已即时生效，此处仅同步缓存）"""
    global _metadata
    _kb_manager.rebuild_index()
    _metadata = _kb_manager.metadata_store
