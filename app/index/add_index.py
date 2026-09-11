# -*- coding: utf-8 -*-
"""
app.index.add_index - 知识库管理器（KnowledgeBaseManager）

提供对知识库条目的增/删/重建索引能力：
- add_document / add_image / add_video / add_text_chunk：基于 doc_id 去重，增量更新
- delete_document：通过 business_id 过滤条件从 Qdrant 删除
- rebuild_index：从 Qdrant 滚动刷新内存 metadata 缓存

设计要点：
- 业务 doc_id（business_id）与 Qdrant point id 解耦：外部按文件/路径/URL 作为业务键，
  Qdrant 维护自增整数 point id。
- 增量检测依赖 doc_hashes 哈希表（独立 JSON 文件），避免重复 embedding 开销。
- metadata_store 作为内存缓存，与 Qdrant 保持同步，供路由层热读取。
"""
import hashlib
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

from app.config import HASHES_FILE, QDRANT_COLLECTION_NAME
from app.index.builder import (
    build_text_chunk_entries,
    get_image_embedding,
    get_text_embedding,
    get_video_embedding,
    metadata_to_payload,
    scroll_all_metadata,
    upsert_points,
)

logger = logging.getLogger(__name__)


class KnowledgeBaseManager:
    """知识库管理器，封装 Qdrant 的增删改操作，并维护内存 metadata 缓存"""

    def __init__(self, qdrant_client, metadata_store, hashes_file=HASHES_FILE):
        """
        Args:
            qdrant_client: QdrantClient 实例
            metadata_store: 内存 metadata 缓存列表（旧结构，兼容上层业务逻辑）
            hashes_file: doc_hashes 文件路径
        """
        self.qdrant_client = qdrant_client
        self.metadata_store = metadata_store
        self.hashes_file = hashes_file
        # 文档hash，用于检测变更
        self.doc_hashes = self._load_hashes()

    def compute_hash(self, text):
        """计算文档内容hash"""
        return hashlib.md5(text.encode()).hexdigest()

    def _next_id(self):
        """生成下一个自增 point id（Qdrant point id 从 1 开始）"""
        if not self.metadata_store:
            return 1
        return max(int(m.get("id", 0)) for m in self.metadata_store) + 1

    def _now_iso(self):
        """当前时间 ISO 格式"""
        return datetime.now().isoformat()

    def _today_str(self):
        """当前日期 YYYY-MM-DD"""
        return datetime.now().strftime('%Y-%m-%d')

    def _load_hashes(self):
        """从磁盘加载文档hash表"""
        if os.path.exists(self.hashes_file):
            try:
                with open(self.hashes_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("加载hash表失败: %s，将使用空表", e)
        return {}

    def _save_hashes(self):
        """将文档hash表持久化到磁盘"""
        os.makedirs(os.path.dirname(self.hashes_file), exist_ok=True)
        with open(self.hashes_file, "w", encoding="utf-8") as f:
            json.dump(self.doc_hashes, f, ensure_ascii=False, indent=2)

    def _upsert_metadata(self, metadata: dict, vector: list) -> int:
        """将单条 metadata 连同向量 upsert 到 Qdrant，并同步到内存缓存

        Returns:
            新分配的 point id
        """
        from qdrant_client import models
        internal_id = self._next_id()
        metadata["id"] = internal_id
        point = models.PointStruct(
            id=internal_id,
            vector=vector,
            payload=metadata_to_payload(metadata),
        )
        upsert_points(self.qdrant_client, [point], batch_size=1)
        self.metadata_store.append(metadata)
        return internal_id

    def add_document(self, doc_id: str, text: str, source: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """添加/更新文本文档

        Args:
            doc_id: 业务文档唯一标识（如文件名）
            text: 文档完整文本内容
            source: 来源描述，如 "1-上海迪士尼门票规则.docx"
            metadata: 额外元数据字典（可选）
        """
        metadata = metadata or {}
        doc_hash = self.compute_hash(text)
        # 检查是否已存在
        if doc_id in self.doc_hashes:
            if self.doc_hashes[doc_id] == doc_hash:
                logger.info("文档 %s 未变更，跳过", doc_id)
                return
            else:
                logger.info("文档 %s 已更新，先删除旧版本", doc_id)
                self.delete_document(doc_id)

        # 生成向量
        embedding = get_text_embedding(text)

        entry = {
            'id': 0,  # 由 _upsert_metadata 分配
            'doc_id': doc_id,
            'source': source,
            'type': 'text',
            'content': text,
            'text': text,
            'created_at': self._now_iso(),
            'last_updated': self._today_str(),
            'hash': doc_hash,
            **metadata
        }
        internal_id = self._upsert_metadata(entry, embedding)
        self.doc_hashes[doc_id] = doc_hash
        self._save_hashes()
        logger.info("文档 %s 已添加，内部ID=%s", doc_id, internal_id)

    def add_text_chunk(self, text: str, source: str, doc_id: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        """添加文本块并建立索引（含多样化改写，逻辑与 builder.build_and_save 一致）

        Args:
            text: 文本内容
            source: 来源描述
            doc_id: 业务文档唯一标识
            metadata: 额外元数据（可选）

        Returns:
            新增的索引条目数量，0 表示内容未变更已跳过
        """
        metadata = metadata or {}
        text = (text or "").strip()
        if not text:
            raise ValueError("文本内容不能为空")

        doc_hash = self.compute_hash(text)
        if doc_id in self.doc_hashes:
            if self.doc_hashes[doc_id] == doc_hash:
                logger.info("文本 %s 未变更，跳过", doc_id)
                return 0
            logger.info("文本 %s 已更新，先删除旧版本", doc_id)
            self.delete_document(doc_id)

        start_id = self._next_id()
        vectors, entries, _ = build_text_chunk_entries(
            chunk=text,
            source=source,
            start_id=start_id,
            today_str=self._today_str(),
            doc_id=doc_id,
            extra_metadata={
                "text": text,
                "created_at": self._now_iso(),
                "hash": doc_hash,
                **metadata,
            },
        )
        if not entries:
            return 0

        # 构建 Qdrant points 并批量写入
        from qdrant_client import models
        points = []
        for i, entry in enumerate(entries):
            points.append(models.PointStruct(
                id=entry["id"],
                vector=vectors[i],
                payload=metadata_to_payload(entry),
            ))
        upsert_points(self.qdrant_client, points, batch_size=100)

        # 同步内存缓存
        self.metadata_store.extend(entries)
        self.doc_hashes[doc_id] = doc_hash
        self._save_hashes()
        logger.info("文本 %s 已添加，共 %s 条索引（主文本 + 多样化问题）", doc_id, len(entries))
        return len(entries)

    def add_image(self, img_path, filename=None, metadata=None):
        """添加/更新图片条目

        Args:
            img_path: 图片本地路径
            filename: 对外显示的文件名，默认从 img_path 提取
            metadata: 额外元数据字典（可选）
        """
        metadata = metadata or {}
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图片不存在: {img_path}")
        if filename is None:
            filename = os.path.basename(img_path)

        # 以图片路径作为业务唯一标识
        doc_id = f"image:{img_path}"
        # 以文件内容的 md5 作为图片 hash（重命名但内容相同可识别）
        with open(img_path, "rb") as f:
            content_bytes = f.read()
        doc_hash = hashlib.md5(content_bytes).hexdigest()

        if doc_id in self.doc_hashes:
            if self.doc_hashes[doc_id] == doc_hash:
                logger.info("图片 %s 未变更，跳过", filename)
                return
            else:
                logger.info("图片 %s 已更新，先删除旧版本", filename)
                self.delete_document(doc_id)

        embedding = get_image_embedding(img_path)
        entry = {
            'id': 0,
            'doc_id': doc_id,
            'source': f"图片: {filename}",
            'type': 'image',
            'path': img_path,
            'content': f"[图片] {filename}",
            'created_at': self._now_iso(),
            'last_updated': self._today_str(),
            'hash': doc_hash,
            **metadata
        }
        internal_id = self._upsert_metadata(entry, embedding)
        self.doc_hashes[doc_id] = doc_hash
        self._save_hashes()
        logger.info("图片 %s 已添加，内部ID=%s", filename, internal_id)

    def add_video(self, url, description="", metadata=None):
        """添加/更新视频条目

        Args:
            url: 视频 URL
            description: 视频描述
            metadata: 额外元数据字典（可选）
        """
        metadata = metadata or {}
        if not url:
            raise ValueError("视频 URL 不能为空")
        if not description:
            description = "自定义视频"

        # 以 URL 作为业务唯一标识
        doc_id = f"video:{url}"
        doc_hash = self.compute_hash(url)

        if doc_id in self.doc_hashes:
            if self.doc_hashes[doc_id] == doc_hash:
                logger.info("视频 %s 未变更，跳过", description)
                return
            else:
                logger.info("视频 %s 已更新，先删除旧版本", description)
                self.delete_document(doc_id)

        embedding = get_video_embedding(url)
        entry = {
            'id': 0,
            'doc_id': doc_id,
            'source': f"视频: {description}",
            'type': 'video',
            'url': url,
            'description': description,
            'content': f"[视频] {description}",
            'created_at': self._now_iso(),
            'last_updated': self._today_str(),
            'hash': doc_hash,
            **metadata
        }
        internal_id = self._upsert_metadata(entry, embedding)
        self.doc_hashes[doc_id] = doc_hash
        self._save_hashes()
        logger.info("视频 %s 已添加，内部ID=%s", description, internal_id)

    def delete_document(self, doc_id: str) -> int:
        """删除文档：从 Qdrant 按 business_id 过滤删除 + 同步内存缓存

        Args:
            doc_id: 业务文档唯一标识

        Returns:
            被删除的条目数
        """
        from qdrant_client import models

        # 1. 从 Qdrant 按 business_id 过滤删除
        self.qdrant_client.delete(
            collection_name=QDRANT_COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="business_id",
                            match=models.MatchValue(value=doc_id),
                        )
                    ]
                )
            ),
        )

        # 2. 同步内存缓存：标记已删除
        deleted_count = 0
        for doc in self.metadata_store:
            if doc.get('doc_id') == doc_id:
                doc['deleted'] = True
                doc['deleted_at'] = datetime.now().isoformat()
                deleted_count += 1

        # 3. 从 doc_hashes 中移除
        if doc_id in self.doc_hashes:
            del self.doc_hashes[doc_id]

        if deleted_count > 0:
            self._save_hashes()
            logger.info("文档 %s 已从 Qdrant 删除，共 %s 条", doc_id, deleted_count)

        return deleted_count

    def refresh_metadata_store(self):
        """从 Qdrant 滚动读取所有 point，刷新内存 metadata 缓存"""
        self.metadata_store = scroll_all_metadata(self.qdrant_client)

    def rebuild_index(self) -> None:
        """重建索引：Qdrant 模式下刷新内存 metadata 缓存即可

        原 FAISS 模式需要重写 faiss/metadata/vectors 三个文件，
        Qdrant 模式下删除已在 delete_document 中即时生效，此处仅刷新缓存。
        """
        self.refresh_metadata_store()
        logger.info("内存 metadata 缓存已刷新，共 %s 条", len(self.metadata_store))
