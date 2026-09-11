# -*- coding: utf-8 -*-
"""
app.config - 全局常量与文件路径

约束：本模块是叶子模块，不导入任何项目内其他模块
"""
import os

# ========== 项目根目录与数据目录 ==========
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

INDEX_DIR = os.path.join(DATA_DIR, "indexes")
DB_DIR = os.path.join(DATA_DIR, "db")
STATS_DIR = os.path.join(DATA_DIR, "stats")
KB_DIR = os.path.join(DATA_DIR, "knowledge_base")

# ========== 路径常量 ==========
DOCS_DIR = KB_DIR
IMG_DIR = os.path.join(KB_DIR, "images")

# doc_hashes 仍用独立文件存储（增量检测），与 Qdrant 解耦
HASHES_FILE = os.path.join(INDEX_DIR, "disney_doc_hashes.json")
DB_FILE = os.path.join(DB_DIR, "users.db")
COST_STATS_FILE = os.path.join(STATS_DIR, "cost_stats.json")

# ========== Qdrant 配置 ==========
QDRANT_PATH = os.path.join(DATA_DIR, "qdrant")
QDRANT_COLLECTION_NAME = "disney_knowledge"

# Qdrant 向量维度：优先环境变量，其次从旧版 disney_vectors.npy 读取，兜底 1024
QDRANT_VECTOR_SIZE = int(os.getenv("QDRANT_VECTOR_SIZE", "0"))
if not QDRANT_VECTOR_SIZE:
    _npy_file = os.path.join(INDEX_DIR, "disney_vectors.npy")
    try:
        import numpy as _np
        if os.path.exists(_npy_file):
            QDRANT_VECTOR_SIZE = int(_np.load(_npy_file).shape[1])
    except Exception:
        pass
if not QDRANT_VECTOR_SIZE:
    # tongyi-embedding-vision-plus 实际输出维度为 1024
    QDRANT_VECTOR_SIZE = 1024

# HNSW 图搜索时扩展的邻居数 ef，控制召回精度与延迟的折中
# 基于 2026-09 小规模实验（63 vectors）推荐，知识库增长至 >1000 后需复测调优
QDRANT_SEARCH_EF = 128

# HNSW 图连接数 m；builder 会根据向量规模自动建议（<500 建议 8），可覆盖
QDRANT_HNSW_M = 16

# ========== 上传文件类型 ==========
UPLOAD_DOC_EXT = {".docx"}
UPLOAD_IMG_EXT = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}

# ========== 媒体匹配阈值 ==========
MEDIA_DISTANCE_THRESHOLD = 3.0

# ========== 文本切分参数 ==========
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ========== 模型与服务 ==========
# 多模态 embedding（DashScope 平台）
MULTIMODAL_EMBEDDING_MODEL = "tongyi-embedding-vision-plus"

# Chat LLM（AGICTO 平台，基于中文训练的性价比模型）
CHAT_BASE_URL = "https://api.agicto.cn/v1"
CHAT_MODEL = "deepseek-v4-flash"

# 多样化改写专用模型（要求较高，索引构建时调用）
DIVERSE_REWRITE_MODEL = "deepseek-v4-pro"

# ========== 媒体意图关键词 ==========
IMAGE_KEYWORDS = ["图片", "海报", "照片", "看看", "长什么样", "图"]
VIDEO_KEYWORDS = ["视频", "录像", "影片", "看一下", "播放"]

# ========== 环境变量读取（fail-fast） ==========
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("错误：请设置 'DASHSCOPE_API_KEY' 环境变量（用于多模态 embedding）。")

AGICTO_API_KEY = os.getenv("AGICTO_API_KEY")
if not AGICTO_API_KEY:
    raise ValueError("错误：请设置 'AGICTO_API_KEY' 环境变量（用于 chat LLM）。")
