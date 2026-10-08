# -*- coding: utf-8 -*-
"""
scripts/build_index.py - 索引重建脚本（完整流程）

流程：
1. clean_index_files()  删除旧 Qdrant 存储目录 + doc_hashes 文件
   （保留 disney_knowledge_base/ 源文档和 users.db，避免脏数据）
2. build_and_save()      重新构建多模态索引
   - 文本 chunk + 多样化问题
   - 图片：qwen-vl-plus 生成多角度语义描述 + 多样化问题
   - 视频：描述 + 多样化问题
3. load_resources()      刷新内存索引（Qdrant 客户端 / metadata / BM25）

使用：
    export DASHSCOPE_API_KEY=xxx
    export AGICTO_API_KEY=xxx
    python scripts/build_index.py
"""
import logging
import os
import sys
import time

# 把工程根加入 path，使 app.* 包可被导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.index.builder import build_and_save, clean_index_files
from app.state import load_resources

# 配置日志：控制台输出，INFO 级别，中文 UTF-8
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
)
logger = logging.getLogger("rebuild_index")


def _progress(stage, current, total, message):
    """build_and_save 的进度回调：实时打印构建进度"""
    print(f"[{stage} {current}/{total}] {message}", flush=True)


def rebuild():
    """执行完整的索引重建流程"""
    start_ts = time.time()
    logger.info("=" * 60)
    logger.info("开始索引重建（清理 -> 构建 -> 刷新内存）")
    logger.info("=" * 60)

    # 1. 删除旧索引文件（Qdrant 存储目录 + doc_hashes）
    logger.info("[1/3] 清理旧索引文件...")
    deleted = clean_index_files()
    for path in deleted:
        logger.info("  已删除: %s", path)

    # 2. 重新构建索引
    logger.info("[2/3] 构建多模态索引（文本/图片/视频 + 多样化问题）...")
    build_and_save(progress_callback=_progress)

    # 3. 刷新内存索引（重建 BM25 等内存态资源）
    logger.info("[3/3] 刷新内存资源（metadata 缓存 + BM25 索引）...")
    load_resources()

    duration = round(time.time() - start_ts, 1)
    logger.info("=" * 60)
    logger.info("索引重建完成，耗时 %s 秒", duration)
    logger.info("=" * 60)


if __name__ == "__main__":
    rebuild()
