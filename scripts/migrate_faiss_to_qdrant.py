# -*- coding: utf-8 -*-
"""
scripts.migrate_faiss_to_qdrant - 从 FAISS 迁移到 Qdrant 的一次性脚本

功能：
1. 检查旧版 data/indexes/ 下的 disney_index.faiss / disney_metadata.json / disney_vectors.npy
2. 临时导入 faiss 读取索引，提取所有向量与元数据
3. 通过 metadata_to_payload 转换为 Qdrant payload 格式
4. 初始化 Qdrant 本地持久化存储，创建 collection
5. 按 batch=100 批量 upsert，point id 从 1 开始自增
6. 迁移完成后验证 client.count() 与原向量数量一致
7. 将原 FAISS 文件移动到 data/indexes/backup/（保留备份，不直接删除）

使用方法：
    python scripts/migrate_faiss_to_qdrant.py
"""
import os
import sys

# 把工程根加入 path，使 app.* 包可被导入
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app.config import INDEX_DIR, QDRANT_COLLECTION_NAME, QDRANT_PATH
from app.index.builder import metadata_to_payload

# 旧版 FAISS 相关文件路径
FAISS_INDEX_FILE = os.path.join(INDEX_DIR, "disney_index.faiss")
FAISS_METADATA_FILE = os.path.join(INDEX_DIR, "disney_metadata.json")
FAISS_VECTORS_FILE = os.path.join(INDEX_DIR, "disney_vectors.npy")

# 备份目录
BACKUP_DIR = os.path.join(INDEX_DIR, "backup")


def check_source_files():
    """检查迁移源文件是否存在"""
    missing = []
    if not os.path.exists(FAISS_INDEX_FILE):
        missing.append(FAISS_INDEX_FILE)
    if not os.path.exists(FAISS_METADATA_FILE):
        missing.append(FAISS_METADATA_FILE)

    if missing:
        print("错误：以下 FAISS 相关文件不存在：")
        for f in missing:
            print(f"  - {f}")
        print("\n请先运行 python scripts/build_index.py 生成索引，或确认文件路径正确。")
        return False

    print(f"[OK] 找到 FAISS 索引: {FAISS_INDEX_FILE}")
    print(f"[OK] 找到元数据文件: {FAISS_METADATA_FILE}")
    return True


def load_faiss_data():
    """加载 FAISS 索引和元数据

    faiss 在本脚本中临时导入，不作为项目依赖保留
    """
    print("\n正在加载 FAISS 索引...")
    try:
        import faiss
    except ImportError:
        print("\n错误: 无法导入 faiss 模块。请先临时安装 faiss-cpu：")
        print("  pip install faiss-cpu==1.7.4 -i https://pypi.tuna.tsinghua.edu.cn/simple")
        sys.exit(1)

    index = faiss.read_index(FAISS_INDEX_FILE)
    print(f"  FAISS 索引记录数: {index.ntotal}")
    print(f"  向量维度: {index.d}")

    print("正在加载元数据...")
    import json
    with open(FAISS_METADATA_FILE, 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)
    print(f"  元数据条目数: {len(metadata_list)}")

    # 提取所有向量（按 FAISS 内部位置顺序）
    vectors = index.reconstruct_n(0, index.ntotal)
    print(f"  已提取向量矩阵: {vectors.shape}")

    return index, metadata_list, vectors


def migrate_to_qdrant(vectors, metadata_list):
    """将数据迁移到 Qdrant

    point id 从 1 开始自增，与 business_id 解耦
    payload 通过 metadata_to_payload 转换，与 builder 使用的格式一致
    """
    from qdrant_client import QdrantClient, models
    from app.index.builder import upsert_points

    vector_size = vectors.shape[1]
    print(f"\n正在初始化 Qdrant (本地持久化模式)...")
    print(f"  存储路径: {QDRANT_PATH}")
    print(f"  Collection: {QDRANT_COLLECTION_NAME}")
    print(f"  向量维度: {vector_size}")
    print(f"  距离度量: EUCLID")

    # 初始化客户端
    client = QdrantClient(path=QDRANT_PATH)

    # 如果 collection 已存在，先删除后重建
    collections = client.get_collections()
    existing_names = [c.name for c in collections.collections]
    if QDRANT_COLLECTION_NAME in existing_names:
        print(f"\n警告: collection '{QDRANT_COLLECTION_NAME}' 已存在，先删除后重建...")
        client.delete_collection(QDRANT_COLLECTION_NAME)

    # 创建 collection
    client.create_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.EUCLID,
        ),
    )
    print(f"  已创建 collection")

    # 构建 points 并批量写入
    print(f"\n开始迁移 {len(metadata_list)} 条数据到 Qdrant...")
    all_points = []
    for idx, meta in enumerate(metadata_list):
        point_id = idx + 1  # Qdrant point id 从 1 开始自增
        payload = metadata_to_payload(meta)
        all_points.append(models.PointStruct(
            id=point_id,
            vector=vectors[idx].tolist(),
            payload=payload,
        ))

    # 批量写入
    upsert_points(client, all_points, batch_size=100)

    # 验证
    count = client.count(collection_name=QDRANT_COLLECTION_NAME)
    print(f"\n迁移验证:")
    print(f"  原 FAISS 向量数: {len(metadata_list)}")
    print(f"  Qdrant collection 记录数: {count.count}")

    if count.count == len(metadata_list):
        print("  [OK] 数量一致，迁移成功！")
        return True
    else:
        print(f"  [WARNING] 数量不一致！(差 {abs(count.count - len(metadata_list))})")
        return False


def backup_faiss_files():
    """将原 FAISS 文件移动到备份目录（保留备份，不直接删除）"""
    import shutil
    from datetime import datetime

    os.makedirs(BACKUP_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    files_to_backup = [
        ("disney_index.faiss", FAISS_INDEX_FILE),
        ("disney_metadata.json", FAISS_METADATA_FILE),
        ("disney_vectors.npy", FAISS_VECTORS_FILE),
    ]

    backed_up = []
    for name, src_path in files_to_backup:
        if os.path.exists(src_path):
            backup_name = f"{timestamp}_{name}"
            dst_path = os.path.join(BACKUP_DIR, backup_name)
            shutil.move(src_path, dst_path)
            backed_up.append((name, backup_name))
            print(f"  已备份: {name} -> data/indexes/backup/{backup_name}")

    if backed_up:
        print(f"\n已将 {len(backed_up)} 个 FAISS 相关文件移动到备份目录: {BACKUP_DIR}")
    else:
        print("\n无文件需要备份。")


def main():
    print("=" * 60)
    print("  FAISS -> Qdrant 迁移工具")
    print("=" * 60)

    # 步骤 1: 检查源文件
    print("\n[步骤 1] 检查迁移源文件...")
    if not check_source_files():
        sys.exit(1)

    # 步骤 2: 加载 FAISS 数据
    print("\n[步骤 2] 加载 FAISS 数据...")
    try:
        index, metadata_list, vectors = load_faiss_data()
    except Exception as e:
        print(f"\n错误: 加载 FAISS 数据失败 - {e}")
        sys.exit(1)

    # 步骤 3: 迁移到 Qdrant
    print("\n[步骤 3] 迁移到 Qdrant...")
    try:
        success = migrate_to_qdrant(vectors, metadata_list)
    except Exception as e:
        print(f"\n错误: 迁移到 Qdrant 失败 - {e}")
        sys.exit(1)

    if not success:
        print("\n迁移未完成，请检查错误信息后重试。")
        sys.exit(1)

    # 步骤 4: 备份原文件
    print("\n[步骤 4] 备份原 FAISS 文件...")
    try:
        backup_faiss_files()
    except Exception as e:
        print(f"\n警告: 备份文件失败 - {e}")
        print("原文件仍保留在原位，请手动处理。")

    # 完成
    print("\n" + "=" * 60)
    print("  迁移完成！")
    print("=" * 60)
    print("""
下一步操作：
1. 确认依赖已更新（已移除 faiss-cpu，新增 qdrant-client）：
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

2. 启动应用验证迁移结果：
   python run.py

3. 如需重新构建索引（可选，会覆盖迁移的数据）：
   python scripts/build_index.py

4. 确认一切正常后，可手动删除备份目录 data/indexes/backup/
""")


if __name__ == "__main__":
    main()
