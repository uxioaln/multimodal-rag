# ADR-001：向量库从 FAISS 迁移到 Qdrant

- **状态**：已采纳（Accepted）
- **日期**：2026-09
- **背景**：项目早期使用 FAISS（`IndexFlatL2` / `IVF`）作为向量库，随着多模态数据与运营诉求增加，需要重新评估向量检索方案。

## 1. 决策

切换到 Qdrant（当前支持本地持久化模式 `QdrantClient(path=...)` 与服务端模式 `QdrantClient(url=...)`，通过环境变量 `QDRANT_URL` 切换），collection 名称为 `disney_knowledge`，距离度量为 L2（EUCLID）。

## 2. 选型理由

### 2.1 元数据原生支持

图片、视频、多样化问题等多模态条目需要携带 `media_type`、`business_id`、`file_path`、`question_type` 等结构化字段。Qdrant 的 payload 原生支持字段过滤与按 `business_id` 映射回业务对象；FAISS 需要外挂独立的 `_metadata.json`，索引与元数据容易出现一致性脏数据。

### 2.2 本地持久化与可重建

Qdrant 本地模式把 collection 落盘到 `data/qdrant/`，进程重启后无需重新构建索引即可继续服务；FAISS 需要在内存中重建 `Index` 对象，启动成本随向量规模线性增长。

### 2.3 统一向量空间

文本 chunk、多样化问题、图片、视频等条目写入同一 collection，召回时通过 payload 区分条目类型。Qdrant 的单 collection + payload 过滤模型天然适配，避免 FAISS 多路索引的拼装逻辑。

### 2.4 运营闭环

重建索引、健康检查等长耗时任务需要按条目维度回写状态，Qdrant 的点级 upsert / delete 比 FAISS 整体重建更友好，配合 SSE 进度推送实现可观测。

### 2.5 召回质量持平

在 `experiments/qdrant_faiss_recall_benchmark.py` 离线对比实验中，对同一批 26 条测试 query、63 条知识库向量，Qdrant 与 FAISS `IndexFlatL2` 暴力基准的 `recall@1/3/5/10`、MRR 完全一致，`Jaccard@5 = 0.9872`、`Jaccard@10 = 0.9930`，召回质量未因切换而退化。完整实验数据与 `hnsw_ef` 档位对比见 [vector-db-tuning.md](vector-db-tuning.md)。

## 3. 迁移指南

旧版 FAISS 数据可通过迁移脚本一键迁移到 Qdrant：

```bash
python scripts/migrate_faiss_to_qdrant.py
```

迁移脚本会临时导入 `faiss-cpu` 读取旧索引（`data/indexes/` 下的 `disney_index.faiss` 等文件），将向量和元数据写入 Qdrant collection，验证数量一致后把原 FAISS 文件移动到 `data/indexes/backup/`。

## 4. 后果

- 构建索引后不再生成独立的 `.faiss` / `.npy` / `_metadata.json` 文件，向量数据统一由 Qdrant 管理。
- `disney_doc_hashes.json` 仍保留在 `data/indexes/`，用于文档增量检测，与向量库解耦。
- 本地持久化模式下 HNSW 的 `hnsw_ef` 参数不生效（退化为暴力搜索），详见 [vector-db-tuning.md](vector-db-tuning.md) 第 3 节；Docker Compose 部署使用 Qdrant 服务端模式，可发挥 HNSW 近似检索能力。
