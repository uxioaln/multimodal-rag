# Qdrant 检索参数调优（hnsw_ef）

本文档记录 Qdrant HNSW 检索参数 `hnsw_ef` 的离线对比实验流程、实测数据与复测时机。实验脚本为 `experiments/qdrant_faiss_recall_benchmark.py`，报告输出到 `data/stats/recall_benchmark_report.json`。

## 1. 参数背景

Qdrant 在 HNSW 索引上的检索通过 `hnsw_ef` 参数控制搜索时扩展的邻居数：值越大召回越精确但延迟越高，值越小延迟越低但可能漏召回。为量化选择合适的 `hnsw_ef`，项目提供离线对比实验，流程如下：

1. 从知识库 docx 与历史对话生成测试 query，批量计算 embedding 并缓存到 `data/stats/benchmark_query_embeddings.npy`。
2. 以 Qdrant 全量向量构建 FAISS `IndexFlatL2` 暴力基准，生成 Top-20 Ground Truth。
3. FAISS 侧加载 backup 索引检索 Top-10；Qdrant 侧遍历 `hnsw_ef ∈ [64, 96, 128, 192, 256, 384, 512]` 七个档位各检索 Top-10。
4. 计算 `recall@1/3/5/10`、MRR、与 FAISS 的 Jaccard 重叠度、P50/P95/P99 延迟。
5. 推荐算法：在 `recall@10` 达到所有档位最高值 99% 阈值的前提下，选择最小的 `hnsw_ef`，兼顾召回质量与搜索效率。

## 2. 最近一次实测结果（2026-09-02）

测试集：26 条测试 query、63 条知识库向量。

| 系统 | recall@5 | recall@10 | MRR | Jaccard@5 | P99 延迟 |
| --- | --- | --- | --- | --- | --- |
| FAISS (Flat) | 0.5462 | 0.5577 | 0.6299 | - | 0.074 ms |
| Qdrant ef=64 | 0.5462 | 0.5577 | 0.6299 | 0.9872 | 1.325 ms |
| Qdrant ef=96 | 0.5462 | 0.5577 | 0.6299 | 0.9872 | 1.141 ms |
| Qdrant ef=128 | 0.5462 | 0.5577 | 0.6299 | 0.9872 | 1.079 ms |
| Qdrant ef=192 | 0.5462 | 0.5577 | 0.6299 | 0.9872 | 1.031 ms |
| Qdrant ef=256 | 0.5462 | 0.5577 | 0.6299 | 0.9872 | 1.018 ms |
| Qdrant ef=384 | 0.5462 | 0.5577 | 0.6299 | 0.9872 | 1.041 ms |
| Qdrant ef=512 | 0.5462 | 0.5577 | 0.6299 | 0.9872 | 1.046 ms |

推荐结果：`best_ef = ef_128`（最高 `recall@10 = 0.5577`，99% 阈值 = 0.5521，满足阈值的最小 ef）。该值已固化为 `app/config.py` 中的 `QDRANT_SEARCH_EF = 128`。

## 3. 重要说明：本地持久化模式下 ef 不生效

Qdrant 本地持久化模式（`QdrantClient(path=...)`）在查询时会回退为精确暴力搜索（brute-force），运行期会输出告警：

> `UserWarning: Local mode performs exact (brute-force) search, so search_params has no effect, with the exception of idf.`

因此当前实测中各 `hnsw_ef` 档位的召回指标完全一致，差异仅来自测量噪声。`ef_128` 的推荐含义为：在未来切换到 Qdrant 服务端模式（gRPC / HTTP）或数据规模增长到触发 HNSW 索引时，`hnsw_ef=128` 是预期满足 99% 召回阈值的最经济档位。如需复现 HNSW 的真实近似搜索行为，请改用 Qdrant Docker 服务端（Docker Compose 已编排，见 [README](../README.md) 快速开始）并显式启用 HNSW 索引。

## 4. HNSW 参数随规模自适应

`app/index/builder.py` 的 `suggest_hnsw_m()` 会按向量规模自动建议 HNSW 图连接数 m：

- 向量数 < 500：建议 m = 8
- 500 ~ 10000：建议 m = 16
- >= 10000：建议 m = 32

构建索引时若实际 m（`QDRANT_HNSW_M`，默认 16）与建议值不一致会打印 WARNING；基准实验报告 `metadata` 中含 `scale_valid` / `scale_warning` / `recommendation` 字段，规模不足时会跳过 ef 网格差异详情并标注待复测。

## 5. 何时需要重跑实验

- 知识库规模发生显著变化（向量数翻倍以上）。
- 切换 embedding 模型或向量维度变化。
- 从本地持久化模式切换到 Qdrant 服务端模式。
- 需要重新校准生产环境 `hnsw_ef` 默认值。

复现命令：

```bash
python experiments/qdrant_faiss_recall_benchmark.py
```

报告输出路径：`data/stats/recall_benchmark_report.json`。

## 6. 相关文档

- 向量库从 FAISS 迁移到 Qdrant 的选型论证：[adr-001-qdrant-migration.md](adr-001-qdrant-migration.md)
- 配置常量与环境变量：[configuration.md](configuration.md)
