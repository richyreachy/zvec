# GloVe-100 最佳已测配置（2026-11-06 历史记录）

> 历史记录：本文保留原实验的说明、参数和实测值。旧 HNSW-QG YAML 已迁至
> [ANN-Benchmarks 历史归档](https://github.com/richyreachy/ann-benchmarks/tree/g9ibench/benchmark_configs/legacy-qg-20261106/)。
> 下文命令描述当时版本，不适用于当前独立 QG 入口；当前测试请使用
> [zvec_qg 配置与说明](../README.md#building-a-quantized-graph)。r7i 配置不在本次清理范围内。

2026-11-06 更新：使用代码 `686f898c`，FP32 / Cosine，原始 HNSW **M64、efConstruction800**，
native 编码、64 个量化邻居、双视图。构建、查询配置与同一快照上的正式复测参数一致。
“最佳”限定为已经扫描和重复确认的选择，不保证其他机器、数据分布或独立新建图上全局最优。
本次收益来自索引重建和查询参数调整，没有新增搜索代码；SIFT、GIST 和 FP16 未验证这组新构建参数。

## 入口与数据准备

- [构建配置](https://github.com/richyreachy/ann-benchmarks/blob/g9ibench/benchmark_configs/legacy-qg-20261106/config/glove_fp32_build.yaml)：M64/800 新图，双视图、持久化量化块和 Router。
- [默认查询配置](https://github.com/richyreachy/ann-benchmarks/blob/g9ibench/benchmark_configs/legacy-qg-20261106/config/glove_fp32_recall_bench.yaml)：官方 ANN Recall@10 90% 档。
- [三档查询配置](https://github.com/richyreachy/ann-benchmarks/blob/g9ibench/benchmark_configs/legacy-qg-20261106/config/glove_fp32_recall_sweep.yaml)：官方 ANN Recall@10 90% / 95% / 99%。

使用 `glove-100-angular.hdf5` 原始向量和零起始行 ID。构建集为 1,183,514×100，查询集为 10,000×100。
在仓库根目录执行，先将 YAML 中 `/data/glove` 替换为本机路径：

```bash
python3 benchmarks/vector/bench/prepare_hdf5.py \
  /data/glove/glove-100-angular.hdf5 /data/glove --dimension 100 --include-train
./build/bin/txt2vecs --input=/data/glove/train.txt \
  --output=/data/glove/glove100_base.zvec.vecs --dimension=100
./build/bin/local_builder tools/core/bench/config/glove_fp32_build.yaml
# 以下 recall 命令检查严格 ID 召回，口径与官方 ANN 召回不同，见下文。
taskset -c 0 ./build/bin/recall tools/core/bench/config/glove_fp32_recall_bench.yaml
taskset -c 0 ./build/bin/bench tools/core/bench/config/glove_fp32_recall_bench.yaml
```

所需工具目标为 `txt2vecs`、`local_builder`、`recall`、`bench`；编译并行度不超过 2。
导出脚本依赖 `numpy`、`h5py`，保留行顺序和 FP32 数值，不归一化、不覆盖已有输出。
已有 `.zvec.vecs` 时省略 `--include-train`，只导出查询及真值，并跳过 `txt2vecs`。
`BuildFile` 必须是 `.zvec.vecs`，不能直接传 HDF5、fvecs 或文本。
归一化由 `CosineFp32Converter` 完成，不要同时设置 `ConverterName` 与 `QuantizerName`。

## 重建索引与配套选项

更新查询参数不能把旧 M32/400 图变成新图。请从 `BuildFile` 重新构建；
默认输出改为 `glove_fp32_m64_ef800_work.index` 和 `glove_fp32_m64_ef800_full.index`，避免覆盖旧索引。
查询 `IndexPath` 必须指向新的 `DumpPath`，不能指向工作索引。
如果通过 `InputIndex` 复用图，必须使用未持久化量化块的原生 HnswStreamer 快照，
且该图已经按 M64/800 构建；已有量化快照不支持重训 Router，重新编码旧 M32 图也不等价。

```yaml
QuantizedGraph:
  Persist: true
  RotationViews: 2
  MaxNeighbors: 64
  ThreadCount: 8
  Codec:
    Type: native
QuantizedGraphRouter:
  Anchors: 65536
  ProjectedDimension: 64
  Seed: 202610261259
```

HNSW 的 M 与量化块的 `MaxNeighbors` 是不同参数，即使此配置两者都为 64。
双视图将 100 维补至 128 维后形成 256 个量化方向，Router 投影维度仍为 64。
视图、编码和 Router 已持久化，查询无需设置 `RotationViews` 或 `QG_*` 环境变量。
图构建和编码训练只使用构建语料。多线程建图可能受插入顺序影响，新建图须重新检查召回。

## 三档查询与正式复测

所有档位均为 TopK=10、`route_beam=1`、`uncertainty_bonus=0`。
默认配置对应 case 0；sweep 的 `QueryParams` 覆盖 `QueryParam`，每项是完整 JSON。
本次将原七档混合口径配置替换为三档官方 ANN 配置，依赖旧 case 编号的脚本应同步更新。

| case | 官方 ANN 目标 | ef_search | margin | 展开上限 | 桶宽 | 实际 ANN Recall@10 | 严格 ID Recall@10 | QPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 90% | 1024 | 0.063 | 384 | 0.006 | 90.684% | 89.061% | 18,566.80 |
| 1 | 95% | 2048 | 0.092 | 768 | 0.002 | 95.911% | 94.665% | 10,877.55 |
| 2 | 99% | 2048 | 0.1257 | 1536 | 0.002 | 99.096% | 98.608% | 4,954.04 |

QPS 来自 2026-11-06 的 ANN-Benchmarks 官方单查询 runner：每点全部 10,000 条查询，
两个独立进程各测 3 次，取 6 次中位数；原→新→新→原交错运行，CPU 0 单线程、预热后计时。
每条曲线共测 12 个参数点，48 个 HDF5 结果均检查 ID 合法性、召回和来源指纹。
本表不能与原生 `bench` 的 QPS 或其它机器上的结果直接相除。
公开查询此前用于调参，这不是独立留出集或盲测。

官方 ANN 召回按返回距离 ≤ 真值第十近邻距离 + 0.001 判断；严格 ID 召回按 Top10 ID 集合交集计算。
**默认 ANN 90% 档的严格 ID 召回为 89.061%，不能当作严格 90% 配置。**
原生 `recall` 的 `CompareById: true` 保留用于核对严格召回，不是官方 ANN 评估器。
要复现表中的官方召回与 QPS，使用匹配的 ANN-Benchmarks runner、原始 HDF5 及相同查询参数。

```bash
taskset -c 0 ./build/bin/recall tools/core/bench/config/glove_fp32_recall_sweep.yaml
taskset -c 0 ./build/bin/bench tools/core/bench/config/glove_fp32_recall_sweep.yaml
```

## 与原配置的性能和内存比较

以下是同一轮、相同查询库的原 M32/400 与新 M64/800 比较。召回是实测值，档位名称不是精确召回。

| 档位 | 原实际 ANN 召回 | 新实际 ANN 召回 | 原 QPS | 新 QPS | 提升 |
|---|---:|---:|---:|---:|---:|
| 90% | 90.595% | 90.684% | 17,492.76 | 18,566.80 | +6.14% |
| 95% | 95.820% | 95.911% | 9,905.34 | 10,877.55 | +9.82% |
| 99% | 99.091% | 99.096% | 4,326.64 | 4,954.04 | +14.50% |

RSS 按阶段采样最大值统计，间隔 0.2 秒；GiB = 2³⁰ 字节。

| 项目 | 原方案字节数（GiB） | 新方案字节数（GiB） |
|---|---:|---:|
| 加载前 RSS 基线 | 542,810,112（0.5055） | 542,875,648（0.5056） |
| 加载后总 RSS | 5,056,540,672（4.7093） | 5,357,850,624（4.9899） |
| 查询峰值总 RSS | 5,155,213,312（4.8012） | 5,457,657,856（5.0828） |
| 索引磁盘大小 | 4,379,101,248（4.0784） | 4,681,409,888（4.3599） |
| 索引借用的外部语料 | 0（0.0000） | 0（0.0000） |
| 框架保留的 FP32 语料 | 473,405,600（0.4409） | 473,405,600（0.4409） |

查询峰值总 RSS 增加约 288 MiB（5.87%），索引磁盘大小增加 6.90%。
总 RSS 包含 Python、测试框架语料与结果缓冲；磁盘大小、总 RSS 或增量 RSS 都不等于独占部署索引内存。
新图构建耗时 588.03 秒，采样峰值 RSS 5.044 GiB；没有本轮基线重建耗时可作比较。
2026-11-05 的两轮确认中，高召回档 QPS 中位数分别提高 12.88% 和 11.16%，
上述 14.50% 是 2026-11-06 的独立重跑结果，不能混成一轮或理解为每次运行的保证。

ANN-Benchmarks 算法/构建任务并发不超过 2，正式查询串行；单个构建任务使用 8 个工作线程。
计时期间不运行构建，查询使用 `BenchThreadCounts: [1]` 或匹配的单线程 ANN runner。

本机证据目录均位于 `~/bench/quantized_graph/environments/exploration/runs/`：

- `ten_percent_20261105/`：构建 YAML、冻结三档查询、两轮确认及索引指纹。
- `glove_two_configs_plot_20261106/`：`summary.json`、`curve.csv`、48 个已核验 HDF5，以及 `results/glove_qps_recall.png`。

历史 2026-11-03 三算法结果及 `annbenchmark_three_datasets.yaml` 保留原参数，见 [验证记录](TESTING.zh-CN.md)。
本目录已有 `results/glove.png` 属于该历史三方比较，不代表本次两配置复测。
