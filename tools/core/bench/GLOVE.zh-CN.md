# GloVe-100 最佳已测配置

对应通用多视图实现 `6ae49f65`，2026-11-03 在保存的完整图上复测。
“最佳”指已扫描和重复确认的候选，不能保证新建图或其他机器上全局最优。
本次厂商对比范围为 SIFT / GIST / GloVe。GIST 保留单视图，GloVe 显式使用双视图；
SIFT 尚无当前代码的可比实测，不能沿用 Cohere 的数据或配置结论。

## 入口与数据准备

- [构建配置](glove_fp32_build.yaml)：FP32、cosine、M32、efConstruction400、native codec、64 个量化邻居、双视图。
- [默认查询配置](glove_fp32_recall_bench.yaml)：目标为严格 ID Recall@10 ≥90%。
- [七档查询配置](glove_fp32_recall_sweep.yaml)：严格 ID 90%/95%/97%，以及 ANN 距离容差 90%/95%/97%/98%。

使用 `glove-100-angular.hdf5` 的原始向量和零起始行 ID。完整数据为 1,183,514×100，查询为 10,000×100。
先安装 `numpy`、`h5py`，然后在仓库根目录执行（`/data/glove` 替换为本机路径）：

```bash
python3 tools/core/bench/prepare_glove.py \
  /data/glove/glove-100-angular.hdf5 /data/glove --include-train
./build/bin/txt2vecs --input=/data/glove/train.txt \
  --output=/data/glove/glove100_base.zvec.vecs --dimension=100
./build/bin/local_builder tools/core/bench/glove_fp32_build.yaml
./build/bin/recall tools/core/bench/glove_fp32_recall_bench.yaml
./build/bin/bench tools/core/bench/glove_fp32_recall_bench.yaml
```

`prepare_glove.py` 分块导出 `id;空格分隔的数值`，保留原始行顺序和 FP32 数值，不归一化，不覆盖已有输出。
已有 `.zvec.vecs` 时省略 `--include-train`，只导出 `queries.txt`、`neighbors.txt`，并跳过 `txt2vecs`。
需要编译的工具目标为 `txt2vecs`、`local_builder`、`recall`、`bench`。
HDF5、fvecs 和文本不能直接作为 `BuilderCommon.BuildFile`。

归一化由 `CosineFp32Converter` 在构建时处理。不要同时设置 `ConverterName` 和 `QuantizerName`。
查询保持 FP32；`IndexConfig` 显式设置 `kCosine`、100 维、`use_id_map: true`、`use_quantized_graph: true`、`kNone` 存储精度。
`IndexPath` 指向构建的 `DumpPath`，不是工作索引。

## 必须一起使用的构建选项

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

M32 是原始 HNSW 的参数，`MaxNeighbors: 64` 是量化邻接块容量；两者不是同一个字段。
双视图将 100 维补至 128 维后形成 256 个量化方向，路由投影仍为 64 维。
视图数、码本和路由均随快照保存，查询不设置 `RotationViews`，也不使用 `QG_RESEARCH_ROTATION_VIEWS`。
历史环境变量原型索引需从原图重建，不能直接用新库加载。

复用原图时把 `BuildFile` 替换为 `InputIndex`，使用尚未持久化量化块的原生 HnswStreamer 快照，
保留 `QuantizedGraphRouter` 节以训练匹配双视图的路由，并指定新的 `DumpPath`。
新建图可能因插入顺序和线程调度改变召回，必须先执行 `recall` 再计时。
本轮完整图比较复用了同一源图；从原始数据新建链路另以 256 条数据完成冒烟验证。

## 查询档位与实测

所有档位 `route_beam=1`、`uncertainty_bonus=0`、TopK=10。
两个独立进程各测 5 轮，单线程绑定 CPU 0，全量查询，表中 QPS 为 10 轮中位数。
这属于本地 Python→C++ 单查询口径，不能与原生 `bench` 或官网排行榜直接相除。
查询集此前已用于调参，不是盲测；FP16 尚无本轮 GloVe 性能验证，因此只提供 FP32 preset。

| case | 目标 | ef_search | margin | 展开上限 | 桶宽 | 严格召回 | ANN召回 | QPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 严格90%（默认） | 2048 | 0.065 | 384 | 0.008 | 90.220% | 91.687% | 15,335 |
| 1 | 严格95% | 2048 | 0.09 | 768 | 0.012 | 95.209% | 96.369% | 8,523 |
| 2 | 严格97% | 2048 | 0.11 | 1024 | 0.004 | 97.046% | 97.923% | 6,047 |
| 3 | ANN90% | 1024 | 0.055 | 384 | 0.016 | 89.188% | 90.686% | 16,725 |
| 4 | ANN95% | 2048 | 0.09 | 768 | 0.002 | 94.361% | 95.578% | 9,813 |
| 5 | ANN97% | 2048 | 0.09 | 768 | 0.028 | 96.168% | 97.150% | 6,960 |
| 6 | ANN98% | 2048 | 0.115 | 1024 | 0.004 | 97.424% | 98.220% | 5,471 |


严格 ID 召回按 Top10 ID 集合交集计算；ANN 召回按返回距离 ≤真实第十近邻距离+0.001 计算。
**ANN 90% 档的严格召回只有 89.188%，不可当作严格 90% 配置。**
原生 `recall` 配置设置 `CompareById: true`，始终报告严格 ID 召回，即使测试的是 ANN 档。

默认配置仅含 `QueryParam`。可选 sweep 的 `QueryParams` 会覆盖 `QueryParam`，按表中 case 编号输出；
每项都是完整 JSON，不会继承缺失的搜索参数。只需单一档位时，将该项复制到默认配置的 `QueryParam`，不要保留 `QueryParams`。

```bash
./build/bin/recall tools/core/bench/glove_fp32_recall_sweep.yaml
./build/bin/bench tools/core/bench/glove_fp32_recall_sweep.yaml
```

## 资源与验证记录

双视图完整索引约 4.078 GiB，单视图约 2.950 GiB，增加 38.3%；构建峰值 RSS 约 4.24 GiB。
默认查询 benchmark 使用 `BenchThreadCounts: [1]`。这是查询线程数；构建的 `ThreadCount: 8` 是单个构建任务的线程数。
ANN-Benchmarks 调度使用 `--parallelism 2`；计时时仍将重任务串行、等待任务不加载索引，避免 OOM 和资源竞争。
此调度参数属于外部 ANN-Benchmarks CLI，不应写进 `QuantizedGraph` 或 `IndexCommon`。

七个档位各 10,000 条查询，当前正式库、原生 recall 文本入口及研究原型的返回 ID 逐项一致；
默认配置与 sweep 的严格 90% 项参数一致。另验证 HDF5→文本→txt2vecs→新建→dump→recall/bench 的链路。
本轮配置验证产物位于本机 `~/bench/quantized_graph/glove_presets_20261103/`，包括 `validated_presets.json`、
每轮计时、全部结果 ID 和原生工具日志；跨数据集与厂商比较见 [验证记录](TESTING.zh-CN.md)。
