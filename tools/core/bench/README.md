# QuantizedGraph benchmark

本目录保存可复现的构建/查询配置、使用说明和测试结果。

```text
bench/
├── config/              # local_builder、recall、bench 的 YAML 配置
├── results/             # sift.png、gist.png、glove.png 三张 ANN Recall/QPS 图
├── prepare_hdf5.py      # 保留原始行 ID 的稠密 HDF5 数据导出
├── prepare_glove.py     # GloVe-100 的兼容入口
├── plot_results.py      # 从本地结果 JSON 重绘三张 PNG
├── TESTING.zh-CN.md     # 构建、运行与验证方法
├── GLOVE.zh-CN.md       # GloVe 参数选择与数据准备
├── FP16.zh-CN.md        # FP16 使用说明
└── ALGORITHM.zh-CN.md   # 算法设计
```

从仓库根目录运行，先将配置中的 `/data/...` 替换为实际路径：

```bash
./build/bin/local_builder tools/core/bench/config/glove_fp32_build.yaml
./build/bin/recall tools/core/bench/config/glove_fp32_recall_bench.yaml
./build/bin/bench tools/core/bench/config/glove_fp32_recall_bench.yaml
```

`results/` 仅保存 [SIFT](results/sift.png)、[GIST](results/gist.png)、
[GloVe](results/glove.png) 三张图，统一采用官方 ANN-Benchmarks 的距离容差召回口径，QPS 轴使用对数刻度。
原始结果、日志、完整报告、数据集和索引保留在本地实验目录。

配置索引及其它数据集的运行方法见 [TESTING.zh-CN.md](TESTING.zh-CN.md)。

2026-11-03 三方对比与测试协议见 [验证记录](TESTING.zh-CN.md)。ANN-Benchmarks 构建并发上限为 2，正式查询串行计时。

使用本地保存的确认点重绘三张图：

```bash
python3 tools/core/bench/plot_results.py /path/to/all-confirmed.json \
  --output tools/core/bench/results
```
