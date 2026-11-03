# QuantizedGraph benchmark

本目录保存可复现的构建/查询配置、使用说明和测试结果。

```text
bench/
├── config/              # local_builder、recall、bench 的 YAML 配置
├── results/             # 按测试批次保存报告、曲线和汇总数据
├── prepare_hdf5.py      # 保留原始行 ID 的稠密 HDF5 数据导出
├── prepare_glove.py     # GloVe-100 的兼容入口
├── plot_results.py      # 从已保存的结果 JSON 重绘曲线
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

数据集、构建中间文件和完整索引保留在本地实验目录。`results/` 收录报告、
可重绘的曲线数据、参数及环境记录；结果中的召回定义和测试批次以对应报告为准。

配置索引及其它数据集的运行方法见 [TESTING.zh-CN.md](TESTING.zh-CN.md)。

最新完整三方对比：[2026-11-03 测试报告](results/20261103-three-datasets/REPORT_ZH.md)；ANN-Benchmarks 构建并发上限为 2，正式查询串行计时。
