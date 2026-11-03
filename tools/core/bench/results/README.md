# Benchmark results

每个子目录对应一次测试批次，包含报告、逐配置召回/QPS 汇总、环境与参数记录，
以及 PNG/SVG 曲线。比较前请核对报告中的召回口径、查询数、线程设置和试验轮数。

有 `points.json` 的批次可以直接重绘，所需依赖为 Python 3 和 matplotlib：

```bash
python3 tools/core/bench/plot_results.py \
  tools/core/bench/results/<batch>/points.json
```

脚本分别输出 ANN 容差召回和严格 ID 召回的三数据集合图、各数据集单图。
曲线采用已实测配置的 Pareto 前沿；配置及原始逐轮记录仍保存在 JSON 中。

原始数据集、完整索引和逐查询 HDF5 保留在 ANN-Benchmarks 的运行目录。
提交的结果通过参数、版本和数据文件信息标明来源。
JSON/CSV 中的 `$ANN_BENCHMARKS_ROOT`、`$NATIVE_ARTIFACTS` 等是来源路径占位符，
使用本地原始产物时需替换为对应目录。

已归档批次：

- [2026-11-03：SIFT / GIST / GloVe × Zvec / Huawei / Descartes](20261103-three-datasets/REPORT_ZH.md)，含三方曲线、CSV、参数和环境记录。
