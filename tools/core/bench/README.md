# QuantizedGraph

本分支保留量化图实现、必要的 C++ 测试、算法说明与验证记录。
数据集最佳实践配置、数据准备脚本和 benchmark Python wrapper 保留在
`feat/quantized-graph-performance-exploration` 分支。

- [算法与精度语义](ALGORITHM.zh-CN.md)
- [回归测试与性能验证](TESTING.zh-CN.md)
- 原优化版三算法对照图：[SIFT](results/sift.png)、[GIST](results/gist.png)、[GloVe](results/glove.png)，采用官方 ANN Recall@10，QPS 轴为对数刻度。

`results/` 仅保存上述三张 PNG。原始测试数据和日志保留在本地实验目录。
