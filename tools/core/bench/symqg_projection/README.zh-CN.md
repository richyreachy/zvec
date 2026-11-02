# SymQG 投影编码研究入口

此目录用于复现实验性的投影编码，不修改现有 Cohere/GIST 默认配置。
当前验证重点是 GIST FP16 的 192 维、384bit 邻居编码。完整测试数据、模型、索引和图表保存在仓库外，不随代码提交。不能据此认为所有数据集、FP32/FP16 或硬件均有收益。

## 算法

对图边残差 `r = x - a`（`a` 是现有仿射/低秩锚点）训练列正交投影 `P`，在保留子空间内作固定随机正交混合。每个保留维度使用 2bit，两个维度组成 16 个笛卡尔码字。标定仍使用完整空间的范数：

```text
beta = ||r||² / (rᵀ P b)
e = beta P b - r
rᵀ e = 0
```

码字选择以数据库代理偏移的协方差 `M = sigma I + B Bᵀ` 为目标，最小化 `eᵀ M e`，并限制相对初始欧氏误差的增长。它保留完整范数和低秩度量，不是用投影后范数替代原距离。原始向量仍用于精排；入口路由不变。

查询投影可使用按行缩放的 INT8 矩阵和动态 UINT8 查询，运行时检查 VNNI 支持，无支持时回退 FP32。该近似会引入额外误差，需要独立召回验证。投影与量化组合已有 [MRQ 等相关工作](https://arxiv.org/abs/2411.06158)，本原型不主张通用优越性或论文新颖性。

## 训练和构建

本研究入口需要两个**数据库训练产物**：

- `residuals.npy`：N×D 的数据库图边残差，使用目标索引相同的旋转及解码后锚点。训练时必须保留完整旋转空间，不能预先裁掉尾部。GIST 的原始 960 维在此空间补齐为 1024 维。
- `metric.bin`：数据库代理偏移估计的 RAD1 低秩度量，文件依次为小端 `uint32[4] = {0x52414431, D, rank, sweeps}`、`double[2] = {sigma, guard}`、按行存放的 `double[D * rank]`。实验设置使用两轮坐标下降和 `guard=1.25`。

不要用 ANN 测试查询或 ground truth 生成这两个输入。采样及低秩度量估计尚未整合进这个训练脚本；脚本只负责从冻结的数据库训练产物生成 PCA/随机混合投影。这是研究接口，不是已经集成到 local_builder 的全自动训练配置。

```bash
python3 train_projection.py \
  --residuals /path/to/residuals.npy --metric /path/to/metric.bin \
  --dimension 192 --output /path/to/projection.bin
```

需要 NumPy。目标 local_builder YAML 继续使用已有的源图和 Codec 配置，将 `QuantizedGraph.Codec.BlockDimension` 设置为 `2`，`Codebook` 设置为 `gaussian`。Codec 的类型必须是 `global` 或 `low_rank`，与残差训练所用锚点一致；不能使用 `native`。其余均值、锚点、训练种子等参数必须与输入残差匹配。

```bash
QG_RESEARCH_PROJECTION=/path/to/projection.bin local_builder build.yaml
```

外部文件只在构建时读取。扩展模型以版本 4 序列化到索引；检索从索引读取，不能把既有索引仅靠修改检索配置变成投影编码索引。旧版本 1 编码继续可读。未发布的版本 2/3 研究布局不在本提交范围内。

## 检索和验证

```bash
QG_RESEARCH_PROJECTION_INT8=1 recall recall_bench.yaml
QG_RESEARCH_PROJECTION_INT8=1 bench recall_bench.yaml
```

不设置 `QG_RESEARCH_PROJECTION_INT8` 则使用 FP32 查询投影。该实验开关以变量是否存在判断，设置为 `0` 也会启用。只有版本 4 模型受此开关影响。

GIST 192 维实验冻结的搜索参数：`ef_search=2048`、`quantized_graph_margin=0.0504295`、`quantized_graph_bucket_width=0.002`、`quantized_graph_max_expansions=256`、`quantized_graph_route_beam=1`、`quantized_graph_uncertainty_bonus=0`。这些是对应模型的实验参数，迁移到新数据或重新训练模型后，应在单独的校准查询上重新选择，并冻结后验证。

验证需同时包含原生召回与性能：两臂使用相同图、原始精排向量及 ground truth；独立确认查询不参与参数选择；单线程/8线程交错重复计时。记录全部查询准备成本、索引大小和峰值 RSS，不能用局部代理候选保留率替代 ANN Recall@10。数学/整数核及模型输入边界测试位于 `tests/turbo/turbo_symqg_projection_test.cc`。
