# Phase 81 候选码与组装归档

本归档保存固定 20 中心候选域的几何构造和码块组装证据。阶段 81 的既定集成门槛未通过，因此这里没有生成百万中心模型或执行新检索；完整 native bank 与局部覆盖方案仅作为[未执行后续提案](path81_native_overlay_proposal.md)保存。

## 原始证据和候选规则

成功原件位于 `~/bench/quantized_graph/research_20261025/path81_candidates_v1`，原始 manifest SHA256 为 `4b72ae11c27a6dac17642771c0753a684467896f94bb77d37e34afb1ee965868`。本次没有改写原始源码、候选文件或成功证明。

固定中心按冻结顺序取全度 32 块。每条边用原始旋转残差绝对值选择最小的 16 个坐标，以物理坐标打破相同值，再按物理坐标排序枚举：候选 0 为实际 native 编码，随后是 16 个单翻转与 120 个双翻转。预算在 FP32 精确整数格点上检查 `102 × flipped_mass ≤ residual_l1`；零残差只允许 native，非法候选按 lane 回填原 native 字节。此次 640 条边的 87,680 个 lane 候选均满足预算，没有短块或零残差。

`candidates.bin` 是 64 字节头之后的 `[20,137,3328]` 完整块；`candidate_compact.u8`、`valid.u8`、坐标与几何 sidecar 的形状和字节序见原始 manifest（本地实验文件：`../results/path81_candidate_artifacts/path81_candidates_v1/manifest.json`）。exact_budget.json（本地实验文件：`../results/path81_candidate_artifacts/path81_candidates_v1/exact_budget.json`）保留每条边的残差 L1 与 16 个坐标质量的精确整数，单位为 `2^-149`。候选 0 直接用实际 codec 编码，非 native 候选沿用既有 coupled 编码实现；不能把理想径向模型的 native 符号直接当作实际 native 因子。

## 归档改动与独立重放

仅格式化归档副本。Python 的语义保持变换为：合并等价嵌套 context manager、将固定字典迭代加索引改为 `.items()`；另有 Ruff 排版与窄范围复杂度注解。C++ 和 header 仅经过 clang-format。重放器对两份 Python 执行限定归一化后的 AST 相等检查，并重新编译归档 C++。

独占重放目录是 `~/bench/quantized_graph/research_20261025/replay81_candidates_formatted_v1`。重放证明（本地实验文件：`../results/stage39_candidates_replay.json`）记录：

- 完整重建 20 × 137 候选块，全部 19 个二进制或 JSON 产物与原成功版本的 SHA256 相同。
- 精确几何汇总相同；实际旋转、native 编码和 2,740 个 native/candidate 块组装校验通过。
- 原机械控制的两套 20 块模型、候选索引二进制及 assembly result 均逐字节相同。
- 原始源码与输入、归档依赖在执行前后保持不变；重放未读取查询、标签、validation 或训练数据。

本次重放约 15.49 秒，只是归档一致性检查，不是 QPS 测试。机械控制 model 0 全部选候选 0，model 1 固定选 `(center_row × 32 + lane) mod 137`；它不是训练模型。选中模型 SHA256 为 `c9fd2f2a571df8a3a6e3f8e345ebf05acf391007e54b87c11e6e17ec8948b912`。

`assemble81_models.py` 只依据被冻结的候选索引逐 lane 拷贝实际因子和符号，再调用实际 FastScan 打包，不重新计算范数或因子。输出 64 字节头加 model-major `[2,20,3328]` 块；每个块通过独立重组字节检查。

## 重放入口与保存范围

```sh
python3 research/20261025/scripts/replay81_candidates_formatted.py \
  --root /home/xingrui/bench/quantized_graph/research_20261025 \
  --archive /home/xingrui/workspace/quantized_graph/symphony_research_20261025/research/20261025 \
  --out /home/xingrui/bench/quantized_graph/research_20261025/replay81_candidates_formatted_v2
```

输出目录必须不存在；`v1` 成功证据应保留。脚本验证原始 manifest、几何文件、冻结生产源码和动态库，再构造新产物；完整命令、退出状态与源文件 SHA 都已保存。大型候选二进制保留在 bench，不加入仓库；归档收录源码、schema、精确预算、小证明与命令。显式归档清单（本地实验文件：`../results/stage39_candidates_archive_manifest.json`）列出原件和归档 SHA，供单独暂存与复核。
