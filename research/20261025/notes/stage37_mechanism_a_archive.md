# 阶段37机制 A：归档与推断范围补充

这是机制 A 原始成功报告的补充说明。原始 bench 脚本、配置、模型、评分和报告保持不变；本次仅归档新副本并验证代码格式化后的复现性。机制 A 的预定判定仍为 **不确定**，样本充分；没有新增训练、调参或检索性能测试。

## Bootstrap 的条件范围

4000 次 bootstrap 以有效 query 为重采样单位，seed8002，使用共享的 query 抽样索引计算 Δpath、Δproxy 与 I_A，报告 97.5% 双侧区间。这里的推断**条件于已经冻结的中心与留出 proxy 表**，不覆盖重新抽取 proxy 所带来的不确定性。

重复中心在不同状态或 query 中复用同一组 16 个 proxy。60688 是映射到状态后的 proxy 观测数，59504 是 3719 个不同中心的 proxy 评分数；二者均不应被解释为独立抽样的代理样本数。共享中心会使不同 query 的 proxy 结果相关，因此该 query bootstrap 只作为固定表、已有查询下的稳定性描述，不能视为针对查询和代理总体的独立泛化保证。

## 归档和验证方法

- 原始脚本保留在 `/home/xingrui/bench/quantized_graph/research_20261025/scripts`；仓库副本经过 Ruff/clang-format。Python 额外改动仅为终端日志输出和将三个合取断言拆为顺序断言，受限 AST 比较检查这些改动。
- 冻结采样配置、缺失状态、原评分证明和逐 query/state 小结果保留原始字节。四张二进制采样表、原始分数、诊断数组、模型和可执行文件保留 bench，归档记录其路径和 SHA256。
- 独立目录 `replay80_matched_formatted_v1` 重新执行归档采样脚本，核对四张表的完整字节及配置语义；只允许采样时间和脚本 SHA 等元数据变化。
- 使用归档的实际 C++ scorer 及其已有依赖副本、原冻结生产编译参数和链接顺序，重算全部 63297 组观测，比较 `observations.bin`、`training_controls.f64`、`score_controls.json` 的 SHA256。生产查询预处理和 Turbo 距离回调保持原验证方式。
- 新证明明确标记为格式化复现证明。分析仍使用原始冻结配置，重新生成逐 query、逐 state 与 4000 次 bootstrap；要求这些输出逐字节一致、诊断数组逐元素一致、报告全部数值与策略字段相同，只允许路径、源码 SHA 与证明 SHA 等来源元数据不同。
- 原 runner 硬绑定初次成功的采样 manifest SHA 和原始来源路径。复现使用专门编排脚本，不改写旧证明来伪装一次原始运行；原 runner 的计算逻辑另外经过受限 AST 比较。

可复现命令（输出目录必须不存在）：

```bash
/home/xingrui/bench/symphony_qg/.venv/bin/python \
  research/20261025/scripts/replay80_matched_formatted_v1.py \
  --bench-root /home/xingrui/bench/quantized_graph/research_20261025 \
  --archive-root /home/xingrui/workspace/quantized_graph/symphony_research_20261025/research/20261025 \
  --out /home/xingrui/bench/quantized_graph/research_20261025/replay80_matched_formatted_v2
```

## 证据位置

- [原始主结果副本](../results/path80_matched_A_v1.json)
- [冻结采样配置](../results/path80_matched_artifacts/path80_matched_v1/sampling_manifest.json)
- [原评分证明](../results/path80_matched_artifacts/path80_matched_v1/score_v1/proof.json)
- [1000 条逐 query 结果](../results/path80_matched_artifacts/path80_matched_v1/analysis_v1/per_query.json)
- [3793 条逐 state 结果](../results/path80_matched_artifacts/path80_matched_v1/analysis_v1/per_state.csv)
- [格式化复现证据](../results/stage37_mechanism_a_replay.json)
- [归档显式清单](../results/stage37_mechanism_a_filelist.json)

原报告副本保留原文和原相对链接；其 `../path80_matched_v1/` 引用指向 bench 原始实验目录。本补充提供仓库内可定位的归档链接。
