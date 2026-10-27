# 阶段36：格式化归档与机械重放验证

本阶段只归档已完成的阶段80机械验证，不包含仍在推进的机制 A/B 统计、训练或性能实验。显式清单见 `../results/stage36_archive_filelist.json`：58份原始成功脚本、协议、命令和小型结果；没有按通配符搜集阶段80文件，也没有复制大二进制输出。新增验证证据清单见 `../results/stage36_postformat_archive_additions.json`。

## 验证结果

隔离目录为 `/home/xingrui/bench/quantized_graph/research_20261025/replay80_formatted_v1`。总结果 `../results/stage36_postformat_validation.json` 的 `passed` 为真。

- 有理数阈值检查的全部算法和数值字段与原结果一致。
- 归档 C++ scorer 对全部1000条查询重放，984388条边的原码评分与真值逐bit通过，364384个实际旋转与冻结数据逐bit通过；学习评分、事件审计、预处理查询、旋转查询四份二进制输出与原成功输出逐字节一致。
- scorer 的编译命令明确编译隔离目录中从归档复制的 `path80_score.cc`，并逐字节验证该快照等于归档源码；查询 probe 同样编译归档源，输出字节与原成功结果相同。编译 flags、链接顺序和冻结 variant 源/库保持原设置。
- native、identity、synthetic、same-bin、learned 五组重放各覆盖全部1000条查询；每组 `queries.jsonl` 和 `proposals.csv` 与原成功证据逐字节一致，摘要算法与数值字段相同。same-bin 对984388个分数作非零扰动而没有改变桶或路径；synthetic 仅产生预定的一次首次分歧。
- actual-prefix 独立比较器的全部算法与数值字段相同，核对全部1000条查询的首次分歧；其实际 trace 输入保持只读。生产桶 LIFO fixture 用归档 C++源重新编译，通过固定候选反转到达次序的控制。
- 重新计算清单中123个原始成功文件、归档文件、生产头及冻结元数据的 SHA，所有列明受保护文件均保持不变。scorer另核对其输入、冻结源清单及链接库，原模型仅作为输入使用；sandbox的符号链接不是操作系统级只读挂载。没有重新执行完整检索、训练或性能计时。

## 格式化差异与证据边界

归档脚本经过 Ruff、clang-format 和 pre-commit 检查。相对于原始成功源码的显式差异保存在 `../results/path80_artifacts/replay80_formatted_v1/source_diffs/`，原始和归档 SHA、实际执行副本 SHA 在同目录 `source_manifest.json` 中。修改仅涉及格式、导入排序、CLI `print` 改为日志、合取 assert 拆分、dict 字面量、保留函数调用但去掉未使用绑定。对长流程使用局部复杂度豁免以避免重构；比较器的 `zip(..., strict=False)` 显式保留原先短序列终止行为。

允许变化是脚本和可执行文件的 SHA、源码/输出/日志路径、执行耗时，以及仅因上述来源信息变化而产生的 proof SHA。数值输出、事件顺序、预测分歧、查询预处理和旋转二进制不允许变化。Sandbox 中脚本字节等于归档脚本；只有编译命令 JSON 的路径被重定向，替换记录保存在 `build_command_path_replacements.json`。

首版汇总 orchestrator 在所有数值重放完成后，误把 LIFO 的命令清单字典当成记录列表读取，出现 Python `TypeError`。该失败不是算法或验证 gate 失败。首版源码快照和完整失败记录保存在 `orchestrator_v1.py.txt`、`first_orchestrator_failure.json`；独立 `finish_v2.py.txt` 按正确字典格式读取命令，仅完成余下 LIFO 控制并复核全部受保护 SHA。首失败文件和先前成功输出未覆盖，最终报告明确记录恢复过程。这两个 `.py.txt` 是原样保存的执行证据，并非新的测试算法。

`trace80_primary.py` 已归档，但本次没有重新采集完整真实检索；使用阶段80既有成功 actual trace 的 SHA、原结果与独立前缀比较。native replay 对停止表达式与展开顺序作重建校验，`path_bits_verified=0` 如实保留，因为该输入不包含单独的 PATH trace。actual MULTI trace 也不直接观察停止原因、待停止目标及获胜来源，因此没有把这些未观察字段声称为直接验证。

历史 `path80_learned_common_prefix_summary_v1.json` 中的 `actual_learned_trace_crosscheck_pending=true` 原样保留；之后的 `path80_actual_prefix_v1.json` 已完成该核验。学习替代评分只在真实共同前缀内用于路径归因；分歧后的 native 轨迹重新评分仍为离线诊断。
