# Stage38：Phase81准备归档与求解器可用性

归档源来自 `scripts/prepare81_hot_centers.py` 的首次成功版本，原始SHA为 `0d0758235fe02042db3ab4905b062632fd16a3aff7209560390db843c2d7e624`。只对repo副本进行Ruff导入排序和格式化，归档SHA为 `9ccd5517e94b9444f442c3e8eb06da1c2e8da5382d72fd0e86efd9669f1109f3`；原脚本和首次成功证据未改。源码差异保存在 `results/stage38_prepare81_source.diff`。

归档版本显式传入原bench `--root`，在全新 `path81_formatted_preflight_v1` 输出。config、split、完整训练中心计数、中心名单、选中训练状态、选中验证状态六份文件均与原输出逐字节相同。result全部算法和数值字段相同；允许差异只有script路径/SHA及其衍生的冻结manifest SHA。没有读取score/GT、安装依赖、执行训练或优化。见 `results/stage38_preflight_postformat_validation.json`。

归档保留小型配置、split、中心名单、计数与结果；原trace、大向量和选中状态完整列表留在bench，后两者的SHA列于冻结manifest/重放报告。本阶段未暂存或提交文件。

Ruff lint/format已通过。首次pre-commit中的gitleaks全局检查扫描到已暂存stage36源码清单中“源码路径→SHA”的generic-api-key误报，涉及phase80两份inputs.json，未报phase81内容。日志完整保留；本阶段未修改历史证据或安全检查配置，此项需root在提交阶段处理。

## 只import检查结果

| Python环境 | NumPy | SciPy | linprog | milp | 可导入HiGHS模块 |
|---|---:|---:|---|---|---|
| bench `.venv` Python3.12.3 | 2.5.4 | 未安装 | 无 | 无 | 无 |
| `/usr/bin/python3` 3.12.3 | 未安装 | 未安装 | 无 | 无 | 无 |
| anaconda Python3.8.5 | 1.19.2 | 1.5.2 | 有 | 无 | 无 |

检查仅导入模块和读取版本/函数可用性；没有调用linprog、milp或求解器，没有安装新依赖。旧SciPy的linprog可用于其支持的连续LP；它不能代替二进制精确最优性证书。三个环境的逐项结果分别保存为 `path81_solver_*import_availability_v1.json`。

## 精确binary hinge小试规模建议（未运行）

建议先验证一个固定中心、一个固定邻居的受限子问题，而不是同时优化20×32个码。所有选择和预算必须在读取训练标签前冻结，例如按既有中心顺序、固定邻居lane以及仅依赖残差几何的坐标顺序选择。不要用验证类比例或损失选择小试。

不新增求解器依赖的最小校验可以固定16个允许翻转坐标、B≤2，枚举 `1+16+C(16,2)=137` 个码并按既定径向质量约束过滤。其余752个坐标固定原符号。每个码在该中心全部训练query上计算已冻结的hinge，保留原码可行解；这能够给出“16坐标受限子问题”的精确最优码，不能声称768维问题最优。B≤3为697个码，仍适合独立穷举对照，但不是建议事后试多个B择优。

若以后授权可用的MILP后端，完整单邻居问题有768个二进制翻转变量、n个非负hinge松弛变量、n条hinge不等式和2条预算约束，其中n为该中心训练query数（64..3072）。先只解一对center/neighbor并与小穷举对照；报告可行最优值、最优界、gap、时限状态。默认浮点MILP的“optimal”是求解器容差意义的证书，不等同有理数精确证明，且不替代生产FP32/FastScan再验证。

对完整768维，B≤1也可直接枚举769个码；B≤2有295297个码，代价明显增大。可用训练query支持数规划计算规模，但不能凭支持数推断两类标签平衡或测试集改进。当前阶段不选择最终B、τ、α、阈值生成器或类别缺失处理策略，也不执行上述小试。
