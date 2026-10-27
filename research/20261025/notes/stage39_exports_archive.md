# Stage39：实际评分导出与独立主指标核算归档

格式化后的训练、验证导出与首次成功输出逐字节一致：两组的states.bin、queries.bin、scores.f32共6个文件全部SHA相同；scorer_result全部字段相同。格式后独立核算的1024query明细JSON也逐字节相同，报告算法和数值字段一致。报告见 `results/stage39_exports_postformat_validation.json`；独占重放目录为 `replay81_exports_formatted_v1`。

## 本次归档范围

只包含 `path81_actual_score.py/.cc`、`audit81_primary_metrics_v1.py`、导出schema、原训练/验证通过证明及小型inputs/plan/result、独立核算结果与本次重放证据。候选库、模型组装及另一份验证分析由其他agent归档，本清单不混入其文件。完整trace、向量、评分二进制和宽泛的冻结源码SHA清单留在bench，不复制到repo；有关输入由通过证明和原路径/SHA关联。

原始成功源码与数据保持不变。Python和C++格式化只作用于archive副本；此外有两处显式路径改动：

- exporter编译依赖从被调用脚本自己的目录读取，而不是固定从原bench `root/scripts`读取，确保实际重放编译归档源；数据根目录仍通过 `--root` 明确指定。
- 独立audit增加 `--root/--export/--output` 路径参数，使其可以核算新的等价导出；主指标规则和常数不变。

原/归档/实际执行副本SHA、源码diff、执行命令都保存在本归档的小型证据中。Ruff与clang-format及最终pre-commit通过。第一次格式检查的E701单行for提示由格式化展开后通过，日志保留；不存在数值门槛调整。

## 导出门禁与覆盖

训练导出包含6625个状态、3072query、20中心、137候选块；全部212000条native边评分/真值与原trace逐bit一致，20个原码块与真实codec逐字节一致，601个实际旋转通过。424000个有限FP32桶跃迁点及邻点通过检查。全部native门禁完成后才开始候选评分。

验证导出包含2183个状态、1024query，先完成69856条native边评分/真值门禁及139712个有限边界核验，再仅评分native、M2、Hinge。冻结训练模型SHA始终为 `adb477c316f1e6af5ef8cedaf5143851711d747866198455c0cb1908b0734251`。格式后验证重放仍绑定原始训练export proof和已封存模型；它是同数据/同模型的等价复现，没有重新训练、重新选模型或追加验收检验。

`eligible_unexpanded` 来自原query完整native展开元数据，包括当前中心；它不是first-seen去重。FP32桶判别独立于理想实数评分。边界为生产locate的可表示跃迁点，机器比较右侧包含等号；不是直接拿float乘积当桶分界。

## 独立主指标核算

`audit81_primary_metrics_v1.py` 不导入或读取另一份分析的实现与效果文件。它使用独立struct布局逐状态/邻居/边界计数、独立FP32除法复现locate，并以有理数精确求和计算1024个query误差比例的均值。主分母为137384行，1164个已展开target proposal被排除；所有1024query保留，无零分母query。true==threshold和prediction==threshold按右侧包含规则计数，本数据这些相等行均为0。

核算得到 native=0.40279537761351986、M2=0.3477660672647139、hinge=0.35072091072517825；hinge−native为−0.05207446688834162，hinge−M2为+0.0029548434604643723。未运行bootstrap、添加验收门槛或选择子组。完整每query分母、错误整数与误差比例保存在小型per_query.json。

独立audit开始实现前曾收到另一agent主动发送的效果摘要，因此**不声称对摘要盲态**。源配置与结果明确记录了这个边界；本实现没有读取其脚本、report、per-query或bootstrap文件。独立数值核算与统计盲态是不同的保证。

## 结果边界

所有状态均来自冻结native路径，是局部阈值分类结果，不是实际新策略路径。step≤8早于STOP_MIN=10，不构成停止决策样本。本次未执行primary-query Recall测试、完整检索、新路径测试或QPS计时；不能将导出成功、局部错误下降或格式重放一致称为重大性能突破。
