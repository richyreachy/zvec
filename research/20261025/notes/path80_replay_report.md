# 阶段80：桶重放、控制实验与真实路径核验

已定位learned78同margin=.020时的**首次路径分歧**：全部1000主查询中，998次首先改变下一展开ID，1次首先提前停止，1次完整native路径和停止动作不变。重放预测已经独立真实learned trace逐查询核验。这解释了改码如何进入搜索决策，尚不能把最终37个净命中损失逐项因果归到第一处分歧，更不构成QPS/recall提升。

## 验证链

1. 对已有phase68完整1000主查询trace按原生产规则重放所有proposal：30,763次展开顺序、984,388条邻边顺序全部一致；保留未展开target的多来源重复，同桶LIFO，pop前清理visited副本；精确Top10和FP32 stop不等式保持原实现。
2. 原1000组Top10成员均来自expanded集合，逐项score bits相同，分数multiset等于expanded节点的精确Top10。所有结束都可由queue empty/cap/margin解释。当前native真实trace又与phase68整个events文件逐字节相同。
3. actual-code scorer先完成全1000 native分数、中心及邻边truth逐bit门槛，再用冻结learned模型重评分。原模型文件SHA256亦独立重查。overlay只替换kind2 estimate，不改变其他字段。
4. learned重放只沿共有前缀推进，到第一次next-ID/stop分歧即停止路径解释；分歧后的原轨迹只标为off-policy。
5. 独立comparator核对真实learned trace：全部1000的首次分歧类型、位置和下一目标均匹配；4,952个共同前缀中心、158,464条共同前缀边完整bits均与overlay一致。新trace两组各10,000行Top10 ID/hex分数分别与冻结native及screen78 learned .020输出相同。

MULTI没有直接记录stop reason、停止时pending target或获胜队列token来源，所以这些字段是通过已核验的重放恢复的，不应写成实际trace直接观测。没有PATH日志，`path_bits_verified=0`如实保留；实际展开选择和终止步的独立观测门槛已经通过。

## 全1000查询的首次分歧

| 项目 | 数值 |
|---|---:|
| 首次改变next ID | 998 |
| 共同状态下learned首先停止 | 1 |
| 整条native路径及停止动作不变 | 1 |
| 平均共有前缀展开数 | 4.952 |
| 共有前缀中位数 / 90分位 | 4 / 11 |
| 共有前缀最小 / 最大 | 1 / 19 |
| next-ID分歧时最低活bin编号相同 | 500 |
| next-ID分歧时learned最低活bin下移 / 上移 | 241 / 257 |
| 路径分歧前已有同target、不同获胜来源 | 40查询 |

1000查询均出现至少一个未展开proposal的bin变化，但只有999查询的展开/停止动作变化。这说明“跨桶”是可能影响动作的必要步骤之一，而非每次跨桶都改变动作。

500次“最低活bin相同”的分歧不表示所有proposal的bin都相同：某些提案跨桶改变了同一最低桶里的栈内容，LIFO可选中不同target。应区分“所有proposal逐个保持bin”和“队首最小bin编号碰巧相同”。

native终止状态重建为981次margin、19次cap。只有1次首次分歧直接是同一共有状态下的停止差异，因此不能将当前损失简单解释成“学习码使同margin直接更早停”。分歧后实际路径上的停止仍可能影响最终结果，此处没有排除这一点。

## 四项控制

| 控制 | 结果 |
|---|---|
| Identity overlay | 全1000无分歧；逐proposal bin与获胜来源一致 |
| 不读GT的单proposal干预 | 只改q0首块最后一个未展开proposal至bin0；仅q0在第2次展开前分歧，共有前缀1；下一ID从205770变为预定56521，其余999查询不变 |
| 非平凡同桶分数干预 | 改变全部984,388条kind2分数：984,383条nextafter(+∞)，5条采用nextafter(−∞)保留bin；所有bin不变，全1000展开顺序、终止动作和获胜来源无分歧 |
| 实际生产bucket LIFO小fixture | 固定同bin32的target1=.5、target2=.501；插入[1,2]则pop[2,1]，交换插入则pop[1,2]；直接include冻结生产bucket，不是只验证Python模拟器 |

上述控制不读取GT、没有训练或参数搜索。人工控制proof明确标记不是模型评分，不冒充actual-code scorer的native逐bit验证。

## 得到的结构性结论与局限

对于当前固定图、入口、参数、有限分数和普通bucket实现，若每个未展开proposal的bin保持不变，则按相同arrival顺序与LIFO归纳，队列状态、展开、精确Top10状态、停止动作和返回结果保持不变。连续分数本身不存入bucket，只决定入桶；因此在桶内降低连续MSE可能完全不改变查询行为。

这个结构性结论已经有非平凡全轨迹控制支持。它提示后续目标函数应单独检查桶边界跨越、竞争proposal相对顺序和多来源探索行为；不能仅凭离线MSE下降声称recall会提高。此阶段没有训练新的目标、选择新阈值或检验新算法收益。

当前1000主查询已有长期调参历史，不是盲测。没有用改善/退化query子集选择模型、控制或阈值。逐查询命中变化可与这份全量诊断事后关联，但不能把关联改写成所有召回损失的因果证明。共同前缀之后的真实路径已由实际trace保留，overlay仍不代表那一段真实路径。

## 证据位置

- 字段/源码行/协议：`notes/diagnostic80_plan.md`；独立审阅：`notes/path80_diagnostic_review.md`。
- Native重放：`path80_native_replay_v1/summary.json`；冻结源 `scripts/trace80_native_v1.py`。
- 真实native逐bit评分proof：`path80_native_v1/proof.json`。
- Learned共有前缀重放：`path80_learned_replay_v2/summary.json`、`queries.jsonl`、`proposals.csv`；冻结源 `scripts/trace80_replay_v2.py`。
- 全量分歧汇总：`results/path80_learned_common_prefix_summary_v1.json`。其中`actual_learned_trace_crosscheck_pending=true`记录生成时状态；之后独立核验已经通过，原证据没有覆盖。
- 四项控制汇总：`results/path80_controls_validation_v2.json`；同桶重放 `path80_same_bin_replay_v3/summary.json`；生产LIFO `results/path80_lifo_control_v1.json`。
- 实际trace输出等价性：`results/path80_actual_trace_equivalence_v1.json`；独立首分歧核验：`results/path80_actual_prefix_v1.json`。
- 所有运行命令/日志分别为 `results/path80_*_command.json` 与 `logs/path80_*.log`；LIFO自身目录含完整commands和编译/运行日志。

冻结版本v1、v2及成功输出未修改；新增同桶proof类型使用独立v3。新报告没有增加检索计时，也没有突破声明。
