# 阶段80草案：解释learned78同margin下的召回损失

状态：**只读代码审阅＋未执行脚本草稿**。confirm78计时窗口内没有编译、数值计算、训练或检索。问题固定为已有主查询、Top10、margin=.020下learned78的Recall .8995与baseline .9032差异；本阶段先解释路径机制，不选择新margin、不用额外查询GT调参。确认实验中的learned .024不是这里的同参数因果比较。

## 现有三种诊断的实际含义

下列源行以当前研究仓库 `src/core/algorithm/quantized_graph/quantized_graph.h` 为准，归档时应固定源码SHA并重新核对行号。

| 诊断/开关 | 输出字段 | 产生点、时机和限制 |
|---|---|---|
| `QG_MULTI_TRACE=1`；运行时 `QG_MULTI_TRACE_FILE` | 文件16B头：u32 magic=`0x51474d68`,version=1,record_bytes=32,dimension=768；记录：u32 query,kind,step,source,target；f32 estimate,exact；u32 reserved=0 | `quantized_graph_multi_trace.h:11`定义；每线程Writer在:45首次读取环境变量，:23以`wbx`独占创建，:28写头。query是线程内搜索调用序号，不是数据集query ID |
| MULTI kind1 | `step=expanded.size()`，`source=target=id`，`estimate=exact=中心精确分数` | `quantized_graph.h:1912`先记录expanded；:2056实际`ctx.distance(center)`；:2066记录kind1。内部ID，不是external key。在StopState observe之前产生 |
| MULTI kind2 | query/step/source=当前展开中心；target=邻居内部ID；estimate=实际codec scan输出；exact=reference_distance | :2132调用实际FastScan；:2144–2153对所有有效lane记录。**发生在accepts/visited检查之前**，包含已经展开的target、重复来源。当前非fusion路径即原始FastScan估计；若开启fusion则位置在融合后，本阶段排除fusion |
| MULTI kind3 | step=总展开数；source/target=UINT32_MAX；两分数0 | :2447在SearchGraph结束后记录并flush。没有stop reason、下一队首或返回Top10字段；发生在:2621的结果补齐之前，不能不检查结果就当作最终返回阶段 |
| `QG_PATH_TRACE=1`；`QG_PATH_FILE` | 无表头CSV：query_serial,results.limit,expanded_step,external_key,exact,popped_estimate（后两项`%.9g`） | :1812–1817创建线程本地serial/FILE，`fopen(...,"w")`会覆盖且打开失败仅不写；:2318在pop前抓取beam.next_distance；:2072在中心精确评分后写一行。**bucket模式popped_estimate是桶下界，不是该条proposal原始FastScan值** |
| `QG_DIAG=1`；`QG_TRACE` | 无表头CSV：QG_EXPERIMENT,topk,results.limit,expanded,popped,pushes,scored,RMSE,bias,top1,batches | :1819定义计数；:2164统计全部被score的lane误差；:2171统计batch argmin是否一致；:2321 pop计数，:2353 push计数；:2615输出。没有query ID，只能按单线程调用序号对齐；fopen为`a`，不独占。即使未设输出环境变量，DIAG额外精确评分仍执行 |
| 补充 `QG_QUEUE_STATS=1`；`QG_STATS_FILE` | topk,limit,expanded,peak,memory_bytes | :2600–2606只提供聚合队列内存/峰值，不提供比较/拒绝原因，最小方案不需要 |

MULTI未设置路径时不额外评估邻居reference_distance；Writer打开失败或写入失败会使查询返回Mismatch。`QG_MULTI_TRACE`排除压缩中心（:93–97）；当前exact FP32中心满足要求。`reference_distance`在 `hnsw_quantized_graph.cc:58–61` 使用显式query距离接口，不消耗生产展开扫描预算；这是阶段68已经验证过的诊断路径，但仍需输出等价性门槛。

所有trace都应单查询线程、每次进程使用新文件。不要在同进程中改变环境变量期望Writer换文件；变量只在首次初始化读取。PATH/DIAG没有独占保护，由外层runner检查路径不存在并创建独立目录。MULTI的writer自身独占也不能让多个查询线程共享一个路径。

## 队列语义：首次来源只是描述字段

当前baseline/import宏均无 `QG_FIRST_ESTIMATE`、dominance、fusion等策略。同一未展开target的每个来源都可能入桶。

- `graph_search.h:156–168`先完整score，再按lane顺序逐条检查accepts与visited。已经展开的target仍有kind2估计，但不入队。
- 当前 `QuantizedGraphBucket::is_full` 恒false（bucket.h:114），接受所有有限分数；并不依据`ef800`裁掉bucket proposal。
- bucket.h:48–53：负分数进0桶；其余`x=FP32(score/width)`，正常桶`size_t(x)+1`，超过范围进入2047溢出桶；桶数默认2048。
- bucket.h:42–43：宽度仅在第一个中心展开时设为 `FP32(max(entry_exact,1e-6f)*.016f)`。入口最初以FLT_MAX在width=1时入队，不能按缩放后宽度重建入口。
- bucket.h:61–69及:86–112：同桶**LIFO**，存储目标ID，不保留连续score。重复target、相同bucket的后到proposal仍改变栈位置；先dedup会改变路径。
- `quantized_graph.h:2288–2305`先丢弃已visited的队首重复proposal，再比较下一桶下界与停止状态；清理重复的pop不会经过Frontier::pop，所以DIAG popped不包含这些清理。
- StopState :1725–1740按精确中心分数维护Top10及stale；:1770–1774按原FP32表达式测试。固定cap64、patience0、min10时，满足expanded≥64即可停；否则比较 `FP32(next_lower−tau) >= FP32(.020f*max(tau,1e−6f))`。**不要改写成FP64的next≥tau*(1+margin)**，这会改变边界舍入。

因此必须保留proposal token：`query,source,expansion_step,lane,arrival_serial,target,score_bits,bucket_id`。额外标记当时是否已visited；记录真正被pop选中的token及source，而不是把第一来源当作“锁定了这个候选”。

## 最小诊断协议

### 门槛A：旧trace能否重建实际native行为

复用 `models68/primary/events.bin`、`models64/full_graph.bin`、`variants/trace68_native/metadata.json`，以及phase68既有1000主查询结果等价性证据。先对**全部1000查询**验证：

1. 每个kind1紧跟degree条kind2，source/target/lane顺序与冻结图一致；query/step/end完整。
2. 按实际bucket/LIFO/重复proposal规则重放，逐步匹配所有展开内部ID及顺序。
3. 重建每步精确Top10、stale、原FP32停止表达式；结束必须能解释为empty/cap/margin。若无法解释，不默认“早停”，应检查scan budget、filter、结果补齐或脚本语义。
4. 如有同次PATH文件，逐bit比对精确中心分数与重建pop桶下界。现有phase68只有MULTI，故仅凭该文件不能声称已观测并逐bit比对队首下界；它可核验展开顺序和原程序允许的终止条件。需要时下一次新诊断运行同时启用MULTI+PATH，以补上这个观测门槛。
5. 将最终返回ID及FP32分数与phase68冻结结果核验，确认结果全部来自展开集合、没有未追踪completion。已有阶段68做过该门槛；新脚本不代替新的输出等价性核验。

历史trace能够用于低成本诊断，但“最终Top10相同”本身不保证当前版本的全部路径相同。若要将结论外推到当前learned78运行，应固定当前源/图/原码，必要时用同一trace variant模型关闭重采一次，证明当前native路径与旧trace相同。

### 门槛B：离线重评分必须复现真实内核

loader_design代理正在写新的path80评分草稿，仅新bench文件，不改生产。

- 原查询TXT经生产 `CosineNormalizeReformer`、实际codec.prepare_query/原旋转；中心分数使用trace记录的FP32 exact，并核验生产metric(dim769)结果。
- native重新encode/scan必须对原kind2估计**逐bit相同**；不满足则先修归一化、旋转、codec或数据对齐，不使用近似几何结果解释路径。
- `models68/primary/queries.f32` 是NumPy FP64 norm后cast的几何分析输入，不能直接拿来承诺此门槛。
- 同一query/source/lane改用已冻结learned77模型码，再运行实际FastScan。所有未展开多来源均保留，短块原codec保持不变。
- 输出同MULTI布局 `native_path_learned_scores.bin`，只替换kind2 estimate，kind1/kind3和其他字段逐字节保留。它是**native路径上的off-policy评分overlay，不是learned真实trace**。
- 成功proof固定字段：`native_scores_bit_identical=true, native_truth_bit_identical=true, off_policy_rescored_native_events=true, actual_learned_path=false, all_queries_native_gate=true, query_count=1000, source_events_sha256, rescored_events_sha256`。sidecar可附global_event_index、degree/lane及native/learned/center bits。

### 固定样本与两个不同的问题

机制样本预先固定为已有主查询q0..127，不按recall下降或误差大小筛选；native门槛覆盖全1000。现有主查询已经用于历史调参，报告必须标注非盲测。本诊断脚本不读取GT；若主任务以后关联已有逐查询hits，只作事后描述，不据此改训练、margin或样本。

**共同前缀解释**：两个队列从相同entry启动，以对应native/learned估计入队。只要next ID和continue/stop一致，后续真实扫描中心仍相同，overlay足够计算下一决策。记录第一次next-ID或stop分歧后，立即停止learned路径解释。可以说“在这个已验证共同状态，改码使最小活桶/同桶LIFO获胜token改变”，不能把之后强制native展开序列说成真实learned路径。

**强制native路径影子统计**：分歧后的native事件仍可比较两种估计的误差、bucket位置及多来源负尾，但这是off-policy样本。若另外模拟强制展开的shadow队列，必须单独输出并命名；不能与共同前缀因果事件混合。本草稿没有继续推进分歧后的learned队列。

## 需要交付的分析字段

| 层级 | 字段及用途 |
|---|---|
| 每个原始估计 | query/source/target/step/lane/global event index、arrival serial、native/learned/exact FP32 bits、三种bucket ID、native是否已展开target |
| 多来源candidate | 第一次遇到来源、第一次展开前来源、全部到达token、来源数、提案数；首次字段仅描述，不代替获胜来源 |
| 每个实际展开 | 获胜token完整来源、当前bucket下界及bits、精确score、Top10 tau、stale、队列proposal数、distinct target数、该winner的重复proposal数 |
| 每次比较/停止 | native与learned next token、原FP32不等式两侧及bits、cap/empty/margin状态、visited重复清理数；现有trace不直接记录，只有通过逐步重放才可恢复 |
| 桶别名 | native/learned/exact bin、到最近名义FP32桶边界的距离；这是描述量，rounded division使它不等于改变bin所需的精确最小扰动 |
| 两个时间点 | 首次**未展开proposal**发生bin改变的step/lane；首次next-ID/stop路径分歧的step。二者通常不相同 |
| 结果关联 | 当前native返回ID与score；可选既有逐query hits变化。未见过的GT邻居、未扫描区域缺失不自动归因于停止 |

核心不变量：相同entry、图、finite分数和参数下，如果native观察路径上每个未展开proposal的bin都未改变，则按入队序与LIFO归纳，两队列状态、pop、stop及返回结果必相同。连续分数在桶内改善不改变当前查询动作。首次bin改变也不必造成路径变化：其proposal可能不是队首、随后因visited被清理、或仍由相同target获胜。

可检验但尚未证实的机制包括：总体MSE改善主要落在同桶内；学习削弱多来源最小桶的负误差尾部，失去探索桥梁；少数桶边界跨越改变LIFO探索顺序；同margin下最小活桶上移，使停止条件更早成立。必须通过获胜token、共同前缀和停止状态分别核查，不能从MSE方向直接推断召回方向。

## 下一次实际trace所需宏/运行方法

优先使用同一冻结variant：`independent78_native_v1`宏集合再加 `QG_MULTI_TRACE=1,QG_PATH_TRACE=1`；不需要DIAG，它重复做reference评分，MULTI已足够计算其聚合量。**不能现在编译，待root解除计时禁令。**

基线组unset全部`QG_INDEPENDENT_*`；learned组用原冻结model/raw/rotated/header/SHA，不训练。相同CPU绑定、8线程预构建、1查询线程、一次进程只跑一种条件/一个margin。诊断调用复用 `research_benchmark_nav search ... 0 -1 <exclusive-prefix>`，环境固定 `BENCH_ROUNDS=1,BENCH_THREADS=1,BENCH_TOPK=10,BENCH_DEGREE=32,BENCH_STOP_GRID=1,BENCH_MARGIN_DENOM=1000,BENCH_EFS=20000,QG_CAP=64,QG_PATIENCE=0`，query_begin=0/end=128或全1000门槛。

`benchmark_nav.cc:240`逐query顺序做唯一一次诊断搜索；duration=-1在:272跳过QPS测量。固定一个ef，避免query serial跨多个grid/round串联；不要误把预热＋正式计时产生的serial当原query ID。现有驱动仍会读已有GT报告Recall，不能把它描述为不接触GT的工具；**本分析/评分脚本本身不读GT、不拟合**。没有新QPS结果可从trace运行得出。

运行前后用对应条件原结果（baseline frozen或screen78 .020子集）核验ID/hex分数完全一致。trace输出只能生成新目录，禁止重用phase68原路径。若新trace改变Top10，诊断门槛失败，不能继续归因。

## 草稿与尚未覆盖的边界

新文件 `scripts/trace80_draft.py` 只写未执行，甚至未做syntax/数值测试。它支持全1000 native重放、可选PATH逐bit检查，以及scorer overlay共同前缀诊断，固定q0..127输出proposal及winning token。所有输出要求新目录，校验metadata仅含冻结bucket宏及trace/import标记，拒绝其他策略。

草稿尚未自动读取原results CSV验证completion/Top10，也未记录所有CPU比较指令；它重建算法级queue动作。没有实际新learned trace就不能知道首次分歧后的真实路径、真正漏掉哪些新邻居或最终召回的因果来源。新trace也无法仅凭已扫描区域保证全图最优近邻。单查询机制或固定128样本不能证明1000查询总体提升，更不能证明论文创新或泛化。

阶段68可复用的证据/脚本：`notes/multianchor68.md`、`scripts/analyze68.py`（逐块图顺序/前两次到达）、`scripts/primary68.py`（主查询输出等价）、`scripts/run68.py`、`results/trace68_primary_equivalence.json`、`results/trace68_validation.json`。其中早期 `analyze_paths.py` 是利用GT搜索停止超参的旧脚本，**本阶段不运行或复用其参数网格**；仅参考其解析CSV的方法。

## 解禁后的native门槛执行记录

主任务随后明确解除计算/编译限制，并要求全1000查询机制分析；此前固定128的建议已被该指令取代，不能按事后坏query筛选。已从draft生成独立冻结版本 `scripts/trace80_native_v1.py`，其SHA256为 `d28248aba839cbf6aa1117234e5ffddd921808d1e06e622387d9bc49a4b62ea7`；原draft未修改。

首跑通过，输出 `path80_native_replay_v1/summary.json`、`queries.jsonl`、`proposals.csv`。30,763次展开顺序和984,388条邻边记录逐项核验；1000组返回Top10都属于expanded集合，每项精确score bits一致，分数multiset等于expanded节点的精确Top10。所有终止均能由原queue empty/cap/margin逻辑解释。没有读取GT或搜索新参数。

新版本还核验原trace SHA、图SHA及phase68 runtime config，并将队列原始entry总数与未visited活entry数分开。为learned共同前缀保留了“next ID相同但winning token来源改变”的独立字段，尚未在本次native-only运行执行learned替换。

现有MULTI文件没有直接观测pop桶下界，因此 `path_bits_verified=0`、`observed_path_score_bits_validated=false` 被如实保留。上述结果不是完整learned路径验证，也不说明learned模型改进召回。
