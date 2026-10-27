# Phase81 后续提案：完整 native 容器加 20 中心覆盖

**未执行。** Phase81 冻结 gate 未通过，本协议的 hinge 模型不进入真实检索；没有构建本提案的完整模型、编译加载器或运行查询。以下只读设计可供另行冻结的独立后续协议评估，不能视作本轮已授权实施。

## 最小诚实接入方案

建议保留现有 QuantizedGraph 的 prebuild、arena、scan 与搜索路径，只为 `independent_code_import.h` 增加一个明确的加载时 header profile。离线生成一次完整 native bank，然后复制为两个独立文件，分别覆盖 20 个中心的已封存 selected block。其他所有中心必须与 native bank 逐字节一致。

不能直接复用 phase77 header 来冒充新模型 provenance。当前 `independent78::Import::load`（约650–676行）硬编码旧训练语义：16 proxies、sweeps=2、每轮4 flips、seed7602、M2 objective、lambda及旧audit/history布局。新 hinge/受限候选训练不满足这些描述。把不同目标的新码塞进该头，虽然可能通过当前字节校验，仍会留下错误的模型来源说明。

可使用相同4096字节头与3600字节记录布局，但新增 `QGIC81` magic/profile，头中准确记录 native-default、覆盖中心列表、selected method、phase81协议/候选/封存训练模型SHA、构建器源码/二进制SHA和实际输入SHA。旧 `QGIC77` 分支继续保持全部严格校验；新 profile 不要求与新模型无关的旧代理训练字段。两者共用外部可信 header 完整字节比对、外部模型SHA、实际raw/rotated SHA和全部记录检查。

不需要改动 `QuantizedGraph::import_codes`（约599行）、`build_block`、query scan 或 arena 发布生命周期。旧入口环境变量仍提供 model/raw/rotation/header/SHA。模型只在prebuild时复制到arena；映射释放后搜索不访问模型文件。

## 可执行的离线构建步骤

1. 新工具 `native81_fullbank.cc` 只读 `models64/full_graph.bin`、`models77/rotated.f32` 及可信 raw/hash。按内部ID0..999999顺序，以graph最终邻居顺序取旋转向量，调用与phase80/81验收相同的实际 `SymphonyQGBlockQuantizer::encode`。不能用coupled73重编码native factors，也不能使用近似norm。8个worker各持有本地codec/scratch。
2. 输出新独占目录的 `native.partial`：4096字节新profile头；每行包含center外部key、degree、full-row flag、32个邻居外部key（不足补UINT64_MAX），以及真实3328字节码块。全部997722个满度块也存native；2278个短块保留native并与现有加载器相同地不覆盖运行时短块。
3. 复用的只是已验证输入/编码器和**记录布局**。不读取models77学习码作为默认码，不重跑phase77优化，也不产生proxy/history/moves文件。已有phase77模型的key/order可作为独立元数据参照，但无需复制它的旧头或学习码。
4. 独立进程全量重编码并核对1M行native bytes与key/order、有限factor；20个实验中心额外与 `path81_candidates_v1/native_blocks.bin` 全字节比较。保存新的SHA、计数和构建命令后才将partial发布为不可变 `native.bin`。
5. 以两个**独立副本**产生M2/hinge容器，不可用硬链接后原位patch。每个中心内部ID为c时，码块offset为 `4096 + c*3600 + 272`，覆盖3328字节，来源必须是封存 `selected_models.bin` 相应method/center块。20中心的key、degree32与邻居顺序必须再次核对。
6. 各文件单独生成真实header/profile及SHA。全量差异审计要求：非20中心的完整3600字节记录与native基准相同；20中心的前272字节相同，码块逐字节等于封存selected block；实际变化中心数可以少于20，但不能有第21个中心变化。
7. 只有独立后续协议的gate允许后才接入真实检索。先比较同编译binary下disabled/native-import的原码行为，再进行selected-import。保留同编译native-import对照，避免将编译差异或加载过程误认为编码收益。

## 已有缓存能复用到什么程度

只读检查未发现持久化的1M全native码库。`models74/codes_native.bin` 与 `models75/codes_native.bin` 各6815744字节，仅2048块，不适合充当全图默认；`path81_candidates_v1/native_blocks.bin` 为20块。phase80 scorer缓存的是进程内访问过的block，未保存完整native bank。

生产 `QuantizedGraph` 的 `arena_` 是进程内分配，clear/destructor释放；`HnswStreamer` 每次打开索引均调用8线程prebuild（约645行），未发现将量化arena序列化到索引的路径。现有 `models77/full/blocks.bin` 是997722个学习满块加2278个native短块，不能当作“其他中心native”的默认库。

## 成本估计与依据

每个完整容器精确为 **3600004096字节**。native及两个overlay全部落盘需要 **10800012288字节**（约10.06GiB），不含旧输入。每个overlay真正替换的码数据最多只有 **66560字节**；但当前bench文件系统由`stat -f`报告ext2/ext3类型，不能承诺reflink节省物理空间，应按完整复制预算。

已有四轮 `experiments/confirm78_r{0..3}_{condition}/run.log` 记录的1M节点8线程prebuild：

| 条件 | 4轮 cost_ms | 中位数 |
|---|---|---:|
| 原baseline | 36939,36827,36802,36815 | 36821ms |
| 同编译disabled | 37193,37250,37221,37214 | 37217.5ms |
| phase78完整模型import | 74411,74356,74253,74226 | 74304.5ms |

这些是既有日志，不是本提案的实测。离线native构建可直接复用已有rotation和最终graph，省去旋转和邻居选择，但会增加约3.6GB顺序写出；合理的工程排期为一次8线程构建加全量核验约1–3分钟，必须用新日志实测，不能将该区间当成保证。当前phase81候选编码0.672秒包含2740个完整候选块及601个rotation检查，操作组成不同，不宜简单外推1M原码吞吐。

每次现有完整import会读取/哈希model+raw+rotation共 **9744004112字节**，并重新构建运行时arena；历史额外启动成本约37秒。它不会因为仅覆盖20中心而自动降低。模型映射只用于prebuild并释放，日志已有 `release_mapped_bytes=9744004112` 证据；实际峰值内存仍需测量。

较省磁盘的稀疏覆盖加载器也可以只在prebuild时使用20行映射，但需要新格式、缺失中心语义和计数/错误路径验证，改动面大于仅加header profile。当前提案优先完整native bank，以复用已有完整记录与生命周期验收。
