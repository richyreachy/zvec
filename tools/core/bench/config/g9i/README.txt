g9i：succinct 三数据集最佳已测配置归档（2026-11-19）

重要：这是已完成全量实验的归档，不是新提交 3634b4a6 的复验结果。
测量版本：b44f0e09 + Python QG binding；归档时分支版本：3634b4a6。
机器由用户标注为 g9i；lscpu 报告 Intel Xeon 6982P-C，完整拓扑见 validation.json。
请勿将 r7i 的硬件、时间或性能数值移入本目录。

范围与选择方法
- SIFT/GIST/GloVe 完整训练集与查询集，FP32，TopK=10。
- 每个数据集同一 HNSW 底图；11个消融变体，总计1,290次参数点评测、6,450轮查询。
- 主指标：3次重复各自 best-of-five QPS 的中位数，不是所有历史批次的单次最高值。
- 在这一批次全部候选中选择达到30/50/70/85/90/95/99%下限的最快点。
- 每轮实际 ANN 距离容差召回均须达标；距离容差为第10真值距离+0.001。
- 参数网格有限，85%和90%可能选中同一测量点；不插值、不保证全局最优。
- 公开查询集同时用于选参和报告，没有独立调参/评估拆分。
- 单核、基线收尾后测量；每点先全查询集预热，perf_counter_ns 计时 Python search。
- 关闭路由的少量优势可能接近运行波动，不能认为所有数据都应该关闭路由。
- 本目录不混入32查询预检，也不混入不同计时批次的精排剪枝消融数据。

文件
- sift.json / gist.json / glove.json：每档查询参数、构建profile、三次QPS、实际召回、证据指纹。
- summary.csv：21个推荐点的概要。
- validation.json：机器信息、原测量二进制指纹、原始测量文件指纹、全量形状及5轮检查。
- validate.py：检查JSON字段、CSV一致性和达标条件，不运行性能测试。
- 原始NPZ、数据和索引仍位于外部ann-benchmarks工程，不提交大文件。

新提交的接口兼容性
3634b4a6 将 QuantizedGraphIndex / QuantizedGraphQuery 从 HNSW 解耦。
旧 use_quantized_graph、quantized_graph_* JSON 以及旧 local_builder 的
QuantizedGraph / QuantizedGraphRouter YAML 被明确拒绝，不能照抄 r7i 命令运行。
本目录因此不提供声称可执行的旧 YAML；legacy_* 仅用于复现历史测量语义。

迁移到原生API时：
1. 用 profiles.<name>.base_graph 构建底图，提供 Graph provider；保留原始向量精度与距离语义。
2. native_build_options 对应 QuantizedGraphBuildOptions 字段；codec 字符串
   native/global/low_rank 对应 Codec::Native/Global/LowRank，codebook 对应枚举。
3. router 字段用于 QuantizedGraphIndex::train_router；rotation_views 必须与编码一致。
4. native_api_mapping.search_options 对应 QuantizedGraphSearchOptions。
5. 历史 ef_search 仅保存在 capacity_candidate；它不是新API的同名字段。
   在 QuantizedGraphQuery::reset(query, topk, capacity, options, scan_limit) 中
   采用该容量仍需调用方适配和实测验证。不要把 legacy_query_params 直接交给 IndexFactory。
6. cosine provider/query 的归一化、精确距离回调、ID映射、快照加载由调用方负责。

所有 native_api_mapping 均显式标记 validated_on_new_api=false。
这些映射经过字段核对，没有新版本原生运行验证；不能据此声称新提交有相同QPS或召回。
后续新版本复验应另存二进制/索引/数据指纹和结果，不能覆盖历史测量版本身份。

数据集 / 召回下限 / 中位数QPS / 实际召回 / profile
gist-960-euclidean / 30% / 68665 / 30.560% / hadamard_low_rank
gist-960-euclidean / 50% / 48550 / 50.030% / full
gist-960-euclidean / 70% / 32938 / 71.930% / no_router
gist-960-euclidean / 85% / 18292 / 90.520% / full
gist-960-euclidean / 90% / 18292 / 90.520% / full
gist-960-euclidean / 95% / 11941 / 95.590% / full
gist-960-euclidean / 99% / 4017 / 99.370% / no_router
glove-100-angular / 30% / 95980 / 35.348% / full
glove-100-angular / 50% / 64767 / 53.785% / no_router
glove-100-angular / 70% / 44784 / 73.153% / no_router
glove-100-angular / 85% / 19009 / 87.336% / one_view
glove-100-angular / 90% / 18083 / 90.685% / full
glove-100-angular / 95% / 10590 / 95.575% / no_router
glove-100-angular / 99% / 3494 / 99.330% / no_router
sift-128-euclidean / 30% / 129348 / 32.093% / no_router
sift-128-euclidean / 50% / 104524 / 51.758% / full
sift-128-euclidean / 70% / 76901 / 73.709% / full
sift-128-euclidean / 85% / 62512 / 88.805% / full
sift-128-euclidean / 90% / 59401 / 90.301% / full
sift-128-euclidean / 95% / 47210 / 95.178% / full
sift-128-euclidean / 99% / 28696 / 99.018% / no_router
