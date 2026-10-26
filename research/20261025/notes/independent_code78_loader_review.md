# 阶段78加载器只读设计审查

审查时研究树HEAD为706ef3d3，阶段77完整构建/验收尚在进行。本文件仅是接入与验证建议，不代表模型已验收或生产加载器已实现。未修改生产代码、编译或运行基准。

## 建议的最小接入

新增研究宏`QG_INDEPENDENT_CODE_IMPORT`，默认完全不编译该路径；宏开启但`QG_INDEPENDENT_CODE_FILE`未设置时保持原构建。模型启用只影响prebuild，搜索扫描热路径和Block布局不变。严格先限制在原始FP32中心、cosine、维度768、最大度32、完整768旋转及Symphony codec；后续整数中心组合另做实验。

可新增一个`quantized_graph_code_import.h`管理文件、头部/记录检查和只读映射。在`QuantizedGraph::prebuild`（quantized_graph.h:987）开始加载候选模型、源向量参考和旋转参考；在并行worker启动前验证schema、数量、支持条件与模型完整性。在`build_block`最终邻居确定后验证key/order，并在原编码完成后只覆盖满度块的3328码字节。短块沿用当前构建的原codec，避免不同编译上下文下原始系数的舍入差异改变baseline语义。

具体改动位置：

| 位置 | 用途 |
|---|---|
| quantized_graph.h:82附近 | 宏约束及新helper include |
| quantized_graph.h:451附近 | 仅在宏下保存导入配置/已验证状态，或将临时映射指针传给build_block |
| quantized_graph.h:521 | clear使已发布模型状态失效并释放临时资源 |
| quantized_graph.h:824 | 过滤、augment、剪枝结束，首次能可靠检查最终邻居序列 |
| quantized_graph.h:917之前 | 度0记录也必须完成中心key、度、padding检查 |
| quantized_graph.h:978–981之后 | 原编码完成；满度flags=1时复制学习码，flags=0继续使用原码 |
| quantized_graph.h:987–1036 | 一次性加载和支持条件验证，线程启动前失败 |
| quantized_graph.h:1076–1093 | 每中心源坐标/旋转验证可并入此worker，使用已pin住的Vector |
| quantized_graph.h:1187–1189 | 全部成功后发布arena/ready，随后释放导入映射 |
| quantized_graph.h:1325 | 若模型被clear或失败prebuild失效，查询边界返回Mismatch |
| symphony_qg_quantizer.h:88附近 | 添加研究宏下的codec能力查询；不改变encode/scan |

## key与最终邻居顺序

HnswGraphAdapter::key(id)在hnsw_quantized_graph.cc:37可用，返回外部key。模型行号是内部ID，中心/邻居字段是外部key，不能用key直接索引行或向量。

`build_block`从原entity.neighbors开始，经valid过滤（548–554）、可选导入（557）、QG_AUGMENT及剪枝（至822）才生成最终顺序。阶段64导出来自最终Block邻居（1191–1195），因此必须检查最终`scratch.neighbors`，不能检查原HNSW邻居后直接套码。

对每行要求：中心key==entity.key(id)，degree==最终neighbor_cnt；有效每槽neighbor_key==entity.key(scratch.neighbors[i])，余槽UINT64_MAX，degree32恰为flags1，其余恰为flags0。禁止排序以“修复”顺序，也不应借导入学习码改变邻接边。仅支持同内部ID布局的已验收模型；重新编号的等价图也先拒绝，不添加隐式key重映射。

`QuantizedGraphImport`现有实现只接收满32边行（quantized_graph_import.h:11–12, count==32检查），不适合直接复用来导入含2278短块的阶段77模型，也不能复用其140字节记录格式。

## Codec隔离及编译约束

`QuantizedGraph`同时被SQ8、PQ、PQResidual实例化（hnsw_quantized_graph.cc:303–316）。不要在泛型模板中无条件访问RaBitQ数据结构或仅Symphony拥有的成员。可沿用Fusion的SFINAE supported模式（quantized_graph_fusion.h:261–268）：只有Symphony提供`independent_codes_supported()`，其他Codec落到false。未设置文件时PQ/SQ8仍完全可用；设置文件而Codec不支持时明确返回Mismatch，禁止沉默忽略。

建议研究宏下拒绝以下组合：缺QG_ROTATION_768；QG_PROJECT_DIM、QG_RESIDUAL_TWO_STAGE、QG_LOWRANK、QG_DIRECTION_BETA、QG_DIRECTION_MSE、QG_RADIAL_CALIBRATION、QG_GEOM_BIAS；首版拒绝QG_CENTER_QUANTIZED及其reanchor变体；拒绝QG_FUSION（现有error_scale使用原sign(r)误差假设，不适用于学习码）；拒绝QG_HIERARCHICAL（upper graph有独立内部ID与记录数，当前全局模型路径会错误套用）；首版也可拒绝QG_GRAPH_IMPORT以保持单一邻接来源。

QG_AUGMENT继续使用冻结baseline配置；最终key/order全量检查应捕获构图规则变化。仅靠dimension、batch_bytes匹配无法区别错误Codec，所以必须做Codec能力隔离。

## 源向量和旋转兼容性：不能只看key

同一组key及邻接边可以对应更新过的向量。模型当前格式没有逐向量摘要，因此至少需要如下完整关联：

1. 使用已验收模型对应的`models64/full_vectors.f32`与`models77/rotated.f32`作为参考。参考文件应按模型header中的SHA256验证；验收JSON的最终模型SHA256也要对实际打开文件核对。文件路径存在或header自行声称某hash不构成校验。
2. 对每个当前entity向量，将前768个FP32坐标与参考内部ID行逐字节比较。Cosine存储的尾部原范数元数据不属于导出768坐标；保持原Block对stored_dimension的复制逻辑，不把该字段当训练坐标。
3. 对当前codec.transform得到的每个中心旋转向量，与参考旋转行逐字节比较。全部中心都会检查，因此不需要对每条邻边重复做数据完整性比较。这同时验证本次二进制真正执行的transform，而非只匹配源码hash或编译宏。维度相同并不能证明Hadamard排列、符号种子或实现一致。
4. 验证在prebuild外部排他锁保护下完成，任何失败都中止模型发布。若当前实现与参考存在SIMD舍入差异，应单独定位并验证，不放宽成任意1e−5“差不多”后继续。

当前源码检索未发现可直接复用的SHA256实现。可选择：A）在helper中引入小型经过独立已知向量验证的流式SHA256工具；或B）作为明确限定的研究加载流程，由基准启动器先核对模型和两个参考文件SHA256，C++再执行上述全量坐标/旋转比较。B不能被描述为“加载器本身完成了文件SHA校验”，并且直接调用加载器时仍需要显式提供可信验收条件。不要用文件名、大小、mtime或普通非加密hash冒充对应的SHA256证据。

JSON解析可使用项目已有`ailego::JsonValue`（src/include/zvec/ailego/encoding/json.h，使用示例src/core/interface/index_factory.cc:76）；需要按类型检查必需字段。不要用字符串包含查找判断字段值。固定研究fixture可用精确header模板作为额外一致性约束，但必须区分结构合法性与固定模型完整性。

## 头、记录与失败处理

先校验文件至少4096字节，再读magic、version、JSON长度≤4064、记录数和3600字节record stride。先做乘法/加法溢出检查，再比较精确长度4096+N*3600；拒绝截断和尾随字节。N必须等于当前doc_cnt且≤UINT32_MAX，header JSON中的database_count/record_count、dimension768、max_neighbors32、codec_bytes3328、record_bytes3600、16代理及目标参数应与二进制字段和支持模式一致。

校验JSON必须合法、数值类型/范围正确，header未使用填充为0。逐行检查degree≤32、flags∈{0,1}及其与degree关系、key/order/padding、有效因子有限；不能把非法flags或缺失行回退为原编码。完整模型应通过外部验收记录与最终哈希确认，`.partial`文件名本身既不是验收证明也不是可靠校验机制。

避免捕获错误后继续部分应用。并行worker使用原有first_error；任何异常转换为明确失败，不能从std::thread中泄漏异常触发terminate。新arena尚未发布时失败，可利用其局部RAII自动释放；新的映射也用RAII。旧arena若仍存在，ready状态必须失效，不能在错误重新加载后误报正在使用新模型。

只在prebuild中复制数据，完成后释放3.6GB模型映射及两个参考映射，查询期保留既有arena与少量状态。否则额外RSS会掩盖相同码长的内存比较。分别报告prebuild峰值RSS和稳定检索RSS。

## clear和惰性重建

现有clear会释放arena；get_block在arena之外会惰性调用原build_block（1283–1310）。HnswStreamer新增向量前确实会clear（hnsw_streamer.cc:838–842）。若仅在prebuild期间使用局部映射，clear后静默回退原码会让“启用学习模型”的语义不成立。

首版建议明确只支持冻结数据快照：记录required/ready状态，clear保留required但将ready置false；模型启用而未就绪时，在search查询边界返回Mismatch，需重新成功prebuild才恢复。未启用模型时保留原lazy行为。查询边界检查不应散布到每个候选扫描中。此策略需测clear、增删/更新、失败重载场景；不要同时扩展到在线增量训练。

## 必要的验证方案

1. **关闭等价性**：原宏关闭与冻结baseline；宏开启但文件未设置。核对实际Top10 ID与hex分数、Recall及合理重复计时，不能只做单元测试。未设置模型时不应读取模型或参考文件。
2. **正确加载**：用小型固定768维、至少33节点的synthetic Graph fixture测试全度、短度及度0；key特意不等于ID；邻居序列不排序。C++模型输出与独立codec参考比较，运行查询确认实际使用学习码。另对完整Cohere输出构建诊断哈希，确认997722学习块真正被复制、2278短块保持当前原码。
3. **错误模型**：短header、错误magic/version、JSON长度越界/语法错误/错类型/缺字段、binary/JSON数量冲突、record size错误、整数溢出、记录截断/额外尾字节、错误flags/degree/padding、非有限因子，均须明确失败且无部分发布。
4. **错配图**：相同大小但中心key置换、单个邻居key改变、邻居顺序交换、删除有效节点、同key同边但一个坐标变化、参考旋转变化，均须拒绝。错误参考文件hash也必须在相应完整性验证层拒绝。
5. **模板/宏**：PQ、SQ8、PQResidual在无模型路径下正常编译运行；有路径时拒绝；不支持维度/metric/degree拒绝；危险宏组合编译失败。RABITQ不支持环境不应引入未声明Symphony类型。
6. **生命周期**：两次成功prebuild、成功后失败重载、clear后search、clear后正确重建；无泄漏、无悬空映射、无静默原码回退。核对初始化失败能传播到HnswStreamer open（645–647已有返回路径）。
7. **实际评分链**：先确认原旋转及原扫描内核与阶段77相同，满度导入码的实际分数对冻结codec逐bit核对；保持原refine/停止/visited行为。之后才运行完整Recall@10≥90%的QPS网格和额外查询验证。

上述验证是待实现事项，未在本次只读审查中执行；阶段77离线验收也不能替代加载器边界和端到端检索验证。
