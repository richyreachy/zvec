# IVF 的 RaBitQ 量化模式

通过现有 IVF 类型选择 RaBitQ，无需创建独立的 `IVF_RABITQ` 索引。

```python
import zvec

index_param = zvec.IVFIndexParam(
    metric_type=zvec.MetricType.L2,
    n_list=1024,
    n_iters=20,
    quantize_type=zvec.QuantizeType.RABITQ,
    total_bits=7,
    sample_count=0,
)
query_param = zvec.IVFQueryParam(nprobe=16)
```

将 `index_param` 传给 `VectorSchema`，将 `query_param` 传给 `Query`。
索引类型和查询类型均保持 `IVF`，参数随 collection manifest 保存。

C++ core 接口：

```cpp
zvec::core_interface::RabitqQuantizerParam quantizer(7);
quantizer.num_clusters = 16;  // RaBitQ 模型中心，与 IVF 列表数独立
quantizer.sample_count = 0;
quantizer.niters = 20;
auto param = zvec::core_interface::IVFIndexParamBuilder()
    .with_data_type(zvec::core_interface::DataType::DT_FP32)
    .with_dimension(128)
    .with_metric_type(zvec::core_interface::MetricType::kL2sq)
    .with_n_list(1024)
    .with_n_iters(20)
    .with_quantizer_param(quantizer)
    .build();
```

- `total_bits` 为每维量化位数，范围 1–9，默认 7。
- `sample_count` 为训练采样数量，0 使用全部训练向量。
- `n_list` 和 `n_iters` 必须为正数；`nprobe` 使用现有 IVF 查询参数，RaBitQ 模式下必须大于 0；超过 `n_list` 时由后端裁剪。
- 支持 FP32 及 L2、IP、Cosine；core 为 2–4095 维，collection 保留 64–4095 维限制。
- 沿用现有 RaBitQ 平台限制：Linux x86_64，AVX2/FMA 或 AVX512F/BW/DQ。
- 支持 group-by、过滤、普通查询与分组查询交替，以及 collection 层 refiner。
- 与独立索引一样，不支持 group-by 与 refiner 同时使用。
- RaBitQ 自带旋转，core 的 `enable_rotate` 不增加额外旋转；collection 仍拒绝该选项。不支持 SOAR。
- core 层 RaBitQ postings 不保存原始向量，不支持 `fetch` / `fetch_vector`。
  collection 层继续通过原始 Flat 存储提供向量和重建输入。

标准 HNSW 和 IVF 均使用 Turbo `RabitqQuantizer`：训练、旋转、编码、
完整距离估计和模型序列化共享一份实现。IVF 使用 `IVFBuilder` / `IVFStreamer`
构建和扫描标准 IVF postings；HNSW 额外使用 1-bit 下界筛选图邻居。
IVF 的分组查询在扫描时保留每组 top-k，支持过滤，不需要保留全部候选。

C++ 推荐通过 `RabitqQuantizerParam` 配置模型，显式类型参数优先。
旧的 `QuantizerParam(kRabitq)` 加 IVF `total_bits` / `sample_count` 仍可用，
此时 `nlist` / `niters` 同时用作模型聚类设置。Python/C 接口继续使用该兼容映射。
IVF 列表聚类和量化器模型训练分别执行；模型中心不强制等于列表中心。

新文件保存标准 IVF header、`RabitqQuantizer` descriptor 和
`ivf.turbo_quantizer` 模型段。加载以磁盘 descriptor 为准，默认量化参数也能重开。
旧的独立 IVF-RaBitQ 文件通过 header 自动选择兼容加载器；独立 API 保留原格式。
独立 API 不支持读取新的 Turbo IVF 文件，需要通过标准 IVF API 打开。

验证入口：`ivf_rabitq_index_test`、`ivf_turbo_index_test`、
`manifest_codec_golden_test`、`schema_test`，以及 Python 的
`test_params.py` 和 `test_collection_ivf_quantized_rabitq.py`。
Linux 专属测试覆盖 L2/IP/Cosine、1/7/9 bit、重开、普通 IVF 与 RaBitQ
交替查询，以及从未量化索引合并构建。

迁移构建配置时保留所需的 `n_list`、`total_bits`、`sample_count` 和 metric。
共享量化器使用自己的模型中心与旋转，因此新旧构建的编码、分数和召回率不保证相同。
加载同一份旧文件的兼容入口应返回与独立入口相同的查询结果。

C API 复用原有 RaBitQ 参数访问器，无需新增函数或改变索引类型：

```c
zvec_index_params_t *p = zvec_index_params_create(ZVEC_INDEX_TYPE_IVF);
zvec_index_params_set_quantize_type(p, ZVEC_QUANTIZE_TYPE_RABITQ);
zvec_index_params_set_ivf_params(p, 1024, 20, false);
zvec_index_params_set_ivf_rabitq_params(p, 1024, 7, 0);
// zvec_index_params_get_ivf_rabitq_params also accepts this IVF object.
zvec_index_params_destroy(p);
```

collection manifest 中的索引类型仍分别是 `IVF` 和 `IVF_RABITQ`，
已有 collection 重开后继续使用对应的查询参数类型。
标准 IVF 的 BufferPool 分页现在同样适用于新 RaBitQ postings。
旧格式兼容加载仍使用原后端的内存布局。

新的 IVF 扫描使用共享量化器的完整距离批处理，不使用独立后端的 FastScan 布局。
运行时、召回率及性能需要在支持 RaBitQ 的 Linux x86_64 机器上验证，
macOS 编译和通用 IVF 测试不能替代 SIMD 路径验证。
