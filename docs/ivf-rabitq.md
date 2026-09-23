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
auto param = zvec::core_interface::IVFIndexParamBuilder()
    .with_data_type(zvec::core_interface::DataType::DT_FP32)
    .with_dimension(128)
    .with_metric_type(zvec::core_interface::MetricType::kL2sq)
    .with_n_list(1024)
    .with_n_iters(20)
    .with_quantizer_param(zvec::core_interface::QuantizerParam(
        zvec::core_interface::QuantizerType::kRabitq))
    .with_total_bits(7)
    .with_sample_count(0)
    .build();
```

- `total_bits` 为每维量化位数，范围 1–9，默认 7。
- `sample_count` 为训练采样数量，0 使用现有 RaBitQ 的自动采样策略。
- `n_list` 和 `n_iters` 必须为正数；`nprobe` 使用现有 IVF 查询参数，RaBitQ 模式下必须大于 0；超过 `n_list` 时由后端裁剪。
- 支持 FP32、64–4095 维及 L2、IP、Cosine。
- 沿用现有 RaBitQ 平台限制：Linux x86_64，AVX2/FMA 或 AVX512F/BW/DQ。
- 支持 group-by、过滤、普通查询与分组查询交替，以及 collection 层 refiner。
- 与独立索引一样，不支持 group-by 与 refiner 同时使用。
- RaBitQ 自带旋转，不支持额外的 `enable_rotate` 或 SOAR。
- core 层 RaBitQ postings 不保存原始向量，不支持 `fetch` / `fetch_vector`。
  collection 层继续通过原始 Flat 存储提供向量和重建输入。

本版在 `IVFIndex` 内选择现有 RaBitQ builder / streamer，复用其聚类、
残差量化、FastScan 和磁盘格式，以及 IVF 的构建重试和 dump/open 生命周期。
磁盘加载根据 RaBitQ header segment 识别格式；core 层用默认量化参数重开
RaBitQ IVF 文件也能正确加载。旧的独立接口和文件格式保留，供已有调用方使用。

验证入口：`ivf_rabitq_index_test`、`ivf_turbo_index_test`、
`manifest_codec_golden_test`、`schema_test`，以及 Python 的
`test_params.py` 和 `test_collection_ivf_quantized_rabitq.py`。
Linux 专属测试覆盖 L2/IP/Cosine、1/7/9 bit、重开、普通 IVF 与 RaBitQ
交替查询，以及从未量化索引合并构建。

与独立入口迁移时，显式使用相同的 `n_list`（旧参数名 `nlist`）、
`total_bits`、`sample_count`、metric 和查询 `nprobe`；设置 `n_iters=20`
以对应独立后端默认的聚类迭代次数。普通 IVF 的既有默认值保留：Python
`n_list=10`、C/C++ `n_list=1024`，`n_iters=10`。因此只切换量化类型
不能保证与独立入口的默认训练配置相同。独立构建仍可能因随机训练产生不同结果；
同一份 postings 通过两个入口加载时应返回相同结果。

C API 复用原有 RaBitQ 参数访问器，无需新增函数或改变索引类型：

```c
zvec_index_params_t *p = zvec_index_params_create(ZVEC_INDEX_TYPE_IVF);
zvec_index_params_set_quantize_type(p, ZVEC_QUANTIZE_TYPE_RABITQ);
zvec_index_params_set_ivf_params(p, 1024, 20, false);
zvec_index_params_set_ivf_rabitq_params(p, 1024, 7, 0);
// zvec_index_params_get_ivf_rabitq_params also accepts this IVF object.
zvec_index_params_destroy(p);
```

core 磁盘文件可由两个入口读取；collection manifest 中的索引类型仍分别是
`IVF` 和 `IVF_RABITQ`，已有 collection 重开后继续使用对应的查询参数类型。
本次没有把 RaBitQ postings 重写成普通 IVF 的底层格式，也没有新增有界内存
分页：现有 RaBitQ 后端仍会加载编码数据，BufferPool 不改变这一点。

新增的 Linux 对照测试包括：同一文件通过两个入口查询时 ID/分数逐项一致，
分组查询一致，负数/零 nprobe 拒绝，以及 collection 两种入口的过滤、
group-by、向量返回、refiner 和重开。运行时与性能一致性需要在支持 RaBitQ
的 Linux x86_64 机器上执行测试及基准确认，macOS 编译检查不能替代它们。
