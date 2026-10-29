# python4bench

实验性 benchmark 专用封装，适用于只读 collection、FP32 dense query，以及
可解析为非负 int64 的唯一主键。代码集中在此目录，生产 Python API 不变。
这不是“性能无损”的保证；请用目标数据、机器和 benchmark runner 实测。

## 构建与测试

在仓库根目录执行，先构建本分支的 Zvec，再安装这个独立 Python 包：

```bash
git submodule update --init --recursive
python -m pip install . -Ccmake.define.BUILD_PYTHON4BENCH=ON
python -m pip install ./python4bench --no-deps
python -m pytest -o addopts='' python4bench/tests
```

默认 `BUILD_PYTHON4BENCH=OFF`。开启时把 `native/searcher.cc` 编译进 `_zvec`，
共享现有 Collection 类型和引擎，避免额外链接一份引擎及跨扩展 ABI 问题。
`python4bench` 是单独安装的包；普通 Zvec wheel 不包含它。

## 使用

```python
import numpy as np
import zvec
from python4bench import Searcher

collection = zvec.open("index", zvec.CollectionOption(read_only=True))
searcher = Searcher(collection, field="vector", param=zvec.HnswQueryParam(ef=100))
q = np.zeros(768, dtype=np.float32)
ids = searcher.search(q, 100)             # owning int64 ndarray
ids = searcher.search_list(q, 100)        # list[int]，包含对象构造成本
rows = searcher.search_batch(q[None, :], 100, threads=1)
```

初始化扫描主键，建立内部 ID → 数据集 ID 映射，绝不假设插入顺序等于 ID。
密集内部 ID 使用直接数组；稀疏 ID 使用哈希表。映射属于当前只读快照；
`1` 与 `01` 这种映射重复会报错。映射准备耗时和内存需要纳入运行记录。
返回结果移除 `-1` 填充值，数量可能小于 k；顺序与普通查询保持一致。

原生入口严格要求 aligned、C-contiguous、native float32，维度必须匹配。
输入在调用返回前必须保持有效且不可被其他线程修改。
返回数组独立拥有底层缓冲区，不会被下一次调用覆盖。参数在初始化时复制，
搜索时释放 GIL，并使用每次调用独立的查询对象。显式关闭 collection 后搜索报错。
字段、过滤和参数在初始化时固定；切换配置需要新建 Searcher。
过滤、refine、多 segment 合并沿用 Collection 的语义和 fallback。
当前仍有每次 C++ 查询对象复制和引擎结果分配，并非完全消除分配。

## ANN-Benchmarks

### SymQG / native index

本分支的 SymQG 配置目前在 core interface，而不是普通 Collection 参数中。
因此额外提供 `CoreSearcher`，直接加载 native bench 的索引和 core JSON 参数：

```python
from python4bench import CoreSearcher
params = dict(index_type="kHNSW", data_type="DT_FP32", metric_type="kCosine",
              dimension=768, use_id_map=True, use_quantized_graph=True)
search = CoreSearcher("graph.index", params,
                      dict(ef_search=100, quantized_graph_max_expansions=200))
ids = search.search(q, 100)
```

参数必须匹配原索引；原索引中的 keys 必须就是数据集 ID。
可通过 `CoreSearcher.build(new_path, X, params)` 从 FP32 矩阵构建行号 ID 的图。
该入口支持 core 的路由模型和 QG 查询参数，当前限 HNSW；它不提供 DB 标量过滤
或外部 reference-index refine。SymQG 内部的精确打分仍由原算法执行。
SymQG 本身要求支持 AVX512/VNNI 的机器，ARM 上仅能测试普通 HNSW core 路径。

ANN-Benchmarks 使用 `module: python4bench.ann`、`constructor: SymQGBench`，
`base_args: ['@metric', {m: 16, ef_construction: 200}, 1]`，以及
`query_arg_groups: [{ef_search: [100, 200], quantized_graph_max_expansions: [200]}]`。
完整配置见 `integrations/symqg/config.yml`；复制到框架算法配置目录后，在对应
Docker 镜像中安装本分支 Zvec 和 python4bench 包（或使用框架本地运行模式）。
每次参数切换重新打开只读图；准备成本在查询计时之外，应另行记录。
`ANNSearchAdapter.from_searcher(search)` 和
`VectorDBSearchAdapter.from_searcher(search)` 可复用这一入口；后者只用于
VectorDBBench 调用路径的 core-only 诊断，不能标为完整数据库过滤测试。

### Collection

完整算法入口为 `python4bench.ann.module.ZvecBench`，继承框架的 `BaseANN`。
构造参数依次是 metric、index_type、index_params、threads。
支持 `euclidean/angular`、`hnsw/vamana`；`fit` 使用训练集行号作为主键。
例如配置中的关键字段：

```yaml
module: python4bench.ann
constructor: ZvecBench
base_args: ['@metric', 'hnsw', {m: 16, ef_construction: 200}, 1]
run_groups:
  search:
    args: []
    query_arg_groups: [{ef: [100, 200]}]
```

把这段放入所使用版本的算法定义结构中，并在对应 Docker 镜像中安装上面的
两个包。量化参数通过 index_params 传入，`quantize_type` 可使用枚举名称字符串；
`set_query_arguments` 接受 query 参数字典，包括 refine 配置。
不使用 prepared-query hooks；输入转换、搜索和输出封装均在 query/batch_query 内。
批量模式用 C++ 循环；threads>1 时每批创建原生工作线程，创建开销包含在计时中。
单条模式不会因 threads 参数而自动并行。线程数应服从框架的 CPU 限制。
已有自定义构建流程可直接使用 `adapters.ANNSearchAdapter`。

## VectorDBBench

`VectorDBSearchAdapter.search_embedding` 保持 list 输入 / list[int] 输出约定。
list→float32、输出 Python int 构造都在调用内计时，不能宣称为零拷贝。
已有 NumPy 输入在 dtype/布局正确时不复制。

上游客户端可通过 mixin 接入，保留 build/insert/optimize/filter 准备逻辑：

```python
from vectordb_bench.backend.clients.zvec.zvec import Zvec
from python4bench.adapters import VectorDBBenchMixin

class ZvecPython4Bench(VectorDBBenchMixin, Zvec):
    pass
```

在所用框架版本的 client 注册处选择这个类。mixin 针对具有 collection、option、
query_param、expr 属性及 init 上下文管理器的上游客户端；它在每个 worker 的
只读 init 会话中建立映射。非只读构建会话不建立 searcher。每请求 filters 参数
若非空会明确报错；固定过滤表达式由上游 prepare_filter 生成并完整传递给引擎。
框架旧版客户端如果与本分支普通 Zvec API 不兼容，需要先适配其建库 API。

## 性能诊断

SymQG/core 索引可直接对照同一个 C++ `run()` 循环（单线程）：

```bash
python python4bench/bench_core.py --path graph.index --index-params index.json \
  --queries queries.npy --k 100 \
  --query-params '{"ef_search":100,"quantized_graph_max_expansions":200}'
```

该脚本报告 native 循环、NumPy 单条、ANN adapter、list adapter 和 batch，
并计算相对 native 循环的 QPS 差距。native 基线保留同样的查询对象/结果分配，
仅不做 Python 调度和结果对象封装；它不是更底层的纯图内核基线。
正确性检查比较调用路径间的 IDs；真实 recall 仍需正式框架与 ground truth 验证。

Collection 路径：

```bash
python python4bench/bench.py --path index --queries queries.npy \
  --field vector --k 100 --threads 8 --index-type hnsw \
  --query-params '{"ef":100}' --repeats 5
```

脚本校验所有结果，轮换路径顺序、预热并报告多轮中位 QPS，包括普通 API、
薄绑定、两个 adapter 和原生 batch（包含输出封装）。它不替代正式 benchmark，
也不把 batch 数值视为纯 C++ 内核性能。单条多线程路径包含 Python executor 调度。
要测封装的绝对损耗，应另跑相同 Collection 查询路径的 C++ 对照；图内核 benchmark
与 Collection 端到端结果不可直接归因比较。固定索引、metric、TopK、搜索参数、
refine、CPU/线程预算、计时边界，并同时比较 recall 和延迟。

原生 batch 与逐条请求、改为 NumPy 输入的 VectorDBBench 与标准 list 输入，
均需分别标注结果。未在目标 x86 数据集复测前，不给出“低于 5%”等性能承诺。

## 本地验证（2026-10-29）

Apple M3 Pro / CPython 3.11 / Release 构建：新增测试 23 项通过，SymQG 硬件测试
1 项跳过；现有 internal-ID 与 refine 回归测试 76 项通过。新增测试包含真实
ANN-Benchmarks BaseANN 的 fit/query/batch/重建生命周期及上游配置解析。
测试通过不代表已经跑完两个框架的完整官方数据集或 Docker 流程。

固定随机种子 20261029，20,000 条 128 维标准正态 FP32 数据，1,000 个独立查询；
HNSW L2，M=16，ef_construction=200，ef_search=100，TopK=100，单线程。
预热后 7 轮中位 QPS（原始记录见 `results/apple-m3-pro-smoke.json`）：

| 路径 | QPS | 相对 native 循环的 QPS 损失 |
| --- | ---: | ---: |
| native 同函数循环 | 14,229 | 基线 |
| NumPy 薄封装 | 14,289 | -0.4%（测量波动） |
| ANN adapter | 13,746 | 3.4% |
| VectorDBBench list adapter | 13,548 | 4.8% |
| native batch 含 Python 输出 | 14,113 | 0.8% |

这是普通 HNSW 的桥接诊断，不能外推到 Cohere-1M、8 线程或 SymQG。
原始样本有明显系统调度波动；薄封装略高于 native 不表示 Python 加速了搜索。
目标 AVX512/VNNI 机器上可设置 `ZVEC_TEST_SYMQG=1` 运行 SymQG 专项测试，
再用真实索引执行 `bench_core.py`，并运行正式框架验证 recall 与吞吐。
