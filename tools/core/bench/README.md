# Quantized graph benchmarks

The native runner in `tools/core/bench.cc` accepts the following commands:

```sh
bench qg build G9I_PROFILE TRAIN.f32 SNAPSHOT_DIRECTORY
bench qg query G9I_PROFILE SNAPSHOT_DIRECTORY QUERIES.f32 NEW_RESULT_DIRECTORY

# Historical provider-based campaigns
bench qg export CONFIG
bench qg build CONFIG PROFILE
bench qg query CONFIG PROFILE OUT
```

`export` prepares a provider from a historical HNSW index. With an existing
provider, start at `build`. Older scripts can invoke the same program as
`qg_bench export|build|query`.

## r7i (Ice Lake)

Use the [c7ibench adapter guide](https://github.com/richyreachy/ann-benchmarks/blob/c7ibench/ann_benchmarks/algorithms/zvec_qg/README.md)
and the [exploration SIFT, GIST and GloVe presets](https://github.com/richyreachy/ann-benchmarks/tree/c7ibench/ann_benchmarks/algorithms/zvec_qg/configs/r7i-exploration).
The [native campaign archive](https://github.com/richyreachy/ann-benchmarks/tree/c7ibench/benchmark-configs/r7i-new-api/exploration)
contains the build inputs, recorded measurements and earlier configurations.

## g9i

Both Zvec worktrees use the `zvec_qg` entry in
[richyreachy/ann-benchmarks](https://github.com/richyreachy/ann-benchmarks).
Develop g9i presets on `g9ibench_test` and promote validated changes to
`g9ibench`. The [test branch guide](https://github.com/richyreachy/ann-benchmarks/tree/g9ibench_test/ann_benchmarks/algorithms/zvec_qg) describes setup and use.

- Current FP32 SIFT, GIST and GloVe presets:
  `ann_benchmarks/algorithms/zvec_qg/configs/g9i/{sift,gist,glove}.json`.
- [Historical measurements and configuration revisions](https://github.com/richyreachy/ann-benchmarks/blob/g9ibench_test/ann_benchmarks/algorithms/zvec_qg/HISTORICAL_RESULTS.md).

The g9i presets build a single-layer independent graph directly from training
rows, with degree 64 and construction list sizes 200 (SIFT/GIST) or 400
(GloVe). The ANN adapter builds in memory by default. Set the corresponding
`ZVEC_QG_*_INDEX` prefix to replay the five fingerprinted independent snapshot
parts. Install the zvec package from the source branch being tested; the adapter
imports `zvec.QuantizedGraph` and records the loaded extension fingerprint.
The native build command above writes a new directory with prefix `index`.
The public improved environment has its own release workflow.

To save resolved copies for a new run, from your `g9ibench_test` checkout:

```bash
# Use the registered exploration Python and a fresh role-specific output path.
/path/to/exploration/.venv-leaders/bin/python scripts/prepare_zvec_qg_profiles.py \
  --output /path/to/exploration/runs/new_run/profiles
```

Record the source branch/commit and runtime for each run. Keep exploration and
succinct measurements labeled separately, and keep the public improved runtime
independent. Query timing is single-threaded, serial on CPU 0, without concurrent
compilation. Build/algorithm job concurrency must not exceed two. Use official
ANN Recall@10 and report actual recall and memory alongside QPS.

`qg_bench.py` and `qg_preflight.py` remain generic native-API tools and accept
external configuration paths. Their native-campaign JSON schema differs from
the g9i independent graph profiles; do not interchange the two formats.

## Query through a Collection

On a QG-capable CPU, build an independent graph on a dense FP32 field with L2
or cosine distance. Given a populated collection and a contiguous FP32 query
vector, the public Python API accepts the same build and search settings as
the standalone graph.

```python
import zvec

collection.create_index(
    "embedding",
    zvec.QuantizedGraphIndexParam(
        metric_type=zvec.MetricType.L2,
        options={"threads": 8, "encoding": {"max_neighbors": 64}},
    ),
)
collection.optimize()
hits = collection.query(
    zvec.Query(
        field_name="embedding",
        vector=query_vector,
        param=zvec.QuantizedGraphQueryParam(capacity=128, max_expansions=64),
    ),
    topk=10,
    include_vector=False,
    output_fields=[],
)
external_ids = [doc.id for doc in hits]
```

Each sealed segment stores its graph, router and row mapping together with the
FP32 rows in the segment index file. Opening a collection restores the graph.
New writes follow the usual mutable-segment lifecycle. Standard queries
materialize documents and external IDs. A read-only collection also supports
`query_internal_ids()` for callers that need numeric internal IDs.

## QG through the core index framework

Link `core_quantized_graph_builder` together with the storage and metric modules.
Use `IndexFactory::CreateBuilder("QuantizedGraphBuilder")`. The previous factory
name `QuantizedGraphIndexBuilder` resolves to the same builder. Initialize it
with dense FP32 metadata and a build-options JSON string in the `options`
parameter, and call `build(holder)`. The holder supplies vectors and external
64-bit keys. Topology construction and codec/router training share a single
build. Dump to a framework dumper and use `IndexFlow::load(path)` to select
`QuantizedGraphSearcher` from the persisted metadata. Alternatively, initialize
`QuantizedGraphStreamer` with the same geometry and open the sealed storage.
Both accept raw FP32 query vectors, including unnormalized cosine queries.
Create one context per concurrent caller, set Top-K, and call `context->update`
with the query parameters (`capacity`, `margin`, `bucket_width`,
`max_expansions`, `route_beam`, `uncertainty_bonus`).

Collection uses the same builder. Its segment file also contains the raw vector
column, so the builder appends named QG segments to that storage. Build-time
readers use the streamer; reopened read-only segments use the searcher.
The streamer queries immutable graphs. Pending inserts use Collection's mutable
segment until the next seal or merge.

## Construct directly from FP32 vectors

Link `core_quantized_graph_builder` and pass row-major FP32 data to
[`QuantizedGraphBuilder`](../../../src/core/algorithm/quantized_graph/quantized_graph_builder.h).
The builder constructs a single-layer Vamana-style graph, trains the codec and
optional router on the database vectors, and encodes the quantized blocks.
The g9i presets select this builder for all three datasets.

```cpp
QuantizedGraphBuilder::Options options;
options.encoding.max_neighbors = 64;
options.search_list_size = 200;

QuantizedGraphBuilder::Result built;
int rc = QuantizedGraphBuilder::build(std::move(vectors), dimension, cosine,
                                     options, built);
```

On success, IDs are input row numbers. Keep `built.data` alive when calling
`built.index->search(*built.data, context)`. For cosine search, normalize the
query before resetting the context and use cosine for exact scoring. To reopen
a snapshot, persist the FP32 rows, adjacency and router alongside the QG blocks.

Construction starts at the database row closest to the centroid. Each shuffled
pass refines outgoing and reverse edges. The first pass uses alpha 1.0, and
later passes use 1.2. The degree cap is `encoding.max_neighbors`. A directed
cycle keeps each node reachable. A single construction thread makes the graph
repeatable for a fixed seed. Parallel builds can produce different graphs.
