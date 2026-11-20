# Quantized graph benchmarks

The native runner in `tools/core/bench.cc` accepts the following commands:

```sh
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

Maintain dataset presets in [richyreachy/ann-benchmarks, branch g9ibench](https://github.com/richyreachy/ann-benchmarks/tree/g9ibench).
Both exploration and succinct use its `zvec_qg` entry and
[adapter instructions](https://github.com/richyreachy/ann-benchmarks/blob/g9ibench/ann_benchmarks/algorithms/zvec_qg/README.md).

- Current FP32 SIFT, GIST and GloVe presets:
  `ann_benchmarks/algorithms/zvec_qg/configs/g9i/{sift,gist,glove}.json`.
- Historical g9i measurements and their checker: `benchmark_configs/g9i/`.
- Retired HNSW-QG YAML recipes: `benchmark_configs/legacy-qg-20261106/`.
  These preserve old experiments; they are not inputs to the current QG API.

Set the three `ZVEC_QG_*_INDEX` paths and `ZVEC_QG_ARTIFACT` as described by the
adapter. It verifies the frozen FP32 corpus/index/runtime fingerprints and
queries the independent QuantizedGraphIndex API. It loads existing indexes;
it does not build a graph or validate FP16 performance.

To save resolved copies for a new run, from your ANN-Benchmarks checkout:

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
the g9i load-only adapter profiles; do not interchange the two formats.

## Construct directly from FP32 vectors

Link `core_quantized_graph_builder` and pass row-major FP32 data to
[`QuantizedGraphBuilder`](../../../src/core/algorithm/quantized_graph/quantized_graph_builder.h).
The builder constructs a single-layer Vamana-style graph, trains the codec and
optional router on the database vectors, and encodes the quantized blocks.
Existing g9i profiles load their original frozen indexes.

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
