# Quantized graph benchmarks

Build the native runner with `-DBUILD_BENCHMARKS=ON` and the `bench` target.
The runner in `benchmarks/vector/bench.cc` accepts the following commands:

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
and the [succinct SIFT, GIST and GloVe presets](https://github.com/richyreachy/ann-benchmarks/tree/c7ibench/ann_benchmarks/algorithms/zvec_qg/configs/r7i).
The [native campaign archive](https://github.com/richyreachy/ann-benchmarks/tree/c7ibench/benchmark-configs/r7i-new-api/succinct)
contains the build inputs, recorded measurements and earlier configurations.
For the frozen M64/800 CPU0 GloVe run, use the [seven recommended settings](https://github.com/richyreachy/ann-benchmarks/blob/c7ibench/ann_benchmarks/algorithms/zvec_qg/configs/r7i-cpu0/glove.json)
and the [CPU0 replay command](https://github.com/richyreachy/ann-benchmarks/blob/c7ibench/ann_benchmarks/algorithms/zvec_qg/validation/cpu0-icelake-20261119/README.md#glove-query-presets).

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

## Collection and standalone QG

Use a dense FP32 field and L2 or cosine distance. On a populated collection:

```python
collection.create_index("v", zvec.QuantizedGraphIndexParam(
    metric_type=zvec.MetricType.L2, options={"encoding": {"max_neighbors": 64}}))
hits = collection.query(zvec.Query(
    field_name="v", vector=query, param=zvec.QuantizedGraphQueryParam(capacity=128)),
    topk=10, output_fields=[], include_vector=False)
```

The standalone API builds directly from FP32 rows with
`zvec.QuantizedGraph.build(rows, metric="l2", options=...)` and returns row IDs
from `graph.search(query, 10)`. Collection persists external keys and the QG
snapshot in each sealed segment. Native framework users can create
`QuantizedGraphBuilder`, then open its dump with `IndexFlow` or
`QuantizedGraphStreamer`.
