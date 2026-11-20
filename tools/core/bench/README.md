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
and the [succinct SIFT, GIST and GloVe presets](https://github.com/richyreachy/ann-benchmarks/tree/c7ibench/ann_benchmarks/algorithms/zvec_qg/configs/r7i).
The [native campaign archive](https://github.com/richyreachy/ann-benchmarks/tree/c7ibench/benchmark-configs/r7i-new-api/succinct)
contains the build inputs, recorded measurements and earlier configurations.
For the frozen M64/800 CPU0 GloVe run, use the [seven recommended settings](https://github.com/richyreachy/ann-benchmarks/blob/c7ibench/ann_benchmarks/algorithms/zvec_qg/configs/r7i-cpu0/glove.json)
and the [CPU0 replay command](https://github.com/richyreachy/ann-benchmarks/blob/c7ibench/ann_benchmarks/algorithms/zvec_qg/validation/cpu0-icelake-20261119/README.md#glove-query-presets).

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
