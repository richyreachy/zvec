# PQ residual graph comparison — 2026-10-23

`pq_residual` is opt-in; existing `pq` encoding, training and scoring are unchanged.
All residual calculations are in Turbo. Core samples graph edges and manages
codebook publication/cache lifetime. L2 uses a 128-byte correction tail per
32-lane block; cosine retains the plain PQ4 block size. Both return exact FP32
scores for visited results, not approximate codec scores.

## Environment and method

- Apple M3 Pro, ARM64, 36 GiB RAM; macOS, Apple Clang, Release `-O3 -march=armv8`.
- Static-linked executables using the same CMake-built objects/libraries.
- 4,096 vectors, 256 independent queries, 64 and 128 dimensions; top-10 exact
  brute-force ground truth. Single query thread, two training/prebuild threads.
- Two synthetic datasets: independent standard Gaussian vectors; and 16 Gaussian
  clusters (cluster centers scaled by 2, unit-noise coordinates). Cosine inputs
  are normalized. Data/queries/graph use seed 20261023.
- Both codecs see the same fixed graph: 22 nearest-neighbor edges, two ring edges
  and eight random long-range edges per node. Same centroid-selected entry.
  This isolates codec effects from HNSW construction randomness; it is not a
  benchmark of an HNSW-built production graph or the complete DB/API pipeline.
- Three independently trained codebooks per configuration, preserving existing
  randomized KMC2 initialization. Codec execution order alternates by trial.
- ef sweep: 16, 32, 64, 128, 256, 512, 1024, 2048, 4096. 32 warm-up queries at
  each ef, then three timing passes with shifted query order. Recall is computed
  outside timed search; training and cache prebuild are measured separately.
- Tables report mean recall and median QPS / median per-run p95 across three
  codebooks. The 95% gate requires **all three** trials to reach recall@10 >= .95.
  This is a threshold comparison with discrete ef values, not interpolation to
  precisely equal recall. The graph beam admits duplicate discoveries, so
  ef=4096 does not imply 4096 distinct expansions or exhaustive search.

## First measured ef meeting the 95% recall gate

| Dim | Data | Metric | pq ef / recall / QPS | pq_residual ef / recall / QPS | QPS ratio | p95 µs, pq → residual |
|---:|---|---|---|---|---:|---|
| 64 | gaussian | l2 | not reached (ef4096: 90.17%) | 1024 / 97.68% / 1,846 | — | — |
| 64 | gaussian | cosine | not reached (ef4096: 94.82%) | 512 / 95.48% / 4,414 | — | — |
| 64 | clustered | l2 | 2048 / 98.10% / 1,742 | 256 / 97.68% / 13,715 | 7.87× | 652.6 → 84.5 |
| 64 | clustered | cosine | 2048 / 98.31% / 1,704 | 256 / 96.22% / 12,222 | 7.17× | 684.3 → 96.1 |
| 128 | gaussian | l2 | not reached (ef4096: 90.29%) | 1024 / 96.60% / 1,668 | — | — |
| 128 | gaussian | cosine | not reached (ef4096: 94.71%) | 1024 / 98.68% / 1,645 | — | — |
| 128 | clustered | l2 | 2048 / 97.06% / 1,813 | 256 / 97.40% / 11,882 | 6.55× | 625.5 → 97.1 |
| 128 | clustered | cosine | 2048 / 96.55% / 1,761 | 256 / 96.20% / 10,889 | 6.18× | 672.3 → 106.5 |

## Same ef=128 (includes slower residual cases)

| Dim | Data | Metric | Recall pq → residual | QPS pq → residual | p95 µs pq → residual |
|---:|---|---|---|---|---|
| 64 | gaussian | l2 | 50.35% → 65.36% | 21,411 → 20,170 | 53.8 → 58.0 |
| 64 | gaussian | cosine | 51.95% → 62.85% | 21,988 → 19,225 | 50.5 → 62.0 |
| 64 | clustered | l2 | 63.95% → 93.05% | 11,193 → 23,740 | 168.0 → 50.7 |
| 64 | clustered | cosine | 63.92% → 88.44% | 9,582 → 21,555 | 190.9 → 55.3 |
| 128 | gaussian | l2 | 49.31% → 60.31% | 18,886 → 16,282 | 59.4 → 72.5 |
| 128 | gaussian | cosine | 50.46% → 51.99% | 18,407 → 15,418 | 61.2 → 78.0 |
| 128 | clustered | l2 | 49.67% → 91.69% | 14,342 → 20,030 | 137.2 → 59.8 |
| 128 | clustered | cosine | 48.45% → 87.55% | 13,148 → 18,802 | 177.2 → 64.2 |

## Training and cache prebuild (median ms)

One observation per trained model (ef=16 rows); cache prebuild excludes training.

| Dim | Data | Metric | Train pq → residual | Prebuild pq → residual |
|---:|---|---|---|---|
| 64 | gaussian | l2 | 24.17 → 26.02 | 13.05 → 18.05 |
| 64 | gaussian | cosine | 24.81 → 26.28 | 12.62 → 13.05 |
| 64 | clustered | l2 | 18.93 → 25.27 | 12.84 → 17.69 |
| 64 | clustered | cosine | 17.90 → 25.55 | 13.53 → 13.19 |
| 128 | gaussian | l2 | 49.95 → 53.38 | 23.89 → 36.25 |
| 128 | gaussian | cosine | 49.26 → 53.62 | 23.85 → 26.20 |
| 128 | clustered | l2 | 35.88 → 50.08 | 24.41 → 35.55 |
| 128 | clustered | cosine | 35.82 → 48.35 | 24.08 → 24.33 |

## Memory accounting

These are layout sizes, **not measured process RSS**. The arena estimate includes
64-byte row headers, cached FP32 centers, IDs and aligned code payloads; it
excludes original vector storage, source adjacency, codebook, training temporaries
and query scratch. Shared codebook centroid payload is 16 * padded_dim * 4 bytes
for either codec, in addition to its pointer/norm caches.

| Dim | Metric | Codes MiB pq → residual | Graph arena MiB pq → residual |
|---:|---|---|---|
| 64 | l2 | 0.50 → 1.00 | 2.25 → 2.75 |
| 64 | cosine | 0.50 → 0.50 | 2.25 → 2.25 |
| 128 | l2 | 1.00 → 1.50 | 3.75 → 4.25 |
| 128 | cosine | 1.00 → 1.00 | 3.75 → 3.75 |

## Interpretation and limits

The residual variant improves the high-recall tradeoff on these fixed synthetic
graphs. On Gaussian data it can be slower at the same ef while achieving higher
recall. Gains include altered traversal order and duplicate discovery behavior,
not just the cost of the block distance kernel. A result on these small,
warm-memory workloads is not a production speedup claim. Real embeddings,
HNSW-built adjacency, larger corpora, filters and concurrent queries remain
unbenchmarked. Large common offsets may amplify affine-LUT/correction rounding
in L2 despite double accumulation; original FP32 result scoring is unchanged.

## Validation

- `turbo_block_quantizer_test`: 8 passed (includes both residual metrics,
  dimensions 1/17/128/4096, partial/unaligned blocks, reconstruction/LUT error
  bounds, padding, zero queries and metadata exclusion).
- `symphony_qg_test`: 26 passed (edge sampling bounded to 4096 pairs,
  deleted/invalid endpoints, read/non-finite failures, empty/isolated corpora,
  concurrent lazy publication, invalidation and exact scores).
- `hnsw_streamer_test` filtered to `PQGraphInsertSearchAndReopen` and
  `ResidualPQCosineInsertSearchAndReopen`: 2 passed, covering both codecs and
  legacy/Turbo FP32 pipelines, insertion before/after training and reopen.
- `turbo_pq_fast_quantizer_test`: 24 passed, 1 AVX512 test skipped on ARM.
- Total: **60 passed, 1 skipped**. Existing PQ tests remain included.
- DB `schema_test` was not run: its normal target pulls in Arrow, whose bundled
  Boost download stalled. That build was stopped; no dependency configuration
  or validation behavior was bypassed. Schema acceptance/conflict tests were
  updated in source. Python integration tests were extended but not run because
  Python bindings were not built; the test file's Python syntax was checked.
- `git diff --check` passed.

## Reproduction

Build target: `cmake --build <build-dir> --target pq_graph_benchmark`.
The executable is not registered as an automatic unit test. This macOS run used
static relinking of the CMake-built component libraries, as with the unit tests.
Run without concurrent builds:

```sh
pq_graph_benchmark 4096 256 64 3 3 > benchmark64.csv
pq_graph_benchmark 4096 256 128 3 3 > benchmark128.csv
```

Raw measurements: [64D CSV](pq_residual_20261023_64d.csv),
[128D CSV](pq_residual_20261023_128d.csv).
Code: [pq_graph_benchmark.cc](../pq_graph_benchmark.cc).
