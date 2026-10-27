# Cohere 1M: end-to-end test of partial native-code scanning

2026-10-27. **The tested fixed-prefix implementation did not improve performance at Recall@10≥90%.** The fastest eligible partial configuration was 512 dimensions with a 128-expansion cap. Across three paired rounds it was 73.55% slower at one thread and 71.35% slower at eight threads than the frozen baseline. The experimental switch remains disabled by default.

This is the next step after the [partial-score arithmetic study](partial82_rabitq_feasibility.md). It tests fixed-prefix query-dot extrapolation through the actual zvec search path. Center-residual correction and interval-driven continuation remain untested. The result rejects the configurations tested here; it is not a proof that every partial-scoring method must lose.

## Implementation and frozen protocol

The research worktree adds `QG_PARTIAL_SCAN_DIM` with values 128, 256, 384, 512 and 768. Default-off and 768 retain the existing query layout and full scan. Shorter variants keep full native codes, factors, rotation, query LUT, exact center-distance calculation and prefetch policy. They expand the corrected prefix query dot product by 768/m. A lightweight prefix reduction computes the query correction without building another LUT; query state grows from 160 to 168 bytes.

All variants use the existing Cohere 1M index, dimension 768, cosine distance, Top10, degree32, bucket width .016, actual ef_search800 and the original graph structure and refinement logic. This is an in-memory search benchmark. Index opening still takes about 37 seconds and reconstructs the quantized graph; opening time is excluded from QPS. Full codes remain stored, and all code bytes are still prefetched. This experiment does not reduce index memory proportionally to the scan length.

The protocol was sealed before new retrieval effects were read:

`partial83_protocol_v1/config.json`, SHA256 `319876ae1ae6a251a84c11c40f0d53347a0ef7af18e6c605671dc73982e507da`.

For each partial length, the first quality sweep used cap64 and margins 0/.01/.02/.04/.08/.16/.32, plus a cap-only point. If no point reached .90 recall, the protocol allowed cap-only128 and256. Patience999 suppresses margin stopping within those caps; the cap remains active. The encoded label is not ef: label20000 means margin .020/patience0; labels1999/2999 mean patience999 with caps128/256. All searches still use ef800.

The original 1,000 primary queries have been used repeatedly for tuning and are not a blind test. All 40 partial quality points were completed. Four reached the target and received a two-second single-thread screen. The highest-QPS point was then frozen for both thread counts and three confirmation rounds. No parameters were changed after selection.

## Correctness before timing

Six builds, including default-off and the full768 control, passed actual codec-interface checks on 3,072 existing training queries and 6,625 states. Each build passed 212,000 scan comparisons against the verified low-level reference. Rotated queries, full LUTs and constants matched; the four partial builds' added prefix fields matched independent reference preparation. Default-off and full768 produced identical score bytes and query object sizes.

The full768 end-to-end quality run then reproduced all 10,000 frozen Top10 ID/hex-score rows byte for byte. Every timed experiment had an untimed complete quality pass. Confirmation results matched the previously frozen IDs and scores for their configurations. Timed calls themselves checked status and result count; they did not re-check every ID inside the timer.

Probe: `partial83_codec_probe_v1/proof.json`, SHA256 `b50cb93fcd4783ff92e6507a0ee267fd5c793626e053a0ffdc899bbbd41cda2f`.

## Recall screening

The table reports the maximum observed recall at cap64 and the two prespecified cap-only rescue points.

| Scanned dimensions | Best Recall@10, cap64 | Recall@10, cap128 | Recall@10, cap256 |
|---:|---:|---:|---:|
| 128 | 25.18% | 49.90% | 70.96% |
| 256 | 57.43% | 78.94% | 90.01% |
| 384 | 74.62% | 88.96% | 94.97% |
| 512 | 85.51% | 94.06% | 97.44% |

The frozen native baseline reaches 90.32% at cap64 with margin .020. No tested partial prefix reaches 90% within cap64, even at the cap-only point.

The four eligible points gave these exploratory single-thread measurements:

| Prefix | Cap | Recall@10 | Screen QPS, one 2-second run |
|---:|---:|---:|---:|
| 256 | 256 | 90.01% | 4,535.94 |
| 384 | 256 | 94.97% | 4,391.74 |
| 512 | 128 | 94.06% | **8,280.05** |
| 512 | 256 | 97.44% | 4,259.32 |

The screen selected prefix512/cap128/margin.001/patience999. The nonzero margin is inactive before the cap because of patience999.

## Three-round paired confirmation

Each measurement ran for at least three seconds after query and per-worker warmup. Every process was restricted to CPUs `0,2,4,6,8,10,12,14`; workers were not individually pinned. Conditions used balanced orders: baseline/partial/full768, full768/baseline/partial, partial/full768/baseline. No compilation or heavy analysis ran during timing. All timed runs recorded zero major page faults.

| Condition | Recall@10 | Median QPS, 1 thread | Median QPS, 8 threads |
|---|---:|---:|---:|
| Frozen `bucket90_w16` | 90.32% | **31,791.07** | **236,188.95** |
| Newly compiled full768 control | 90.32% | 30,962.10 | 232,624.53 |
| Selected partial512, cap128 | 94.06% | 8,409.82 | 67,670.85 |

Relative to the frozen baseline, the ratio of QPS medians changes by **−73.55% / −71.35%** at 1/8 threads. Relative to the newly compiled full768 control, the changes are **−72.84% / −70.91%**. The partial candidate lost in every paired comparison with both controls.

| Condition | 1-thread QPS across rounds | 8-thread QPS across rounds |
|---|---|---|
| Frozen baseline | 31,791.07; 31,742.11; 31,866.86 | 237,028.76; 234,724.93; 236,188.95 |
| Full768 control | 30,962.10; 30,884.55; 31,007.29 | 231,497.93; 232,864.58; 232,624.53 |
| Partial512 | 8,469.99; 8,383.77; 8,409.82 | 67,670.85; 67,403.60; 67,697.09 |

These compare a frozen grid of configurations satisfying **Recall≥90%**, not identical recall. In particular, the selected partial point has higher recall than either control. A finer cap sweep near 90% would be a separate experiment; it was not inserted into this protocol. The 256-dimensional screen point at 90.01% is also much slower, but it was not the selected three-round candidate. Three rounds are a confirmation pilot, not a statistical-significance claim.

## Interpretation and decision

The reduced scan arithmetic did not translate into a better end-to-end trade-off in this grid. Partial scoring degraded navigation quality enough to require substantially larger search budgets. Exact center work, graph traversal, queue work, full-code prefetch and full query preparation remain part of the total cost. This experiment does not separately profile their contributions or prove that one source of numerical error caused all of the recall loss.

The implementation samples the query dot product while retaining the full center correction. The earlier mathematical study identified a center-residual control variate as a possible alternative; its variance and implementation costs would need separate measurement. Likewise, adaptive continuation would need to demonstrate that its intervals resolve decisions early enough to recover the overhead. The present negative result supplies a comparison point for those ideas, not evidence that they work.

Keep this experimental feature disabled by default. The reserve-query gate failed, so the 1,000 self-excluded additional queries were not run or used for retuning. The 25% breakthrough criterion remains unmet.

## Audit, provenance and artifacts

The independent raw-file audit reconstructed 54 quality passes, 54,000 query evaluations, 540,000 ID/hex rows and **22 new timing rows**: four screening rows plus 18 confirmation rows. It independently verified requested thread sets, positive finite timing fields, QPS arithmetic, recall hit totals, caps/margins, candidate selection and paired summaries. It matched all root results. The audit was not statistically blind and made no new performance selection.

All six frozen source manifests match the research worktree's 1,117 source files. Recorded probe dependencies and runner/driver/helper hashes remain unchanged. Variant shared-library hashes were collected after timing, with modification times predating the quality phase; no pre/post library-digest equality is claimed for hashes that were not captured before timing. See `results/partial83_final_provenance.json` for this distinction.

- `partial83_quality_v1/result.json`: full768 identity and all 40 partial quality points.
- `partial83_screen_v1/result.json`: four eligible measurements and the frozen selection.
- `partial83_confirm_v1/result.json`: three paired rounds, medians and failed reserve gate.
- `partial83_screen_audit_v1/report.json` and `partial83_final_audit_v1/report.json`: independent audits.
- `experiments/partial83_*`: actual configs, logs, per-query hits, ID/hex rows and timing CSVs.
- `variants/partial83_*`: frozen sources, compiler commands and binaries.
- `scripts/partial83_end_to_end.py`, `partial83_codec_probe.*`, `audit83_end_to_end.py`: reproducible drivers and checks.

Artifacts remain under `/home/xingrui/bench/quantized_graph/research_20261025`. The original `zvec` working tree is untouched. Total accumulated retrieval timing rows: **1,362**.

For individual-run reproduction, use the saved command/environment in its `config.json` with a fresh output prefix. The phase driver deliberately reserves its original v1 names and refuses overwriting frozen outputs.
