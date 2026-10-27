# Phase 82 partial-score mechanical verification

Status: passed on all 6,625 frozen phase81 training states, 3,072 training queries, 20 centers and 32 lanes. No new retrieval, validation evaluation, training, QPS measurement or production edit was performed. This is an implementation/algebra check, not an accuracy or speed improvement claim.

## Fixed inputs and scope

The only query/score data came from `path81_training_export_v1/{states.bin,queries.bin,scores.f32}`. Original native blocks and candidate-0 compact signs came from `path81_candidates_v1/{native_blocks.bin,candidate_compact.u8}`. The runner checked the training proof, linked candidate proof and all input/source/library hashes before and after execution. Saved exact-distance labels were not decoded or used.

Dimensions were fixed at 128, 256, 384, 512 and 768. Mode 0 uses the coordinate prefix. Mode 1 uses prefixes of one global PCG64 seed-8201 permutation of the 192 four-coordinate codebooks. That single permutation is shared by every state/lane; this is a mechanical subset control, not fresh independent randomness per candidate. The plan and subset were saved before compiling/running the scoring checks.

Full original LUT, delta, sum_vl, f_add and f_rescale are retained. The original full-dimensional factor offsets are always used; a short dimension is passed directly to the actual low-level `zvec::turbo::symqg_scan`, never to a full-block mapper that would move the factors.

Write `N_S` for the selected integer LUT sum, `rho=768/m`, `V` for full sum_vl, `K_S=-0.5 sum(q_S)`, `K` for the original full query correction, `A/B` for the full original lane factors, and `g` for its actual center distance. The signed-query variant keeps `A+g` and uses `rho*delta*N_S + V + rho*K_S` before multiplying by `B`. It estimates a partial query-dot expression, not the Split residual expression involving `(q-c)`; no center control variate is implemented. The separately checked uncentered HT alternative uses `delta*(rho*N_S)+V+K`. These formulas are distinct. Their real-arithmetic interpretation must be distinguished from the explicit FP32 order below.

At m=768 both variants explicitly use the original delta and K, including for reordered codebooks. Integer summation has no rounding-order ambiguity. Partial correction reset is used only to obtain K_S: its new LUT/delta/V are discarded.

## Failed v1 and actual binary rounding

The original v1 remains unchanged and failed at the first explicit scalar native epilogue check. Its actual native scan already matched the saved score. At state 0, query 0, center row 0, lane 0, the production accumulator and independently decoded compact sum both equal 26905. Native score bits were 1041253840 (0.140891314); separate multiply/add produced 1041253842 (0.140891343). The two-FMA oracle produced 1041253840 exactly.

Disassembly of the existing native executable shows two fused operations. The authoritative FP32 sequence is:

1. `t = fma(delta, float(N), V)`;
2. `t = fp32_add(t, K)`;
3. `base = fp32_add(A, g)`;
4. `score = fma(B, t, base)`.

The existing benchmark build script copied the fp32 squared_euclidean compilation recipe, with `-O3 -march=icelake-server`, without the CMake `-ffp-contract=off` flag. This is a property of the frozen baseline binary, not a new algorithm change. No library was rebuilt to change contraction. Original compile argv, source SHA, object/library SHA, disassembly instruction addresses, and the first mismatch are recorded in `native_build_provenance.json` and the diagnostic evidence.

V2 models this existing behavior explicitly with `std::fma`. For partial signed-query scans the rho*delta and rho*K_S products are rounded separately before passing them to the actual production kernel. Its independent scalar oracle spells out the same rounds separately. The full-dimension gate was not weakened.

## Successful gates

| Gate | Passed checks |
|---|---:|
| Native score bits against saved state and saved candidate-0 scores | 212,000 |
| Actual shortened production vector scan versus scalar signed-query score | 2,120,000 |
| Generic accumulator versus independent uint32 scalar sum | 2,120,000 |
| Packed lane nibble versus independent compact-code nibble | 217,088,000 |
| Two independent partial FP32 formula checks | 4,240,000 |
| Native LUT-tail interval containment | 2,120,000 |
| m=768 score bit identity and collapsed interval | 424,000 |
| Zero-delta / zero-scale algebra controls | 2 |

For each unscanned codebook, the actual 16-entry uint8 LUT supplies its minimum and maximum. Adding these integer tail endpoints to N_S brackets the full integer sum. Passing both endpoints through the exact native two-FMA epilogue and ordering their scores handles either sign of B. The raw closed interval had to contain the actual full native score before any expansion; partial intervals were then conservatively expanded by one `nextafter` at each endpoint. At m=768 the interval had to collapse bit-exactly. The interval bounds the native quantized score, not the true distance.

## Evidence and reproduction

Successful proof: `partial82_mechanics_v2/proof.json` (SHA256 `63374a9a17d750ca7ac4c5caeb02d01a9d3756efeb717276b482f0c36181901c`). Source files were formatted before this successful execution. V1 failed output and its diagnostic directory are preserved. The output contains the fixed plan, source snapshots, first-state diagnostic samples, compile/link/run argv and logs, and concise source/input hashes.

The runner accepts `--root`, `--output` and `--source-dir`, so a byte-identical archived source can be replayed against the original frozen inputs into a fresh exclusive directory without path editing. The accepted old compile-plan module and path80 dependency files remain fixed under `root/scripts` and have their hashes recorded. The completed invocation was:

```text
/home/xingrui/bench/symphony_qg/.venv/bin/python R/scripts/partial82_mechanics_v2.py --root R --output R/partial82_mechanics_v2 --source-dir R/scripts
```

Here R denotes `/home/xingrui/bench/quantized_graph/research_20261025`. Reproduction must choose a new output directory. This experiment does not show improved recall/QPS, unbiasedness of a fixed prefix, coordinate exchangeability of the structured rotation, or applicability of the original RaBitQ error bound.
