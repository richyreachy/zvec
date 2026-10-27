# Phase80 offline scorer draft

The draft was prepared without compilation or numerical execution during
confirm78 timing. Root subsequently ended that timing window and authorized
small-prefix validation followed by the full native bitwise gate. Acceptance
is recorded only in each exclusive run's `proof.json`.

The scorer reuses `models68/primary/events.bin`, `models64/full_graph.bin`,
`models64/full_vectors.f32`, `models77/rotated.f32`, and the accepted phase77
full model. Raw queries are read from the same Cohere text file with C++ float
stream extraction. `models68/primary/queries.f32` is deliberately excluded: it
was prepared with NumPy normalization for geometry analysis.

The query route is the linked actual `Index::_prepare_dense_query`, using its
FP32 Turbo quantizer, then actual `SymphonyQGBlockQuantizer::prepare_query`.
Prepared bytes are independently compared against the production
`CosineNormalizeReformer::transform` result. Exact center and edge distances
use that same initialized `Fp32Quantizer::calc_distance_dp_query` callback,
which dispatches over 768 coordinates and does not read the stored norm tail.
The native
codec TU retains the frozen `trace68_native` compiler flags, avoiding an
unreviewed `-march=native` or contraction-policy change. Actual base-vector
rotations are compared bitwise with the accepted rotated bank before use.

For all 1000 primary queries, every center score, every edge's native FastScan
estimate, and every edge's exact reference distance must match the trace bits.
Graph center keys, final neighbor-key order, target IDs, lane order and query
boundaries are checked. Learned bytes replace only full degree-32 blocks via
the accepted import helper; short blocks retain the newly encoded native bytes.

Only after this entire gate succeeds are score artifacts emitted. The runner
publishes `proof.json` after checking the success result and computing hashes.
The replay consumer must require that proof; partial files without it are not
accepted. `native_path_learned_scores.bin` retains the original 16-byte header
and 32-byte event records, modifying only kind-2 estimate bits. It is explicitly
an **off-policy rescoring of native events**, not a learned visitation trace.
The integration agent will separately inspect fixed native-path shadow state
and stop same-prefix replay at its first next-node or stop-decision difference.
Mechanism queries are preselected as primary IDs 0 through 127 inclusive.

Commands after the timing window, subject to root's execution decision:

```bash
python3 scripts/path80_score.py --name path80_native_v1
python3 scripts/path80_score.py --name path80_preflight8_v3 --query-count 8 --execute
python3 scripts/path80_score.py --name path80_native_v1 --execute
```

The first command only prints a plan. The other commands create a new exclusive
directory with frozen local script/helper copies, `inputs.json`, each compile,
link and score command/log/result, output hashes, and a success proof. Existing
directories and output files are refused. Neither command runs a search index
or measures QPS.

Small-fixture acceptance plan before full scoring (not run):

1. Reuse the 40-node phase78 synthetic fixture with full, degree-0, 1, 7 and 31
   blocks, plus several raw float queries. Produce events through an independent
   synthetic driver using the real query reformer, actual codec and metric.
   This would require a separate synthetic fixture mode; the main scorer's
   `--query-count` selects a prefix of the actual frozen 1000-query trace.
2. Verify baseline estimates and exact distances bitwise; verify short outputs
   remain byte-identical and learned full outputs match actual FastScan.
3. Flip one native estimate bit, exact center bit, source ID, target order,
   query order, model key, raw coordinate and rotation coordinate separately.
   Each must reject without publishing a proof.
4. Mutate kind-1/kind-3 bytes and nonestimate edge bytes in a rescored fixture;
   the replay consumer must reject them. Test file truncation and existing
   output-directory collisions.
5. Check record-layout round trips with Python `struct`, including audit
   global-event index, native/learned/center bits, degree and lane sentinels.

The full primary bitwise gate remains mandatory even if synthetic checks pass.
No new-query generalization, learned search recall or QPS follows from this
offline scorer alone.

## Preserved preflight failures and diagnosis

`path80_preflight8_v1` and `path80_preflight8_v2` refused publication at event 4:
the initially chosen `IndexFactory::CreateMetric("Cosine")` exact distance was
4 ULP from the trace. Matching the original driver's full link order did not
resolve it. Neither failed attempt evaluated learned scores.

`path80_query_probe_v3/probe.jsonl` used the actual public-index initialization
and protected query preparation route, without opening an index or searching.
It found `uses_turbo=true`, prepared dimension 768 with 3076 stored bytes, and
prepared bytes identical to the standalone reformer. The actual Turbo callback
matched all first 40 trace truth scores bitwise; the separate legacy Cosine
metric did not. The scorer was corrected to use the actual Turbo callback.
This is a tooling fidelity correction, not an algorithm change or relaxed
acceptance tolerance. All earlier failure directories remain intact.

## Native fidelity acceptance

`path80_preflight8_v3/proof.json` passed 8 queries, 218 expansions and 6976
scored edges. `path80_native_v1/proof.json` then passed the full 1000-query
gate: 1,016,151 events, 30,763 expansion blocks and 984,388 scored edges, with
all native estimates and recomputed exact center/edge distances bit-identical.
Actual rotation was checked against the frozen bank for 364,384 unique vectors.
The learned second pass began only after that complete native gate. It covered
30,760 full-block visits and 3 unchanged native-short-block visits, changing
984,316 edge-score bit patterns. The consolidated evidence is
`results/path80_score_acceptance.json`. No synthetic malformed-file tests or
new retrieval/QPS runs were performed by this scorer task.
