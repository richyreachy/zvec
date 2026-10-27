# Phase81 actual query / score export schema v1

Implementation: `scripts/path81_actual_score.cc` and `.py`. New implementation starts as an uncompiled draft; only a resulting exclusive `proof.json` with `passed:true` certifies a successful run. The frozen protocol is `path81_protocol_v1/config.json`, SHA `ced04e30f808f016156bacdabfd89ad27ddd238e8388a4ba798457b4eb849f29`.

## Partition and provenance gates

Training exports all6625 frozen selected states and137 candidateblocks×32lanes perstate. It prepares only the3072 training query serials using raw `models64/full_vectors.f32[(serial*2654435761+6819)%1000000]` as the actual DB driver's input to `_prepare_dense_query`. It never uses NumPy query normalization, `models68/primary/queries.f32`, or old learned models77 codes. The model77 rotated base bank is geometry input only and is compared to actual production rotation for all touched center/neighbor vectors.

Trace indexing decodes16B kind1/end prefixes across full native paths, to recover state positions and the complete expanded-ID set. All other partition score payloads are skipped. On the active partition, actual entry/center truth,32 edge truths and32 native FastScan values must match original trace bits for ALL selected states before ANY candidate scoring. Candidate0 and `native_blocks.bin` must byte-match actual production `codec.encode` for all20centers.

Validation is a separate invocation and requires `--frozen-models` and `--training-export-proof`. The training-frozen manifest must contain:

- `protocol_sha256`, `candidate_manifest_sha256`, `training_export_proof_sha256`;
- `validation_authorized_after_freeze:true`;
- `artifacts_sha256` including `selected_indices.json`, `objectives.npz`, `selected_models.bin`, `training_result.json` (relative to manifest parent).

`selected_indices.json` must contain `method_order:['second_moment','hinge']`, `indices:[2][20][32]` integer candidate indices, and the same three provenance hashes. The wrapper converts only these frozen indices to its exclusive `selected_inputs/selected_indices.u32`, copies frozen selected_models.bin, and C++ reassembles each selected block from the candidate bank to verify byte identity. Validation scans only native,M2,Hinge, never137 candidates. No validation export is allowed before this gate. Both modes perform their own full native bit gate.

## Input candidate bank

Builder interface: candidates.bin64B u32header `{0x51474381,1,64,768,20,137,32,3328,0×8}`, then `[center20,candidate137,block3328]` bytes. Headerless sidecars: centers.u32 `[20]`; neighbors.u32 `[20,32]`; valid.u8 `[20,137,32]`; native_blocks.bin `[20,3328]`; candidate_compact.u8 `[20,137,32,96]`. Additional geometry sidecars remain builder-owned. Per-lane invalid candidates contain native fallback bytes and cannot be selected. Successful candidate manifest must expose protocol/centers SHA, all artifact hashes, passed/geometry_only/native_blocks_bit_identical booleans and counts20/137.

Selected_models.bin64B u32header `{0x51474d81,1,64,768,20,2,32,3328,0×8}`, then `[model2,center20,block3328]`, model orderM2,Hinge.

## Common output file header

Every file below starts with64B little-endian uint32[16]:

`magic, version=1, header_bytes=64, record_bytes, dimension=768, lanes=32, records, models, phase, zero×7`.

phase0=training,1=validation. All floats are IEEE little-endian FP32, and score bits in u32 fields are reinterpretations, not numerical integer conversions. `records` is the number of states except queries.bin, where it is the number of prepared queries. Output order follows the frozen selected-state JSON order (queryserial ascending,step ascending).

## states.bin

Magic0x51474181, recordbytes3136, records6625training/2183validation, models137training/3validation. Each record:

- StateHeader64B: uint32[8] `(query_serial,query_internal_id,center_internal_id,step,degree,entry_exact_bits,width_bits,center_exact_bits)`; uint64 native_global_event_index (excludes16B trace header); uint32[6] `(center_row,gate_flags,entry_internal_id,phase,query_record_index,score_model_count)`.
-32 Lane records,96B each: uint32[8] `(target_internal_id,native_score_bits,exact_score_bits,native_bucket_id,eligible_unexpanded,valid_candidate_count,0,0)` followed by lowerBoundary32B and upperBoundary32B.

State gateflags7 mean native-block byte identity, native score gate and exact score gate; no direct learned path claim. `eligible_unexpanded` uses all native kind1 centers through and including the current expansion. Full32lanes remain available descriptively; primary support uses only eligible_unexpanded=1.

Boundary32B consists of uint32[8]:

`transition_bits, predecessor_bits, nextafter_bits, predecessor_bucket, at_bucket, nextafter_bucket, flags, zero`.

flags0 means no finite boundary: all fields zero. flags7 means finite transition, predecessor check passed, nextafter check passed. The successor may be+infinity in synthetic extreme cases; flags1 certifies the transition is finite, not all neighbors. Actual scored candidates/truth must always be finite.

`at_bucket` is the right-side bucket. transition is the first finite IEEEFP32 value in ordered-float order whose production `locate(score)>=at_bucket`. For boundary0→1 this is−0; both−0 and+0 classify on the right. Each present boundary is checked against direct production locate at predecessor,transition,nextafter. For a positive finite width whose target bucket has no finite crossing, the boundary is absent, not a dropped row. Nonfinite/nonpositive width fails.

Native/truth/candidate predicates `locate(score)>=at_bucket` are checked equivalent to `score>=float32(transition_bits)` for every present boundary. Machine comparisons include equality on the right. Do not substitute `float(bucket_id−1)*width` for this transition: production FP32 division can move the representable crossing. No assertion is made that ideal real-valued scoring shares actual FastScan rounding.

## queries.bin

Magic0x51475181, recordbytes6164, records3072training/1024validation, models0. Each record is uint32[4] `(query_serial,query_internal_id,entry_exact_bits,width_bits)`, followed by FP32[769] actual preparedquery storage and FP32[768] actual rotatedquery. StateHeader.query_record_index selects this record. These exported rotatedqueries and StateHeader.center_exact_bits allow the frozen ideal objective `C=actual_center_exact+R/2+nu`.

## scores.f32

Magic0x51474581, recordbytes128, records=statecount, models137training/3validation. Payload is float32 `[state,model,32]` in C row-major order. Training model index is the candidateindex0..136 and native0 is bit-compared to states.bin. Validation model order is native,M2,Hinge. Invalid training candidates contain the builder's native fallback score; the validmask must still be enforced by the trainer.

## Successful proof

Python writes `proof.json` only after the C++ native gates, full requested export, and rehash of protected protocol/preflight/candidate/source/library inputs succeed. It records source hashes, command argv/logs, candidate/protocol hashes, partition, state/query counts, native bit counts, and hashes for states.bin,queries.bin,scores.f32,scorer_result.json. No search, performance measurement or optimizer runs in the exporter. Any failure stays in its exclusive directory with failure.json and logs; it must not be consumed as training evidence.
