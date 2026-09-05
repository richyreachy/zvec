# collection_test SEGFAULT — Root Cause & Fix Report

**Date:** 2026-09-04
**Issue:** `62 - collection_test (SEGFAULT)` on Windows (MSVC 2022 / Ninja)
**Failing test:** `CollectionTest.Feature_Optimize_Concurrent_ReadWrite_NonBlocking` (`tests/db/collection_test.cc:2935`)
**Status:** Fixed — 45/45 clean runs, full suite green.

---

## 1. Summary

The crash was a **data race on shared index state between the segment flush path and
concurrent reader/indexer-open paths**, not heap corruption by a buffer overflow. Two
unsynchronized structures were involved:

1. `FlatStreamerEntity::segments_` — a `std::vector` lazily filled by `get_segment()`.
   A concurrent append reallocates the buffer while another thread copies a
   `Segment::Pointer` (shared_ptr) out of it, yielding a stale/garbage pointer.
2. `SegmentImpl`'s memory→persist component swap in `flush()` /
   `init_memory_components()` mutated the indexer maps (`vector_indexers_`,
   `memory_vector_indexers_`) and persist block offsets while readers
   (`Fetch`, `fetch`, `get_combined_vector_indexer`) traversed them unlocked.

The fix adds a dedicated mutex + address-stable `std::deque` for `segments_`, and
takes `seg_col_mtx_` (shared for readers, unique for the flush/init swap) around all
affected paths.

## 2. Symptom & Reproduction

- Running the filtered test directly crashed **5 of 20 runs (25%)** on pristine code.
  Under `ctest` it manifests as the reported intermittent SEGFAULT.
- Crash happens **~37 ms into the test**, during the *initial single-threaded
  20,000-doc load* — before the concurrent optimizer/writer/reader phase begins.
- The test config (`CollectionOptions{false, true, 64 * 1024 * 1024}`) sets a 64 MB
  in-memory buffer limit. `ready_for_dump_block()` (segment.cc:2901) fires once the
  forward store crosses it, so the load loop repeatedly triggers the
  `flush() → finish_memory_components() → init_memory_components() →
  create_vector_indexer() → VectorColumnIndexer::Open → core_interface::Index::open
  → FlatStreamer<32>::open` cycle while other threads (buffer-pool eviction, further
  inserts) touch the same state.

## 3. Evidence Chain

### 3.1 Symbolized minidump

Captured via a custom debug-loop launcher (`tools/dump_launcher.cc`) writing a
full-memory minidump on first-chance access violation, then symbolized offline with
`tools/dump_analyze.cc`:

```
EXCEPTION code=0xc0000005  AV type=WRITE target=0x1
faulting instruction: memcpy+0xe9  (mov [rax], r8; RAX=1)
R8 = 0x6f725072656e6e49  ("InnerPro" — the "InnerProduct" metric name)

stack:
  std::string::assign
  zvec::core::IndexMeta::operator=
  zvec::core::FlatStreamer<32>::open+0x327        (flat_streamer.cc:150, *entity_->mutable_meta() = meta_)
  zvec::core_interface::Index::open
  zvec::VectorColumnIndexer::CreateProximaIndex / Open
  zvec::SegmentImpl::create_vector_indexer        (init_memory_components)
  ... CollectionTest TestBody → CreateCollectionWithDoc (initial load)
```

A freshly `make_unique`'d entity whose meta assignment writes through a wild pointer
(RAX = 1) means the object/pointer pair was invalidated between creation and use —
the signature of a race, not of an overflowing copy.

### 3.2 Ruling out classic heap corruption

- **Full PageHeap** (IFEO `PageHeapFlags=3`, catches OOB/UAF writes at the exact
  write site): **5/5 clean runs**. No allocation-boundary violation exists.
- **Attached debugger**: 0 crashes in debugged runs — the race window is timing
  sensitive and perturbation masks it. This also explains why the crash "disappears"
  when investigated interactively.

### 3.3 Root cause confirmed by an ODR accident

After applying the fixes, an incremental rebuild crashed **10/10** with the *same*
signature. Investigation showed **ninja's header dependency tracking is broken on
this machine** (Chinese-locale MSVC `/showIncludes` output is not parsed — touching
`flat_streamer_entity.h` dirties zero targets). The binary mixed old-layout
`flat_streamer.cc.obj` with new-layout `flat_streamer_entity.cc.obj`: an ODR/layout
mismatch that produces exactly this wild-write signature. A clean rebuild of the
identical sources passes 45/45.

This accident demonstrates the crash mechanism precisely: whenever the entity's
in-memory state (pointer/layout) does not match what the reading thread expects, the
meta copy in `FlatStreamer::open` writes to garbage. In the pristine build that
mismatch is produced by the unsynchronized `segments_` reallocation and the unlocked
flush/init swap, rather than by stale objects.

## 4. The Fix

### 4.1 `src/core/algorithm/flat/flat_streamer_entity.{h,cc}`

- `segments_` changed from `std::vector` to `std::deque` — element addresses stay
  stable while a writer appends segments, so `shared_ptr` copies taken by concurrent
  searches remain valid.
- New `mutable std::mutex segments_mutex_` serializing all `segments_` access:
  `get_segment()` (lazy fill), `alloc_segment()`, `alloc_block()` (restructured so it
  does not hold the mutex while calling `alloc_segment()`, which takes it itself),
  `add_to_block()`, and `clone()`.
- `update_head_block()` / `get_head_block()` now obtain segment 0 through the locked
  `get_segment(0)` instead of raw `segments_[0]`.

### 4.2 `src/db/index/segment/segment.cc`

- `flush()`: `unique_lock(seg_col_mtx_)` around `finish_memory_components()` — the
  memory→persist swap now runs exclusively; readers observe either the old memory
  block or the fully migrated persisted block, never a half-migrated mix.
- `init_memory_components()`: `unique_lock(seg_col_mtx_)` — the rebuild of memory
  components and indexer maps is exclusive with readers.
- `get_combined_vector_indexer()` / `get_quant_combined_vector_indexer()`: `shared_lock`
  — the indexer maps are traversed under the column lock.
- `Fetch()` (vector-field loop) and single-row `fetch()`: `shared_lock` around the
  block-offset/indexer-map traversal; the multi-block fallback path was moved *after*
  lock release because it recursively acquires the lock — a nested shared acquisition
  could deadlock against a flush waiting for the unique lock.

No public API, on-disk format, or single-threaded behavior changes.

## 5. Verification

| Build | Runs | Result |
|---|---|---|
| Pristine (before fix), filtered test | 20 | **5 crashes (25%)** |
| Fixed, clean rebuild, filtered test | 45 | **45 PASS, 0 crashes** |
| Fixed, full `collection_test` suite | 91 tests | **89 passed, 2 skipped¹, 0 failed** |
| Fixed, `flat_streamer_test` | 19 tests | **18 passed, 1 skipped¹, 0 failed** |

¹ Skips are pre-existing platform skips (`Feature_DropIndex_Scalar_FailureKeepsPersistedOldSchema`,
`Feature_DropFtsIndex_FailureKeepsPersistedOldSchema`, `FlatStreamerTest.TestMaxIndexSize`),
unrelated to this change.

All runs executed sequentially with `test_collection/` removed between runs (the test
requires a non-existing path). Build: `build.rel` (Release + PDBs, MSVC 2022 BuildTools,
Ninja), full clean rebuild after the fixes.

## 6. Files Changed

```
src/core/algorithm/flat/flat_streamer_entity.h   | 17 +-
src/core/algorithm/flat/flat_streamer_entity.cc  | 46 +--
src/db/index/segment/segment.cc                  | 262 ++++++------
```

Diagnostic tools written during the investigation (untracked, kept for future use):
- `tools/dump_launcher.cc` — runs a child under `DEBUG_ONLY_THIS_PROCESS`, writes a
  full minidump on first-chance AV (gtest's SEH swallows first-chance AVs in the test
  body, so waiting for second-chance never fires).
- `tools/dump_analyze.cc` — offline minidump symbolizer (modules, exception record,
  registers, faulting instruction, RSP stack scan) using dbghelp.

## 7. Environment Notes

1. **ninja header dependency tracking is broken in this workspace** (Chinese-locale
   MSVC: `/showIncludes` output not parsed; touching a header dirties no targets).
   After any header change, do a clean rebuild of the affected target — incremental
   builds can silently produce mixed-layout binaries that crash deterministically.
   Worth fixing properly (e.g., `/deps` .d files or English locale for the compiler).
2. Registry keys used transiently during diagnosis (PageHeap IFEO key, WER
   LocalDumps for `collection_test.exe`) have been removed.
3. The `tests/db/CMakeLists.txt` PDB flags used to symbolize test binaries during
   diagnosis were reverted — they are not part of the fix.
