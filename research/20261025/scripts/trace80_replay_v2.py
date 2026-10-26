"""Frozen phase80 native and common-prefix replay v2: replay existing native trace and score overlays.

No retrieval, training, GT, source mutation, or parameter search. The learned
stream is an off-policy score overlay on native events, NOT a learned trace.
Run only after the root agent releases the current timing exclusion window.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import mmap
import struct
from pathlib import Path

import numpy as np

HEADER = struct.Struct("<4I")
EVENT = struct.Struct("<5I2fI")
UINT_MAX = 2**32 - 1
REQUIRED = {
    "QG_BUCKET=1",
    "QG_BUCKET_WIDTH=.016f",
    "QG_ADAPTIVE=1",
    "QG_RUNTIME_ADAPTIVE=1",
    "QG_STOP_MIN=10",
    "QG_STOP_PATIENCE=0",
    "QG_STOP_MARGIN=.04f",
    "QG_ROTATION_768=1",
    "QG_AUGMENT=3",
}
OPTIONAL = {
    "QG_MULTI_TRACE=1",
    "QG_PATH_TRACE=1",
    "QG_DIAG=1",
    "QG_INDEPENDENT_CODE_IMPORT=1",
}


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def f32(value):
    return np.float32(value)


def finite_json(value):
    return float(value) if math.isfinite(float(value)) else None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def queries(path):
    """Yield native bytes in complete per-query groups; preserve global index."""
    with path.open("rb") as stream:
        need(stream.read(16) == HEADER.pack(0x51474D68, 1, 32, 768), "trace header")
        serial, index, rows = 0, 0, []
        while raw := stream.read(32):
            need(len(raw) == 32, "truncated trace")
            row = EVENT.unpack(raw)
            need(
                row[0] == serial and row[7] == 0 and row[1] in (1, 2, 3),
                "trace order/reserved/kind",
            )
            need(
                math.isfinite(row[5]) and math.isfinite(row[6]),
                "nonfinite trace scores",
            )
            rows.append((index, raw, row))
            index += 1
            if row[1] == 3:
                need(row[3] == row[4] == UINT_MAX, "end sentinels")
                yield serial, rows
                serial += 1
                rows = []
        need(not rows, "missing end event")


class Graph:
    def __init__(self, path):
        self.stream = path.open("rb")
        self.data = mmap.mmap(self.stream.fileno(), 0, access=mmap.ACCESS_READ)
        header = struct.unpack_from("<8I", self.data)
        need(header[0] == 0x51474764 and header[3] == 768, "graph header")
        self.count = header[2]
        need(len(self.data) == 32 + self.count * 140, "graph size")

    def row(self, node):
        need(0 <= node < self.count, "graph node range")
        values = struct.unpack_from("<QI32I", self.data, 32 + node * 140)
        need(values[1] <= 32, "graph degree")
        return values[0], values[2 : 2 + values[1]]


class Queue:
    """Only the frozen ordinary bucket+adaptive path; every duplicate retained."""

    def __init__(self, entry, margin, cap, patience):
        self.width = f32(1)
        self.margin = f32(margin)
        self.cap, self.patience = cap, patience
        self.bins, self.active, self.visited, self.best = {}, [], set(), []
        self.expanded = self.stale = self.arrival = self.stale_pops = 0
        self.proposals = self.first = 0
        seed = {
            "target": entry,
            "source": None,
            "step": 0,
            "lane": None,
            "arrival_serial": 0,
            "score_bits": 0x7F7FFFFF,
            "bin": 2047,
        }
        self.insert(seed)

    def locate(self, distance):
        value = f32(distance)
        if value < 0:
            return 0
        quotient = f32(value / self.width)
        return 2047 if quotient >= f32(2046) else int(quotient) + 1

    def lower(self, index):
        return f32(-np.inf) if index == 0 else f32(f32(index - 1) * self.width)

    def boundary_distance(self, distance):
        # Descriptive nominal FP32 boundaries, NOT the exact perturbation needed
        # to cross locate(): rounded division also affects the boundary.
        bucket = self.locate(distance)
        if bucket == 0:
            return abs(float(distance))
        low = float(self.lower(bucket))
        if bucket == 2047:
            return abs(float(distance) - low)
        high = float(f32(f32(bucket) * self.width))
        return min(abs(float(distance) - low), abs(float(distance) - high))

    def insert(self, token):
        bucket = token["bin"]
        if bucket not in self.bins:
            self.bins[bucket] = []
            heapq.heappush(self.active, bucket)
        self.bins[bucket].append(token)

    def pop_raw(self):
        bucket = self.active[0]
        token = self.bins[bucket].pop()
        if not self.bins[bucket]:
            del self.bins[bucket]
            heapq.heappop(self.active)
        return token

    def decision(self):
        while self.active and self.bins[self.active[0]][-1]["target"] in self.visited:
            self.pop_raw()
            self.stale_pops += 1
        if not self.active:
            return {"action": "stop", "reason": "empty", "next": None}
        token = self.bins[self.active[0]][-1]
        lower = self.lower(token["bin"])
        tau = self.best[-1] if len(self.best) == 10 else None
        left = f32(lower - f32(tau)) if tau is not None else None
        right = f32(self.margin * max(f32(tau), f32(1e-6))) if tau is not None else None
        eligible = len(self.best) == 10 and self.expanded >= 10
        cap_hit = eligible and self.expanded >= self.cap
        margin_hit = eligible and self.stale >= self.patience and left >= right
        stop = cap_hit or margin_hit
        return {
            "action": "stop" if stop else "expand",
            "reason": "cap" if cap_hit else "margin" if margin_hit else None,
            "next": token,
            "bucket_lower": finite_json(lower),
            "bucket_lower_bits": bits(lower),
            "tau": None if tau is None else float(tau),
            "stop_left": None if left is None else finite_json(left),
            "stop_left_bits": None if left is None else bits(left),
            "stop_right": None if right is None else float(right),
            "stop_right_bits": None if right is None else bits(right),
            "expanded": self.expanded,
            "stale": self.stale,
            "raw_queue_entries": sum(len(v) for v in self.bins.values()),
            "live_unvisited_entries": sum(
                t["target"] not in self.visited for v in self.bins.values() for t in v
            ),
            "distinct_target_count": len(
                {t["target"] for v in self.bins.values() for t in v}
            ),
            "winner_target_proposals": sum(
                t["target"] == token["target"] for v in self.bins.values() for t in v
            ),
        }

    def expand(self, node, exact):
        token = self.pop_raw()
        need(
            token["target"] == node and node not in self.visited,
            "replay expansion mismatch",
        )
        self.visited.add(node)
        self.expanded += 1
        if self.expanded == 1:
            self.width = f32(max(f32(exact), f32(1e-6)) * f32(0.016))
        if len(self.best) < 10 or exact < self.best[-1]:
            self.best.append(exact)
            self.best.sort()
            self.best = self.best[:10]
            self.stale = 0
        else:
            self.stale += 1
        return token

    def observe(self, source, target, step, lane, score):
        self.arrival += 1
        if target in self.visited:
            return None
        token = {
            "source": source,
            "target": target,
            "step": step,
            "lane": lane,
            "arrival_serial": self.arrival,
            "score_bits": bits(score),
            "bin": self.locate(score),
        }
        self.insert(token)
        self.proposals += 1
        return token


def main():  # noqa: PLR0915 - preserve the frozen validation protocol
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-events", type=Path, required=True)
    parser.add_argument("--native-metadata", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--rescored-events", type=Path)
    parser.add_argument("--rescore-proof", type=Path)
    parser.add_argument("--allow-control-proof", action="store_true")
    parser.add_argument("--native-path", type=Path)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--native-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mechanism-queries", type=int, default=1000)
    parser.add_argument("--expected-queries", type=int, default=1000)
    args = parser.parse_args()
    need(
        args.mechanism_queries == 1000 and args.expected_queries == 1000,
        "Draft protocol freezes all1000 native gate and mechanism all primary q0..999",
    )
    need(
        sha(args.native_events)
        == "4787aa0f822bb2ddf35f4af28c04579283fac5d557fbea74c9f1802047d3d04c",
        "Frozen phase68 native trace hash",
    )
    need(
        sha(args.graph)
        == "c04308b5445a666026bd460100ef0a649fd3dfd7e75a1b57561ab40a1e7dc53b",
        "Frozen full graph hash",
    )
    runtime = json.loads(args.runtime_config.read_text())
    expected_runtime = {
        "QG_CAP": "64",
        "QG_PATIENCE": "0",
        "BENCH_STOP_GRID": "1",
        "BENCH_MARGIN_DENOM": "1000",
        "BENCH_EFS": "20000",
        "BENCH_ROUNDS": "1",
        "BENCH_THREADS": "1",
        "BENCH_QUERY_BEGIN": "0",
        "BENCH_QUERY_END": "1000",
        "BENCH_TOPK": "10",
        "BENCH_DEGREE": "32",
    }
    need(
        all(runtime["env"].get(k) == v for k, v in expected_runtime.items()),
        "Frozen runtime does not match replay protocol",
    )
    need(runtime["command"][-2] == "-1", "Expected one non-timing query pass")
    expected_results = {}
    with args.native_results.open() as stream:
        for label, query, key, score in csv.reader(stream):
            need(label == "20000", "Expected fixed margin result rows")
            expected_results.setdefault(int(query), []).append(
                (int(key), float.fromhex(score))
            )
    need(
        set(expected_results) == set(range(1000))
        and all(len(v) == 10 for v in expected_results.values()),
        "Native result rows",
    )
    defines = set(json.loads(args.native_metadata.read_text())["defines"])
    need(
        REQUIRED.issubset(defines) and defines.issubset(REQUIRED | OPTIONAL),
        "Unsupported queue/search macros",
    )
    need("QG_MULTI_TRACE=1" in defines, "Expected real native MULTI trace")
    need(
        bool(args.rescored_events) == bool(args.rescore_proof),
        "Overlay requires scorer proof",
    )
    if args.rescore_proof:
        proof = json.loads(args.rescore_proof.read_text())
        control = proof.get("proof_kind") in {
            "identity_control",
            "synthetic_last_lane_control",
        }
        need(
            not control or args.allow_control_proof,
            "Control proof requires explicit flag",
        )
        need(
            not args.allow_control_proof or control,
            "Control flag is only for explicit controls",
        )
        if control:
            need(
                proof.get("control_native_scores_from_original_trace") is True
                and proof.get("no_codec_rescore_claim") is True,
                "Control provenance",
            )
            need(
                proof.get("model_sha256") is None,
                "Control must not impersonate learned model",
            )
        else:
            for key in [
                "native_scores_bit_identical",
                "native_truth_bit_identical",
                "all_queries_native_gate",
            ]:
                need(proof.get(key) is True, "Missing scorer gate: " + key)
            expected_model = (
                "2ac03543d4a09ae655d38130419df41668e4d62bdd3231b49135d2603c39f138"
            )
            need(
                proof.get("model_sha256") == expected_model,
                "Frozen learned model hash in proof",
            )
            need(
                sha(Path(proof["model_path"])) == expected_model,
                "Frozen learned model current bytes",
            )
        need(
            proof.get("off_policy_rescored_native_events") is True,
            "Expected off-policy overlay",
        )
        need(
            proof.get("actual_learned_path") is False,
            "Overlay must not claim actual learned trace",
        )
        need(
            proof["source_events_sha256"] == sha(args.native_events),
            "Native trace hash",
        )
        need(
            proof["rescored_events_sha256"] == sha(args.rescored_events),
            "Score overlay hash",
        )
        need(proof["query_count"] == 1000, "Score overlay query count")
    args.output.mkdir(parents=True, exist_ok=False)
    graph = Graph(args.graph)
    overlays = iter(queries(args.rescored_events)) if args.rescored_events else None
    path_rows = None
    if args.native_path:
        path_rows = iter(csv.reader(args.native_path.open()))
    counters = {
        "native_queries": 0,
        "native_expansions_order_verified": 0,
        "native_score_records": 0,
        "path_bits_verified": 0,
        "top10_members_and_score_bits_verified_queries": 0,
        "prefix_diverged_queries": 0,
        "same_target_different_provenance_queries": 0,
        "queries_with_eligible_bin_change": 0,
        "same_bins_implies_same_actions_checked_queries": 0,
    }
    with (
        (args.output / "queries.jsonl").open("x") as out,
        (args.output / "proposals.csv").open("x", newline="") as propout,
    ):
        columns = [
            "query",
            "event_index",
            "source",
            "step",
            "lane",
            "arrival_serial",
            "target",
            "native_unexpanded",
            "shared_prefix",
            "native_score_bits",
            "learned_score_bits",
            "exact_bits",
            "native_bin",
            "learned_bin",
            "exact_bin",
            "native_nominal_boundary_distance",
            "learned_nominal_boundary_distance",
        ]
        proposals = csv.DictWriter(propout, fieldnames=columns)
        proposals.writeheader()
        for query, rows in queries(args.native_events):
            need(query < 1000, "Unexpected additional native queries")
            learned_rows = None
            if overlays:
                overlay_query, learned_rows = next(overlays)
                need(
                    overlay_query == query and len(learned_rows) == len(rows),
                    "Overlay alignment",
                )
                for (index, raw, row), (other_index, other_raw, _other_row) in zip(
                    rows, learned_rows, strict=True
                ):
                    need(
                        index == other_index
                        and raw[:20] == other_raw[:20]
                        and raw[24:] == other_raw[24:],
                        "Overlay changed non-score fields",
                    )
                    if row[1] != 2:
                        need(raw == other_raw, "Overlay changed begin/end event")
            need(rows[0][2][1] == 1, "Missing first expansion")
            entry = rows[0][2][3]
            native = Queue(entry, 0.020, 64, 0)
            shadow = (
                Queue(entry, 0.020, 64, 0) if learned_rows and query < 1000 else None
            )
            first_encounter, first_preexpansion, winners = {}, {}, []
            expanded_key_scores = {}
            same_target_different_provenance = []
            first_bin_change = divergence = None
            cursor, step = 0, 0
            while cursor < len(rows) - 1:
                _, _, row = rows[cursor]
                _, kind, current_step, source, target, estimate, exact, _ = row
                step += 1
                need(
                    kind == 1
                    and current_step == step
                    and source == target
                    and bits(estimate) == bits(exact),
                    "Begin block",
                )
                decision = native.decision()
                need(
                    decision["action"] == "expand"
                    and decision["next"]["target"] == source,
                    "Native queue replay diverged; no learned interpretation permitted",
                )
                if shadow is not None:
                    other = shadow.decision()
                    if other["action"] != "expand" or other["next"]["target"] != source:
                        divergence = {
                            "before_expansion_step": step,
                            "native": decision,
                            "learned": other,
                            "kind": "stop" if other["action"] == "stop" else "next_id",
                        }
                        need(
                            first_bin_change is not None,
                            "Different actions without earlier eligible bin change",
                        )
                        shadow = None  # Stop learned path interpretation at first divergence.
                    else:
                        if any(
                            other["next"][field] != decision["next"][field]
                            for field in [
                                "source",
                                "step",
                                "lane",
                                "arrival_serial",
                                "target",
                            ]
                        ):
                            same_target_different_provenance.append(
                                {
                                    "step": step,
                                    "native": decision["next"],
                                    "learned": other["next"],
                                }
                            )
                        shadow.expand(source, exact)
                winner = native.expand(source, exact)
                winners.append(winner)
                key, neighbors = graph.row(source)
                expanded_key_scores[key] = exact
                if path_rows is not None:
                    p = next(path_rows)
                    need(
                        [int(p[0]), int(p[2]), int(p[3])] == [query, step, key],
                        "PATH identity/order",
                    )
                    need(
                        bits(float(p[4])) == bits(exact)
                        and bits(float(p[5])) == decision["bucket_lower_bits"],
                        "PATH exact/bucket lower bits",
                    )
                    counters["path_bits_verified"] += 1
                degree = len(neighbors)
                for lane in range(degree):
                    position = cursor + 1 + lane
                    event_index, _, edge = rows[position]
                    need(
                        edge[1] == 2
                        and edge[2] == step
                        and edge[3] == source
                        and edge[4] == neighbors[lane],
                        "Graph edge order",
                    )
                    target, score, truth = edge[4:7]
                    other_score = learned_rows[position][2][5] if learned_rows else None
                    pending = target not in native.visited
                    first_encounter.setdefault(
                        target,
                        {
                            "source": source,
                            "step": step,
                            "lane": lane,
                            "score_bits": bits(score),
                        },
                    )
                    if pending:
                        first_preexpansion.setdefault(
                            target,
                            {
                                "source": source,
                                "step": step,
                                "lane": lane,
                                "score_bits": bits(score),
                            },
                        )
                    native_bin = native.locate(score)
                    other_bin = (
                        native.locate(other_score) if other_score is not None else None
                    )
                    if (
                        pending
                        and other_bin is not None
                        and native_bin != other_bin
                        and first_bin_change is None
                    ):
                        first_bin_change = {
                            "event_index": event_index,
                            "step": step,
                            "lane": lane,
                            "source": source,
                            "target": target,
                            "native_bin": native_bin,
                            "learned_bin": other_bin,
                        }
                    native.observe(source, target, step, lane, score)
                    if shadow is not None:
                        shadow.observe(source, target, step, lane, other_score)
                    if query < 1000:
                        proposals.writerow(
                            {
                                "query": query,
                                "event_index": event_index,
                                "source": source,
                                "step": step,
                                "lane": lane,
                                "arrival_serial": native.arrival,
                                "target": target,
                                "native_unexpanded": int(pending),
                                "shared_prefix": int(divergence is None),
                                "native_score_bits": bits(score),
                                "learned_score_bits": None
                                if other_score is None
                                else bits(other_score),
                                "exact_bits": bits(truth),
                                "native_bin": native_bin,
                                "learned_bin": other_bin,
                                "exact_bin": native.locate(truth),
                                "native_nominal_boundary_distance": native.boundary_distance(
                                    score
                                ),
                                "learned_nominal_boundary_distance": None
                                if other_score is None
                                else native.boundary_distance(other_score),
                            }
                        )
                    counters["native_score_records"] += 1
                cursor += degree + 1
            end = rows[-1][2]
            need(end[1] == 3 and end[2] == step, "End step")
            final = native.decision()
            need(
                final["action"] == "stop",
                "Unexplained native termination; scan-limit/filter/completion protocol must be checked",
            )
            if shadow is not None:
                other = shadow.decision()
                if other["action"] != "stop":
                    divergence = {
                        "after_expansion_step": step,
                        "kind": "native_stop_learned_continue",
                        "native": final,
                        "learned": other,
                    }
                    need(
                        first_bin_change is not None,
                        "Different termination without eligible bin change",
                    )
                if first_bin_change is None:
                    need(divergence is None, "Same-bin invariant failed")
                    counters["same_bins_implies_same_actions_checked_queries"] += 1
            returned = expected_results[query]
            need(len({key for key, _ in returned}) == 10, "Duplicate result keys")
            need(
                all(
                    key in expanded_key_scores
                    and bits(score) == bits(expanded_key_scores[key])
                    for key, score in returned
                ),
                "Result not in expanded set with exact FP32 score",
            )
            need(
                [bits(v) for v in sorted(score for _, score in returned)]
                == [bits(v) for v in sorted(expanded_key_scores.values())[:10]],
                "Top10 score multiset mismatch",
            )
            counters["top10_members_and_score_bits_verified_queries"] += 1
            counters["native_queries"] += 1
            counters["native_expansions_order_verified"] += step
            counters["prefix_diverged_queries"] += divergence is not None
            counters["same_target_different_provenance_queries"] += bool(
                same_target_different_provenance
            )
            counters["queries_with_eligible_bin_change"] += first_bin_change is not None
            record = {
                "query": query,
                "native_steps": step,
                "native_stop": final,
                "native_duplicate_tokens_discarded_after_expansion": native.stale_pops,
                "native_enqueued_proposals_excluding_seed": native.proposals,
                "native_distinct_first_seen_targets": len(first_encounter),
                "same_target_different_winning_provenance": same_target_different_provenance,
                "native_distinct_preexpansion_targets": len(first_preexpansion),
                "first_eligible_bin_change": first_bin_change,
                "first_prefix_divergence": divergence,
                "common_prefix_expansions": (
                    divergence.get(
                        "before_expansion_step",
                        divergence.get("after_expansion_step", step) + 1,
                    )
                    - 1
                )
                if divergence
                else step,
                "mechanism_sample": query < 1000,
                "overlay_is_actual_learned_trace": False,
                "after_divergence_data_are_native_path_off_policy_only": divergence
                is not None,
            }
            if query < 1000:
                record["native_expansion_winning_tokens"] = winners
                record["native_first_preexpansion_sources_descriptive_only"] = (
                    first_preexpansion
                )
            out.write(json.dumps(record, allow_nan=False) + "\n")
    need(counters["native_queries"] == 1000, "Incomplete native query gate")
    if overlays:
        need(next(overlays, None) is None, "Extra overlay queries")
    if path_rows is not None:
        need(next(path_rows, None) is None, "Extra PATH rows")
    report = {
        "version": "native_overlay_replay_v2",
        "counters": counters,
        "fixed_protocol": {
            "topk": 10,
            "margin": 0.020,
            "cap": 64,
            "patience": 0,
            "bucket_width_ratio": 0.016,
            "mechanism_queries": "primary 0..999",
            "native_gate_queries": 1000,
        },
        "trace_hash": sha(args.native_events),
        "graph_hash": sha(args.graph),
        "metadata_hash": sha(args.native_metadata),
        "runtime_config_hash": sha(args.runtime_config),
        "native_results_hash": sha(args.native_results),
        "script_hash": sha(Path(__file__)),
        "overlay_present": bool(args.rescored_events),
        "proof_kind": proof.get("proof_kind", "learned_model")
        if args.rescore_proof
        else None,
        "rescore_proof_sha256": sha(args.rescore_proof) if args.rescore_proof else None,
        "rescored_events_sha256": sha(args.rescored_events)
        if args.rescored_events
        else None,
        "no_gt_read": True,
        "no_training_or_parameter_search": True,
        "no_claim_of_learned_path_after_first_divergence": True,
        "observed_path_score_bits_validated": bool(args.native_path),
    }
    with (args.output / "summary.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
