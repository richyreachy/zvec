"""Independently compare predicted shared prefixes with real MULTI traces.

Only reads three completed trace files and replay JSONL. No graph search,
GT, training, code loading, or claims about the path after first divergence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

HEADER = struct.pack("<4I", 0x51474D68, 1, 32, 768)
WORDS = struct.Struct("<8I")
COUNT = 1000
NODES = 1000000
UINT_MAX = 2**32 - 1


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def finite(bits):
    return bits & 0x7F800000 != 0x7F800000


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, "Duplicate replay JSON key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError("Nonfinite replay JSON constant: " + value)


def signature(stream):
    state = os.fstat(stream.fileno())
    return state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns


class TraceReader:
    def __init__(self, stream, path):
        self.stream, self.path = stream, str(path)
        self.before = signature(stream)
        self.hash = hashlib.sha256()
        self.records = self.queries = 0

    def read(self, count):
        data = self.stream.read(count)
        self.hash.update(data)
        return data

    def groups(self):
        need(self.read(16) == HEADER, self.path + ": invalid MULTI header")
        query, blocks, current, expanded = 0, [], None, set()
        while raw := self.read(32):
            need(len(raw) == 32, self.path + ": truncated MULTI record")
            row = WORDS.unpack(raw)
            qi, kind, step, source, target, estimate, exact, reserved = row
            context = f"{self.path}: query={query} record={self.records}"
            self.records += 1
            need(qi == query and query < COUNT, context + ": query order/range")
            need(
                reserved == 0 and finite(estimate) and finite(exact),
                context + ": invalid score/reserved",
            )
            if kind == 1:
                need(step == len(blocks) + 1, context + ": expansion step")
                need(source == target and source < NODES, context + ": expansion node")
                need(
                    source not in expanded and estimate == exact,
                    context + ": duplicate expansion/center bits",
                )
                current = {"begin": raw, "node": source, "edges": []}
                blocks.append(current)
                expanded.add(source)
            elif kind == 2:
                need(
                    current is not None and step == len(blocks),
                    context + ": edge before expansion/wrong step",
                )
                need(
                    source == current["node"] and target < NODES,
                    context + ": edge source/target",
                )
                need(len(current["edges"]) < 32, context + ": more than 32 lanes")
                current["edges"].append(raw)
            elif kind == 3:
                need(bool(blocks) and step == len(blocks), context + ": end step")
                need(
                    source == target == UINT_MAX and estimate == exact == 0,
                    context + ": end sentinels/bits",
                )
                self.queries += 1
                yield {"query": query, "blocks": blocks, "end": raw}
                query += 1
                blocks, current, expanded = [], None, set()
            else:
                need(False, context + ": unknown event kind")
        need(not blocks, self.path + ": missing query end")
        need(query == COUNT, self.path + ": incomplete query count")
        need(
            signature(self.stream) == self.before,
            self.path + ": file changed during read",
        )

    def metadata(self):
        return {
            "path": self.path,
            "bytes": self.before[2],
            "sha256": self.hash.hexdigest(),
            "queries": self.queries,
            "records": self.records,
            "file_unchanged_during_read": signature(self.stream) == self.before,
        }


def check_overlay(native, overlay, counts):
    need(native["query"] == overlay["query"], "Overlay query mismatch")
    need(
        len(native["blocks"]) == len(overlay["blocks"]),
        "Overlay expansion count changed",
    )
    need(native["end"] == overlay["end"], "Overlay changed end record")
    for block, other in zip(native["blocks"], overlay["blocks"], strict=True):
        need(block["begin"] == other["begin"], "Overlay changed expansion record")
        need(len(block["edges"]) == len(other["edges"]), "Overlay changed lane count")
        for edge, replacement in zip(block["edges"], other["edges"], strict=True):
            need(
                edge[:20] == replacement[:20] and edge[24:] == replacement[24:],
                "Overlay changed non-estimate edge fields",
            )
            counts["all_native_overlay_non_score_edge_records_checked"] += 1


def token_target(decision, action):
    need(
        isinstance(decision, dict) and decision.get("action") == action,
        "Predicted action schema mismatch",
    )
    token = decision.get("next")
    if action == "expand":
        need(
            isinstance(token, dict) and type(token.get("target")) is int,
            "Missing predicted expansion target",
        )
        return token["target"]
    return None


def verify_provenance(token, trace, prefix):
    """Check predicted pop token originates in a scored shared-prefix lane.

    This does not independently observe the winning token in actual MULTI;
    actual MULTI only records the expanded node, not queue provenance.
    """
    need(isinstance(token, dict), "Missing predicted token")
    step, lane = token.get("step"), token.get("lane")
    need(
        type(step) is int and 1 <= step <= prefix,
        "Predicted token source is outside shared prefix",
    )
    block = trace["blocks"][step - 1]
    need(
        type(lane) is int and 0 <= lane < len(block["edges"]),
        "Predicted token lane range",
    )
    row = WORDS.unpack(block["edges"][lane])
    serial = sum(len(item["edges"]) for item in trace["blocks"][: step - 1]) + lane + 1
    need(
        token.get("source") == row[3] and token.get("target") == row[4],
        "Predicted token source/target",
    )
    need(
        token.get("score_bits") == row[5] and token.get("arrival_serial") == serial,
        "Predicted token score/arrival",
    )
    need(
        token["target"] not in {item["node"] for item in trace["blocks"][:prefix]},
        "Predicted next target is already expanded",
    )


def compare_prediction(native, overlay, actual, prediction, counts):
    query = native["query"]
    need(
        actual["query"] == query and prediction.get("query") == query,
        f"Query {query}: prediction/actual alignment",
    )
    native_nodes = [block["node"] for block in native["blocks"]]
    actual_nodes = [block["node"] for block in actual["blocks"]]
    shared = 0
    for first, second in zip(native_nodes, actual_nodes, strict=False):
        if first != second:
            break
        shared += 1
    need(
        type(prediction.get("common_prefix_expansions")) is int,
        "Missing integer prefix prediction",
    )
    need(
        prediction["common_prefix_expansions"] == shared,
        f"Query {query}: actual/predicted common prefix differs",
    )
    need(
        prediction.get("native_steps") == len(native_nodes),
        "Predicted native expansion count mismatch",
    )
    need(
        prediction.get("overlay_is_actual_learned_trace") is False,
        "Prediction improperly labels overlay as actual trace",
    )
    for index in range(shared):
        nblock, oblock, ablock = (
            native["blocks"][index],
            overlay["blocks"][index],
            actual["blocks"][index],
        )
        need(
            nblock["begin"] == ablock["begin"],
            f"Query {query}: shared center exact bits/order mismatch",
        )
        need(
            len(ablock["edges"]) == len(oblock["edges"]),
            f"Query {query}: shared block lane count mismatch",
        )
        for lane, (observed, expected) in enumerate(
            zip(ablock["edges"], oblock["edges"], strict=True)
        ):
            need(
                observed == expected,
                f"Query {query}: step {index + 1} lane {lane}: actual/overlay record bits differ",
            )
            counts["shared_prefix_edge_records_bit_identical"] += 1
        counts["shared_prefix_center_records_bit_identical"] += 1
    divergence = prediction.get("first_prefix_divergence")
    if divergence is None:
        need(
            len(native_nodes) == len(actual_nodes) == shared,
            f"Query {query}: unpredicted actual divergence",
        )
        need(native["end"] == actual["end"], "Equal-path end bits mismatch")
        kind = "no_divergence"
    else:
        kind = divergence.get("kind")
        if kind in {"next_id", "stop"}:
            need(
                divergence.get("before_expansion_step") == shared + 1
                and shared < len(native_nodes),
                "Predicted before-step mismatch",
            )
            need(
                token_target(divergence.get("native"), "expand")
                == native_nodes[shared],
                "Predicted native next target mismatch",
            )
            verify_provenance(divergence["native"]["next"], native, shared)
            if kind == "stop":
                token_target(divergence.get("learned"), "stop")
                need(
                    len(actual_nodes) == shared,
                    f"Query {query}: predicted stop but actual continued",
                )
            else:
                need(
                    shared < len(actual_nodes),
                    f"Query {query}: predicted next-ID but actual stopped",
                )
                need(
                    token_target(divergence.get("learned"), "expand")
                    == actual_nodes[shared],
                    "Predicted learned next target mismatch",
                )
                need(
                    native_nodes[shared] != actual_nodes[shared],
                    "Predicted next-ID divergence has equal IDs",
                )
                verify_provenance(divergence["learned"]["next"], overlay, shared)
        elif kind == "native_stop_learned_continue":
            need(
                divergence.get("after_expansion_step") == shared == len(native_nodes),
                "Predicted native-stop step mismatch",
            )
            token_target(divergence.get("native"), "stop")
            need(
                shared < len(actual_nodes),
                "Predicted continuation but actual also stopped",
            )
            need(
                token_target(divergence.get("learned"), "expand")
                == actual_nodes[shared],
                "Continuation target mismatch",
            )
            verify_provenance(divergence["learned"]["next"], overlay, shared)
        else:
            need(False, "Unknown predicted divergence kind: " + str(kind))
    counts["queries_checked"] += 1
    counts["divergence_kind_" + kind] += 1
    return {
        "query": query,
        "native_expansions": len(native_nodes),
        "actual_expansions": len(actual_nodes),
        "common_prefix_expansions": shared,
        "verified_divergence_kind": kind,
        "native_next_target": native_nodes[shared]
        if shared < len(native_nodes)
        else None,
        "actual_next_target": actual_nodes[shared]
        if shared < len(actual_nodes)
        else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "native-events",
        "overlay-events",
        "actual-events",
        "replay-queries",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    need(not args.output.exists(), "Output exists; refusing overwrite")
    paths = [args.native_events, args.overlay_events, args.actual_events]
    counts, rows, prediction_hash = Counter(), [], hashlib.sha256()
    with ExitStack() as stack:
        readers = [
            TraceReader(stack.enter_context(path.open("rb")), path) for path in paths
        ]
        iterators = [iter(reader.groups()) for reader in readers]
        predictions = stack.enter_context(args.replay_queries.open("rb"))
        before = signature(predictions)
        for query in range(COUNT):
            group = [next(iterator, None) for iterator in iterators]
            need(
                all(item is not None for item in group), "Incomplete trace query group"
            )
            raw = predictions.readline()
            need(
                bool(raw) and raw.endswith(b"\n"),
                "Missing/truncated replay prediction line",
            )
            prediction_hash.update(raw)
            prediction = json.loads(
                raw, object_pairs_hook=strict_object, parse_constant=reject_constant
            )
            need(
                isinstance(prediction, dict)
                and type(prediction.get("query")) is int
                and prediction["query"] == query,
                "Replay prediction query order",
            )
            native, overlay, actual = group
            check_overlay(native, overlay, counts)
            rows.append(compare_prediction(native, overlay, actual, prediction, counts))
        need(not predictions.read(1), "Extra replay prediction lines")
        need(signature(predictions) == before, "Replay predictions changed during read")
        for iterator in iterators:
            need(next(iterator, None) is None, "Extra trace query group")
        inputs = [reader.metadata() for reader in readers]
    report = {
        "status": "passed",
        "version": "independent_actual_prefix_v1",
        "counts": dict(counts),
        "queries": rows,
        "inputs_native_overlay_actual": inputs,
        "replay_predictions": {
            "path": str(args.replay_queries),
            "bytes": before[2],
            "sha256": prediction_hash.hexdigest(),
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "all_1000_predicted_first_divergences_match_actual": True,
        "all_shared_prefix_actual_scores_equal_overlay_bits": True,
        "actual_stop_reason_or_pending_stop_target_observed": False,
        "actual_winning_queue_provenance_observed": False,
        "actual_path_after_first_divergence_compared_to_native_overlay": False,
        "external_gates_required": [
            "Same query order/input normalization, graph, rotation and learned model",
            "Actual run fixed margin .020/cap64/patience0/bucket width .016",
            "Successful complete actual trace run, native replay proof and actual-code scorer proof",
            "Tracing disabled/enabled output equivalence checked separately",
        ],
        "no_gt_training_or_retrieval_executed": True,
        "no_qps_or_recall_claim": True,
    }
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
