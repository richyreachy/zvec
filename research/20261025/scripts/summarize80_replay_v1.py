"""Describe all1000 frozen common-prefix replay outcomes; never reads GT."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--replay", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
summary = json.loads((args.replay / "summary.json").read_text())
assert summary["proof_kind"] == "learned_model"
counts = Counter()
stop_reasons = Counter()
first_steps = Counter()
common_prefix = []
unchanged = []
source_change_queries = []
queries = 0
with (args.replay / "queries.jsonl").open() as stream:
    for line in stream:
        row = json.loads(line)
        assert row["query"] == queries
        queries += 1
        common_prefix.append(row["common_prefix_expansions"])
        stop_reasons[row["native_stop"]["reason"]] += 1
        event = row["first_prefix_divergence"]
        if row["same_target_different_winning_provenance"]:
            source_change_queries.append(row["query"])
        if event is None:
            unchanged.append(row["query"])
            counts["unchanged_entire_native_path_and_stop_action"] += 1
            continue
        counts[event["kind"]] += 1
        first_steps[str(row["common_prefix_expansions"] + 1)] += 1
        if event["kind"] == "next_id":
            original = event["native"]["next"]["bin"]
            changed = event["learned"]["next"]["bin"]
            counts[
                "next_id_divergence_same_min_bin"
                if original == changed
                else "next_id_divergence_learned_min_bin_up"
                if changed > original
                else "next_id_divergence_learned_min_bin_down"
            ] += 1
assert queries == 1000
result = {
    "queries": queries,
    "all_primary_queries_used_no_subset_selection": True,
    "first_divergence_counts": dict(counts),
    "common_prefix_expansions_mean": statistics.mean(common_prefix),
    "common_prefix_expansions_percentiles_0_10_25_50_75_90_100": np.percentile(
        common_prefix, [0, 10, 25, 50, 75, 90, 100]
    ).tolist(),
    "first_divergence_decision_step_histogram": dict(
        sorted(first_steps.items(), key=lambda x: int(x[0]))
    ),
    "native_stop_reasons_reconstructed": dict(stop_reasons),
    "same_target_different_winning_provenance_query_count": len(source_change_queries),
    "unchanged_native_path_query_ids": unchanged,
    "no_GT_read": True,
    "no_tuning": True,
    "actual_learned_trace_crosscheck_pending": True,
    "no_actual_learned_path_claim_after_first_divergence": True,
    "source_replay_summary_sha256": hashlib.sha256(
        (args.replay / "summary.json").read_bytes()
    ).hexdigest(),
    "source_replay_queries_sha256": hashlib.sha256(
        (args.replay / "queries.jsonl").read_bytes()
    ).hexdigest(),
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
with args.output.open("x") as stream:
    json.dump(result, stream, indent=2)
    stream.write("\n")
logging.info("%s", json.dumps(result, indent=2))
