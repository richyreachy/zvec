"""Frozen mechanism-B statistics on all existing native query trajectories.

Learned scores are a forced-native/off-policy overlay after first divergence.
No retrieval, training, GT reads, parameter selection, or fresh-query claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import struct
from collections import Counter
from pathlib import Path

import numpy as np

F32 = struct.Struct("<f")
WORDS = struct.Struct("<8I")
FIELDS = (
    "blocks",
    "primary_blocks",
    "short_blocks",
    "candidate_pairs",
    "duplicate_target_pairs",
    "visited_filtered_pairs",
    "pending_true_tie_pairs",
    "n_pairs",
    "continuous_native_errors",
    "continuous_learned_errors",
    "bucket_native_errors",
    "bucket_learned_errors",
    "native_approx_ties",
    "learned_approx_ties",
    "continuous_corrections",
    "continuous_regressions",
    "continuous_corrections_bucket_unchanged",
    "bucket_corrections",
    "bucket_regressions",
    "negative_control_pairs",
    "exact_control_pairs",
)


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_verified(path, expected, name):
    need(sha(path) == expected, "Validated module changed: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def outcome(
    first_score, second_score, first_bin, second_bin, first_serial, second_serial
):
    return (
        first_score < second_score,
        (first_bin, -first_serial) < (second_bin, -second_serial),
    )


def count_pair(
    row, native, learned, truth, bins_native, bins_learned, serials, anchor, other
):
    ncont, nbucket = outcome(
        native[anchor],
        native[other],
        bins_native[anchor],
        bins_native[other],
        serials[anchor],
        serials[other],
    )
    lcont, lbucket = outcome(
        learned[anchor],
        learned[other],
        bins_learned[anchor],
        bins_learned[other],
        serials[anchor],
        serials[other],
    )
    duplicate = outcome(
        native[anchor],
        native[other],
        bins_native[anchor],
        bins_native[other],
        serials[anchor],
        serials[other],
    )
    need(duplicate == (ncont, nbucket), "Native-vs-native negative control failed")
    exact_cont, _ = outcome(
        truth[anchor],
        truth[other],
        bins_native[anchor],
        bins_native[other],
        serials[anchor],
        serials[other],
    )
    need(exact_cont, "Exact-reference continuous positive control failed")
    row["n_pairs"] += 1
    row["continuous_native_errors"] += not ncont
    row["continuous_learned_errors"] += not lcont
    row["bucket_native_errors"] += not nbucket
    row["bucket_learned_errors"] += not lbucket
    row["native_approx_ties"] += native[anchor] == native[other]
    row["learned_approx_ties"] += learned[anchor] == learned[other]
    row["continuous_corrections"] += not ncont and lcont
    row["continuous_regressions"] += ncont and not lcont
    row["continuous_corrections_bucket_unchanged"] += (
        not ncont and lcont and nbucket == lbucket
    )
    row["bucket_corrections"] += not nbucket and lbucket
    row["bucket_regressions"] += nbucket and not lbucket
    row["negative_control_pairs"] += 1
    row["exact_control_pairs"] += 1


def query_statistics(native, overlay, bucket_module, tolerance):
    row = dict.fromkeys(FIELDS, 0)
    row["query"] = native["query"]
    first = native["blocks"][0]
    mapper = bucket_module.Queue(first["node"], 0.020, 64, 0)
    mapper.expand(first["node"], F32.unpack_from(first["begin"], 24)[0])
    visited, arrival = set(), 0
    for block, replacement in zip(native["blocks"], overlay["blocks"], strict=True):
        visited.add(block["node"])
        row["blocks"] += 1
        degree = len(block["edges"])
        serials = list(range(arrival + 1, arrival + degree + 1))
        arrival += degree
        if degree != 32:
            row["short_blocks"] += 1
            continue
        row["primary_blocks"] += 1
        targets = [WORDS.unpack(edge)[4] for edge in block["edges"]]
        truth = [F32.unpack_from(edge, 24)[0] for edge in block["edges"]]
        scores_n = [F32.unpack_from(edge, 20)[0] for edge in block["edges"]]
        scores_l = [F32.unpack_from(edge, 20)[0] for edge in replacement["edges"]]
        bins_n = [mapper.locate(value) for value in scores_n]
        bins_l = [mapper.locate(value) for value in scores_l]
        anchor = min(range(32), key=lambda index: truth[index])
        for other in range(32):
            if other == anchor:
                continue
            row["candidate_pairs"] += 1
            if targets[anchor] == targets[other]:
                row["duplicate_target_pairs"] += 1
                continue
            if targets[anchor] in visited or targets[other] in visited:
                row["visited_filtered_pairs"] += 1
                continue
            gap = truth[other] - truth[anchor]
            need(gap >= 0, "Anchor is not exact Top1")
            if gap <= tolerance:
                row["pending_true_tie_pairs"] += 1
                continue
            count_pair(
                row, scores_n, scores_l, truth, bins_n, bins_l, serials, anchor, other
            )
    need(
        row["candidate_pairs"]
        == row["duplicate_target_pairs"]
        + row["visited_filtered_pairs"]
        + row["pending_true_tie_pairs"]
        + row["n_pairs"],
        "Pair filtering counts do not reconcile",
    )
    return row


def statistics(rows, config):
    valid = [row for row in rows if row["n_pairs"]]
    need(bool(valid), "No eligible query pairs")
    rates = np.asarray(
        [
            [
                (row["continuous_learned_errors"] - row["continuous_native_errors"])
                / row["n_pairs"],
                (row["bucket_learned_errors"] - row["bucket_native_errors"])
                / row["n_pairs"],
                row["continuous_corrections_bucket_unchanged"] / row["n_pairs"],
                row["continuous_corrections"] / row["n_pairs"],
            ]
            for row in valid
        ],
        dtype=np.float64,
    )
    mean = rates.mean(axis=0)
    points = {
        "delta_cont": float(mean[0]),
        "delta_bucket": float(mean[1]),
        "I_B": float(mean[1] - mean[0]),
        "E": float(mean[2] / mean[3]) if mean[3] > 0 else None,
    }
    bootstrap = config["bootstrap"]
    rng = np.random.default_rng(bootstrap["seed"])
    samples = np.empty((bootstrap["samples"], 4), dtype=np.float64)
    zero_denominator = 0
    for i in range(bootstrap["samples"]):
        draw = rates[rng.integers(0, len(valid), size=len(valid))].mean(axis=0)
        samples[i, :3] = [draw[0], draw[1], draw[1] - draw[0]]
        samples[i, 3] = draw[2] / draw[3] if draw[3] > 0 else np.nan
        zero_denominator += int(draw[3] <= 0)
    intervals = {}
    for i, name in enumerate(("delta_cont", "delta_bucket", "I_B", "E")):
        intervals[name] = (
            None
            if name == "E" and zero_denominator
            else np.quantile(
                samples[:, i], bootstrap["quantiles"], method="linear"
            ).tolist()
        )
    totals = {name: sum(row[name] for row in rows) for name in FIELDS}
    enough = (
        len(valid) >= config["minimum_valid_queries"]
        and totals["continuous_corrections"] >= config["minimum_continuous_corrections"]
        and sum(row["continuous_corrections"] > 0 for row in valid)
        >= config["minimum_queries_with_continuous_corrections"]
        and points["E"] is not None
    )
    support = (
        enough
        and points["delta_cont"] <= -0.01
        and points["delta_bucket"] >= 0
        and points["E"] >= 0.75
        and intervals["I_B"][0] > 0
    )
    refutation = (
        enough
        and points["E"] <= 0.25
        and points["delta_bucket"] <= -0.01
        and intervals["delta_bucket"][1] < 0
    )
    need(not (support and refutation), "Conflicting mechanism gates")
    means = {
        name: float(np.mean([row[name] / row["n_pairs"] for row in valid]))
        for name in (
            "continuous_native_errors",
            "continuous_learned_errors",
            "bucket_native_errors",
            "bucket_learned_errors",
        )
    }
    return {
        "query_equal_weight_error_rates": means,
        "point_estimates": points,
        "query_bootstrap_97_5_intervals": intervals,
        "E_query_equal_weight_numerator": float(mean[2]),
        "E_query_equal_weight_denominator": float(mean[3]),
        "E_raw_unchanged_corrections": totals[
            "continuous_corrections_bucket_unchanged"
        ],
        "E_raw_continuous_corrections": totals["continuous_corrections"],
        "E_raw_ratio_descriptive_only": totals[
            "continuous_corrections_bucket_unchanged"
        ]
        / totals["continuous_corrections"]
        if totals["continuous_corrections"]
        else None,
        "bootstrap_zero_E_denominator_draws": zero_denominator,
        "queries_with_pairs": len(valid),
        "queries_without_pairs": [row["query"] for row in rows if not row["n_pairs"]],
        "queries_with_continuous_corrections": sum(
            row["continuous_corrections"] > 0 for row in valid
        ),
        "totals": totals,
        "minimum_counts_satisfied": enough,
        "support_gate_passed": support,
        "refutation_gate_passed": refutation,
        "mechanism_b_decision": "supported_on_frozen_diagnostic"
        if support
        else "refuted_on_frozen_diagnostic"
        if refutation
        else "inconclusive"
        if enough
        else "insufficient",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-query", type=Path, required=True)
    args = parser.parse_args()
    need(
        not args.output.exists() and not args.per_query.exists(),
        "Refusing to overwrite output",
    )
    config = json.loads(args.config.read_text())
    need(
        config["version"] == "mechanism_b_forced_native_v1"
        and config["frozen_before_effect_statistics"],
        "Configuration version/freeze",
    )
    need(
        config["query_count"] == 1000
        and config["primary_block_degree"] == 32
        and config["true_tie_tolerance"] == 1e-7,
        "Population changed",
    )
    need(
        config["bootstrap"]["samples"] == 4000
        and config["bootstrap"]["seed"] == 8002
        and config["bootstrap"]["quantiles"] == [0.0125, 0.9875],
        "Bootstrap changed",
    )
    bucket = load_verified(
        config["validated_bucket_module"],
        config["validated_bucket_module_sha256"],
        "mechanism80_bucket",
    )
    trace = load_verified(
        config["validated_parser_module"],
        config["validated_parser_module_sha256"],
        "mechanism80_trace",
    )
    audit_counts, rows = Counter(), []
    with (
        Path(config["native_events"]).open("rb") as ns,
        Path(config["overlay_events"]).open("rb") as ls,
    ):
        native_reader = trace.TraceReader(ns, config["native_events"])
        overlay_reader = trace.TraceReader(ls, config["overlay_events"])
        natives, overlays = iter(native_reader.groups()), iter(overlay_reader.groups())
        for query in range(1000):
            native, overlay = next(natives, None), next(overlays, None)
            need(
                native is not None and overlay is not None and native["query"] == query,
                "Incomplete query groups",
            )
            trace.check_overlay(native, overlay, audit_counts)
            rows.append(
                query_statistics(native, overlay, bucket, config["true_tie_tolerance"])
            )
        need(
            next(natives, None) is None and next(overlays, None) is None,
            "Extra query groups",
        )
        inputs = {
            "native": native_reader.metadata(),
            "overlay": overlay_reader.metadata(),
        }
    need(
        inputs["native"]["sha256"] == config["native_events_sha256"]
        and inputs["overlay"]["sha256"] == config["overlay_events_sha256"],
        "Input content hash changed",
    )
    analysis = statistics(rows, config)
    with args.per_query.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["query", *FIELDS])
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "status": "passed",
        "configuration_path": str(args.config),
        "configuration_sha256": sha(args.config),
        "configuration": config,
        "script_sha256": sha(Path(__file__)),
        "numpy_version": np.__version__,
        "inputs": inputs,
        "overlay_alignment_checks": dict(audit_counts),
        "analysis": analysis,
        "per_query_file": str(args.per_query),
        "per_query_sha256": sha(args.per_query),
        "controls": {
            "native_vs_native_passed": True,
            "native_vs_native_delta_cont": 0,
            "native_vs_native_delta_bucket": 0,
            "native_vs_native_I_B": 0,
            "native_vs_native_corrections": 0,
            "exact_reference_continuous_passed": True,
            "exact_reference_continuous_errors": 0,
            "pairs_checked_each_control": analysis["totals"]["n_pairs"],
        },
        "scope": "All native full trajectories; learned comparisons after divergence are forced-native/off-policy local proposal comparisons",
        "does_not_attribute_end_to_end_recall_or_qps_causally": True,
        "does_not_test_mechanism_A_matched_proxy_control": True,
        "no_gt_training_parameter_search_or_retrieval": True,
    }
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
