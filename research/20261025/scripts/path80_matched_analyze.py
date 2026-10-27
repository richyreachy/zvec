"""Frozen mechanism-A query-cluster analysis; no fitting or subgroup selection."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")

R = Path(__file__).resolve().parent.parent
table = R / (sys.argv[1] if len(sys.argv) > 1 else "path80_matched_v1")
score = table / (sys.argv[2] if len(sys.argv) > 2 else "score_v1")
out = table / (sys.argv[3] if len(sys.argv) > 3 else "analysis_v1")
out.mkdir(exist_ok=False)


def need(yes, message):
    if not yes:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(4 << 20), b""):
            h.update(data)
    return h.hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


manifest = json.loads((table / "sampling_manifest.json").read_text())
config = manifest["analysis"]
proof = json.loads((score / "proof.json").read_text())
need(
    proof["passed"] and proof["source_and_inputs_unchanged"],
    "Successful matched-score proof required",
)
need(
    sha(table / "sampling_manifest.json") == proof["frozen_sampling_manifest_sha256"],
    "Frozen configuration changed",
)
for name, expected in manifest["table_sha256"].items():
    need(sha(table / name) == expected, "Frozen sampled table changed")
for name, expected in proof["artifact_sha256"].items():
    need(sha(score / name) == expected, "Matched score artifact changed")
states = np.fromfile(table / "states.u32", dtype="<u4").reshape(-1, 5)
centers = np.fromfile(table / "centers_proxies.u32", dtype="<u4").reshape(-1, 17)
ns, nc = len(states), len(centers)
header = np.fromfile(score / "observations.bin", dtype="<u4", count=8)
need(
    np.array_equal(header, [0x51474180, 1, 424, 768, ns, nc, 32, 16]),
    "Observation header",
)
dtype = np.dtype(
    [
        (name, "<u4")
        for name in [
            "kind",
            "owner",
            "center_row",
            "proxy_index",
            "center",
            "query_id",
            "center_bits",
            "reserved",
        ]
    ]
    + [
        ("delta_norm2", "<f8"),
        ("exact", "<f4", (32,)),
        ("native", "<f4", (32,)),
        ("learned", "<f4", (32,)),
    ]
)
need(dtype.itemsize == 424, "Observation dtype")
records = np.memmap(score / "observations.bin", dtype=dtype, mode="r", offset=32)
need(
    len(records) == ns + nc * 16 and np.all(records["reserved"] == 0),
    "Observation count/reserved",
)
need(
    np.all(records[:ns]["kind"] == 0)
    and np.array_equal(records[:ns]["owner"], np.arange(ns))
    and np.array_equal(records[:ns]["query_id"], states[:, 0])
    and np.array_equal(records[:ns]["center"], states[:, 2])
    and np.array_equal(records[:ns]["center_row"], states[:, 3]),
    "Path observation/state association",
)
need(
    np.all(records[ns:]["kind"] == 1)
    and np.array_equal(records[ns:]["center_row"], np.repeat(np.arange(nc), 16))
    and np.array_equal(records[ns:]["proxy_index"], np.tile(np.arange(16), nc))
    and np.array_equal(records[ns:]["query_id"], centers[:, 1:].reshape(-1)),
    "Frozen heldout proxy association",
)
truth = records["exact"].astype(np.float64)
native = records["native"].astype(np.float64)
learned = records["learned"].astype(np.float64)
need(
    np.isfinite(truth).all()
    and np.isfinite(native).all()
    and np.isfinite(learned).all(),
    "Finite scores",
)
tol = float(config["top1_exact_tie_tolerance"])
rows = np.arange(len(records))
true_lane = truth.argmin(axis=1)
true_min = truth[rows, true_lane]


def errors(values):
    choice = values.argmin(axis=1)  # stable first minimum lane, frozen rule
    return choice, truth[rows, choice] - true_min > tol


native_choice, native_error = errors(native)
learned_choice, learned_error = errors(learned)
repeat_choice, repeat_error = errors(native.copy())
exact_choice, exact_error = errors(truth)
need(
    np.array_equal(native_choice, repeat_choice)
    and np.array_equal(native_error, repeat_error),
    "Native/native choice control",
)
need(
    proof["native_native_bit_identical_lanes"] == len(records) * 32,
    "Native/native actual-score bit control",
)
need(not exact_error.any(), "Exact Top1 positive control")
training = np.fromfile(score / "training_controls.f64", dtype="<f8").reshape(nc, 8)
need(np.array_equal(training[:, 0], centers[:, 0]), "Training control centers")
need(
    np.all(training[:, 3] <= training[:, 2])
    and np.all(training[:, 6] <= 1.02 + 1e-12)
    and np.all(training[:, 7] <= 8),
    "Full training objective and budgets",
)
need(
    np.all(
        np.abs(training[:, 2:4] - training[:, 4:6])
        <= 1e-12 * np.maximum(1, np.abs(training[:, 4:6]))
    ),
    "Objective accepted-audit identity tolerance",
)
proxy_native = native_error[ns:].reshape(nc, 16)
proxy_learned = learned_error[ns:].reshape(nc, 16)
proxy_corrected = proxy_native & ~proxy_learned
proxy_worsened = ~proxy_native & proxy_learned
state_center = states[:, 3].astype(int)
state_proxy_native = proxy_native.sum(axis=1)[state_center]
state_proxy_learned = proxy_learned.sum(axis=1)[state_center]
state_proxy_corrected = proxy_corrected.sum(axis=1)[state_center]
state_proxy_worsened = proxy_worsened.sum(axis=1)[state_center]
query_id = states[:, 0].astype(int)
state_count = np.bincount(query_id, minlength=1000)
eligible = state_count > 0
need(
    int(eligible.sum()) == manifest["eligible_query_count"],
    "Frozen eligible-query count",
)


def total(values):
    return np.bincount(
        query_id, weights=np.asarray(values, dtype=np.float64), minlength=1000
    )


path_native_count = total(native_error[:ns])
path_learned_count = total(learned_error[:ns])
proxy_native_count = total(state_proxy_native)
proxy_learned_count = total(state_proxy_learned)
proxy_correction_count = total(state_proxy_corrected)
proxy_worsening_count = total(state_proxy_worsened)
path_corrected_count = total(native_error[:ns] & ~learned_error[:ns])
path_worsened_count = total(~native_error[:ns] & learned_error[:ns])
denom = np.where(eligible, state_count, 1)
delta_path = (path_learned_count - path_native_count) / denom
delta_proxy = (proxy_learned_count - proxy_native_count) / (16 * denom)
interaction = delta_path - delta_proxy
matrix = np.column_stack((delta_path, delta_proxy, interaction))[eligible]
rng = np.random.default_rng(int(config["bootstrap_seed"]))
bootstrap = np.empty((int(config["bootstrap_replicates"]), 3), dtype=np.float64)
for i in range(len(bootstrap)):
    indices = rng.integers(0, len(matrix), size=len(matrix))
    bootstrap[i] = matrix[indices].mean(axis=0)
with (out / "bootstrap.f64").open("xb") as stream:
    bootstrap.astype("<f8").tofile(stream)
ci = np.quantile(bootstrap, config["ci_quantiles"], axis=0, method="linear")
effects = matrix.mean(axis=0)
effect_names = ["delta_path", "delta_proxy", "I_A"]
intervals = {
    name: {"estimate": float(effects[i]), "ci97_5": [float(ci[0, i]), float(ci[1, i])]}
    for i, name in enumerate(effect_names)
}
corrections = int(proxy_correction_count.sum())
correction_queries = int((proxy_correction_count > 0).sum())
sufficient = (
    int(eligible.sum()) >= config["sufficiency"]["eligible_queries_min"]
    and corrections >= config["sufficiency"]["proxy_corrections_min"]
    and correction_queries
    >= config["sufficiency"]["proxy_correction_source_queries_min"]
)
support = effects[1] <= -0.02 and effects[0] >= 0.005 and ci[0, 2] > 0
refute = effects[0] <= -0.02 and ci[1, 2] <= 0.01
decision = (
    "insufficient_sample"
    if not sufficient
    else "supports_A"
    if support
    else "refutes_A"
    if refute
    else "inconclusive"
)
per_query = []
for q in range(1000):
    per_query.append(
        {
            "query": q,
            "eligible": bool(eligible[q]),
            "states": int(state_count[q]),
            "path_native_errors": int(path_native_count[q]),
            "path_learned_errors": int(path_learned_count[q]),
            "path_denominator": int(state_count[q]),
            "proxy_native_errors": int(proxy_native_count[q]),
            "proxy_learned_errors": int(proxy_learned_count[q]),
            "proxy_denominator": int(16 * state_count[q]),
            "proxy_corrections": int(proxy_correction_count[q]),
            "proxy_worsenings": int(proxy_worsening_count[q]),
            "path_corrections": int(path_corrected_count[q]),
            "path_worsenings": int(path_worsened_count[q]),
            **{
                name: float(value[q]) if eligible[q] else None
                for name, value in zip(
                    effect_names, [delta_path, delta_proxy, interaction], strict=True
                )
            },
        }
    )
save(out / "per_query.json", per_query)
with (out / "per_state.csv").open("x", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(
        [
            "state",
            "query",
            "step",
            "center",
            "center_row",
            "native_path_error",
            "learned_path_error",
            "native_proxy_errors",
            "learned_proxy_errors",
            "proxy_count",
            "proxy_corrections",
            "proxy_worsenings",
        ]
    )
    for i, state in enumerate(states):
        writer.writerow(
            [
                i,
                *map(int, state[:4]),
                int(native_error[i]),
                int(learned_error[i]),
                int(state_proxy_native[i]),
                int(state_proxy_learned[i]),
                16,
                int(state_proxy_corrected[i]),
                int(state_proxy_worsened[i]),
            ]
        )
true_gap = truth - true_min[:, None]
native_gap = native - native[rows, true_lane, None]
learned_gap = learned - learned[rows, true_lane, None]
non_tied = true_gap > tol
with (out / "signed_gap_diagnostics.npz").open("xb") as stream:
    np.savez_compressed(
        stream,
        true_lane=true_lane.astype("<u4"),
        native_choice=native_choice.astype("<u4"),
        learned_choice=learned_choice.astype("<u4"),
        exact_gap=true_gap,
        native_gap=native_gap,
        learned_gap=learned_gap,
        non_tied_pair=non_tied,
    )
descriptive = {}
for name, begin, end in [
    ("path", 0, ns),
    ("proxy_unique_center_observations", ns, len(records)),
]:
    mask = non_tied[begin:end]
    ng = native_gap[begin:end]
    lg = learned_gap[begin:end]
    descriptive[name] = {
        "observations": end - begin,
        "true_top1_tied_observations": int(
            ((true_gap[begin:end] <= tol).sum(axis=1) > 1).sum()
        ),
        "non_tied_true_top1_pairs": int(mask.sum()),
        "native_signed_gap_wrong_or_zero": int(((ng <= 0) & mask).sum()),
        "learned_signed_gap_wrong_or_zero": int(((lg <= 0) & mask).sum()),
        "signed_gap_negative_to_positive": int(((ng < 0) & (lg > 0) & mask).sum()),
        "signed_gap_positive_to_negative": int(((ng > 0) & (lg < 0) & mask).sum()),
        "query_center_exact_distance_percentiles": np.percentile(
            records["center_bits"][begin:end].view("<f4"), [0, 25, 50, 75, 100]
        ).tolist(),
        "rotated_delta_norm_percentiles": np.percentile(
            np.sqrt(records["delta_norm2"][begin:end]), [0, 25, 50, 75, 100]
        ).tolist(),
        "native_actual_mse": float(
            np.mean((native[begin:end] - truth[begin:end]) ** 2)
        ),
        "learned_actual_mse": float(
            np.mean((learned[begin:end] - truth[begin:end]) ** 2)
        ),
    }
report = {
    "passed_controls": True,
    "decision": decision,
    "mechanism": "A: benefit mismatch between native path query and matched heldout database proxies",
    "effects_query_equal_weight": intervals,
    "error_rates_query_equal_weight": {
        "path_native": float((path_native_count / denom)[eligible].mean()),
        "path_learned": float((path_learned_count / denom)[eligible].mean()),
        "proxy_native": float((proxy_native_count / (16 * denom))[eligible].mean()),
        "proxy_learned": float((proxy_learned_count / (16 * denom))[eligible].mean()),
    },
    "raw_numerators_denominators": {
        "path_native_errors": int(path_native_count.sum()),
        "path_learned_errors": int(path_learned_count.sum()),
        "path_states": ns,
        "matched_proxy_native_errors": int(proxy_native_count.sum()),
        "matched_proxy_learned_errors": int(proxy_learned_count.sum()),
        "matched_proxy_observations": ns * 16,
        "unique_center_proxy_observations": nc * 16,
        "proxy_corrections": corrections,
        "proxy_correction_source_queries": correction_queries,
        "proxy_worsenings": int(proxy_worsening_count.sum()),
        "path_corrections": int(path_corrected_count.sum()),
        "path_worsenings": int(path_worsened_count.sum()),
        "path_worsening_source_queries": int((path_worsened_count > 0).sum()),
    },
    "eligible_queries": int(eligible.sum()),
    "ineligible_query_ids": np.flatnonzero(~eligible).tolist(),
    "sufficiency_passed": sufficient,
    "support_conditions_passed": bool(support),
    "refute_conditions_passed": bool(refute),
    "frozen_analysis_configuration": config,
    "bootstrap": {
        "replicates": len(bootstrap),
        "seed": 8002,
        "cluster": "eligible query",
        "two_sided_percent": 97.5,
    },
    "controls": {
        "actual_native_native_bit_identical_lanes": proof[
            "native_native_bit_identical_lanes"
        ],
        "native_native_choice_or_indicator_changes": 0,
        "native_native_delta_path": 0,
        "native_native_delta_proxy": 0,
        "native_native_I_A": 0,
        "exact_top1_errors": int(exact_error.sum()),
        "complete_training_objective_nonincrease_centers": nc,
        "training_objective_relative_changes_percentiles": np.percentile(
            training[:, 3] / training[:, 2] - 1, [0, 25, 50, 75, 100]
        ).tolist(),
    },
    "descriptive_not_new_hypothesis_tests": descriptive,
    "ideal_score_error_and_training_subspace_projection": "deferred descriptive follow-up; not required for this preregistered main effect",
    "sampling_manifest": str(table / "sampling_manifest.json"),
    "sampling_manifest_sha256": proof["frozen_sampling_manifest_sha256"],
    "score_proof": str(score / "proof.json"),
    "score_proof_sha256": sha(score / "proof.json"),
    "analysis_script_sha256": sha(Path(__file__)),
    "primary_queries_already_tuned_not_blind": True,
    "off_policy_native_states": True,
    "new_training_or_retrieval": False,
    "limitations": [
        "Bootstrap describes stability on reused queries, not independent generalization",
        "Same-center holdouts do not isolate query-center distance, candidate difficulty or arrival selection",
        "No attribution of all recall/QPS differences and no learned-path claim",
    ],
    "artifact_sha256": {
        name: sha(out / name)
        for name in [
            "per_query.json",
            "per_state.csv",
            "bootstrap.f64",
            "signed_gap_diagnostics.npz",
        ]
    },
}
save(out / "report.json", report)
logging.info(
    json.dumps(
        {
            "report": str(out / "report.json"),
            "decision": decision,
            "effects": intervals,
            "sufficiency": sufficient,
            "counts": report["raw_numerators_denominators"],
        },
        indent=2,
    )
)
