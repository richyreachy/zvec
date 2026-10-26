from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
lam = json.loads((R / "results/multianchor67_calibrated.json").read_text())[
    "lambda_fitted_only_on_training_targets"
]
reports = {}
for mode in ("database", "primary"):
    root = R / ("models68" if mode == "database" else "models68/primary")
    M = root / "geometry"
    p = np.load(M / "sample.npy")
    n = len(p)
    mom = np.fromfile(M / "moments.f64", dtype="<f8").reshape(n, 16)
    e = np.memmap(M / "error_vectors.f64", dtype="<f8", mode="r", shape=(n, 2, 768))
    delta = np.memmap(M / "proxy_deltas.f64", dtype="<f8", mode="r", shape=(n, 768))
    ideal = -np.einsum("nd,nad->na", delta, e)
    assert np.max(np.abs(ideal - mom[:, 5:7])) < 1e-12
    a, b, c = (mom[:, 0], mom[:, 1], mom[:, 2])
    denom = a + b - 2 * c
    w = np.clip(np.divide(b - c, denom, out=np.full(n, 0.5), where=denom > 1e-20), 0, 1)
    norm2 = w * w * a + (1 - w) ** 2 * b + 2 * w * (1 - w) * c
    assert (norm2 <= np.minimum(a, b) + 1e-12).all()
    api = np.column_stack((p["estimate1"] - p["exact"], p["estimate2"] - p["exact"]))
    difference = api - ideal
    rho = np.divide(c, np.sqrt(a * b), out=np.zeros(n), where=a * b > 0)
    fused = w * p["estimate1"] + (1 - w) * p["estimate2"]
    scale = np.sqrt(2 * np.maximum(fused, 0) * np.maximum(norm2, 0) / 768)
    lower = fused - lam * scale
    minimum = np.minimum(p["estimate1"], p["estimate2"])
    source_self = (
        (p["source1"] == p["query_id"]) | (p["source2"] == p["query_id"])
        if mode == "database"
        else np.zeros(n, dtype=bool)
    )
    summary = {
        "pairs": n,
        "queries": len(np.unique(p["query"])),
        "lambda_frozen_from_stage67": lam,
        "source_equals_database_query_pairs": int(source_self.sum()),
        "rho_percentiles": np.percentile(rho, [0, 10, 50, 90, 100]).tolist(),
        "actual_vs_ideal_error_abs_percentiles": np.percentile(
            np.abs(difference), [50, 90, 99, 100]
        ).tolist(),
        "actual_error_correlation": float(np.corrcoef(api.T)[0, 1]),
        "no_online_coefficient_or_policy_implementation": True,
        "strata": {},
    }
    for name, mask in [
        ("all", np.ones(n, dtype=bool)),
        ("sources_not_query", ~source_self),
        ("eventually_expanded", p["expanded"] == 1),
        ("eventually_returned", p["returned"] == 1),
    ]:
        r = {}
        r["pairs"] = int(mask.sum())
        if not mask.any():
            continue
        for label, scores in [
            ("first", p["estimate1"]),
            ("minimum", minimum),
            ("weighted", fused),
            ("calibrated", lower),
        ]:
            error = (scores - p["exact"])[mask]
            r[label] = {
                "mse": float(np.mean(error**2)),
                "bias": float(error.mean()),
                "overestimate_fraction": float(np.mean(error > 0)),
            }
        r["calibrated_mse_gain_vs_minimum"] = (
            1 - r["calibrated"]["mse"] / r["minimum"]["mse"]
        )
        summary["strata"][name] = r
    threshold = p["threshold_at_second"]
    threshold = threshold + 0.02 * np.maximum(threshold, 1e-06)
    eligible = np.isfinite(threshold)
    near = eligible & (p["exact"] <= threshold)
    far = eligible & ~near
    min_accept = minimum <= threshold
    cal_accept = lower <= threshold
    decisions = {
        "finite_threshold_pairs": int(eligible.sum()),
        "true_within_threshold": int(near.sum()),
        "true_outside_threshold": int(far.sum()),
        "minimum_false_reject": int((near & ~min_accept).sum()),
        "calibrated_false_reject": int((near & ~cal_accept).sum()),
        "minimum_false_accept": int((far & min_accept).sum()),
        "calibrated_false_accept": int((far & cal_accept).sum()),
        "previously_accepted_near_now_rejected": int(
            (near & min_accept & ~cal_accept).sum()
        ),
        "previously_rejected_near_now_accepted": int(
            (near & ~min_accept & cal_accept).sum()
        ),
        "threshold_is_unbucketed_snapshot_not_actual_algorithm": True,
    }
    query_n = int(p["query"].max()) + 1
    counts = np.bincount(p["query"][near], minlength=query_n)
    delta_fn = np.bincount(
        p["query"][near & ~cal_accept], minlength=query_n
    ) - np.bincount(p["query"][near & ~min_accept], minlength=query_n)
    rng = np.random.default_rng(6802)
    boot = []
    for _ in range(500):
        idx = rng.integers(0, query_n, query_n)
        boot.append(float(delta_fn[idx].sum() / counts[idx].sum()))
    decisions["false_reject_rate_difference_query_bootstrap95"] = np.percentile(
        boot, [2.5, 97.5]
    ).tolist()
    summary["snapshot_decisions"] = decisions
    v = np.memmap(
        R / "models64/full_vectors.f32",
        dtype="<f4",
        mode="r",
        offset=16,
        shape=(1000000, 768),
    )
    q = (
        np.memmap(root / "queries.f32", dtype="<f4", mode="r", shape=(1000, 768))
        if mode == "primary"
        else v
    )
    max_truth_error = 0
    for lo in range(0, n, 1024):
        pp = p[lo : lo + 1024]
        qq = q[pp["query"] if mode == "primary" else pp["query_id"]].astype(np.float64)
        vv = v[pp["target"]].astype(np.float64)
        truth = 1 - np.sum(qq * vv, axis=1)
        max_truth_error = max(
            max_truth_error, float(np.max(np.abs(truth - pp["exact"])))
        )
    assert max_truth_error < 2e-06
    summary["fp64_reference_distance_max_error"] = max_truth_error
    reports[mode] = summary
(R / "results/trace68_geometry.json").write_text(json.dumps(reports, indent=2) + "\n")
logging.info("%s", json.dumps(reports, indent=2))
