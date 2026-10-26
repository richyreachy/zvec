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
    mom = np.fromfile(M / "moments.f64", dtype="<f8").reshape(-1, 16)
    a = mom[:, 0].astype(np.float16).astype(np.float64)
    b = mom[:, 1].astype(np.float16).astype(np.float64)
    total = a + b
    w = np.divide(b, total, out=np.full(len(p), 0.5), where=total > 0)
    variance = np.divide(a * b, total, out=np.zeros(len(p)), where=total > 0)
    mean = w * p["estimate1"] + (1 - w) * p["estimate2"]
    score = mean - lam * np.sqrt(2 * np.maximum(mean, 0) * variance / 768)
    minimum = np.minimum(p["estimate1"], p["estimate2"])
    fallback = (p["estimate1"] == p["estimate2"]) | (total == 0)
    score[fallback] = minimum[fallback]
    threshold = p["threshold_at_second"]
    threshold = threshold + 0.02 * np.maximum(threshold, 1e-06)
    near = np.isfinite(threshold) & (p["exact"] <= threshold)
    far = np.isfinite(threshold) & ~near
    before = minimum <= threshold
    after = score <= threshold
    qn = int(p["query"].max()) + 1
    cnt = np.bincount(p["query"][near], minlength=qn)
    change = np.bincount(p["query"][near & ~after], minlength=qn) - np.bincount(
        p["query"][near & ~before], minlength=qn
    )
    rng = np.random.default_rng(6803)
    boot = []
    for _ in range(500):
        ix = rng.integers(0, qn, qn)
        boot.append(float(change[ix].sum() / cnt[ix].sum()))
    report = {
        "pairs": len(p),
        "lambda_frozen": lam,
        "covariance_assumed_zero_not_certified": True,
        "metadata": "2 bytes per directed edge FP16 error_norm_squared;64 bytes per32-edge row,not yet integrated",
        "exact_equal_or_zero_variance_fallbacks": int(fallback.sum()),
        "max_error_norm_abs_FP16_rounding": float(
            np.max(np.abs(np.column_stack((a, b)) - mom[:, :2]))
        ),
        "minimum_mse": float(np.mean((minimum - p["exact"]) ** 2)),
        "cheap_mse": float(np.mean((score - p["exact"]) ** 2)),
        "minimum_overestimate_fraction": float(np.mean(minimum > p["exact"])),
        "cheap_overestimate_fraction": float(np.mean(score > p["exact"])),
        "snapshot": {
            "true_within": int(near.sum()),
            "true_outside": int(far.sum()),
            "minimum_false_reject": int((near & ~before).sum()),
            "cheap_false_reject": int((near & ~after).sum()),
            "minimum_false_accept": int((far & before).sum()),
            "cheap_false_accept": int((far & after).sum()),
            "false_reject_rate_change_query_bootstrap95": np.percentile(
                boot, [2.5, 97.5]
            ).tolist(),
        },
        "no_online_policy_or_qps_claim": True,
    }
    reports[mode] = report
(R / "results/trace68_cheap_scale.json").write_text(
    json.dumps(reports, indent=2) + "\n"
)
logging.info("%s", json.dumps(reports, indent=2))
