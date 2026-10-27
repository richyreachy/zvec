from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
M = R / "models67"
n = 8192
mom = np.fromfile(M / "moments.f64", dtype="<f8").reshape(n, 16)
delta = np.memmap(M / "proxy_deltas.f64", dtype="<f8", mode="r", shape=(n, 4, 768))
errors = mom[:, 5:13].reshape(n, 4, 2)
v1, v2, c = (mom[:, 0], mom[:, 1], mom[:, 2])
w = np.clip((v2 - c) / (v1 + v2 - 2 * c), 0, 1)
norm2 = w * w * v1 + (1 - w) ** 2 * v2 + 2 * w * (1 - w) * c
d = 0.5 * np.sum(delta * delta, axis=2)
mean_error = w[:, None] * errors[:, :, 0] + (1 - w[:, None]) * errors[:, :, 1]
mean_score = d + mean_error
scale = np.sqrt(2 * np.maximum(mean_score, 0) * norm2[:, None] / 768)
positive = scale > 0
assert (mean_error[~positive] <= 0).all()
minimum = errors.min(axis=2)
split = np.load(M / "split.npz")
train = split["train"]
valid = split["valid"]
train_over = float(np.mean(minimum[train] > 0))
fraction_positive = float(positive[train].mean())
standardized = np.divide(
    mean_error, scale, out=np.zeros_like(mean_error), where=positive
)
lam = float(
    np.quantile(
        standardized[train][positive[train]], 1 - train_over / fraction_positive
    )
)
calibrated = mean_error - lam * scale
base = np.mean(minimum[valid] ** 2, axis=1)
new = np.mean(calibrated[valid] ** 2, axis=1)
over_delta = np.mean(calibrated[valid] > 0, axis=1) - np.mean(
    minimum[valid] > 0, axis=1
)
rng = np.random.default_rng(6703)
boot = []
overboot = []
for _ in range(500):
    idx = rng.integers(0, len(valid), len(valid))
    boot.append(float(np.mean(base[idx] - new[idx]) / np.mean(base[idx])))
    overboot.append(float(np.mean(over_delta[idx])))
reference_sigma = np.sqrt(2 * d * 0.5 * (v1 + v2)[:, None] / 768)
tail = []
for gap in (0, 0.5, 1, 2, 3):
    tail.append(
        {
            "gap_in_reference_sigma": gap,
            "minimum_accept_fraction": float(
                np.mean(minimum[valid] <= -gap * reference_sigma[valid])
            ),
            "calibrated_accept_fraction": float(
                np.mean(calibrated[valid] <= -gap * reference_sigma[valid])
            ),
        }
    )
z = NormalDist().inv_cdf(0.75)
gaussian = []
for gap in (0, 0.5, 1, 2, 3):
    cdf = NormalDist().cdf
    gaussian.append(
        {
            "gap_sigma": gap,
            "minimum_accept_probability": 1 - cdf(gap) ** 2,
            "mean_lcb_accept_probability": cdf(z - math.sqrt(2) * gap),
        }
    )
assert all(
    row["mean_lcb_accept_probability"] <= row["minimum_accept_probability"] + 1e-14
    for row in gaussian
)
report = {
    "nonpositive_estimated_distance_cases": int((~positive).sum()),
    "zero_scale_handling": "Keep original fused score;its error is nonpositive;include every case in validation and adjust training quantile mass.",
    "training_targets": len(train),
    "validation_targets": len(valid),
    "lambda_fitted_only_on_training_targets": lam,
    "training_minimum_overestimate_fraction": train_over,
    "validation_minimum_overestimate_fraction": float(np.mean(minimum[valid] > 0)),
    "validation_calibrated_overestimate_fraction": float(
        np.mean(calibrated[valid] > 0)
    ),
    "overestimate_rate_difference_cluster_bootstrap95": np.percentile(
        overboot, [2.5, 97.5]
    ).tolist(),
    "validation_minimum_mse": float(np.mean(base)),
    "validation_calibrated_mse": float(np.mean(new)),
    "relative_validation_mse_gain_vs_minimum": float(
        np.mean(base - new) / np.mean(base)
    ),
    "relative_gain_target_bootstrap95": np.percentile(boot, [2.5, 97.5]).tolist(),
    "validation_calibrated_bias": float(np.mean(calibrated[valid])),
    "synthetic_threshold_decisions": tail,
    "independent_gaussian_model": {
        "same_threshold_false_rejection_rate": 0.25,
        "normal_quantile": z,
        "mean_lcb_mse_over_sigma2": 0.5 * (1 + z * z),
        "minimum_mse_over_sigma2": 1,
        "gap_examples": gaussian,
    },
    "no_online_cooccurrence_or_ranking_verification": True,
    "no_anytime_or_conditional_risk_guarantee": True,
    "geometry_coefficient_cost_not_implemented_online": True,
    "proxy_not_external_query_distribution": True,
    "no_qps_or_recall_claim": True,
}
(R / "results/multianchor67_calibrated.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
logging.info("%s", json.dumps(report, indent=2))
