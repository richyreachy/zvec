from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
M = R / "models67"
n = 8192
mom = np.fromfile(M / "moments.f64", dtype="<f8").reshape(n, 16)
e = np.memmap(M / "error_vectors.f64", dtype="<f8", mode="r", shape=(n, 2, 768))
delta = np.memmap(M / "proxy_deltas.f64", dtype="<f8", mode="r", shape=(n, 4, 768))
errors = mom[:, 5:13].reshape(n, 4, 2)
check = np.einsum("nqd,nad->nqa", delta, e) * -1
err = float(np.max(np.abs(check - errors)))
assert err < 1e-12
gram = np.einsum("nad,nbd->nab", e, e)
assert np.max(np.abs(gram[:, 0, 0] - mom[:, 0])) < 1e-12
assert np.max(np.abs(gram[:, 1, 1] - mom[:, 1])) < 1e-12
assert np.max(np.abs(gram[:, 0, 1] - mom[:, 2])) < 1e-12
v1, v2, c = (mom[:, 0], mom[:, 1], mom[:, 2])
denom = v1 + v2 - 2 * c
w = np.clip(np.divide(v2 - c, denom, out=np.full(n, 0.5), where=denom > 1e-20), 0, 1)
fused = e[:, 0] * w[:, None] + e[:, 1] * (1 - w[:, None])
norm2 = np.sum(fused * fused, axis=1)
assert (norm2 <= np.minimum(v1, v2) + 1e-12).all()
identity = float(
    np.max(np.abs(norm2 - (w * w * v1 + (1 - w) ** 2 * v2 + 2 * w * (1 - w) * c)))
)
assert identity < 1e-12
d = 0.5 * np.sum(delta * delta, axis=2)
fusion = w[:, None] * errors[:, :, 0] + (1 - w[:, None]) * errors[:, :, 1]
dhat = d + fusion
rad = norm2[:, None]
disc = rad + 2 * dhat
assert disc.min() > -1e-12
lower = dhat + rad - np.sqrt(np.maximum(rad * disc, 0))
upper = dhat + rad + np.sqrt(np.maximum(rad * disc, 0))
assert (lower <= d + 1e-12).all()
assert (upper >= d - 1e-12).all()
for i in range(1000):
    grid = np.linspace(0, 1, 1001)
    objective = grid**2 * v1[i] + (1 - grid) ** 2 * v2[i] + 2 * grid * (1 - grid) * c[i]
    assert norm2[i] <= objective.min() + 1e-12
splits = np.load(M / "split.npz")
train = splits["train"]
valid = splits["valid"]
a = errors[:, :, 0]
b = errors[:, :, 1]
C11 = float(np.mean(a[train] ** 2))
C22 = float(np.mean(b[train] ** 2))
C12 = float(np.mean(a[train] * b[train]))
wg = float(np.clip((C22 - C12) / (C11 + C22 - 2 * C12), 0, 1))
modes = {
    "first": a,
    "second": b,
    "minimum": np.minimum(a, b),
    "average": 0.5 * (a + b),
    "geometry_weighted": fusion,
    "smaller_geometric_variance": np.where((v1 < v2)[:, None], a, b),
    "train_global_mse_weight": wg * a + (1 - wg) * b,
}
rows = []
for name, x in modes.items():
    values = x[valid]
    rows.append(
        {
            "mode": name,
            "validation_mse": float(np.mean(values**2)),
            "bias": float(np.mean(values)),
            "overestimate_fraction": float(np.mean(values > 0)),
            "mae": float(np.mean(np.abs(values))),
        }
    )
rho = np.divide(c, np.sqrt(v1 * v2), out=np.zeros(n), where=v1 * v2 > 0)
base = np.mean(a[valid] ** 2, axis=1)
new = np.mean(fusion[valid] ** 2, axis=1)
rng = np.random.default_rng(6702)
boot = []
for _ in range(500):
    idx = rng.integers(0, len(valid), len(valid))
    boot.append(float(np.mean(base[idx] - new[idx]) / np.mean(base[idx])))
report = {
    "database_targets": n,
    "proxy_queries": n * 4,
    "validation_targets": len(valid),
    "validation_proxy_queries": len(valid) * 4,
    "actual_online_cooccurrence_not_measured": True,
    "no_search_recall_or_qps_claim": True,
    "no_global_recall_or_novelty_claim": True,
    "formula_checks": {
        "error_dot_product_max_difference": err,
        "covariance_quadratic_identity_max_error": identity,
        "norm_bound_cases": n,
        "dense_grid_optimum_cases": 1000,
        "distance_interval_cases": n * 4,
        "all_intervals_contain_exact_distance": True,
        "mean_interval_width": float(np.mean(upper - lower)),
        "mean_exact_half_l2": float(d.mean()),
    },
    "error_direction_correlation_percentiles": dict(
        zip(
            ["min", "p10", "p25", "p50", "p75", "p90", "max"],
            np.percentile(rho, [0, 10, 25, 50, 75, 90, 100]).tolist(),
            strict=True,
        )
    ),
    "mean_geometry_norm2_ratio_vs_first": float(np.mean(norm2 / v1)),
    "geometry_weight_percentiles": np.percentile(w, [0, 10, 50, 90, 100]).tolist(),
    "rows": rows,
    "validation_geometry_mse_gain_vs_first": float(np.mean(base - new) / np.mean(base)),
    "target_cluster_bootstrap95_gain_vs_first": np.percentile(
        boot, [2.5, 97.5]
    ).tolist(),
    "train_global_mse_weight": wg,
    "validation_empirical_error_correlation": float(
        np.corrcoef(a[valid].ravel(), b[valid].ravel())[0, 1]
    ),
    "model_isotropic_mean_both_overestimate_probability": float(
        np.mean(0.25 + np.arcsin(np.clip(rho, -1, 1)) / (2 * np.pi))
    ),
    "observed_validation_both_overestimate_fraction": float(
        np.mean((a[valid] > 0) & (b[valid] > 0))
    ),
    "isotropic_gaussian_model_not_asserted_for_cohere": True,
    "decision_gate": "Do not integrate from MSE alone:need online repeat availability,ranking/tail-risk and stopping-policy tests.",
}
(R / "results/multianchor67_probe.json").write_text(json.dumps(report, indent=2) + "\n")
logging.info("%s", json.dumps(report, indent=2))
