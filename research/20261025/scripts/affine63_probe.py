from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
assert "RECOVER62 COMPLETE" in (R / "logs/recover62.log").read_text()
assert not (R / "results/affine63_probe.json").exists()
data = np.load(R / "models61/directional62_pair_scores.npz")
X = data["X"]
Y = data["Y"]
gap = data["gaps"]
train = data["train"]
valid = data["valid"]
n = len(X)
dtype = np.dtype(
    [
        ("ids", "<u4", (4,)),
        ("keys", "<u8", (4,)),
        ("degree", "<u4"),
        ("offsets", "<u4", (3,)),
        ("vectors", "<f4", (4, 768)),
    ]
)
records = np.memmap(
    R / "models61/edges.bin", dtype=dtype, mode="r", offset=24, shape=(n,)
)
C = np.empty(n)
for start in range(0, n, 512):
    v = records["vectors"][start : start + 512].astype(np.float64)
    C[start : start + len(v)] = np.sum(v[:, 0] * (v[:, 1] - v[:, 2]), axis=1)
features = np.stack([X, -C], axis=1)
coef, residuals, rank, singular = np.linalg.lstsq(features[train], Y[train], rcond=None)
assert rank == 2
beta, gamma = map(float, coef)
gradient = features[train].T @ (Y[train] - features[train] @ coef)
assert np.max(np.abs(gradient)) < 1e-08
rows = []
old = None
mask = np.abs(gap[valid]) > 1e-12
for name, b, g in [
    ("baseline", 1.0, 0.0),
    ("directional62", 0.9359792556459329, 0.0),
    ("affine63", beta, gamma),
]:
    error = Y[valid] - b * X[valid] + g * C[valid]
    flips = (np.sign(gap[valid] + error) != np.sign(gap[valid]))[mask]
    mse = error**2
    if old is None:
        old = (mse, flips)
    delta = flips.astype(float) - old[1]
    rng = np.random.default_rng(6301)
    boot = []
    mboot = []
    for _ in range(500):
        ix = rng.integers(len(delta), size=len(delta))
        boot.append(float(np.mean(delta[ix])))
        ix = rng.integers(len(mse), size=len(mse))
        mboot.append(float(1 - np.mean(mse[ix]) / np.mean(old[0][ix])))
    rows.append(
        {
            "name": name,
            "beta": b,
            "gamma": g,
            "gap_mse": float(np.mean(mse)),
            "relative_mse_gain": float(1 - np.mean(mse) / np.mean(old[0])),
            "mse_gain_bootstrap95": np.percentile(mboot, [2.5, 97.5]).tolist(),
            "flip_rate": float(np.mean(flips)),
            "flip_improved": int(np.sum(delta < 0)),
            "flip_worsened": int(np.sum(delta > 0)),
            "flip_difference_bootstrap95": np.percentile(boot, [2.5, 97.5]).tolist(),
            "unit_endpoint_bias_coefficient": 1 - b - 0.5 * g,
        }
    )
summary = {
    "diagnostic_only": True,
    "not_search_recall_or_qps": True,
    "model_fit_train_only": True,
    "train_cases": len(train),
    "validation_cases": len(valid),
    "family": "error=Y-beta*X+gamma*C; C=center.dot(residual_a-residual_b)",
    "normal_equation_max_residual": float(np.max(np.abs(gradient))),
    "design_singular_values": singular.tolist(),
    "rows": rows,
    "no_novelty_claim": True,
    "validation_shared_endpoints_possible": True,
}
(R / "results/affine63_probe.json").write_text(json.dumps(summary, indent=2) + "\n")
np.save(R / "models61/affine63_coefficients.npy", coef)
logging.info("%s", json.dumps(summary, indent=2))
logging.info("%s", "AFFINE63 PROBE COMPLETE")
