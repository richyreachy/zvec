from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
M = R / "models61"
config = json.loads((M / "training_config.json").read_text())
n = int(np.fromfile(M / "edges.bin", dtype="<u4", count=6)[2])
res = np.memmap(M / "residuals.f32", dtype="<f4", mode="r", shape=(n, 3, 768))
gaps = np.fromfile(M / "local_gaps.f64", dtype="<f8")
X = np.empty(n)
Y = np.empty(n)
for start in range(0, n, 512):
    a = res[start : start + 512, 0].astype(np.float64)
    b = res[start : start + 512, 1].astype(np.float64)
    q = res[start : start + 512, 2].astype(np.float64)
    h = []
    for v in [a, b]:
        norm = np.sum(v * v, axis=1)
        l1_norm = np.sum(np.abs(v), axis=1)
        scale = np.divide(norm, l1_norm, out=np.zeros_like(norm), where=l1_norm > 0)
        h.append(scale[:, None] * np.where(v > 0, 1.0, -1.0))
    X[start : start + len(a)] = np.sum(q * (h[0] - h[1]), axis=1)
    Y[start : start + len(a)] = np.sum(q * (a - b), axis=1)
order = np.random.default_rng(config["split_seed"]).permutation(n)
train = order[: int(0.75 * n)]
valid = order[int(0.75 * n) :]
beta = float(np.sum(X[train] * Y[train]) / np.sum(X[train] ** 2))
clipped = float(np.clip(beta, 0.5, 1))
rows = []
for label, value in [("baseline", 1.0), ("trained_beta", clipped)]:
    errors = Y[valid] - value * X[valid]
    predicted = gaps[valid] + errors
    mask = np.abs(gaps[valid]) > 1e-12
    rows.append(
        {
            "mode": label,
            "beta": value,
            "gap_error_mse": float(np.mean(errors[mask] ** 2)),
            "flip_rate": float(
                np.mean(np.sign(predicted[mask]) != np.sign(gaps[valid][mask]))
            ),
            "local_cases": int(np.sum(mask)),
        }
    )
base = (Y[valid] - X[valid]) ** 2
new = (Y[valid] - clipped * X[valid]) ** 2
rng = np.random.default_rng(6262)
bootstrap = []
for _ in range(500):
    j = rng.integers(0, len(valid), size=len(valid))
    bootstrap.append(float(np.mean(base[j] - new[j]) / np.mean(base[j])))
summary = {
    "diagnostic_only": True,
    "no_search_qps_or_recall_claim": True,
    "model_fit_reads_training_rows_only": True,
    "training_cases": len(train),
    "validation_cases": len(valid),
    "unclipped_beta": beta,
    "beta_clip": [0.5, 1.0],
    "rows": rows,
    "relative_validation_mse_gain": float(np.mean(base - new) / np.mean(base)),
    "relative_gain_bootstrap95": np.percentile(bootstrap, [2.5, 97.5]).tolist(),
    "source": "stage61 actual graph local triples;ideal directional estimator,no query LUT rounding",
    "query_or_gt_inputs": [],
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(R / "results/directional62_probe.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
np.savez(
    M / "directional62_pair_scores.npz", X=X, Y=Y, gaps=gaps, train=train, valid=valid
)
logging.info("%s", json.dumps(summary, indent=2))
