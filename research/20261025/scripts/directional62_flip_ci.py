from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
data = np.load(R / "models61/directional62_pair_scores.npz")
probe = json.loads((R / "results/directional62_probe.json").read_text())
beta = probe["rows"][1]["beta"]
valid = data["valid"]
train = data["train"]
X = data["X"]
Y = data["Y"]
gap = data["gaps"]
gradient = float(np.sum(X[train] * (Y[train] - beta * X[train])))
assert abs(gradient) < 1e-10 * max(1.0, float(np.sum(X[train] ** 2)))
old = np.sign(gap[valid] + Y[valid] - X[valid]) != np.sign(gap[valid])
new = np.sign(gap[valid] + Y[valid] - beta * X[valid]) != np.sign(gap[valid])
mask = np.abs(gap[valid]) > 1e-12
delta = new[mask].astype(float) - old[mask]
rng = np.random.default_rng(6263)
values = []
for _ in range(500):
    values.append(float(np.mean(delta[rng.integers(0, len(delta), size=len(delta))])))
result = {
    "cases": len(delta),
    "improved_cases": int(np.sum(delta < 0)),
    "worsened_cases": int(np.sum(delta > 0)),
    "flip_rate_difference": float(np.mean(delta)),
    "paired_bootstrap95": np.percentile(values, [2.5, 97.5]).tolist(),
    "training_normal_equation_residual": gradient,
    "not_a_search_recall_test": True,
}
(R / "results/directional62_flip_ci.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
logging.info("%s", json.dumps(result, indent=2))
