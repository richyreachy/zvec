from __future__ import annotations

import json
import logging
from pathlib import Path
from statistics import NormalDist

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
rng = np.random.default_rng(6704)
n = 1000000
x = rng.standard_normal((n, 2))
z = NormalDist().inv_cdf(0.75)
minimum = x.min(axis=1)
fused = x.mean(axis=1) - z / np.sqrt(2)
assert abs(np.mean(minimum**2) - 1) < 0.01
assert abs(np.mean(fused**2) - 0.5 * (1 + z * z)) < 0.01
assert abs(np.mean(minimum > 0) - 0.25) < 0.002
assert abs(np.mean(fused > 0) - 0.25) < 0.002
rows = []
for gap in (0, 0.5, 1, 2, 3):
    p1 = 1 - NormalDist().cdf(gap) ** 2
    p2 = NormalDist().cdf(z - np.sqrt(2) * gap)
    observed1 = float(np.mean(minimum <= -gap))
    observed2 = float(np.mean(fused <= -gap))
    assert abs(observed1 - p1) < 0.002
    assert abs(observed2 - p2) < 0.002
    rows.append(
        {
            "gap": gap,
            "minimum_observed": observed1,
            "minimum_formula": p1,
            "fused_observed": observed2,
            "fused_formula": p2,
        }
    )
result = {
    "synthetic_only": True,
    "normal_pair_samples": n,
    "seed": 6704,
    "mse_minimum": float(np.mean(minimum**2)),
    "mse_fused": float(np.mean(fused**2)),
    "tails": rows,
    "not_a_model_validation_on_cohere": True,
}
(R / "results/multianchor67_gaussian_check.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
logging.info("%s", json.dumps(result, indent=2))
