from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
from normalrisk72 import correlated_risk, predict

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
rng = np.random.default_rng(7273)
a = rng.uniform(-4, 4, 256)
b = rng.uniform(-4, 4, 256)
rho = rng.uniform(-0.95, 0.95, 256)
got = correlated_risk(a, b, rho, 128)
nodes, weights = np.polynomial.legendre.leggauss(512)
expected = []
for aa, bb, rr in zip(a, b, rho, strict=True):
    z = -12 + (aa + 12) * (nodes + 1) / 2

    def phi(x):
        return 0.5 * math.erfc(-float(x) / math.sqrt(2))

    cdf2 = (
        (aa + 12)
        / 2
        * np.sum(
            weights
            * np.exp(-z * z / 2)
            / math.sqrt(2 * math.pi)
            * np.array([phi((bb - rr * t) / math.sqrt(1 - rr * rr)) for t in z])
        )
    )
    expected.append(phi(aa) + phi(bb) - 2 * cdf2)
max_error = float(np.max(np.abs(got - expected)))
assert max_error < 1e-11
rr = np.array([-0.999999, -0.95, -0.5, 0.0, 0.5, 0.95, 0.999999])
angular = correlated_risk(np.zeros_like(rr), np.zeros_like(rr), rr, 128)
assert np.max(np.abs(angular - np.arccos(rr) / math.pi)) < 1e-12
x = np.array(
    [[-1.0, 1.0, -2.0, 2.0]] * 5
    + [
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [-1.0, -1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0, 1.0],
    ]
)
y = np.array(
    [
        [-1.0, 1.0, -2.0, 2.0],
        [1.0, -1.0, 2.0, -2.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [-1.0, -1.0, -1.0, -1.0],
        [-1.0, 1.0, -2.0, 2.0],
        [-1.0, 1.0, -2.0, 2.0],
        [-1.0, 1.0, -2.0, 2.0],
        [-1.0, -1.0, -1.0, -1.0],
    ]
)
p, stats = predict(x, y)
assert np.allclose(p, [0, 1, 0, 0.5, 0.5, 0, 0.5, 0.5, 1], atol=1e-12)
result = {
    "cdf_independent_conditional_integral_cases": 256,
    "max_cdf_absolute_error": max_error,
    "near_perfect_correlation_angular_cases": 7,
    "zero_variance_perfect_correlation_cases": 9,
    "degenerate_predictions": p.tolist(),
    "predict_stats": stats,
    "no_distribution_fit_validation_claim": True,
}
(R / "results/risk72_math_checks.json").write_text(json.dumps(result, indent=2) + "\n")
logging.info("%s", json.dumps(result, indent=2))
logging.info("%s", "CHECK72 COMPLETE")
