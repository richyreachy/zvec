from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
SQRT2 = math.sqrt(2)


def phi(x):
    return 0.5 * math.erfc(-float(x) / SQRT2)


def bivar_cdf(a, b, rho, order):
    assert abs(rho) < 1
    assert -10 < a < 10
    nodes, weights = np.polynomial.legendre.leggauss(order)
    z = -12 + (a + 12) * (nodes + 1) / 2
    conditional = np.array([phi((b - rho * t) / math.sqrt(1 - rho * rho)) for t in z])
    return float(
        (a + 12)
        / 2
        * np.sum(weights * np.exp(-z * z / 2) / math.sqrt(2 * math.pi) * conditional)
    )


def flip_risk(ma, mb, sa, sb, rho, order=256):
    a = -ma / sa
    b = -mb / sb
    return phi(a) + phi(b) - 2 * bivar_cdf(a, b, rho, order)


rng = np.random.default_rng(7172)
cases = []
max_identity = 0.0
max_quad = 0.0
for i in range(24):
    d = 32
    c = rng.normal(size=d) * 0.08
    r = rng.normal(size=(2, d))
    r *= np.array([0.7, 1.3])[:, None] / np.linalg.norm(r, axis=1)[:, None]
    U = np.linalg.qr(rng.normal(size=(d, d)))[0]
    z = r @ U.T
    h = (
        (np.sum(r * r, axis=1) / np.sum(np.abs(z), axis=1))[:, None]
        * np.where(z > 0, 1.0, -1.0)
        @ U
    )
    v = c + r
    a = r[0] - r[1]
    b = h[0] - h[1]
    t = 0.5 * (r[0] @ r[0] - r[1] @ r[1] - v[0] @ v[0] + v[1] @ v[1])
    mu = rng.normal(size=d) * 0.13
    A = rng.normal(size=(d, d)) / math.sqrt(d)
    sigma = A @ A.T + 0.1 * np.eye(d)
    L = np.linalg.cholesky(sigma)
    sa = math.sqrt(a @ sigma @ a)
    sb = math.sqrt(b @ sigma @ b)
    rho = float(a @ sigma @ b / (sa * sb))
    ma = float(t - mu @ a)
    mb = float(t - mu @ b)
    p = flip_risk(ma, mb, sa, sb, rho)
    p2 = flip_risk(ma, mb, sa, sb, rho, 512)
    max_quad = max(max_quad, abs(p - p2))
    delta = rng.normal(size=(100000, d)) @ L.T + mu
    exact = t - delta @ a
    approx = t - delta @ b
    empirical = float(np.mean((exact < 0) != (approx < 0)))
    se = math.sqrt(p * (1 - p) / len(delta))
    assert abs(empirical - p) < 6 * se + 1 / len(delta)
    err = -(delta @ (b - a))
    max_identity = max(max_identity, float(np.max(np.abs(approx - exact - err))))
    cosine = (1 - (delta + c) @ v[0]) - (1 - (delta + c) @ v[1])
    max_identity = max(max_identity, float(np.max(np.abs(cosine - exact))))
    cases.append(
        {
            "case": i,
            "rho": rho,
            "mean_exact": ma,
            "mean_quantized": mb,
            "analytic_flip_probability": p,
            "empirical_flip_probability": empirical,
            "samples": len(delta),
            "absolute_error": abs(p - empirical),
            "binomial_standard_error": se,
        }
    )
# The zero-mean zero-threshold special case reduces to acos(rho)/pi.
angular = []
for rho in [-0.95, -0.5, 0.0, 0.5, 0.95]:
    p = flip_risk(0.0, 0.0, 1.0, 1.0, rho, 512)
    expected = math.acos(rho) / math.pi
    assert abs(p - expected) < 1e-11
    angular.append({"rho": rho, "numeric": p, "acos": expected})
assert max_quad < 1e-11
assert max_identity < 1e-12
result = {
    "conditional_gaussian_pair_formula_cases": cases,
    "zero_threshold_angular_cases": angular,
    "max_quadrature_256_512_difference": max_quad,
    "max_pair_and_cosine_identity_error": max_identity,
    "total_gaussian_queries": 2400000,
    "covariance_positive_definite_by_construction": True,
    "unequal_radius_and_nonzero_mean_cases": 24,
    "degenerate_zero_variance_cases_excluded": True,
    "actual_cohere_gaussian_assumption_tested": False,
    "formula_not_claimed_novel": True,
    "no_qps_or_recall_claim": True,
}
(R / "results/local71_risk_math.json").write_text(json.dumps(result, indent=2) + "\n")
logging.info("%s", json.dumps(result, indent=2))
logging.info("%s", "RISK71 MATH COMPLETE")
