from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from coupled73_math import flip_costs, objective, optimize, reconstruct

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
rng = np.random.default_rng(7301)
nchecks = 0
max_relative = 0.0
max_radial = 0.0
pair_error = 0.0
accepted_total = 0
for trial in range(40):
    K = 8 if trial < 32 else 32
    d = [8, 16, 32, 64][trial % 4] if trial < 32 else 768
    p = d if trial % 2 else min(8, d)
    r = rng.normal(size=(K, d)) / np.sqrt(d)
    Q = rng.normal(size=(p, d)) / np.sqrt(p)
    lam = 0.1
    if trial % 7 == 0:
        r[0] = 0
    s = np.where(r > 0, 1.0, -1.0)
    s[rng.random(s.shape) < 0.01] *= -1
    h, alpha = reconstruct(r, s)
    max_radial = max(
        max_radial, float(np.max(np.abs(np.sum(r * h, axis=1) - np.sum(r * r, axis=1))))
    )
    e = h - r
    z = e @ Q.T
    left, right = np.triu_indices(K, 1)
    pair = float(np.sum((z[left] - z[right]) ** 2) / K)
    centered = float(np.sum((z - z.mean(axis=0)) ** 2))
    pair_error = max(pair_error, abs(pair - centered) / max(1, pair))
    before = objective(r, s, Q, lam)
    for i in range(K):
        costs = flip_costs(r, s, Q, lam, i)
        for k in range(d) if d <= 64 else rng.choice(d, 8, replace=False):
            if np.isfinite(costs[k]):
                trial_s = s.copy()
                trial_s[i, k] *= -1
                after = objective(r, trial_s, Q, lam)
                err = abs(after - before - costs[k]) / max(1, abs(before), abs(after))
                max_relative = max(max_relative, err)
                nchecks += 1
    if trial < 12:
        learned, history, accepted = optimize(r, Q, lam, sweeps=2, per_edge_per_sweep=2)
        assert np.all(np.diff(history) < 0)
        accepted_total += len(accepted)
        _, aa = reconstruct(r, learned)
        _, a0 = reconstruct(r, np.where(r > 0, 1.0, -1.0))
        assert np.all(aa <= a0 * 1.02 + 1e-14)
assert max_relative < 1e-10
assert max_radial < 1e-12
assert pair_error < 1e-12
isotropic_checks = 0
for _ in range(100):
    r = rng.normal(size=(8, 32))
    s = np.where(r > 0, 1.0, -1.0)
    Q = np.zeros((1, 32))
    base = objective(r, s, Q, 1.0)
    for i in range(8):
        assert np.min(flip_costs(r, s, Q, 1.0, i)) >= -1e-10
        isotropic_checks += 32
    learned, history, accepted = optimize(r, Q, 1.0)
    assert not accepted
    assert np.array_equal(learned, s)
r = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.5, 0.5 + 1e-12, 0.0], [0.0, 0.0, 0.0, 0.0]])
s = np.where(r > 0, 1.0, -1.0)
Q = np.eye(4)
_, alpha = reconstruct(r, s)
assert np.isinf(flip_costs(r, s, Q, 0.1, 0)[0])
assert np.isinf(flip_costs(r, s, Q, 0.1, 1, alpha[1] * 1.02)[0])
assert np.isinf(flip_costs(r, s, Q, 0.1, 2)).all()
r2 = np.array([[1.0, 0.5, 0.5, 0.0], [1.0, 1.0, 1.0, 1.0]])
assert np.isinf(flip_costs(r2, np.where(r2 > 0, 1.0, -1.0), Q, 0.1, 0)[0])
result = {
    "finite_flip_direct_objective_checks": nchecks,
    "max_relative_objective_delta_error": max_relative,
    "max_radial_identity_error": max_radial,
    "max_pairwise_centered_identity_relative_error": pair_error,
    "accepted_monotone_synthetic_updates": accepted_total,
    "isotropic_single_flip_checks": isotropic_checks,
    "isotropic_optimizer_no_change_cases": 100,
    "zero_negative_tiny_positive_denominators_checked": True,
    "scale_cap_checked": True,
    "no_cohere_or_qps_claim": True,
}
(R / "results/coupled73_math_checks.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
logging.info("%s", json.dumps(result, indent=2))
logging.info("%s", "CHECK73 COMPLETE")
