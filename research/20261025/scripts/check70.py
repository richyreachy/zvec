from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from rank70_math import evaluate
from rotation61_math import forward

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
rng = np.random.default_rng(7003)
c = rng.normal(size=(3, 1, 768)) * 0.03
v = rng.normal(size=(3, 5, 768)) * 0.03
q = rng.normal(size=(3, 4, 768)) * 0.03
r = v - c
delta = q - c
norm2 = np.sum(r * r, axis=2)
offset = 0.5 * (norm2 - np.sum(v * v, axis=2))
truth = 1 - np.einsum("bqd,bkd->bqk", q, v)
theta = rng.normal(scale=0.03, size=(3, 384))
loss, grad = evaluate(r, delta, offset, truth, theta, True)
maximum = 0
for flat in rng.choice(theta.size, 120, replace=False):
    index = np.unravel_index(flat, theta.shape)
    a = theta.copy()
    b = theta.copy()
    a[index] += 1e-07
    b[index] -= 1e-07
    numeric = (
        evaluate(r, delta, offset, truth, a)[0]
        - evaluate(r, delta, offset, truth, b)[0]
    ) / 2e-07
    error = abs(numeric - grad[index])
    maximum = max(maximum, float(error))
    assert error < 2e-07
x = rng.normal(size=(1000, 768))
y = rng.normal(size=(1000, 768))
xx = forward(x, theta)[0]
yy = forward(y, theta)[0]
norm_error = float(np.max(np.abs(np.sum(x * x, axis=1) - np.sum(xx * xx, axis=1))))
dot_error = float(np.max(np.abs(np.sum(x * y, axis=1) - np.sum(xx * yy, axis=1))))
assert norm_error < 1e-10
assert dot_error < 1e-10
identity_error = float(np.max(np.abs(forward(x, np.zeros_like(theta))[0] - x)))
assert identity_error == 0
r[:, 0] = 0
r[:, 1] = r[:, 2]
delta[:] = 0
offset[:, 0] = 0
offset[:, 1] = offset[:, 2]
truth = np.broadcast_to(offset[:, None, :], (3, 4, 5)).copy()
zero_loss, zero_grad = evaluate(r, delta, offset, truth, theta, True)
assert np.isfinite(zero_loss)
assert np.isfinite(zero_grad).all()
assert np.max(np.abs(zero_grad)) == 0
result = {
    "gradient_probes": 120,
    "max_gradient_abs_error": maximum,
    "norm_cases": 1000,
    "max_norm2_error": norm_error,
    "inner_product_cases": 1000,
    "max_inner_product_error": dot_error,
    "zero_angle_identity_cases": 1000,
    "zero_tie_gradient_check": True,
    "piecewise_gradient_only_no_global_smoothness_claim": True,
}
(R / "results/rank70_math_checks.json").write_text(json.dumps(result, indent=2) + "\n")
logging.info("%s", json.dumps(result, indent=2))
logging.info("%s", "CHECK70 COMPLETE")
