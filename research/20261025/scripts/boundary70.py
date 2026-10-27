from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from rank70_math import evaluate
from rotation61_math import forward

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
M = R / "models70"
rows = np.load(M / "train_rows.npy")[:64]
x = np.memmap(M / "rotated.f32", mode="r", dtype="<f4", shape=(8192, 37, 768))
a = np.asarray(x[rows], dtype=np.float64)
r = a[:, 1:33] - a[:, 0, None]
q = a[:, 33:] - a[:, 0, None]
truth = np.memmap(M / "truth.f64", mode="r", dtype="<f8", shape=(8192, 4, 32))[rows]
offset = np.memmap(M / "offsets.f64", mode="r", dtype="<f8", shape=(8192, 32))[rows]
zero = np.zeros((3, 384))
baseline, grad = evaluate(r, q, offset, truth, zero, True)
direction = -grad / np.max(np.abs(grad))
slope = float(np.sum(grad * direction))
norm2 = np.sum(r * r, axis=2)
sign = np.where(r > 0, 1.0, -1.0)
target = truth <= truth.min(axis=2, keepdims=True) + 1e-07
weights = target / target.sum(axis=2, keepdims=True)
results = []
for step in [1e-07, 1e-05, 0.0001, 0.001, 0.003]:
    theta = step * direction
    zr = forward(r.reshape(-1, 768), theta)[0].reshape(r.shape)
    zq = forward(q.reshape(-1, 768), theta)[0].reshape(q.shape)
    l1 = np.sum(zr * sign, axis=2)
    assert np.all(l1 > 0)
    h = (norm2 / l1)[:, :, None] * sign
    scores = offset[:, None, :] - np.einsum("bqd,bkd->bqk", zq, h)
    logits = -scores / 0.01
    logits -= logits.max(axis=2, keepdims=True)
    logprob = logits - np.log(np.exp(logits).sum(axis=2, keepdims=True))
    branch = float(np.mean(-np.sum(weights * logprob, axis=2)))
    actual = evaluate(r, q, offset, truth, theta)[0]
    results.append(
        {
            "max_angle_step": step,
            "linear_predicted_loss_change": step * slope,
            "same_sign_branch_loss_change": branch - baseline,
            "actual_hard_quantizer_loss_change": actual - baseline,
            "residual_coordinates_crossing_zero": int(
                np.sum(np.where(zr > 0, 1.0, -1.0) != sign)
            ),
        }
    )
report = {
    "training_centers": 64,
    "fixed_first_training_rows": True,
    "baseline_crossentropy": baseline,
    "direction_negative_piecewise_gradient_max_coordinate_normalized": True,
    "rows": results,
    "diagnostic_only_no_optimizer_or_parameter_selection": True,
    "does_not_isolate_all_causes_of_failed_training": True,
}
(R / "results/rank70_boundary_probe.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
logging.info("%s", json.dumps(report, indent=2))
logging.info("%s", "BOUNDARY70 COMPLETE")
