from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
rng = np.random.default_rng(6607)


def reconstruction(r):
    norm2 = np.sum(r * r, axis=-1, keepdims=True)
    l1 = np.sum(np.abs(r), axis=-1, keepdims=True)
    return np.divide(norm2, l1, out=np.zeros_like(norm2), where=l1 > 0) * np.where(
        r > 0, 1.0, -1.0
    )


def pair_metrics(residuals):
    h = reconstruction(residuals)
    e = h - residuals
    a = residuals[1] - residuals[0]
    b = h[1] - h[0]
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return None
    rho = float(np.clip(a @ b / denominator, -1, 1))
    return {
        "individual_error2": np.sum(e * e, axis=1).tolist(),
        "pair_error2": float(np.sum((e[1] - e[0]) ** 2)),
        "correlation": rho,
        "gaussian_equal_radius_rank_disagreement": float(np.arccos(rho) / np.pi),
        "endpoint_max": float(np.max(np.abs(np.sum(residuals * e, axis=1)))),
    }


example = None
for trial in range(10000):
    residuals = rng.normal(size=(2, 8))
    residuals /= np.linalg.norm(residuals, axis=1)[:, None]
    rotation, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    rotated = residuals @ rotation.T
    a = pair_metrics(residuals)
    b = pair_metrics(rotated)
    if a is None or b is None:
        continue
    if (
        all(
            (
                y < x * 0.9
                for x, y in zip(
                    a["individual_error2"], b["individual_error2"], strict=True
                )
            )
        )
        and b["gaussian_equal_radius_rank_disagreement"]
        > a["gaussian_equal_radius_rank_disagreement"] + 0.03
    ):
        example = {
            "trial": trial,
            "residuals": residuals.tolist(),
            "rotation": rotation.tolist(),
            "original": a,
            "rotated": b,
        }
        break
assert example is not None
assert np.max(np.abs(rotation.T @ rotation - np.eye(8))) < 1e-12
assert a["endpoint_max"] < 1e-12
assert b["endpoint_max"] < 1e-12
max_identity = 0.0
max_endpoint = 0.0
for _batch in range(16):
    c = rng.normal(size=(256, 768))
    vectors = rng.normal(size=(256, 2, 768))
    q = rng.normal(size=(256, 768))
    r = vectors - c[:, None, :]
    h = reconstruction(r)
    e = h - r
    delta = q - c
    exact = 0.5 * np.sum((q[:, None, :] - vectors) ** 2, axis=2)
    estimate = (
        0.5 * np.sum(delta * delta, axis=1)[:, None]
        + 0.5 * np.sum(r * r, axis=2)
        - np.einsum("bd,bkd->bk", delta, h)
    )
    expected = -np.einsum("bd,bd->b", delta, e[:, 1] - e[:, 0])
    observed = estimate[:, 1] - estimate[:, 0] - (exact[:, 1] - exact[:, 0])
    max_identity = max(max_identity, float(np.max(np.abs(observed - expected))))
    max_endpoint = max(max_endpoint, float(np.max(np.abs(np.sum(r * e, axis=2)))))
assert max_identity < 1e-10
assert max_endpoint < 1e-10
queries = rng.normal(size=(200000, 8))
metrics = []
for mode, matrix in [("original", np.eye(8)), ("rotated", rotation)]:
    transformed = residuals @ matrix.T
    h = reconstruction(transformed)
    q = queries @ matrix.T
    exact = 0.5 * np.sum(transformed * transformed, axis=1) - q @ transformed.T
    estimate = 0.5 * np.sum(transformed * transformed, axis=1) - q @ h.T
    wrong = (exact[:, 1] - exact[:, 0]) * (estimate[:, 1] - estimate[:, 0]) < 0
    observed = float(wrong.mean())
    analytic = example[mode]["gaussian_equal_radius_rank_disagreement"]
    se = np.sqrt(analytic * (1 - analytic) / len(q))
    assert abs(observed - analytic) < 6 * se + 0.0001
    metrics.append(
        {
            "mode": mode,
            "queries": len(q),
            "empirical_rank_disagreement": observed,
            "analytic_rank_disagreement": analytic,
            "binomial_standard_error": float(se),
        }
    )
report = {
    "seed": 6607,
    "onebit_reconstruction_actual_formula": True,
    "counterexample_equal_residual_radius": 1.0,
    "counterexample": example,
    "shared_anchor_identity_cases": 4096,
    "shared_anchor_identity_max_error": max_identity,
    "endpoint_identity_max_error": max_endpoint,
    "gaussian_checks": metrics,
    "not_cohere_query_model": True,
    "not_a_novelty_claim": True,
    "no_online_speed_or_recall_claim": True,
}
(R / "results/graph66_rank_math.json").write_text(json.dumps(report, indent=2) + "\n")
logging.info("%s", json.dumps(report, indent=2))
logging.info("%s", "MATH66 COMPLETE")
