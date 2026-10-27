from __future__ import annotations

import json
import logging
from itertools import pairwise
from pathlib import Path

import numpy as np
from normalrisk72 import predict

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
M = R / "models72"
assert "RUN72 COMPLETE" in (R / "logs/run72.log").read_text()
assert "CHECK72 COMPLETE" in (R / "logs/check72_final.log").read_text()
config = json.loads((M / "config.json").read_text())
names = config["models"]
ids = np.fromfile(M / "ids.u32", dtype="<u4").reshape(2048, 65)
v = np.memmap(
    R / "models64/full_vectors.f32",
    mode="r",
    dtype="<f4",
    offset=16,
    shape=(1000000, 768),
)
truth = np.memmap(M / "truth.f64", mode="r", dtype="<f8", shape=(2048, 32, 32))
sel = np.fromfile(M / "selection_positions.u32", dtype="<u4")
val = np.fromfile(M / "validation_positions.u32", dtype="<u4")
left, right = np.triu_indices(32, 1)
pairs = np.empty((2048, 2, 16, 2), dtype="<u4")
pairs[:, 0] = np.arange(32, dtype=np.uint32).reshape(16, 2)
for i in range(2048):
    vectors = np.asarray(v[ids[i, 1:33]], dtype=np.float64)
    difference = vectors[left] - vectors[right]
    distance = np.sum(difference * difference, axis=1)
    order = np.lexsort((right, left, distance))[:16]
    pairs[i, 1, :, 0] = left[order]
    pairs[i, 1, :, 1] = right[order]
pairs.tofile(M / "pairs.u32")


def gaps(scores):
    return np.stack(
        [
            np.take_along_axis(scores, pairs[:, None, f, :, 0], axis=2)
            - np.take_along_axis(scores, pairs[:, None, f, :, 1], axis=2)
            for f in range(2)
        ],
        axis=1,
    ).transpose(0, 1, 3, 2)


xbase = gaps(truth)
x = np.broadcast_to(xbase, (4, *xbase.shape))
ys = []
for name in names:
    scores = np.fromfile(M / f"codec_scores_{name}.f32", dtype="<f4").reshape(
        2048, 32, 32
    )
    assert np.isfinite(scores).all()
    ys.append(gaps(scores.astype(np.float64)))
y = np.stack(ys)
flips = x * y < 0
fit_x = x[..., sel]
fit_y = y[..., sel]
p, stats = predict(fit_x, fit_y)
np.save(M / "gaussian_predictions.npy", p)
fit_flips = flips[..., sel].mean(axis=-1)
smoothed = (flips[..., sel].sum(axis=-1) + 0.5) / 17
observed = flips[..., val].mean(axis=-1)
assert p.shape == (4, 2048, 2, 16)
assert np.isfinite(p).all()
np.save(M / "heldout_flip_rates.npy", observed)
np.save(M / "empirical_predictions.npy", fit_flips)
np.save(M / "jeffreys_predictions.npy", smoothed)
changed = y.copy()
changed[..., val] = -changed[..., val] + 123
assert np.array_equal(changed[..., sel], fit_y)
pr2, st2 = predict(x[..., sel], changed[..., sel])
assert np.array_equal(pr2, p)
assert stats == st2


def brier(pr):
    return pr * pr + (1 - 2 * pr) * observed


briers = {
    "gaussian": brier(p),
    "empirical": brier(fit_flips),
    "jeffreys": brier(smoothed),
}
families = []
all_gates = []
edges = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 1.000000001]
for f, family in enumerate(config["pair_families"]):
    comparison = []
    for baseline in ["empirical", "jeffreys"]:
        base = briers[baseline][:, :, f, :].mean(axis=(0, 2))
        change = base - briers["gaussian"][:, :, f, :].mean(axis=(0, 2))
        gain = float(change.mean() / base.mean())
        rng = np.random.default_rng(7272)
        boot = []
        for _ in range(4000):
            sample = rng.integers(0, 2048, 2048)
            boot.append(float(change[sample].mean() / base[sample].mean()))
        ci = np.quantile(boot, [0.025, 0.975]).tolist()
        gate = gain >= 0.05 and ci[0] > 0
        all_gates.append(gate)
        comparison.append(
            {
                "baseline": baseline,
                "relative_brier_reduction": gain,
                "paired_center_bootstrap95": ci,
                "quality_gate_passed": gate,
            }
        )
    calibration = []
    for lo, hi in pairwise(edges):
        mask = (p[:, :, f, :] >= lo) & (p[:, :, f, :] < hi)
        count = int(mask.sum())
        calibration.append(
            {
                "low": lo,
                "high": min(hi, 1.0),
                "pair_models": count,
                "predicted": float(p[:, :, f, :][mask].mean()) if count else None,
                "observed": float(observed[:, :, f, :][mask].mean()) if count else None,
            }
        )
    families.append(
        {
            "family": family,
            "heldout_flip_rate": float(observed[:, :, f, :].mean()),
            "predicted_gaussian_flip_rate": float(p[:, :, f, :].mean()),
            "brier": {name: float(z[:, :, f, :].mean()) for name, z in briers.items()},
            "comparisons": comparison,
            "calibration": calibration,
            "per_model_brier": {
                name: {
                    method: float(z[i, :, f, :].mean()) for method, z in briers.items()
                }
                for i, name in enumerate(names)
            },
        }
    )
bounds = []
for gamma in config["margin_grid"]:
    fit_bound = np.minimum(
        1.0,
        np.mean(np.abs(fit_x) < gamma, axis=-1)
        + np.mean((fit_y - fit_x) ** 2, axis=-1) / (gamma * gamma),
    )
    validation_bound = np.minimum(
        1.0,
        np.mean(np.abs(x[..., val]) < gamma, axis=-1)
        + np.mean((y[..., val] - x[..., val]) ** 2, axis=-1) / (gamma * gamma),
    )
    assert np.all(validation_bound + 1e-12 >= observed)
    for f, family in enumerate(config["pair_families"]):
        bounds.append(
            {
                "gamma": gamma,
                "family": family,
                "selection_bound_mean": float(fit_bound[:, :, f, :].mean()),
                "selection_bound_below1_fraction": float(
                    np.mean(fit_bound[:, :, f, :] < 1)
                ),
                "selection_bound_below_observed_validation_rate_fraction": float(
                    np.mean(fit_bound[:, :, f, :] + 1e-12 < observed[:, :, f, :])
                ),
                "validation_empirical_bound_mean": float(
                    validation_bound[:, :, f, :].mean()
                ),
                "validation_empirical_bound_below1_fraction": float(
                    np.mean(validation_bound[:, :, f, :] < 1)
                ),
                "validation_empirical_bound_above_flip_rate_checked": True,
            }
        )
result = {
    "config": config,
    "families": families,
    "all_prediction_gates_passed": all(all_gates),
    "gaussian_numerics": stats,
    "selection_validation_input_isolation_checked": True,
    "geometry_pair_selection_uses_queries": False,
    "true_zero_gaps": int(np.sum(x == 0)),
    "actual_codec_zero_gaps": int(np.sum(y == 0)),
    "strict_flip_labels_true_gap_times_codec_gap_negative": True,
    "paired_center_bootstrap_not_iid_guarantee": True,
    "margin_bounds": bounds,
    "margin_bounds_use_total_actual_codec_error": True,
    "new_retrieval_timing_rows": 0,
    "cumulative_retrieval_timing_rows": 1310,
    "no_full_search_recall_or_qps_claim": True,
    "no_new_quantizer_training_or_frame_selection": True,
}
(R / "results/risk72_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
logging.info("%s", json.dumps(result, indent=2))
logging.info("%s", "ANALYZE72 COMPLETE")
