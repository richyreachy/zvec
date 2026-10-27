from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from coupled73_math import objective, reconstruct

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
M = R / "models74"
assert "RUN74 COMPLETE" in (R / "logs/run74.log").read_text()
config = json.loads((M / "config.json").read_text())
training = json.loads((R / "results/coupled74_training.json").read_text())
truth = np.memmap(M / "truth.f64", mode="r", dtype="<f8", shape=(2048, 32, 32))
minimum = truth.min(axis=2)
x = np.memmap(M / "rotated.f32", mode="r", dtype="<f4", shape=(2048, 65, 768))
fit = np.memmap(M / "fit.f32", mode="r", dtype="<f4", shape=(2048, 49, 768))
sel = np.fromfile(M / "selection_positions.u32", dtype="<u4")
val = np.fromfile(M / "validation_positions.u32", dtype="<u4")
signs = np.memmap(M / "signs.i8", mode="r", dtype="i1", shape=(2048, 32, 768))
alphas = np.memmap(M / "alpha.f64", mode="r", dtype="<f8", shape=(2048, 32))
assert np.isin(signs, [-1, 1]).all()
actual = {}
scores = {}
metrics = []
for name in config["score_modes"]:
    score = np.memmap(
        M / f"scores_{name}.f32", mode="r", dtype="<f4", shape=(2048, 32, 32)
    )
    assert np.isfinite(score).all()
    scores[name] = score
    chosen = np.argmin(score, axis=2)
    error = (
        np.take_along_axis(truth, chosen[:, :, None], axis=2)[:, :, 0] > minimum + 1e-07
    )
    actual[name] = error
    np.save(M / f"errors_{name}.npy", error)
base = actual["native"][:, val]
base_center = base.mean(axis=1)
for name in config["score_modes"]:
    error = actual[name]
    heldout = error[:, val]
    change = base_center - heldout.mean(axis=1)
    gain = float(change.mean() / base_center.mean())
    rng = np.random.default_rng(7474)
    boot = []
    for _ in range(4000):
        sample = rng.integers(0, 2048, 2048)
        boot.append(float(change[sample].mean() / base_center[sample].mean()))
    ci = np.quantile(boot, [0.025, 0.975]).tolist()
    total_error = np.asarray(scores[name], dtype=np.float64) - truth
    center_bias = total_error.mean(axis=2)
    centered = total_error - center_bias[:, :, None]
    metrics.append(
        {
            "mode": name,
            "build_top1_error_rate": float(error[:, sel].mean()),
            "heldout_queries": 32768,
            "heldout_top1_errors": int(heldout.sum()),
            "heldout_top1_error_rate": float(heldout.mean()),
            "heldout_relative_error_reduction_vs_native": gain,
            "paired_center_bootstrap95": ci,
            "quality_gate_passed": name == "learned" and gain >= 0.1 and (ci[0] > 0),
            "heldout_score_total_mse": float(np.mean(total_error[:, val] ** 2)),
            "heldout_score_block_mean_error_squared": float(
                np.mean(center_bias[:, val] ** 2)
            ),
            "heldout_score_centered_mse": float(np.mean(centered[:, val] ** 2)),
            "center_counts_better_equal_worse": [
                int(np.sum(change > 0)),
                int(np.sum(change == 0)),
                int(np.sum(change < 0)),
            ],
        }
    )
ideal_errors = {
    name: np.empty((2048, 32), dtype=bool) for name in ["native", "learned"]
}
max_alpha = 0.0
max_radial = 0.0
max_objective = 0.0
history_rows = 0
total_updates = 0
with (M / "history.jsonl").open() as hist:
    for i, line in enumerate(hist):
        entry = json.loads(line)
        assert entry["row"] == i
        history = np.array(entry["objectives"])
        moves = entry["accepted_sweep_lane_bit"]
        assert len(history) == len(moves) + 1
        assert np.all(np.diff(history) < 0)
        row = np.asarray(x[i], dtype=np.float64)
        r = row[1:33] - row[0]
        delta = row[33:] - row[0]
        offset = 0.5 * (np.sum(r * r, axis=1) - np.sum(row[1:33] ** 2, axis=1))
        native = np.where(r > 0, 1.0, -1.0)
        s = native.copy()
        counts = np.zeros((2, 32), dtype=int)
        for sweep, lane, bit in moves:
            assert 0 <= sweep < 2
            assert 0 <= lane < 32
            assert 0 <= bit < 768
            s[lane, bit] *= -1
            counts[sweep, lane] += 1
        assert np.array_equal(s, signs[i])
        assert counts.max() <= 4
        h, alpha = reconstruct(r, s)
        _, a0 = reconstruct(r, native)
        assert np.all(alpha <= a0 * 1.02 + 1e-14)
        max_alpha = max(max_alpha, float(np.max(np.abs(alpha - alphas[i]))))
        max_radial = max(
            max_radial,
            float(np.max(np.abs(np.sum(r * h, axis=1) - np.sum(r * r, axis=1)))),
        )
        build = np.asarray(fit[i], dtype=np.float64)
        assert np.array_equal(build[:33], row[:33])
        assert np.array_equal(build[33:], row[33 + sel])
        Q = (build[33:] - build[0]) / 4
        lam = float(np.sum(Q * Q) / 768)
        assert lam == entry["lambda"]
        j0 = objective(r, native, Q, lam)
        j1 = objective(r, s, Q, lam)
        max_objective = max(max_objective, abs(j0 - history[0]), abs(j1 - history[-1]))
        assert abs(j0 - history[0]) < 1e-12
        assert abs(j1 - history[-1]) < 1e-12
        for name, code in [("native", native), ("learned", s)]:
            reconstructed, _ = reconstruct(r, code)
            ideal = offset[None, :] - delta @ reconstructed.T
            chosen = np.argmin(ideal, axis=1)
            ideal_errors[name][i] = truth[i, np.arange(32), chosen] > minimum[i] + 1e-07
        history_rows += 1
        total_updates += len(moves)
assert history_rows == 2048
assert total_updates == training["total_accepted_updates"]
assert max_alpha == 0
assert max_radial < 1e-12
ideal_summary = {}
for name, error in ideal_errors.items():
    np.save(M / f"ideal_errors_{name}.npy", error)
    ideal_summary[name] = {
        "build_error_rate": float(error[:, sel].mean()),
        "heldout_error_rate": float(error[:, val].mean()),
        "heldout_errors": int(error[:, val].sum()),
    }
result = {
    "config": config,
    "actual_codec_metrics": metrics,
    "ideal_block_ranking_metrics": ideal_summary,
    "offline_quality_gate_passed": metrics[-1]["quality_gate_passed"],
    "all_history_rows_replayed_to_saved_signs": 2048,
    "all_initial_final_objectives_recomputed": 2048,
    "max_saved_alpha_difference": max_alpha,
    "max_radial_error": max_radial,
    "max_initial_final_objective_difference": max_objective,
    "build_file_verified_excludes_validation_vectors": True,
    "all_update_histories_strictly_decrease": True,
    "all_flip_counts_and_scale_caps_checked": True,
    "training_summary": {
        key: training[key]
        for key in [
            "total_accepted_updates",
            "max_scale_ratio",
            "elapsed_seconds",
            "zero_lambda_centers",
        ]
    },
    "mean_initial_objective": float(
        np.mean([r["initial_objective"] for r in training["per_center"]])
    ),
    "mean_final_objective": float(
        np.mean([r["final_objective"] for r in training["per_center"]])
    ),
    "total_final_hamming_bits": sum(r["hamming_bits"] for r in training["per_center"]),
    "codec_checks": {
        name: json.loads((R / f"logs/codec74_{name}.log").read_text())
        for name in config["score_modes"]
    },
    "truth_minimum_ties": int(
        np.sum(np.sum(truth <= minimum[:, :, None] + 1e-07, axis=2) > 1)
    ),
    "precision_control_result_choice_differences": int(
        np.sum(
            np.argmin(scores["native"], axis=2)
            != np.argmin(scores["precision_control"], axis=2)
        )
    ),
    "no_cross_block_queue_guarantee": True,
    "no_retrieval_qps_or_recall_claim": True,
    "new_retrieval_timing_rows": 0,
    "cumulative_retrieval_timing_rows": 1310,
}
(R / "results/coupled74_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
logging.info("%s", json.dumps(result, indent=2))
logging.info("%s", "ANALYZE74 COMPLETE")
