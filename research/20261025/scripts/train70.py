from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
from rank70_math import evaluate

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
M = R / "models70"
config = json.loads((M / "config.json").read_text())
assert "DATA70 COMPLETE" in (R / "logs/validate_data70.log").read_text()
assert "CHECK70 COMPLETE" in (R / "logs/check70.log").read_text()
x = np.memmap(M / "rotated.f32", mode="r", dtype="<f4", shape=(8192, 37, 768))
truth = np.memmap(M / "truth.f64", mode="r", dtype="<f8", shape=(8192, 4, 32))
offsets = np.memmap(M / "offsets.f64", mode="r", dtype="<f8", shape=(8192, 32))
train = np.load(M / "train_rows.npy")
valid = np.load(M / "validation_rows.npy")
started = time.monotonic()


def batch(rows):
    a = np.asarray(x[rows], dtype=np.float64)
    return (
        a[:, 1:33] - a[:, 0, None],
        a[:, 33:] - a[:, 0, None],
        np.asarray(offsets[rows]),
        np.asarray(truth[rows]),
    )


def metrics(rows, theta):
    errors = []
    near_errors = 0
    near_count = 0
    losses = []
    tied = 0
    for start in range(0, len(rows), 64):
        r, q, o, t = batch(rows[start : start + 64])
        loss, scores = evaluate(r, q, o, t, theta)
        losses.append(loss * len(r))
        chosen = np.argmin(scores, axis=2)
        minimum = np.min(t, axis=2)
        actual = np.take_along_axis(t, chosen[:, :, None], axis=2)[:, :, 0]
        errors.extend((actual > minimum + 1e-07).tolist())
        tied += int(np.sum(np.sum(t <= minimum[:, :, None] + 1e-07, axis=2) > 1))
        order = np.argsort(t, axis=2, kind="stable")
        best = order[:, :, :1]
        nexts = order[:, :, 1:5]
        gap = np.take_along_axis(t, nexts, axis=2) - np.take_along_axis(t, best, axis=2)
        estimate = np.take_along_axis(scores, nexts, axis=2) - np.take_along_axis(
            scores, best, axis=2
        )
        eligible = gap > 1e-07
        near_errors += int(np.sum((estimate < 0) & eligible))
        near_count += int(eligible.sum())
    e = np.asarray(errors, dtype=bool)
    return (
        {
            "crossentropy": sum(losses) / len(rows),
            "top1_error_rate": float(e.mean()),
            "top1_errors": int(e.sum()),
            "queries": int(e.size),
            "near_pair_error_rate": near_errors / near_count,
            "near_pair_count": near_count,
            "tied_minimum_queries": tied,
        },
        e,
    )


baseline, base_error = metrics(valid, np.zeros((3, 384)))
np.save(M / "baseline_validation_errors.npy", base_error)
summary = {
    "config": config,
    "baseline_validation": baseline,
    "seeds": [],
    "primary_or_extra_query_files_read": False,
    "no_qps_claim": True,
}
logging.info("%s %s", "BASELINE", json.dumps(baseline))
for seed in config["seeds"]:
    theta = np.zeros((3, 384))
    moment = np.zeros_like(theta)
    variance = np.zeros_like(theta)
    rng = np.random.default_rng(seed)
    history = []
    updates = []
    for step in range(config["steps"]):
        chosen = rng.choice(train, size=config["batch_centers"], replace=False)
        r, q, o, t = batch(chosen)
        before, gradient = evaluate(r, q, o, t, theta, True)
        assert np.isfinite(gradient).all()
        moment = 0.9 * moment + 0.1 * gradient
        variance = 0.999 * variance + 0.001 * gradient * gradient
        j = step + 1
        lr = config["initial_lr"] * (
            config["final_lr_fraction"]
            + (1 - config["final_lr_fraction"]) * (1 - step / config["steps"])
        )
        direction = moment / (1 - 0.9**j) / (np.sqrt(variance / (1 - 0.999**j)) + 1e-08)
        accepted = 0.0
        after = before
        attempts = []
        for factor in config["backtrack_factors"]:
            candidate = theta - lr * factor * direction
            candidate_loss = evaluate(r, q, o, t, candidate)[0]
            assert np.isfinite(candidate_loss)
            attempts.append(candidate_loss)
            if candidate_loss <= before:
                theta = candidate
                accepted = factor
                after = candidate_loss
                break
        updates.append(
            {
                "step": j,
                "before": before,
                "after": after,
                "accepted_factor": accepted,
                "attempted_losses": attempts,
            }
        )
        if j in [1, 100, 300, 600]:
            train_metric, _ = metrics(train[:256], theta)
            history.append(
                {
                    "step": j,
                    "train_fixed_subset": train_metric,
                    "accepted_updates": sum(u["accepted_factor"] > 0 for u in updates),
                    "elapsed_seconds": time.monotonic() - started,
                }
            )
            logging.info("%s %s %s %s", "CHECK", seed, j, json.dumps(history[-1]))
    theta.astype("<f8").tofile(M / f"angles_seed{seed}.f64")
    theta.astype("<f4").tofile(M / f"angles_seed{seed}.f32")
    np.stack([np.cos(theta), np.sin(theta)], axis=1).astype("<f4").tofile(
        M / f"coefficients_seed{seed}.f32"
    )
    final, error = metrics(valid, theta)
    np.save(M / f"validation_errors_seed{seed}.npy", error)
    change = base_error.mean(axis=1) - error.mean(axis=1)
    base = base_error.mean(axis=1)
    boot = []
    bootrng = np.random.default_rng(7070)
    for _i in range(4000):
        sample = bootrng.integers(0, len(valid), len(valid))
        boot.append(float(change[sample].mean() / base[sample].mean()))
    gain = float(change.mean() / base.mean())
    ci = np.quantile(boot, [0.025, 0.975]).tolist()
    passed = gain >= 0.05 and ci[0] > 0
    result = {
        "seed": seed,
        "history": history,
        "validation": final,
        "relative_top1_error_reduction": gain,
        "paired_center_bootstrap95": ci,
        "quality_gate_passed": passed,
        "accepted_updates": sum(u["accepted_factor"] > 0 for u in updates),
        "rejected_updates": sum(u["accepted_factor"] == 0 for u in updates),
        "backtrack_attempts": sum(len(u["attempted_losses"]) for u in updates),
        "max_abs_angle": float(np.max(np.abs(theta))),
    }
    (M / f"updates_seed{seed}.json").write_text(json.dumps(updates, indent=2) + "\n")
    summary["seeds"].append(result)
    (R / "results/rank70_training.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    logging.info("%s %s", "SEED COMPLETE", json.dumps(result))
summary["all_seeds_quality_gate_passed"] = all(
    x["quality_gate_passed"] for x in summary["seeds"]
)
summary["elapsed_seconds"] = time.monotonic() - started
(R / "results/rank70_training.json").write_text(json.dumps(summary, indent=2) + "\n")
logging.info("%s", "TRAIN70 COMPLETE")
