from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
from rotation61_math import forward, local_comparisons, objective

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
M = R / "models61"
started = time.monotonic()
config = {
    "seeds": [61, 62, 63],
    "split_seed": 202610261361,
    "steps": 600,
    "batch_size": 1024,
    "initial_lr": 0.01,
    "final_lr_fraction": 0.1,
    "pair_strides": [1, 16, 384],
    "train_fraction": 0.75,
    "checkpoint_steps": [0, 100, 300, 600],
    "integration_gate": {
        "all_seeds_relative_validation_error_gain_lower95": 0.05,
        "all_seeds_local_flip_increase_max": 0.005,
        "incremental_transform_ns_max": 500,
    },
    "query_or_gt_inputs": [],
    "gate_is_diagnostic_not_breakthrough": True,
}
assert not (M / "training_config.json").exists()
(M / "training_config.json").write_text(json.dumps(config, indent=2) + "\n")
rng = np.random.default_rng(6100)
x = rng.normal(size=(16, 768))
x /= np.linalg.norm(x, axis=1)[:, None]
theta = rng.normal(scale=0.1, size=(3, 384))
value, grad = objective(x, theta, True)
maxgrad = 0
for index in rng.choice(theta.size, 120, replace=False):
    idx = np.unravel_index(index, theta.shape)
    a = theta.copy()
    b = theta.copy()
    a[idx] += 1e-07
    b[idx] -= 1e-07
    numeric = (np.mean(objective(x, a)) - np.mean(objective(x, b))) / 2e-07
    maxgrad = max(maxgrad, abs(numeric - grad[idx]))
    assert abs(numeric - grad[idx]) < 2e-07
z = forward(x, theta)[0]
orth = float(np.max(np.abs(np.sum(z * z, axis=1) - 1)))
assert orth < 2e-14
l1_norm = np.sum(np.abs(z), axis=1)
h = np.where(z > 0, 1.0, -1.0) / l1_norm[:, None]
assert np.max(np.abs(np.sum(z * h, axis=1) - 1)) < 1e-14
assert np.max(np.abs(np.sum((h - z) ** 2, axis=1) - (768 / l1_norm**2 - 1))) < 2e-14
unit = {
    "gradient_probes": 120,
    "max_gradient_abs_error": maxgrad,
    "orthogonality_cases": 16,
    "max_norm2_error": orth,
    "reconstruction_identity_cases": 16,
}
(R / "results/rotation61_math_checks.json").write_text(
    json.dumps(unit, indent=2) + "\n"
)
header = np.fromfile(M / "edges.bin", dtype="<u4", count=6)
n = int(header[2])
assert list(header[[0, 1, 3, 4, 5]]) == [1363625313, 1, 768, 1000000, 3]
dtype = np.dtype(
    [
        ("ids", "<u4", (4,)),
        ("keys", "<u8", (4,)),
        ("degree", "<u4"),
        ("offsets", "<u4", (3,)),
        ("vectors", "<f4", (4, 768)),
    ]
)
assert (M / "edges.bin").stat().st_size == 24 + n * dtype.itemsize
records = np.memmap(M / "edges.bin", dtype=dtype, mode="r", offset=24, shape=(n,))
assert len(np.unique(records["ids"][:, 0])) == n
assert np.all(np.diff(np.sort(records["ids"], axis=1), axis=1) > 0)
assert np.all(records["offsets"] < records["degree"][:, None])
assert np.all(np.diff(np.sort(records["offsets"], axis=1), axis=1) > 0)
base = np.memmap(
    "/home/xingrui/bench/symphony_qg/cohere/original/data/base.f32",
    dtype="<f4",
    mode="r",
    shape=(1000000, 768),
)
maxdiff = 0.0
normerror = 0.0
gaperror = 0.0
residuals = np.memmap(M / "residuals.f32", dtype="<f4", mode="r", shape=(n, 3, 768))
gaps = np.fromfile(M / "local_gaps.f64", dtype="<f8")
assert len(gaps) == n
for i in range(0, n, 512):
    keys = records["keys"][i : i + 512]
    raw = records["vectors"][i : i + 512]
    assert np.all(keys < 1000000)
    assert np.isfinite(raw).all()
    maxdiff = max(maxdiff, float(np.max(np.abs(raw - base[keys]))))
    normerror = max(
        normerror,
        float(np.max(np.abs(np.sum(raw.astype(np.float64) ** 2, axis=2) - 1))),
    )
    exact = np.sum(
        raw[:, 3].astype(np.float64) * (raw[:, 2].astype(np.float64) - raw[:, 1]),
        axis=1,
    )
    gaperror = max(gaperror, float(np.max(np.abs(exact - gaps[i : i + len(raw)]))))
assert maxdiff < 2e-06
assert normerror < 0.0001
assert gaperror < 1e-12
rawx = residuals[:, 0].copy()
norms = np.linalg.norm(rawx.astype(np.float64), axis=1)
good = norms > 1e-09
x = np.divide(
    rawx, norms[:, None], out=np.zeros_like(rawx), where=good[:, None]
).astype(np.float32)
order = np.random.default_rng(config["split_seed"]).permutation(n)
cut = int(0.75 * n)
train_all = order[:cut]
valid_all = order[cut:]
train = train_all[good[train_all]]
valid = valid_all[good[valid_all]]
np.save(M / "train_centers.npy", records["ids"][train, 0])
np.save(M / "validation_centers.npy", records["ids"][valid_all, 0])
baseline = objective(x[valid], np.zeros((3, 384), dtype=np.float32))
valid_res = residuals[valid_all].copy()
valid_gaps = gaps[valid_all]
baseflips, basemse, ties = local_comparisons(
    valid_res, valid_gaps, np.zeros((3, 384), dtype=np.float32)
)
summary = {
    "config": config,
    "samples": n,
    "train_edges": len(train),
    "validation_edges": len(valid),
    "excluded_zero_first_residual": int(np.sum(~good)),
    "base_mapping_max_abs_error": maxdiff,
    "norm2_error_max": normerror,
    "local_gap_error_max": gaperror,
    "local_ties_excluded": ties,
    "baseline_validation_J": float(np.mean(baseline)),
    "baseline_local_flip_rate": float(np.mean(baseflips)),
    "baseline_local_gap_mse": basemse,
    "seeds": [],
}
for seed in config["seeds"]:
    rng = np.random.default_rng(seed)
    theta = np.zeros((3, 384), dtype=np.float32)
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    history = []
    for step in range(601):
        if step in config["checkpoint_steps"]:
            trainloss = float(np.mean(objective(x[train[:8192]], theta)))
            valloss = float(np.mean(objective(x[valid], theta)))
            history.append(
                {"step": step, "train_subset_J": trainloss, "validation_J": valloss}
            )
            logging.info("%s %s %s %s %s", "CHECK", seed, step, trainloss, valloss)
        if step == 600:
            break
        batch = x[rng.choice(train, size=1024, replace=False)]
        loss, grad = objective(batch, theta, True)
        t = step + 1
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad * grad
        lr = 0.01 * (0.1 + 0.9 * (1 - step / 600))
        theta -= lr * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-08)
    theta.astype("<f4").tofile(M / f"angles_seed{seed}.f32")
    np.stack([np.cos(theta), np.sin(theta)], axis=1).astype("<f4").tofile(
        M / f"coefficients_seed{seed}.f32"
    )
    final = objective(x[valid], theta)
    gains = baseline - final
    bootrng = np.random.default_rng(6161)
    boot = []
    for _ in range(500):
        indices = bootrng.integers(0, len(valid), size=len(valid))
        boot.append(float(np.mean(gains[indices]) / np.mean(baseline[indices])))
    flips, mse, unused_ties = local_comparisons(valid_res, valid_gaps, theta)
    assert unused_ties == ties
    result = {
        "seed": seed,
        "history": history,
        "validation_J": float(np.mean(final)),
        "relative_validation_gain": float(np.mean(gains) / np.mean(baseline)),
        "relative_validation_gain_bootstrap95": np.percentile(
            boot, [2.5, 97.5]
        ).tolist(),
        "local_flip_rate": float(np.mean(flips)),
        "local_gap_mse": mse,
        "max_abs_angle": float(np.max(np.abs(theta))),
        "diagnostic_quality_gate_passed": float(np.percentile(boot, 2.5)) >= 0.05
        and float(np.mean(flips)) - float(np.mean(baseflips)) <= 0.005,
    }
    summary["seeds"].append(result)
    (R / "results/rotation61_training.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    logging.info("%s %s", "SEED COMPLETE", json.dumps(result))
summary["elapsed_seconds"] = time.monotonic() - started
summary["source_hashes"] = {
    str(p.relative_to(R)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in [
        M / "edges.bin",
        M / "residuals.f32",
        M / "local_gaps.f64",
        R / "scripts/rotation61_math.py",
        R / "scripts/train61.py",
        M / "training_config.json",
    ]
}
summary["all_seeds_quality_gate_passed"] = all(
    r["diagnostic_quality_gate_passed"] for r in summary["seeds"]
)
summary["no_qps_claim"] = True
(R / "results/rotation61_training.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
logging.info("%s", "TRAIN61 COMPLETE")
