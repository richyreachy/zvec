from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
from coupled73_math import objective, optimize, reconstruct

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
M = R / "models74"
assert "FITDATA74 COMPLETE" in (R / "logs/fitdata74.log").read_text()
x = np.memmap(M / "fit.f32", mode="r", dtype="<f4", shape=(2048, 49, 768))
for name in ["signs.i8", "alpha.f64", "history.jsonl"]:
    assert not (M / name).exists()
saved = np.memmap(M / "signs.i8", mode="w+", dtype="i1", shape=(2048, 32, 768))
alphas = np.memmap(M / "alpha.f64", mode="w+", dtype="<f8", shape=(2048, 32))
rows = []
started = time.monotonic()
max_radial = 0.0
max_ratio = 1.0
total_updates = 0
with (M / "history.jsonl").open("x") as log:
    for i in range(2048):
        a = np.asarray(x[i], dtype=np.float64)
        r = a[1:33] - a[0]
        Q = (a[33:] - a[0]) / 4
        lam = float(np.sum(Q * Q) / 768)
        native = np.where(r > 0, 1.0, -1.0)
        _, alpha0 = reconstruct(r, native)
        if lam == 0:
            s = native
            history = [objective(r, s, Q, lam)]
            accepted = []
        else:
            s, history, accepted = optimize(
                r, Q, lam, sweeps=2, per_edge_per_sweep=4, scale_cap=1.02
            )
        h, alpha = reconstruct(r, s)
        radial = float(np.max(np.abs(np.sum(r * h, axis=1) - np.sum(r * r, axis=1))))
        ratio = np.divide(alpha, alpha0, out=np.ones_like(alpha), where=alpha0 > 0)
        counts = np.bincount([t[1] for t in accepted], minlength=32)
        assert np.all(np.isin(s, [-1, 1]))
        assert np.all(np.diff(history) < 0)
        assert np.max(counts) <= 8
        assert np.all(ratio <= 1.02 + 1e-12)
        assert radial < 1e-12
        saved[i] = s.astype(np.int8)
        alphas[i] = alpha
        entry = {
            "row": i,
            "lambda": lam,
            "objectives": history,
            "accepted_sweep_lane_bit": accepted,
        }
        log.write(json.dumps(entry, separators=(",", ":")) + "\n")
        max_radial = max(max_radial, radial)
        max_ratio = max(max_ratio, float(ratio.max()))
        total_updates += len(accepted)
        rows.append(
            {
                "row": i,
                "initial_objective": history[0],
                "final_objective": history[-1],
                "updates": len(accepted),
                "hamming_bits": int(np.sum(s != native)),
                "max_scale_ratio": float(ratio.max()),
                "zero_residual_edges": int(np.sum(np.sum(r * r, axis=1) == 0)),
            }
        )
        if (i + 1) % 128 == 0:
            logging.info(
                "%s %s %s %s %s %s",
                "TRAIN74",
                i + 1,
                "updates",
                total_updates,
                "seconds",
                time.monotonic() - started,
            )
saved.flush()
alphas.flush()
summary = {
    "centers": 2048,
    "optimizer_input": "models74/fit.f32:33 center/neighbor vectors plus16 build queries only",
    "no_validation_vectors_or_truth_read": True,
    "config_frozen_in_coupled_code74_plan": True,
    "total_accepted_updates": total_updates,
    "max_radial_error": max_radial,
    "max_scale_ratio": max_ratio,
    "zero_lambda_centers": sum(r["initial_objective"] == 0 for r in rows),
    "elapsed_seconds": time.monotonic() - started,
    "per_center": rows,
    "no_retrieval_qps_claim": True,
}
(R / "results/coupled74_training.json").write_text(json.dumps(summary, indent=2) + "\n")
logging.info("%s", "TRAIN74 COMPLETE")
