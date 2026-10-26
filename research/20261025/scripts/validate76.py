from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
M = R / sys.argv[1]
signs = np.fromfile(M / "signs.i8", dtype="i1").reshape(2048, 32, 768)
alpha = np.fromfile(M / "alpha.f64", dtype="<f8").reshape(2048, 32)
hist = np.fromfile(M / "history.f64", dtype="<f8").reshape(2048, 257)
moves = np.fromfile(M / "moves.u32", dtype="<u4").reshape(2048, 256, 3)
counts = np.fromfile(M / "counts.u32", dtype="<u4")
lam = np.fromfile(M / "lambda.f64", dtype="<f8")
refsigns = np.fromfile(R / "models75/signs_independent.i8", dtype="i1").reshape(
    signs.shape
)
refalpha = np.fromfile(R / "models75/alpha_independent.f64", dtype="<f8").reshape(
    alpha.shape
)
assert np.array_equal(signs, refsigns), (
    f"sign bit differences: {np.sum(signs != refsigns)}"
)
assert np.allclose(alpha, refalpha, rtol=1e-12, atol=1e-14)
max_history = 0.0
max_lambda = 0.0
total = 0
min_step = float("inf")
for row, line in enumerate((R / "models75/history_independent.jsonl").open()):
    ref = json.loads(line)
    n = int(counts[row])
    assert ref["row"] == row
    assert n == len(ref["accepted_sweep_lane_bit"])
    assert np.array_equal(
        moves[row, :n].reshape(-1),
        np.asarray(ref["accepted_sweep_lane_bit"]).reshape(-1),
    ), f"move difference row{row}"
    assert np.all(moves[row, n:] == np.iinfo(np.uint32).max)
    assert np.all(np.isnan(hist[row, n + 1 :]))
    assert np.all(np.diff(hist[row, : n + 1]) < 0)
    assert np.allclose(hist[row, : n + 1], ref["objectives"], rtol=1e-11, atol=1e-12)
    assert np.isclose(lam[row], ref["lambda"], rtol=1e-12, atol=1e-15)
    max_history = max(
        max_history, float(np.max(np.abs(hist[row, : n + 1] - ref["objectives"])))
    )
    max_lambda = max(max_lambda, abs(lam[row] - ref["lambda"]))
    if n:
        min_step = min(min_step, float(np.min(-np.diff(hist[row, : n + 1]))))
    total += n
assert row == 2047
assert total == 524119
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
a = {
    "directory": str(M),
    "centers": 2048,
    "signs_bit_identical": True,
    "all_accepted_positions_and_order_identical": True,
    "total_accepted_updates": total,
    "all_history_steps_strictly_decreasing": True,
    "max_history_absolute_error": max_history,
    "max_lambda_absolute_error": max_lambda,
    "max_alpha_absolute_error": float(np.max(np.abs(alpha - refalpha))),
    "minimum_accepted_objective_decrease": min_step,
    "tolerances": {
        "history_rtol": 1e-11,
        "history_atol": 1e-12,
        "alpha_rtol": 1e-12,
        "alpha_atol": 1e-14,
        "lambda_rtol": 1e-12,
        "lambda_atol": 1e-15,
    },
    "files": {p.name: sha(p) for p in sorted(M.iterdir()) if p.is_file()},
    "no_retrieval_qps_claim": True,
}
(R / "results" / f"{M.name}_checks.json").write_text(json.dumps(a, indent=2) + "\n")
logging.info(json.dumps(a, indent=2))
logging.info("VALIDATE76 COMPLETE")
