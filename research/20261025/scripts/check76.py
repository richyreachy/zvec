from __future__ import annotations

import ctypes as ct
import json
import logging
from pathlib import Path

import numpy as np
from coupled75_math import flip_costs, objective, optimize, reconstruct

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
lib = ct.CDLL(str(R / "lib/independent76.so"))
F = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
SignArray = np.ctypeslib.ndpointer(dtype=np.int8, flags="C_CONTIGUOUS")
U = np.ctypeslib.ndpointer(dtype=np.uint32, flags="C_CONTIGUOUS")
lib.costs76.argtypes = [
    ct.c_size_t,
    ct.c_size_t,
    F,
    ct.c_double,
    F,
    F,
    ct.c_double,
    F,
    F,
]
lib.optimize76.argtypes = [
    ct.c_size_t,
    ct.c_size_t,
    ct.c_size_t,
    F,
    ct.c_double,
    F,
    SignArray,
    F,
    F,
    U,
    U,
]
lib.error76.restype = ct.c_char_p
rng = np.random.default_rng(7600)
max_delta = 0.0
max_objective = 0.0
flips = 0
trajectories = 0
updates = 0
rejections = 0
for d in [2, 7, 32, 64]:
    for p in [1, 3, d]:
        for case in range(5):
            q = rng.normal(size=(p, d))
            r = rng.normal(size=(3, d))
            s = np.where(r > 0, 1.0, -1.0)
            if case == 1:
                q[:] = 0
            if case == 2:
                q[:] = q[0]
            if case == 3:
                r[0] = 0
            lam = float(np.sum(q * q) / d) if case != 1 else 1.0
            _, a0 = reconstruct(r, s)
            for i in range(3):
                for cap in [np.inf, a0[i] * 1.02]:
                    costs = np.empty(d)
                    obj = np.empty(1)
                    assert (
                        lib.costs76(d, p, q, lam, r[i], s[i], cap, costs, obj) == 0
                    ), lib.error76()
                    ref = flip_costs(r, s, q, lam, i, cap, coupled=False)
                    assert np.array_equal(np.isfinite(costs), np.isfinite(ref))
                    before = objective(
                        r[i : i + 1], s[i : i + 1], q, lam, coupled=False
                    )
                    max_objective = max(
                        max_objective, abs(obj[0] - before) / max(1, abs(before))
                    )
                    for k in np.flatnonzero(np.isfinite(costs)):
                        trial = s[i : i + 1].copy()
                        trial[0, k] *= -1
                        after = objective(r[i : i + 1], trial, q, lam, coupled=False)
                        error = abs(costs[k] - (after - before)) / max(
                            1, abs(before), abs(after)
                        )
                        max_delta = max(max_delta, float(error))
                        assert error < 1e-11
                        flips += 1
            signs = np.empty((3, d), dtype=np.int8)
            alpha = np.empty(3)
            hist = np.empty(25)
            moves = np.empty((24, 3), dtype=np.uint32)
            count = np.zeros(1, dtype=np.uint32)
            assert (
                lib.optimize76(d, p, 3, q, lam, r, signs, alpha, hist, moves, count)
                == 0
            ), lib.error76()
            ref, refhist, refmoves = optimize(r, q, lam, coupled=False)
            n = int(count[0])
            assert np.array_equal(signs, ref)
            assert np.array_equal(
                moves[:n].reshape(-1), np.asarray(refmoves).reshape(-1)
            )
            assert np.allclose(hist[: n + 1], refhist, rtol=1e-11, atol=1e-12)
            assert np.all(np.diff(hist[: n + 1]) < 0)
            assert np.allclose(alpha, reconstruct(r, ref)[1], rtol=1e-12, atol=1e-14)
            trajectories += 1
            updates += n
q = np.eye(2)
r = np.array([1.0, 1.0 - 1e-12])
s = np.array([1.0, -1.0])
cost = np.empty(2)
obj = np.empty(1)
assert lib.costs76(2, 2, q, 1.0, r, s, np.inf, cost, obj) == 0
assert np.isfinite(obj[0])
for rr, ss in [
    (np.array([1.0, 1.0]), s),
    (np.array([1.0, 2.0]), s),
    (r, np.array([1.0, 0.0])),
    (np.array([np.nan, 1.0]), s),
]:
    assert lib.costs76(2, 2, q, 1.0, rr, ss, np.inf, cost, obj) != 0
    rejections += 1
r = rng.normal(size=(3, 32))
q = np.zeros((16, 32))
signs = np.empty((3, 32), dtype=np.int8)
alpha = np.empty(3)
hist = np.empty(25)
moves = np.empty((24, 3), dtype=np.uint32)
count = np.zeros(1, dtype=np.uint32)
assert lib.optimize76(32, 16, 3, q, 0.0, r, signs, alpha, hist, moves, count) == 0
assert count[0] == 0
assert np.array_equal(signs, np.where(r > 0, 1, -1))
a = {
    "seed": 7600,
    "finite_flips_compared_with_direct_objective": flips,
    "max_normalized_finite_delta_error": max_delta,
    "max_normalized_objective_error": max_objective,
    "synthetic_trajectories_matching_python": trajectories,
    "accepted_updates_matching_python": updates,
    "invalid_inputs_rejected": rejections,
    "zero_lambda_native_signs_unchanged": True,
    "positive_tiny_denominator_checked": True,
    "tolerances": {
        "normalized_finite_delta": 1e-11,
        "history_rtol": 1e-11,
        "history_atol": 1e-12,
        "alpha_rtol": 1e-12,
        "alpha_atol": 1e-14,
    },
    "no_retrieval_qps_claim": True,
}
assert max_objective < 1e-11
(R / "results/independent76_math_checks.json").write_text(
    json.dumps(a, indent=2) + "\n"
)
logging.info(json.dumps(a, indent=2))
logging.info("CHECK76 COMPLETE")
