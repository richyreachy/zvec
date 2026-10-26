from __future__ import annotations

import json
import logging
from pathlib import Path

import coupled73_math as old
import coupled75_math as new
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
rng = np.random.default_rng(7500)
checks = 0
max_relative = 0.0
updates = 0
for t in range(24):
    K = 8
    d = [8, 16, 32, 64][t % 4]
    r = rng.normal(size=(K, d)) / np.sqrt(d)
    Q = rng.normal(size=(d if t % 2 else 4, d)) / np.sqrt(d)
    lam = 0.1
    if t % 4 == 0:
        r[0] = 0
    s = np.where(r > 0, 1.0, -1.0)
    s[rng.random(s.shape) < 0.005] *= -1
    for mode in [True, False]:
        before = new.objective(r, s, Q, lam, coupled=mode)
        for i in range(K):
            cost = new.flip_costs(r, s, Q, lam, i, coupled=mode)
            if mode:
                assert np.array_equal(cost, old.flip_costs(r, s, Q, lam, i))
            for k in range(d):
                if np.isfinite(cost[k]):
                    code = s.copy()
                    code[i, k] *= -1
                    after = new.objective(r, code, Q, lam, coupled=mode)
                    max_relative = max(
                        max_relative,
                        abs(after - before - cost[k]) / max(1, abs(before), abs(after)),
                    )
                    checks += 1
        learned, history, moves = new.optimize(r, Q, lam, coupled=mode)
        assert np.all(np.diff(history) < 0)
        updates += len(moves)
        if mode:
            ref = old.optimize(r, Q, lam)
            assert np.array_equal(learned, ref[0])
            assert history == ref[1]
            assert moves == ref[2]
r = rng.normal(size=(8, 32))
Q = np.zeros((1, 32))
for mode in [True, False]:
    s, h, m = new.optimize(r, Q, 1.0, coupled=mode)
    assert not m
    assert np.array_equal(s, np.where(r > 0, 1.0, -1.0))
r = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.5, 0.5 + 1e-12, 0.0], [0.0, 0.0, 0.0, 0.0]])
s = np.where(r > 0, 1.0, -1.0)
Q = np.eye(4)
_, alpha = new.reconstruct(r, s)
for mode in [True, False]:
    assert np.isinf(new.flip_costs(r, s, Q, 0.1, 0, coupled=mode)[0])
    assert np.isinf(new.flip_costs(r, s, Q, 0.1, 1, alpha[1] * 1.02, coupled=mode)[0])
    assert np.isinf(new.flip_costs(r, s, Q, 0.1, 2, coupled=mode)).all()
x = np.memmap(R / "models74/fit.f32", mode="r", dtype="<f4", shape=(2048, 49, 768))
saved = np.memmap(R / "models74/signs.i8", mode="r", dtype="i1", shape=(2048, 32, 768))
entries = []
with (R / "models74/history.jsonl").open() as f:
    for _ in range(8):
        entries.append(json.loads(next(f)))
for i in range(8):
    a = np.asarray(x[i], dtype=np.float64)
    r = a[1:33] - a[0]
    Q = (a[33:] - a[0]) / 4
    lam = float(np.sum(Q * Q) / 768)
    s, h, m = new.optimize(r, Q, lam, coupled=True)
    assert np.array_equal(s, saved[i])
    assert h == entries[i]["objectives"]
    assert [list(t) for t in m] == entries[i]["accepted_sweep_lane_bit"]
assert max_relative < 1e-10
result = {
    "finite_delta_checks_both_objectives": checks,
    "max_relative_error": max_relative,
    "monotone_synthetic_updates": updates,
    "synthetic_coupled_exact_old_matches": 24,
    "frozen_phase74_full_trajectory_and_sign_matches": 8,
    "isotropic_zero_negative_tiny_positive_denominators_both_modes_checked": True,
    "no_new_dataset_validation_used": True,
}
(R / "results/coupled75_math_checks.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
logging.info("%s", json.dumps(result, indent=2))
logging.info("%s", "CHECK75 COMPLETE")
