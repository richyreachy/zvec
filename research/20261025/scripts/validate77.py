from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
from coupled75_math import optimize

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
M = R / "models77"
out = R / sys.argv[1]
scope = sys.argv[2]
assert scope in ["pilot", "full"]
N = 1000000
count = 8192 if scope == "pilot" else N
model_filename = sys.argv[3] if len(sys.argv) > 3 else "blocks.partial"
assert (out / model_filename).stat().st_size == 4096 + count * 3600
rot = np.memmap(M / "rotated.f32", mode="r", dtype="<f4", shape=(N, 768))
proxy = np.memmap(M / "proxies.u32", mode="r", dtype="<u4", shape=(N, 16))
g = np.memmap(
    R / "models64/full_graph.bin",
    mode="r",
    dtype=np.dtype([("key", "<u8"), ("degree", "<u4"), ("ids", "<u4", (32,))]),
    offset=32,
    shape=(N,),
)
audit = np.memmap(
    out / "audit.bin",
    mode="r",
    dtype=np.dtype([("updates", "<u4"), ("flags", "<u4"), ("values", "<f8", (7,))]),
    shape=(count,),
)
history = np.memmap(out / "history.f64", mode="r", dtype="<f8", shape=(count, 257))
moves = np.memmap(out / "moves.u16", mode="r", dtype="<u2", shape=(count, 256, 3))
compact = np.memmap(out / "compact.u8", mode="r", dtype="u1", shape=(count, 32, 96))
for name, size in [
    ("audit.bin", count * 64),
    ("history.f64", count * 257 * 8),
    ("moves.u16", count * 256 * 3 * 2),
    ("compact.u8", count * 3072),
]:
    assert (out / name).stat().st_size == size
full = np.flatnonzero(g["degree"][:count] == 32)
short = np.flatnonzero(g["degree"][:count] < 32)
assert np.all(audit["flags"][full] == 1)
assert np.all(audit["flags"][short] == 0)
assert np.all(audit["updates"] <= 256)
assert np.all(audit["updates"][short] == 0)
assert np.all(np.isfinite(audit["values"]))
assert np.all(audit["values"] >= 0)
assert np.array_equal(audit["values"][:, 6], g["degree"][:count])
assert np.all(audit["values"][:, 3] <= 1.02 + 1e-12)
assert np.all(audit["values"][:, 4] < 1e-12)
assert np.all(audit["values"][:, 5] < 1e-05)
assert np.all(audit["values"][full, 2] <= audit["values"][full, 1])
for start in range(0, count, 4096):
    rows = np.arange(start, min(start + 4096, count))
    ns = audit["updates"][rows]
    isfull = audit["flags"][rows] == 1
    h = np.asarray(history[rows])
    mv = np.asarray(moves[rows])
    valid = np.arange(257)[None, :] <= ns[:, None]
    valid &= isfull[:, None]
    assert np.all(np.isfinite(h[valid]))
    assert np.all(np.isnan(h[~valid]))
    diff = np.diff(h, axis=1)
    accepted = np.arange(256)[None, :] < ns[:, None]
    assert np.all(diff[accepted] < 0)
    assert np.all(mv[~accepted] == 65535)
    assert np.all(mv[:, :, 0][accepted] < 2)
    assert np.all(mv[:, :, 1][accepted] < 32)
    assert np.all(mv[:, :, 2][accepted] < 768)
    assert np.array_equal(h[isfull, 0], audit["values"][rows[isfull], 1])
    assert np.array_equal(
        h[np.flatnonzero(isfull), ns[isfull]], audit["values"][rows[isfull], 2]
    )
    if len(mv):
        labels = mv[:, :, 0].astype(np.uint32) * 32 + mv[:, :, 1]
        for local in range(len(rows)):
            n = int(ns[local])
            lab = labels[local, :n]
            assert np.all(np.diff(lab.astype(np.int32)) >= 0)
            if n:
                assert np.max(np.bincount(lab, minlength=64)) <= 4
samples = (
    full[:64]
    if scope == "pilot"
    else np.fromfile(M / "validation_ids.u32", dtype="<u4")
)
assert len(samples) == (64 if scope == "pilot" else 2048)
max_history = 0.0
max_lambda = 0.0
total = 0
for i, row in enumerate(samples):
    ids = np.r_[row, g["ids"][row], proxy[row]]
    x = np.asarray(rot[ids], dtype=np.float64)
    r = x[1:33] - x[0]
    q = (x[33:] - x[0]) / 4
    lam = float(np.sum(q * q) / 768)
    ref, hist, accepted = optimize(r, q, lam, coupled=False)
    s = np.unpackbits(compact[row], axis=1, bitorder="big").astype(np.int8) * 2 - 1
    assert np.array_equal(s, ref), f"signs row{row}"
    n = int(audit["updates"][row])
    assert n == len(accepted)
    assert np.array_equal(
        moves[row, :n].reshape(-1), np.asarray(accepted).reshape(-1)
    ), f"moves row{row}"
    assert np.allclose(history[row, : n + 1], hist, rtol=1e-11, atol=1e-12)
    assert np.isclose(audit["values"][row, 0], lam, rtol=1e-12, atol=1e-15)
    max_history = max(max_history, float(np.max(np.abs(history[row, : n + 1] - hist))))
    max_lambda = max(max_lambda, abs(audit["values"][row, 0] - lam))
    total += n
    if (i + 1) % 128 == 0:
        logging.info("%s %s %s %s", "VALIDATE77", i + 1, "of", len(samples))
a = {
    "scope": scope,
    "centers_audited": count,
    "learned_blocks": len(full),
    "native_short_blocks": len(short),
    "all_history_steps_monotone_and_budgets_valid": True,
    "max_scale_ratio": float(audit["values"][:, 3].max()),
    "max_radial_error": float(audit["values"][:, 4].max()),
    "max_actual_fp32_radial_error": float(audit["values"][:, 5].max()),
    "total_accepted_updates": int(audit["updates"].sum()),
    "python_oracle_sample_centers": len(samples),
    "python_oracle_sample_rule": "first64 fulldegree"
    if scope == "pilot"
    else "fixed seed7701 before training",
    "python_signs_and_all_moves_bit_identical": True,
    "oracle_updates_compared": total,
    "max_history_absolute_error": max_history,
    "max_lambda_absolute_error": max_lambda,
    "codec_check_separately_required": True,
    "no_retrieval_qps_claim": True,
}
(R / "results" / f"independent77_{scope}_validation.json").write_text(
    json.dumps(a, indent=2) + "\n"
)
logging.info(json.dumps(a, indent=2))
logging.info("VALIDATE77 COMPLETE")
