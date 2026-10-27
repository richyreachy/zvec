from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from coupled75_math import objective, reconstruct

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
M = R / "models75"
assert "RUN75 COMPLETE" in (R / "logs/run75.log").read_text()
config = json.loads((M / "config.json").read_text())
names = config["score_modes"]
truth = np.memmap(M / "truth.f64", mode="r", dtype="<f8", shape=(2048, 32, 32))
minimum = truth.min(axis=2)
sel = np.fromfile(M / "selection_positions.u32", dtype="<u4")
val = np.fromfile(M / "validation_positions.u32", dtype="<u4")
ids = np.fromfile(M / "ids.u32", dtype="<u4").reshape(2048, 65)
pairs = np.fromfile(M / "pairs.u32", dtype="<u4").reshape(1024, 2)
cross_truth = np.memmap(
    M / "cross_truth.f64", mode="r", dtype="<f8", shape=(1024, 2, 16, 32)
)
single_errors = {}
cross_errors = {}
single = []
cross = []
training_checks = {}


def comparison(base, other, seed):
    b = base.mean(axis=1)
    change = b - other.mean(axis=1)
    gain = float(change.mean() / b.mean())
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(4000):
        sample = rng.integers(0, len(b), len(b))
        boot.append(float(change[sample].mean() / b[sample].mean()))
    ci = np.quantile(boot, [0.025, 0.975]).tolist()
    return {
        "relative_error_reduction": gain,
        "paired_bootstrap95": ci,
        "quality_gate_passed": gain >= 0.1 and ci[0] > 0,
        "better_equal_worse_units": [
            int(np.sum(change > 0)),
            int(np.sum(change == 0)),
            int(np.sum(change < 0)),
        ],
    }


for name in names:
    scores = np.memmap(
        M / f"scores_{name}.f32", mode="r", dtype="<f4", shape=(2048, 32, 32)
    )
    assert np.isfinite(scores).all()
    chosen = np.argmin(scores, axis=2)
    errors = (
        np.take_along_axis(truth, chosen[:, :, None], axis=2)[:, :, 0] > minimum + 1e-07
    )
    single_errors[name] = errors
    np.save(M / f"errors_{name}.npy", errors)
    e = scores.astype(np.float64) - truth
    mean = e.mean(axis=2)
    stats = {
        "mode": name,
        "build_error_rate": float(errors[:, sel].mean()),
        "heldout_error_rate": float(errors[:, val].mean()),
        "heldout_errors": int(errors[:, val].sum()),
        "queries": 32768,
        "heldout_total_score_mse": float(np.mean(e[:, val] ** 2)),
        "heldout_block_mean_error_squared": float(np.mean(mean[:, val] ** 2)),
        "heldout_centered_score_mse": float(
            np.mean((e - mean[:, :, None])[:, val] ** 2)
        ),
    }
    stats.update(comparison(single_errors["native"][:, val], errors[:, val], 7574))
    single.append(stats)
indices = np.full((2, 1024, 64), -1, dtype="<i4")
unique_ids = np.full((2, 1024, 64), np.iinfo(np.uint32).max, dtype="<u4")
counts = np.empty((2, 1024), dtype="<u4")
for p, rows in enumerate(pairs):
    flat_ids = ids[rows, 1:33].ravel()
    for order in range(2):
        visits = (
            list(range(64)) if order == 0 else list(range(32, 64)) + list(range(32))
        )
        seen = set()
        keep = []
        for j in visits:
            if int(flat_ids[j]) not in seen:
                seen.add(int(flat_ids[j]))
                keep.append(j)
        count = len(keep)
        indices[order, p, :count] = keep
        unique_ids[order, p, :count] = flat_ids[keep]
        counts[order, p] = count
        assert len(seen) == count
    assert set(unique_ids[0, p, : counts[0, p]]) == set(
        unique_ids[1, p, : counts[1, p]]
    )
indices.tofile(M / "cross_dedup_indices.i32")
unique_ids.tofile(M / "cross_unique_ids.u32")
counts.tofile(M / "cross_unique_counts.u32")
for name in names:
    scores = np.memmap(
        M / f"cross_scores_{name}.f32", mode="r", dtype="<f4", shape=(1024, 2, 16, 32)
    )
    assert np.isfinite(scores).all()
    errors = np.empty((2, 1024, 16), dtype=bool)
    chosen_ids = np.empty((2, 1024, 16), dtype="<u4")
    saved = np.memmap(
        M / f"cross_unique_scores_{name}.f32",
        mode="w+",
        dtype="<f4",
        shape=(2, 1024, 16, 64),
    )
    saved[:] = np.inf
    for p in range(1024):
        raw = scores[p].transpose(1, 0, 2).reshape(16, 64)
        exact = cross_truth[p].transpose(1, 0, 2).reshape(16, 64)
        for order in range(2):
            n = int(counts[order, p])
            keep = indices[order, p, :n]
            selected = raw[:, keep]
            target = exact[:, keep]
            saved[order, p, :, :n] = selected
            best = np.argmin(selected, axis=1)
            chosen_ids[order, p] = unique_ids[order, p, best]
            errors[order, p] = target[np.arange(16), best] > target.min(axis=1) + 1e-07
    saved.flush()
    chosen_ids.tofile(M / f"cross_chosen_ids_{name}.u32")
    cross_errors[name] = errors
    np.save(M / f"cross_errors_{name}.npy", errors)
    e = scores.astype(np.float64) - cross_truth
    mean = e.mean(axis=3)
    rows = []
    for order in range(2):
        entry = {
            "order": "A_first" if order == 0 else "B_first",
            "queries": 16384,
            "errors": int(errors[order].sum()),
            "error_rate": float(errors[order].mean()),
        }
        entry.update(comparison(cross_errors["native"][order], errors[order], 7575))
        rows.append(entry)
    cross.append(
        {
            "mode": name,
            "orders": rows,
            "total_errors_both_orders": int(errors.sum()),
            "total_score_mse": float(np.mean(e * e)),
            "block_mean_error_squared": float(np.mean(mean * mean)),
            "centered_score_mse": float(np.mean((e - mean[:, :, :, None]) ** 2)),
            "order_sensitive_query_choices": int(
                np.sum(chosen_ids[0] != chosen_ids[1])
            ),
        }
    )
x = np.memmap(M / "fit.f32", mode="r", dtype="<f4", shape=(2048, 49, 768))
for variant in ["coupled", "independent"]:
    s_saved = np.memmap(
        M / f"signs_{variant}.i8", mode="r", dtype="i1", shape=(2048, 32, 768)
    )
    a_saved = np.memmap(
        M / f"alpha_{variant}.f64", mode="r", dtype="<f8", shape=(2048, 32)
    )
    train = json.loads((R / f"results/coupled75_training_{variant}.json").read_text())
    updates = 0
    max_radial = 0.0
    max_objective = 0.0
    max_ratio = 1.0
    rows = 0
    with (M / f"history_{variant}.jsonl").open() as f:
        for i, line in enumerate(f):
            entry = json.loads(line)
            assert entry["row"] == i
            a = np.asarray(x[i], dtype=np.float64)
            r = a[1:33] - a[0]
            Q = (a[33:] - a[0]) / 4
            lam = float(np.sum(Q * Q) / 768)
            assert lam == entry["lambda"]
            native = np.where(r > 0, 1.0, -1.0)
            s = native.copy()
            count = np.zeros((2, 32), dtype=int)
            history = np.array(entry["objectives"])
            moves = entry["accepted_sweep_lane_bit"]
            assert len(history) == len(moves) + 1
            assert np.all(np.diff(history) < 0)
            for sweep, lane, bit in moves:
                assert 0 <= sweep < 2
                assert 0 <= lane < 32
                assert 0 <= bit < 768
                s[lane, bit] *= -1
                count[sweep, lane] += 1
            assert count.max() <= 4
            assert np.array_equal(s, s_saved[i])
            h, alpha = reconstruct(r, s)
            _, alpha0 = reconstruct(r, native)
            assert np.array_equal(alpha, a_saved[i])
            assert np.all(alpha <= alpha0 * 1.02 + 1e-14)
            max_radial = max(
                max_radial,
                float(np.max(np.abs(np.sum(r * h, axis=1) - np.sum(r * r, axis=1)))),
            )
            ratio = np.divide(alpha, alpha0, out=np.ones_like(alpha), where=alpha0 > 0)
            max_ratio = max(max_ratio, float(ratio.max()))
            before = objective(r, native, Q, lam, coupled=variant == "coupled")
            after = objective(r, s, Q, lam, coupled=variant == "coupled")
            max_objective = max(
                max_objective, abs(before - history[0]), abs(after - history[-1])
            )
            assert max_objective < 1e-12
            updates += len(moves)
            rows += 1
    assert rows == 2048
    assert updates == train["total_accepted_updates"]
    assert max_radial < 1e-12
    training_checks[variant] = {
        "centers": rows,
        "updates": updates,
        "max_radial_error": max_radial,
        "max_objective_difference": max_objective,
        "max_scale_ratio": max_ratio,
        "elapsed_seconds": train["elapsed_seconds"],
        "all_histories_match_saved_signs_and_alpha": True,
        "all_initial_final_objectives_checked": True,
        "all_step_histories_strictly_descend": True,
    }
mode_gates = {
    name: single[names.index(name)]["quality_gate_passed"]
    and all(x["quality_gate_passed"] for x in cross[names.index(name)]["orders"])
    for name in ["coupled", "independent"]
}
eligible = [name for name in mode_gates if mode_gates[name]]
selected = (
    min(
        eligible,
        key=lambda name: (
            cross[names.index(name)]["total_errors_both_orders"],
            name != "independent",
        ),
    )
    if eligible
    else None
)
result = {
    "config": config,
    "single_block": single,
    "cross_block": cross,
    "mode_all_gates_passed": mode_gates,
    "selected_for_fullgraph_feasibility": selected,
    "selection_rule_fixed_before_data": True,
    "coupled_vs_independent_single": comparison(
        single_errors["independent"][:, val], single_errors["coupled"][:, val], 7574
    ),
    "coupled_vs_independent_cross_orders": [
        comparison(cross_errors["independent"][o], cross_errors["coupled"][o], 7575)
        for o in range(2)
    ],
    "training_checks": training_checks,
    "codec_checks": {
        name: json.loads((R / f"logs/codec75_{name}.log").read_text()) for name in names
    },
    "cross_codec_checks": {
        name: json.loads((R / f"logs/cross75_{name}.log").read_text()) for name in names
    },
    "unique_score_padding_positive_infinity_not_a_candidate": True,
    "first_seen_dedup_both_orders_without_score_selection": True,
    "no_full_search_equivalence_or_iid_claim": True,
    "no_retrieval_qps_or_recall_claim": True,
    "new_retrieval_timing_rows": 0,
    "cumulative_retrieval_timing_rows": 1310,
}
result["coupled_vs_independent_single"].pop("quality_gate_passed")
for item in result["coupled_vs_independent_cross_orders"]:
    item.pop("quality_gate_passed")
(R / "results/coupled75_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
logging.info("%s", json.dumps(result, indent=2))
logging.info("%s", "ANALYZE75 COMPLETE")
