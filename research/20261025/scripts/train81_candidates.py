"""Fit two frozen objectives on training queries only; never open validation data."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(4 << 20), b""):
            h.update(data)
    return h.hexdigest()


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def bits_float(values):
    return np.ascontiguousarray(values, dtype="<u4").view("<f4").astype(np.float64)


def header(path, magic, record_bytes):
    h = np.fromfile(path, dtype="<u4", count=16)
    need(
        len(h) == 16 and tuple(h[:6]) == (magic, 1, 64, record_bytes, 768, 32),
        f"invalid training export header: {path}",
    )
    need(h[8] == 0 and np.all(h[9:] == 0), "training-only export required")
    return h


LANE = np.dtype([("head", "<u4", (8,)), ("boundaries", "<u4", (2, 8))])
STATE = np.dtype(
    [
        ("head", "<u4", (8,)),
        ("event", "<u8"),
        ("extra", "<u4", (6,)),
        ("lanes", LANE, (32,)),
    ]
)
QUERY = np.dtype(
    [("head", "<u4", (4,)), ("prepared", "<f4", (769,)), ("rotated", "<f4", (768,))]
)


def select(objective, valid):
    need(np.all(np.isfinite(objective[valid])), "nonfinite feasible objective")
    masked = np.where(valid, objective, np.inf)
    index = int(np.argmin(masked))
    need(
        valid[0] and valid[index] and masked[index] <= masked[0],
        "native feasible bound",
    )
    ranked = np.sort(masked[valid])
    gap = float(ranked[1] - ranked[0]) if len(ranked) > 1 else None
    return index, gap


def fit(protocol_path, candidate_manifest_path, export, out):  # noqa: PLR0915
    # Keep the single training transaction and its frozen input gates together.
    protocol = json.loads(protocol_path.read_text())
    need(
        protocol["phase"] == 81 and protocol["frozen"],
        "frozen phase81 protocol required",
    )
    need(protocol["candidate"]["candidate_count"] == 137, "fixed candidate count")
    pf = Path(protocol["preflight"]["root"])
    # Only training split/state contents are decoded here. The protocol contains
    # the pre-existing validation hash, but that file is not opened by this tool.
    for name in ["split.json", "centers.json", "training_selected_states.json"]:
        need(
            sha(pf / name) == protocol["preflight"]["files_sha256"][name],
            "preflight changed",
        )
    split = json.loads((pf / "split.json").read_text())
    train_serials = np.sort(np.asarray(split["train_query_serials"], dtype=np.uint32))
    need(
        len(train_serials) == 3072 and len(np.unique(train_serials)) == 3072,
        "training split",
    )
    manifest = json.loads(candidate_manifest_path.read_text())
    bank = candidate_manifest_path.parent
    need(
        manifest.get("passed") is True
        and manifest.get("geometry_only") is True
        and manifest.get("native_blocks_bit_identical") is True
        and manifest["protocol_sha256"] == sha(protocol_path),
        "candidate bank must bind frozen protocol",
    )
    for name, expected in manifest["artifacts_sha256"].items():
        need(sha(bank / name) == expected, f"candidate artifact changed: {name}")
    proof_path = export / "proof.json"
    proof = json.loads(proof_path.read_text())
    need(
        proof.get("passed") is True or proof.get("status") == "passed",
        "training export failed",
    )
    need(
        proof["partition"] == "training"
        and proof["protocol_sha256"] == sha(protocol_path)
        and proof["candidate_manifest_sha256"] == sha(candidate_manifest_path),
        "training export must bind the same frozen protocol/candidate bank",
    )
    need(
        proof["native_gate_completed_before_candidate_scoring"]
        and proof["full_frozen_partition_scope"]
        and not proof["validation_scores_read_in_training_mode"],
        "full training-only native gates required",
    )
    for name, expected_hash in proof["artifacts_sha256"].items():
        need(
            sha(export / name) == expected_hash,
            f"training export artifact changed: {name}",
        )
    inputs = {
        str(p): sha(p)
        for p in [
            protocol_path,
            candidate_manifest_path,
            proof_path,
            export / "states.bin",
            export / "queries.bin",
            export / "scores.f32",
        ]
    }
    centers = np.fromfile(bank / "centers.u32", dtype="<u4")
    neighbors = np.fromfile(bank / "neighbors.u32", dtype="<u4").reshape(20, 32)
    coords = np.fromfile(bank / "coords.u16", dtype="<u2").reshape(20, 32, 16)
    local = np.fromfile(bank / "flip_local.u16", dtype="<u2").reshape(137, 2)
    valid = np.fromfile(bank / "valid.u8", dtype="u1").reshape(20, 137, 32).astype(bool)
    need(
        centers.tolist() == protocol["preflight"]["center_order"], "frozen center order"
    )
    need(
        np.all(valid[:, 0, :]) and np.all(coords < 768),
        "native validity/coordinate domain",
    )
    need(np.all((local < 16) | (local == 65535)), "flip-local domain")
    need(np.all(local[0] == 65535), "native candidate0")
    hs = header(export / "states.bin", 0x51474181, STATE.itemsize)
    hq = header(export / "queries.bin", 0x51475181, QUERY.itemsize)
    hc = header(export / "scores.f32", 0x51474581, 128)
    n = int(hs[6])
    need(n == 6625 and hs[7] == hc[7] == 137 and hc[6] == n, "fixed training shape")
    need(
        (export / "states.bin").stat().st_size == 64 + n * STATE.itemsize,
        "state byte count",
    )
    need(
        (export / "queries.bin").stat().st_size == 64 + int(hq[6]) * QUERY.itemsize,
        "query byte count",
    )
    need(
        (export / "scores.f32").stat().st_size == 64 + n * 137 * 32 * 4,
        "score byte count",
    )
    states = np.memmap(
        export / "states.bin", mode="r", offset=64, dtype=STATE, shape=(n,)
    )
    queries = np.memmap(
        export / "queries.bin", mode="r", offset=64, dtype=QUERY, shape=(int(hq[6]),)
    )
    scores = np.memmap(
        export / "scores.f32", mode="r", offset=64, dtype="<f4", shape=(n, 137, 32)
    )
    qserial = states["head"][:, 0].astype(np.int64)
    ci = states["extra"][:, 0].astype(np.int64)
    qri = states["extra"][:, 4].astype(np.int64)
    need(
        np.array_equal(np.unique(qserial), train_serials),
        "training-only query coverage",
    )
    need(np.all(ci < 20) and np.all(qri < len(queries)), "state association range")
    need(np.array_equal(queries["head"][qri, 0], qserial), "query record identity")
    need(np.array_equal(states["head"][:, 2], centers[ci]), "state center identity")
    need(
        np.array_equal(states["lanes"]["head"][:, :, 0], neighbors[ci]),
        "neighbor identity",
    )
    expected_states = json.loads((pf / "training_selected_states.json").read_text())
    observed = [(int(h[0]), int(h[1]), int(h[2]), int(h[3])) for h in states["head"]]
    expected = [
        (s["query_serial"], s["query_internal_id"], s["center"], s["step"])
        for s in expected_states
    ]
    need(observed == expected, "exact preflight training state order")
    boundaries = states["lanes"]["boundaries"]
    finite = (boundaries[:, :, :, 6] & 1) != 0
    threshold = bits_float(boundaries[:, :, :, 0])
    eligible = states["lanes"]["head"][:, :, 4].astype(bool)
    active = eligible[:, :, None] & finite
    truth = bits_float(states["lanes"]["head"][:, :, 2])
    need(
        np.array_equal(
            scores[:, 0, :].view("<u4"),
            np.ascontiguousarray(states["lanes"]["head"][:, :, 1]),
        ),
        "native score export identity",
    )
    width = bits_float(states["head"][:, 6])
    center_exact = bits_float(states["head"][:, 7])
    need(np.all(np.isfinite(width) & (width > 0)), "positive finite width")
    nq = np.bincount(qserial, weights=np.sum(active, axis=(1, 2)), minlength=4096)
    per_row_weight = np.zeros(n, dtype=np.float64)
    nonempty = nq[qserial] > 0
    per_row_weight[nonempty] = 1.0 / (3072 * nq[qserial[nonempty]])
    raw_rotated = Path(protocol["provenance"]["bench_root"]) / "models77/rotated.f32"
    need(raw_rotated.stat().st_size == 1000000 * 768 * 4, "rotated vector shape")
    # Builder has verified this immutable dataset against its trusted hash.
    rotated = np.memmap(raw_rotated, dtype="<f4", mode="r", shape=(1000000, 768))
    before_rotated_stat = (raw_rotated.stat().st_size, raw_rotated.stat().st_mtime_ns)
    objectives = np.full((2, 20, 32, 137), np.inf, dtype=np.float64)
    selections = np.zeros((2, 20, 32), dtype=np.uint32)
    lambda_values = np.zeros((20, 32), dtype=np.float64)
    details = []
    formula_checks = 0
    max_formula_error = 0.0
    max_second_moment_error = 0.0
    flip_indices = np.minimum(local, 15)
    flip_present = local != 65535
    for cindex, center_id in enumerate(centers):
        rows = np.flatnonzero(ci == cindex)
        c = rotated[int(center_id)].astype(np.float64)
        qrot = queries["rotated"][qri[rows]].astype(np.float64)
        delta = qrot - c
        delta_norm2 = np.sum(delta * delta, axis=1)
        cn = float(np.dot(c, c))
        for lane in range(32):
            v = rotated[int(neighbors[cindex, lane])].astype(np.float64)
            residual = v - c
            rr = float(np.dot(residual, residual))
            u0 = float(np.sum(np.abs(residual)))
            s0 = np.where(residual > 0, 1.0, -1.0)
            allowed = valid[cindex, :, lane]
            dimensions = coords[cindex, lane].astype(np.int64)
            ranked = np.lexsort((np.arange(768), np.abs(residual)))[:16]
            need(
                np.array_equal(np.sort(ranked), dimensions),
                "geometry-only coordinate rule",
            )
            edge_active = active[rows, lane, :]
            state_weights = per_row_weight[rows] * np.sum(edge_active, axis=1)
            hinge_rows = edge_active & (
                truth[rows, lane, None] != threshold[rows, lane, :]
            )
            if rr == 0:
                need(not np.any(allowed[1:]), "zero residual must retain native only")
                objectives[:, cindex, lane, 0] = 0.0
                details.append(
                    {"center": int(center_id), "lane": lane, "zero_residual": True}
                )
                continue
            need(u0 > 0, "positive residual denominator")
            mass = np.sum(
                np.abs(residual[dimensions])[flip_indices] * flip_present, axis=1
            )
            u = u0 - 2 * mass
            need(np.all(u[allowed] > 0), "allowed positive denominators")
            alpha = np.divide(rr, u, out=np.zeros_like(u), where=u > 0)
            base_dot = delta @ s0
            flip_terms = delta[:, dimensions] * s0[dimensions]
            ds = base_dot[:, None] - 2 * np.sum(
                flip_terms[:, flip_indices] * flip_present, axis=2
            )
            dr = delta @ residual
            error_projection = alpha[None, :] * ds - dr[:, None]
            error_norm2 = 768 * alpha * alpha - rr
            nu = (cn - float(np.dot(v, v))) / 2
            constant = center_exact[rows] + rr / 2 + nu
            ideal_scores = constant[:, None] - alpha[None, :] * ds
            # Independent direct reconstruction for every candidate on three
            # deterministic training states, with an explicit FP64 tolerance.
            signs = np.broadcast_to(s0, (137, 768)).copy()
            for code in range(137):
                for pos in local[code]:
                    if pos != 65535:
                        signs[code, dimensions[int(pos)]] *= -1
            inspect = np.unique([0, len(rows) // 2, len(rows) - 1])
            direct_u = signs @ residual
            direct_alpha = rr / direct_u
            direct_scores = (
                constant[inspect, None]
                - (delta[inspect] @ signs.T) * direct_alpha[None, :]
            )
            direct_error = direct_alpha[:, None] * signs - residual
            direct_projection = delta[inspect] @ direct_error.T
            difference = float(
                np.max(
                    np.abs(
                        direct_scores[:, allowed] - ideal_scores[inspect][:, allowed]
                    )
                )
            )
            difference2 = float(
                np.max(
                    np.abs(
                        direct_projection[:, allowed]
                        - error_projection[inspect][:, allowed]
                    )
                )
            )
            need(
                difference <= 1e-10 and difference2 <= 1e-10,
                "independent radial formula mismatch",
            )
            need(
                np.max(
                    np.abs(
                        np.sum(direct_error[allowed] ** 2, axis=1)
                        - error_norm2[allowed]
                    )
                )
                <= 1e-10,
                "radial norm identity mismatch",
            )
            formula_checks += len(inspect) * int(np.sum(allowed))
            max_formula_error = max(max_formula_error, difference)
            max_second_moment_error = max(max_second_moment_error, difference2)
            weight_sum = float(np.sum(state_weights))
            if weight_sum:
                lam = float(np.dot(state_weights, delta_norm2) / (768 * weight_sum))
                m2 = np.sum(state_weights[:, None] * error_projection**2, axis=0)
                m2 += weight_sum * lam * error_norm2
            else:
                lam = 0.0
                m2 = np.zeros(137)
            hinge = np.zeros(137, dtype=np.float64)
            for side in range(2):
                t = threshold[rows, lane, side]
                y = np.sign(truth[rows, lane] - t)
                numerator = (constant - t)[:, None] * u[None, :] - rr * ds
                value = np.maximum(
                    0, 1 - y[:, None] * numerator / ((width[rows] / 4 * u0)[:, None])
                )
                weights = per_row_weight[rows] * hinge_rows[:, side]
                hinge += np.sum(weights[:, None] * value, axis=0)
            objectives[0, cindex, lane] = np.where(allowed, m2, np.inf)
            objectives[1, cindex, lane] = np.where(allowed, hinge, np.inf)
            chosen_m2, gap_m2 = select(m2, allowed)
            chosen_hinge, gap_hinge = select(hinge, allowed)
            if not np.any(hinge_rows):
                chosen_hinge = 0
            selections[:, cindex, lane] = (chosen_m2, chosen_hinge)
            lambda_values[cindex, lane] = lam
            details.append(
                {
                    "center": int(center_id),
                    "lane": lane,
                    "target": int(neighbors[cindex, lane]),
                    "feasible_candidates": int(np.sum(allowed)),
                    "eligible_boundary_rows": int(np.sum(edge_active)),
                    "hinge_rows": int(np.sum(hinge_rows)),
                    "second_moment_index": chosen_m2,
                    "hinge_index": chosen_hinge,
                    "second_moment_runnerup_gap": gap_m2,
                    "hinge_runnerup_gap": gap_hinge,
                }
            )
        logging.info("TRAIN81 completed center %s/20", cindex + 1)
    need(
        before_rotated_stat
        == (raw_rotated.stat().st_size, raw_rotated.stat().st_mtime_ns),
        "rotated input changed",
    )
    for path, expected_hash in inputs.items():
        need(sha(path) == expected_hash, "input changed during training")
    np.savez(
        out / "objectives.npz",
        objectives=objectives,
        selected_indices=selections,
        lambda_values=lambda_values,
        nq=nq,
        row_weights=per_row_weight,
    )
    selected = {
        "method_order": ["second_moment", "hinge"],
        "indices": selections.tolist(),
        "protocol_sha256": inputs[str(protocol_path)],
        "candidate_manifest_sha256": inputs[str(candidate_manifest_path)],
        "training_export_proof_sha256": inputs[str(proof_path)],
    }
    write_json(out / "selected_indices.json", selected)
    chosen_totals = []
    native_totals = []
    for method in range(2):
        chosen_totals.append(
            float(
                sum(
                    objectives[method, c, lane, selections[method, c, lane]]
                    for c in range(20)
                    for lane in range(32)
                )
            )
        )
        native_totals.append(float(np.sum(objectives[method, :, :, 0])))
    result = {
        "passed": True,
        "protocol_sha256": inputs[str(protocol_path)],
        "method_order": selected["method_order"],
        "centers": 20,
        "edges": 640,
        "candidate_combinations_per_edge": 137,
        "training_states": n,
        "training_query_count": 3072,
        "training_boundary_rows": int(np.sum(active)),
        "empty_training_queries": int(np.sum(nq[train_serials] == 0)),
        "changed_edges": {
            name: int(np.count_nonzero(selections[i]))
            for i, name in enumerate(selected["method_order"])
        },
        "native_objectives": native_totals,
        "selected_objectives": chosen_totals,
        "direct_radial_checks": formula_checks,
        "max_direct_score_abs_error": max_formula_error,
        "max_direct_projection_abs_error": max_second_moment_error,
        "objective_scope": "exhaustive computed FP64 objective over frozen137-candidate domain;not full768D or rational optimum",
        "second_moment_scope": "delta-dot-radial-error proxy;not actual anchored-score MSE",
        "inputs_sha256": inputs,
        "source_sha256": sha(Path(__file__)),
        "validation_score_or_labels_read": False,
        "retrieval_or_QPS_run": False,
        "assembly_and_training_freeze_pending": True,
        "edge_details": details,
    }
    write_json(out / "training_result.json", result)
    logging.info(
        "%s",
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                in [
                    "passed",
                    "changed_edges",
                    "selected_objectives",
                    "native_objectives",
                ]
            }
        ),
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--training-export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    try:
        fit(
            args.protocol.resolve(),
            args.candidate_manifest.resolve(),
            args.training_export.resolve(),
            args.output,
        )
    except BaseException as error:
        write_json(
            args.output / "failure.json", {"passed": False, "error": repr(error)}
        )
        raise


if __name__ == "__main__":
    main()
