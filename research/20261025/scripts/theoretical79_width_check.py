#!/usr/bin/env python3
# Standalone experiment ledger intentionally reports JSON and retains sequential checks.
# ruff: noqa: PLR0915, T201
"""Read a fixed tiny phase75 prefix to diagnose ideal Gram interval widths.

Not an online/finite-precision safety certificate; no full dataset search.
The first 64 already-used phase75 centers and their fixed held-out proxy
positions are deliberately reused as an exploratory diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def describe(values):
    a = np.asarray(values).ravel()
    return {
        "count": int(a.size),
        "quantiles_0_25_50_75_95_100": np.quantile(
            a, [0, 0.25, 0.5, 0.75, 0.95, 1]
        ).tolist(),
        "mean": float(a.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    model = args.root / "models75"
    count, dimension = 64, 768
    source_checks = []

    def prefix(name, dtype, shape):
        path = model / name
        length = int(np.prod(shape)) * np.dtype(dtype).itemsize
        with path.open("rb") as stream:
            raw = stream.read(length)
        assert len(raw) == length
        source_checks.append(
            {
                "path": str(path),
                "prefix_bytes": length,
                "prefix_sha256": hashlib.sha256(raw).hexdigest(),
                "full_file_hash": False,
            }
        )
        return np.frombuffer(raw, dtype=dtype).reshape(shape)

    vectors = prefix("rotated.f32", "<f4", (count, 65, dimension)).astype(np.float64)
    signs = prefix("signs_independent.i8", "i1", (count, 32, dimension)).astype(
        np.float64
    )
    positions = np.fromfile(model / "validation_positions.u32", dtype="<u4")
    assert positions.size == 16
    results = {}
    for mode in ("native", "independent"):
        radii, gaps, errors, widths_over_gap = [], [], [], []
        certified = 0
        violations = 0
        query_count = 0
        max_recovery_error = 0.0
        for row in range(count):
            c, neighbors, queries = (
                vectors[row, 0],
                vectors[row, 1:33],
                vectors[row, 33 + positions],
            )
            residual = neighbors - c
            delta = queries - c
            R = np.sum(residual * residual, axis=1)
            s = np.where(residual >= 0, 1.0, -1.0) if mode == "native" else signs[row]
            alpha = R / np.sum(residual * s, axis=1)
            h = alpha[:, None] * s
            H2 = np.sum(h * h, axis=1)
            L = np.sum(delta * delta, axis=1)
            p = delta @ h.T
            g = 1 - queries @ c
            nu = 0.5 * (c @ c - np.sum(neighbors * neighbors, axis=1))
            estimated = g[:, None] + 0.5 * R[None, :] + nu[None, :] - p
            recovered_p = g[:, None] + 0.5 * R[None, :] + nu[None, :] - estimated
            max_recovery_error = max(
                max_recovery_error, float(np.max(np.abs(p - recovered_p)))
            )
            truth = 1 - queries @ neighbors.T
            center = (
                g[:, None]
                + 0.5 * R[None, :]
                + nu[None, :]
                - R[None, :] / H2[None, :] * p
            )
            radius = np.sqrt(
                np.maximum(0.0, R - R * R / H2)[None, :]
                * np.maximum(0.0, L[:, None] - p * p / H2[None, :])
            )
            lower, upper = center - radius, center + radius
            violations += int(np.sum((truth < lower - 1e-12) | (truth > upper + 1e-12)))
            for j in range(len(queries)):
                best = int(np.argmin(estimated[j]))
                competitors = np.arange(32) != best
                certified += int(upper[j, best] < lower[j, competitors].min())
            sorted_truth = np.sort(truth, axis=1)
            gap = sorted_truth[:, 1] - sorted_truth[:, 0]
            assert np.all(gap > 0)
            radii.extend(radius.ravel().tolist())
            gaps.extend(gap.tolist())
            errors.extend(np.abs(estimated - truth).ravel().tolist())
            widths_over_gap.extend((2 * radius / gap[:, None]).ravel().tolist())
            query_count += len(queries)
        results[mode] = {
            "queries": query_count,
            "distances": len(radii),
            "ideal_real_arithmetic_containment_violations": violations,
            "certified_top1_comparisons": certified,
            "half_width": describe(radii),
            "absolute_ideal_score_error": describe(errors),
            "exact_top1_top2_gap": describe(gaps),
            "full_width_divided_by_top1_gap": describe(widths_over_gap),
            "max_p_recovery_error": max_recovery_error,
        }
        assert violations == 0
    report = {
        "status": "passed",
        "scope": "first 64 previously used phase75 centers; 16 fixed held-out database proxies each; exploratory ideal FP64 geometry only",
        "center_rows": [0, count - 1],
        "validation_positions": positions.tolist(),
        "source_prefixes": source_checks,
        "results": results,
        "deployed_or_finite_precision_certificate": False,
        "fresh_iid_query_claim": False,
        "qps_or_recall_claim": False,
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {"status": report["status"], "out": str(args.out), "results": results},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
