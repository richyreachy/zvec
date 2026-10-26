#!/usr/bin/env python3
# Standalone experiment ledger intentionally reports JSON and retains sequential checks.
# ruff: noqa: PLR0915, T201
"""Small independent real-arithmetic checks; not a Cohere/QPS experiment.

Run only outside retrieval timing. The only output mutation is the requested
JSON report. No frozen models, production code, or earlier scripts are read.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(790026)
    counts = {}
    residuals = {}

    def check(name, actual, expected, *, atol=2e-10, rtol=2e-10):
        actual = np.asarray(actual)
        expected = np.asarray(expected)
        err = float(np.max(np.abs(actual - expected)))
        residuals[name] = max(residuals.get(name, 0.0), err)
        assert np.allclose(actual, expected, atol=atol, rtol=rtol), (name, err)
        counts[name] = counts.get(name, 0) + 1

    # Exhaustive sign-code global optimum for small dimensions, including zeros.
    for d in range(1, 9):
        for trial in range(12):
            r = rng.normal(size=d)
            if trial == 0:
                r[0] = 0
            R = r @ r
            if R == 0:
                continue
            native = np.where(r >= 0, 1.0, -1.0)
            alpha0 = R / (r @ native)
            e0 = alpha0 * native - r
            best = float("inf")
            for bits in itertools.product((-1.0, 1.0), repeat=d):
                s = np.asarray(bits)
                u = r @ s
                if u <= 1e-14:
                    continue
                alpha = R / u
                e = alpha * s - r
                best = min(best, e @ e)
                check("euclidean_error_identity", e @ e, d * alpha**2 - R)
            check("exhaustive_sign_global_minimum", e0 @ e0, best)

    # Score identities and both known-R and no-R radial geometry.
    for d in (1, 2, 3, 8, 64, 768):
        for trial in range(64):
            c, q, v = rng.normal(size=(3, d))
            if trial % 2:
                c /= np.linalg.norm(c)
                q /= np.linalg.norm(q)
                v /= np.linalg.norm(v)
            r, delta = v - c, q - c
            R, L = r @ r, delta @ delta
            if R < 1e-24:
                continue
            s = np.where(r >= 0, 1.0, -1.0)
            alpha = R / (r @ s)
            h = alpha * s
            e = h - r
            H2, p, x = h @ h, delta @ h, delta @ r
            g = 1 - q @ c
            nu = 0.5 * (c @ c - v @ v)
            F = 0.5 * R + c @ h + nu
            estimated = g + F - q @ h
            exact = 1 - q @ v
            check("radial_orthogonality", r @ e, 0)
            check("score_error", estimated - exact, -delta @ e)
            check("factor_equals_center_error", F, c @ e)
            check("p_recovery", g + 0.5 * R + nu - estimated, p)
            check("L_recovery", q @ q + c @ c - 2 + 2 * g, L)
            center = R * p / H2
            perp_r2 = max(0.0, R - R * R / H2)
            perp_delta2 = max(0.0, L - p * p / H2)
            radius = math.sqrt(perp_r2 * perp_delta2)
            assert center - radius - 1e-9 <= x <= center + radius + 1e-9
            counts["known_R_interval_contains_x"] = (
                counts.get("known_R_interval_contains_x", 0) + 1
            )
            beta = (H2 - R) / R
            radical = max(0.0, beta * ((1 + beta) * R * L - p * p))
            check("quadratic_center_equivalence", p / (1 + beta), center)
            check(
                "quadratic_radius_equivalence",
                math.sqrt(radical) / (1 + beta),
                radius,
                atol=2e-7,
                rtol=2e-8,
            )
            # Construct the two extrema, retaining h and R but not any native-
            # sign/Hamming constraints. In d=2 these are the two feasible r's.
            delta_perp = delta - p / H2 * h
            if d >= 2 and perp_r2 > 1e-12 and np.linalg.norm(delta_perp) > 1e-8:
                direction = delta_perp / np.linalg.norm(delta_perp)
                for orientation in (-1, 1):
                    extremal = R / H2 * h + orientation * math.sqrt(perp_r2) * direction
                    check("known_R_endpoint_norm", extremal @ extremal, R)
                    check("known_R_endpoint_radial", extremal @ h, R)
                    check(
                        "known_R_endpoint_attainment",
                        delta @ extremal,
                        center + orientation * radius,
                    )
            # Without R, r lies on the sphere with diameter [0,h].
            sphere_center = 1 - q @ c - 0.5 * q @ h
            sphere_radius = 0.5 * np.linalg.norm(q) * math.sqrt(H2)
            assert (
                sphere_center - sphere_radius - 1e-9
                <= exact
                <= sphere_center + sphere_radius + 1e-9
            )
            counts["zero_extra_scalar_interval_contains_distance"] = (
                counts.get("zero_extra_scalar_interval_contains_distance", 0) + 1
            )

    # Explicit nondegenerate d=2 example: fixed h=(2,0), R=1 gives
    # exactly r=(1/2,+/-sqrt(3)/2); the interval interior is not attainable.
    h = np.array([2.0, 0.0])
    endpoints = np.array([[0.5, math.sqrt(3) / 2], [0.5, -math.sqrt(3) / 2]])
    check("d2_two_point_norms", np.sum(endpoints**2, axis=1), np.ones(2))
    check("d2_two_point_radial", endpoints @ h, np.ones(2))
    assert np.linalg.norm(np.array([0.5, 0.0])) ** 2 != 1.0

    # Ellipsoidal radial projection bound is attained in the continuous error
    # set, with fixed r and arbitrary SPD A. No claim about discrete-code tightness.
    for d in (3, 8, 32):
        for _ in range(32):
            B = rng.normal(size=(4, d))
            A = B.T @ B + 0.5 * np.eye(d)
            r, delta = rng.normal(size=(2, d))
            inv_delta, inv_r = np.linalg.solve(A, np.stack((delta, r), axis=1)).T
            t = (delta @ inv_r) / (r @ inv_r)
            ell = delta @ inv_delta - (delta @ inv_r) ** 2 / (r @ inv_r)
            J = 1.7
            extremal_e = math.sqrt(J / ell) * (inv_delta - t * inv_r)
            check("ellipsoid_endpoint_radial", r @ extremal_e, 0)
            check("ellipsoid_endpoint_energy", extremal_e @ A @ extremal_e, J)
            check("ellipsoid_endpoint_attainment", (delta @ extremal_e) ** 2, J * ell)

    # A deterministic transfer from native to learned signs, allowing arbitrary
    # data-dependent flip selection when final Hamming and scale bounds hold.
    max_budget = 8
    tau = 1.02
    for d in (32, 64, 768):
        for trial in range(96):
            r, delta = rng.normal(size=(2, d))
            s0 = np.where(r >= 0, 1.0, -1.0)
            R, l1 = r @ r, np.abs(r).sum()
            alpha0 = R / l1
            order = np.argsort(np.abs(r))
            m = trial % (max_budget + 1)
            while m and l1 / (l1 - 2 * np.abs(r[order[:m]]).sum()) > tau:
                m -= 1
            flipped = order[:m]
            s = s0.copy()
            s[flipped] *= -1
            alpha = R / (r @ s)
            h0, h = alpha0 * s0, alpha * s
            difference = h - h0
            weight = np.abs(r[flipped]).sum()
            check("flip_scale_identity", alpha / alpha0, l1 / (l1 - 2 * weight))
            assert weight <= 0.5 * (1 - 1 / tau) * l1 + 1e-12
            exact_norm2 = d * (alpha - alpha0) ** 2 + 4 * m * alpha * alpha0
            check("flip_difference_norm_identity", difference @ difference, exact_norm2)
            endpoint_bound = alpha**2 * max(
                4 * max_budget, d * (1 - 1 / tau) ** 2 + 4 * max_budget / tau
            )
            assert exact_norm2 <= endpoint_bound + 1e-10
            radial_delta = delta - (delta @ r) / R * r
            penalty = np.linalg.norm(radial_delta) * math.sqrt(max(0.0, exact_norm2))
            native_error = -delta @ (h0 - r)
            learned_error = -delta @ (h - r)
            assert abs(learned_error - native_error) <= penalty + 1e-10
            assert abs(learned_error) <= abs(native_error) + penalty + 1e-10
            counts["deterministic_native_error_transfer"] = (
                counts.get("deterministic_native_error_transfer", 0) + 1
            )

    report = {
        "status": "passed",
        "seed": 790026,
        "numpy_version": np.__version__,
        "scope": "synthetic float64 algebra checks; no production-code/model validation or retrieval benchmark",
        "counts": counts,
        "maximum_absolute_identity_residuals": residuals,
        "known_R_endpoint_domain": "continuous radial-compatible vectors; no final-Hamming/native-sign feasibility claim",
        "d2_interval_warning": "nondegenerate feasible set has only the two endpoints, not every interval value",
        "finite_precision_certificate_implemented": False,
        "qps_or_recall_claim": False,
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {"status": report["status"], "out": args.out, "check_counts": counts},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
