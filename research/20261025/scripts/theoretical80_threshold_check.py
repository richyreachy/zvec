"""Exact-rational exhaustive checks of radial threshold linearization.

No training, model reads, retrieval, or timing claims. This checks real-algebra
statements with rational arithmetic, not the production FP32/LUT scorer.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
from fractions import Fraction as F
from pathlib import Path


def dot(left, right):
    return sum((a * b for a, b in zip(left, right, strict=True)), F(0))


def radial_score(r, delta, constant, signs):
    radius = dot(r, r)
    denominator = dot(r, signs)
    if radius == 0:
        raise ValueError("zero residual requires separate exact fallback")
    if denominator <= 0:
        raise ValueError("nonpositive radial denominator")
    return constant - radius * dot(delta, signs) / denominator


def decision_label(exact, threshold):
    if exact == threshold:
        raise ValueError("zero exact margin requires explicit tie policy")
    return 1 if exact > threshold else -1


def expect_rejection(function, *args):
    try:
        function(*args)
    except ValueError:
        return
    raise AssertionError("required rejection did not occur")


def check_case(r, delta, constant, exact, counts):
    dimension = len(r)
    radius, l1 = dot(r, r), sum(map(abs, r), F(0))
    if radius == 0:
        expect_rejection(radial_score, r, delta, constant, [1] * dimension)
        counts["zero_residual_cases_rejected"] += 1
        return
    native = [1 if value >= 0 else -1 for value in r]
    tau, gamma = F(51, 50), F(1, 100)
    kappa = l1 * gamma
    expect_rejection(decision_label, exact, exact)
    counts["zero_exact_margin_labels_rejected"] += 1
    for signs in itertools.product((-1, 1), repeat=dimension):
        counts["enumerated_codes"] += 1
        denominator = dot(r, signs)
        flips = [
            int(first != second) for first, second in zip(native, signs, strict=True)
        ]
        mass = sum(
            (abs(value) * flip for value, flip in zip(r, flips, strict=True)), F(0)
        )
        assert denominator == l1 - 2 * mass
        for budget in sorted({1, 2, min(8, dimension)}):
            radial_feasible = (
                denominator > 0 and l1 / denominator <= tau and sum(flips) <= budget
            )
            linear_feasible = sum(flips) <= budget and mass <= (1 - 1 / tau) * l1 / 2
            assert radial_feasible == linear_feasible
            counts["budget_set_equivalence_checks"] += 1
        if denominator <= 0:
            expect_rejection(radial_score, r, delta, constant, signs)
            counts["nonpositive_denominator_codes_rejected"] += 1
            continue
        estimated = radial_score(r, delta, constant, signs)
        for threshold in (exact - gamma, exact, exact + gamma, estimated):
            linear = dot(
                [
                    (constant - threshold) * x - radius * d
                    for x, d in zip(r, delta, strict=True)
                ],
                signs,
            )
            assert estimated - threshold == linear / denominator
            assert (estimated <= threshold) == (linear <= 0)
            assert (estimated < threshold) == (linear < 0)
            assert (estimated == threshold) == (linear == 0)
            assert (estimated >= threshold) == (linear >= 0)
            assert (estimated > threshold) == (linear > 0)
            counts["threshold_sign_equivalence_checks"] += 1
            counts["estimated_threshold_ties_checked"] += int(estimated == threshold)
            if exact == threshold:
                counts["zero_exact_margin_hinge_cases_excluded"] += 1
                continue
            label = decision_label(exact, threshold)
            hinge = max(F(0), 1 - label * linear / kappa)
            flip = int(label * (estimated - threshold) <= 0)
            assert F(flip) <= hinge
            counts["hinge_flip_upper_bound_checks"] += 1
            if hinge == 0:
                assert label * (estimated - threshold) >= gamma
                counts["zero_hinge_implies_score_margin_checks"] += 1
            if (
                l1 / denominator <= tau
                and label * (estimated - threshold) >= tau * gamma
            ):
                assert hinge == 0
                counts["sufficient_margin_implies_zero_hinge_checks"] += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rng = random.Random(8003)
    names = [
        "enumerated_codes",
        "budget_set_equivalence_checks",
        "zero_residual_cases_rejected",
        "zero_exact_margin_labels_rejected",
        "nonpositive_denominator_codes_rejected",
        "threshold_sign_equivalence_checks",
        "estimated_threshold_ties_checked",
        "zero_exact_margin_hinge_cases_excluded",
        "hinge_flip_upper_bound_checks",
        "zero_hinge_implies_score_margin_checks",
        "sufficient_margin_implies_zero_hinge_checks",
    ]
    counts = dict.fromkeys(names, 0)
    for dimension in range(1, 9):
        for trial in range(8):
            r = [
                F(rng.choice((0, 1, 2, 3, 5, 50, 100, 200)) * rng.choice((-1, 1)), 100)
                for _ in range(dimension)
            ]
            if trial == 0:
                r = [F(0)] * dimension
            delta = [F(rng.randrange(-7, 8), 7) for _ in range(dimension)]
            center = [F(rng.randrange(-5, 6), 7) for _ in range(dimension)]
            query = [c + d for c, d in zip(center, delta, strict=True)]
            vector = [c + value for c, value in zip(center, r, strict=True)]
            radius = dot(r, r)
            norm_correction = (dot(center, center) - dot(vector, vector)) / 2
            constant = 1 - dot(query, center) + radius / 2 + norm_correction
            exact = 1 - dot(query, vector)
            assert constant - dot(delta, r) == exact
            check_case(r, delta, constant, exact, counts)
    report = {
        "status": "passed",
        "seed": 8003,
        "arithmetic": "fractions.Fraction exact rational",
        "dimensions": list(range(1, 9)),
        "cases_per_dimension": 8,
        "scale_ratio": "51/50",
        "gamma": "1/100",
        "kappa": "||r||_1 * gamma",
        "counts": counts,
        "scope": "exhaustive signs on fixed synthetic cases; no code optimization",
        "production_fp32_lut_equivalence_claim": False,
        "qps_or_recall_claim": False,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
