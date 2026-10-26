#!/usr/bin/env python3
# ruff: noqa: B905, T201 -- preserve frozen zip semantics and CLI summary
"""Exact toy checks of hinge Lagrange bounds; no data, training, or solver."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import random
from dataclasses import dataclass
from fractions import Fraction as Q
from pathlib import Path

ZERO = Q(0)
ONE = Q(1)
SEED = 8102


@dataclass(frozen=True)
class Problem:
    a: tuple
    b: tuple
    weight: tuple
    mass: tuple
    count_budget: Q
    mass_budget: Q

    def validate(self):
        dimension = len(self.mass)
        if dimension == 0 or len(self.b) != len(self.a):
            raise ValueError("invalid dimensions")
        if len(self.weight) != len(self.b):
            raise ValueError("invalid weight dimension")
        if any(len(row) != dimension for row in self.a):
            raise ValueError("invalid row dimension")
        values = [
            *self.b,
            *self.weight,
            *self.mass,
            self.count_budget,
            self.mass_budget,
        ]
        values.extend(value for row in self.a for value in row)
        if any(type(value) is not Q for value in values):
            raise ValueError("all problem coefficients must be exact Fractions")
        if any(value < 0 for value in self.weight + self.mass):
            raise ValueError("negative weight or mass")
        if self.count_budget < 0 or self.mass_budget < 0:
            raise ValueError("negative budget")

    def feasible(self, z):
        return (
            len(z) == len(self.mass)
            and all(value in (0, 1) for value in z)
            and sum(z) <= self.count_budget
            and sum(m * value for m, value in zip(self.mass, z)) <= self.mass_budget
        )

    def objective(self, z):
        return sum(
            (
                w * max(ZERO, b + sum(coef * value for coef, value in zip(row, z)))
                for row, b, w in zip(self.a, self.b, self.weight)
            ),
            ZERO,
        )


def certificate(problem, multipliers, mu, nu):
    """Return raw L and matching reduced costs, never clip L before fixing."""
    problem.validate()
    if len(multipliers) != len(problem.weight):
        raise ValueError("invalid multiplier dimension")
    if any(type(value) is not Q for value in (*multipliers, mu, nu)):
        raise ValueError("multipliers must be exact Fractions")
    if mu < 0 or nu < 0:
        raise ValueError("negative budget multiplier")
    if any(lam < 0 or lam > w for lam, w in zip(multipliers, problem.weight)):
        raise ValueError("hinge multiplier outside [0, weight]")
    reduced = tuple(
        sum((row[k] * lam for row, lam in zip(problem.a, multipliers)), ZERO)
        + mu
        + nu * mass
        for k, mass in enumerate(problem.mass)
    )
    lower = (
        sum((lam * b for lam, b in zip(multipliers, problem.b)), ZERO)
        - mu * problem.count_budget
        - nu * problem.mass_budget
        + sum((min(ZERO, value) for value in reduced), ZERO)
    )
    return lower, reduced


def conditional_lower(lower, reduced, coordinate, value):
    if value not in (0, 1):
        raise ValueError("fixed coordinate must be binary")
    d = reduced[coordinate]
    return lower - min(ZERO, d) + value * d


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def encode(value):
    if isinstance(value, Q):
        return str(value)
    if isinstance(value, dict):
        return {key: encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    return value


def serial_problem(problem):
    return encode(
        {
            "a": problem.a,
            "b": problem.b,
            "weight": problem.weight,
            "mass": problem.mass,
            "count_budget": problem.count_budget,
            "mass_budget": problem.mass_budget,
        }
    )


def check_problem(problem, candidates, incumbent_index, counters):
    problem.validate()
    dimension = len(problem.mass)
    points = [
        z for z in itertools.product((0, 1), repeat=dimension) if problem.feasible(z)
    ]
    require(bool(points), "zero code must be feasible")
    values = [problem.objective(z) for z in points]
    incumbent = points[incumbent_index % len(points)]
    upper = problem.objective(incumbent)
    optimum = min(values)
    counters["problems"] += 1
    counters["enumerated_binary_codes"] += 2**dimension
    counters["feasible_binary_codes"] += len(points)
    bounds = []
    for lam, mu, nu in candidates:
        lower, reduced = certificate(problem, lam, mu, nu)
        counters["candidate_certificates"] += 1
        require(lower <= optimum <= upper, "sandwich inequality")
        for z, objective in zip(points, values):
            require(objective >= lower, "global lower bound failed")
            counters["pointwise_global_checks"] += 1
            for coordinate, value in enumerate(z):
                forced = conditional_lower(lower, reduced, coordinate, value)
                require(objective >= forced, "conditional lower bound failed")
                counters["pointwise_conditional_checks"] += 1
        screened = []
        for coordinate in range(dimension):
            for value in (0, 1):
                forced = conditional_lower(lower, reduced, coordinate, value)
                if forced > upper:
                    allowed = [
                        objective
                        for z, objective in zip(points, values)
                        if z[coordinate] == value and objective <= upper
                    ]
                    require(not allowed, "strict screening removed <= U code")
                    counters["strict_screened_branches"] += 1
                    if any(z[coordinate] == value for z in points):
                        counters["nonempty_strict_screened_branches"] += 1
                    screened.append([coordinate, value, forced])
                if forced == upper:
                    counters["equality_branches_not_screened"] += 1
        bounds.append(
            {
                "lambda": lam,
                "mu": mu,
                "nu": nu,
                "raw_lower": lower,
                "reduced_costs": reduced,
                "screened": screened,
            }
        )
    return encode(
        {
            "dimension": dimension,
            "feasible_codes": len(points),
            "optimum": optimum,
            "incumbent": incumbent,
            "upper": upper,
            "certificates": bounds,
        }
    )


def fixtures(counters):
    decreasing = Problem(((Q(-1),),), (Q(2),), (ONE,), (ONE,), ONE, ONE)
    increasing = Problem(((ONE,),), (ONE,), (ONE,), (ONE,), ONE, ONE)
    zero_budget = Problem(((Q(-2), Q(1)),), (Q(3),), (Q(2),), (Q(1), Q(2)), ZERO, ZERO)
    zero_weight = Problem(((Q(7),),), (Q(-3),), (ZERO,), (ZERO,), ONE, ZERO)
    results = {
        "nonzero_tight_optimum": check_problem(
            decreasing, [((ONE,), ZERO, ZERO)], 1, counters
        ),
        "strict_force_one_screen": check_problem(
            increasing, [((ONE,), ZERO, ZERO)], 0, counters
        ),
        "zero_budgets": check_problem(
            zero_budget, [((Q(2),), Q(4), Q(1))], 0, counters
        ),
        "zero_weight_and_zero_mass": check_problem(
            zero_weight, [((ZERO,), ZERO, ZERO)], 0, counters
        ),
    }
    low, reduced = certificate(decreasing, (ONE,), ZERO, ZERO)
    upper = decreasing.objective((1,))
    require(low == upper == ONE, "nonzero tight certificate fixture")
    require(
        conditional_lower(low, reduced, 0, 1) == upper,
        "equality must keep tied optimum",
    )
    low_bad, d_bad = certificate(decreasing, (ZERO,), Q(6), ZERO)
    correct = conditional_lower(low_bad, d_bad, 0, 1)
    bad_clipped = max(ZERO, low_bad) + max(ZERO, d_bad[0])
    bad_mixed = low + max(ZERO, d_bad[0])
    require(
        correct <= upper < bad_clipped < bad_mixed,
        "must expose clipping and mixed-certificate pitfall",
    )
    results["invalid_clipping_and_mixing_counterexample"] = encode(
        {
            "raw_lower": low_bad,
            "matching_reduced_cost": d_bad[0],
            "correct_force_one_lower": correct,
            "incumbent_upper": upper,
            "incorrect_clipped_force_one_lower": bad_clipped,
            "incorrect_mixed_force_one_lower": bad_mixed,
            "optimum_code_incorrectly_removed_by_both": [1],
        }
    )
    gap_problem = Problem(
        ((ONE, ZERO), (Q(-1), ZERO), (ZERO, ONE), (ZERO, Q(-1))),
        (Q(-1, 2), Q(1, 2), Q(-1, 2), Q(1, 2)),
        (ONE,) * 4,
        (ONE, ONE),
        ONE,
        ONE,
    )
    gap_problem.validate()
    binary_values = [
        gap_problem.objective(z)
        for z in itertools.product((0, 1), repeat=2)
        if gap_problem.feasible(z)
    ]
    continuous_point = (Q(1, 2), Q(1, 2))
    lp_lower, _ = certificate(gap_problem, (ZERO,) * 4, ZERO, ZERO)
    continuous_upper = gap_problem.objective(continuous_point)
    require(
        sum(continuous_point) <= gap_problem.count_budget,
        "fractional count feasibility",
    )
    require(
        sum(m * z for m, z in zip(gap_problem.mass, continuous_point))
        <= gap_problem.mass_budget,
        "fractional mass feasibility",
    )
    require(
        lp_lower == continuous_upper == ZERO and min(binary_values) == ONE,
        "integrality gap fixture",
    )
    results["positive_additive_integrality_gap"] = encode(
        {
            "lp_lower": lp_lower,
            "fractional_feasible_code": continuous_point,
            "fractional_upper": continuous_upper,
            "integer_optimum": min(binary_values),
        }
    )
    rejected = []
    invalid = [
        ("lambda_negative", decreasing, (Q(-1),), ZERO, ZERO),
        ("lambda_above_weight", decreasing, (Q(2),), ZERO, ZERO),
        ("mu_negative", decreasing, (ZERO,), Q(-1), ZERO),
        ("nu_negative", decreasing, (ZERO,), ZERO, Q(-1)),
        ("zero_weight_nonzero_lambda", zero_weight, (ONE,), ZERO, ZERO),
        ("float_multiplier", decreasing, (0.5,), ZERO, ZERO),
        ("missing_multiplier", decreasing, (), ZERO, ZERO),
    ]
    for name, problem, lam, mu, nu in invalid:
        try:
            certificate(problem, lam, mu, nu)
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError(f"invalid input accepted: {name}")
    results["fail_closed_inputs"] = rejected
    return results


def run():
    rng = random.Random(SEED)
    counters = dict.fromkeys(
        (
            "problems",
            "enumerated_binary_codes",
            "feasible_binary_codes",
            "candidate_certificates",
            "pointwise_global_checks",
            "pointwise_conditional_checks",
            "strict_screened_branches",
            "nonempty_strict_screened_branches",
            "equality_branches_not_screened",
        ),
        0,
    )
    cases = []
    inputs = []
    for dimension in range(1, 9):
        for case in range(8):
            rows = rng.randint(1, 6)
            a = tuple(
                tuple(
                    Q(rng.randint(-7, 7), rng.randint(1, 5)) for _ in range(dimension)
                )
                for _ in range(rows)
            )
            b = tuple(Q(rng.randint(-7, 7), rng.randint(1, 5)) for _ in range(rows))
            weight = tuple(Q(rng.randint(1, 5), rng.randint(1, 5)) for _ in range(rows))
            mass = tuple(
                Q(rng.randint(1, 5), rng.randint(1, 5)) for _ in range(dimension)
            )
            problem = Problem(
                a,
                b,
                weight,
                mass,
                Q(rng.randint(0, 2 * dimension), 2),
                sum(mass) * Q(rng.randint(0, 4), 4),
            )
            candidates = [((ZERO,) * rows, ZERO, ZERO), (weight, ZERO, ZERO)]
            for _ in range(3):
                lam = tuple(w * Q(rng.randint(0, 4), 4) for w in weight)
                candidates.append(
                    (lam, Q(rng.randint(0, 5), 3), Q(rng.randint(0, 5), 3))
                )
            result = check_problem(
                problem, candidates, rng.randrange(2**dimension), counters
            )
            result["case_in_dimension"] = case
            cases.append(result)
            inputs.append(serial_problem(problem))
    special = fixtures(counters)
    require(
        counters["nonempty_strict_screened_branches"] >= 2,
        "screening verification must be nonvacuous",
    )
    source = Path(__file__).resolve()
    return {
        "status": "passed",
        "scope": "synthetic exact rational weak-duality and coordinate-screening checks",
        "not_evaluated": [
            "Cohere",
            "training",
            "LP_or_MIP_solver",
            "production_FP32",
            "held_out_generalization",
            "recall",
            "QPS",
        ],
        "protocol": {
            "seed": SEED,
            "rng": "stdlib random.Random",
            "dimensions": list(range(1, 9)),
            "cases_per_dimension": 8,
            "random_certificates_per_problem": 5,
            "screen_rule": "conditional_raw_lower > incumbent_upper",
            "arithmetic": "fractions.Fraction only; no numerical tolerance",
            "comparison": "all feasible binary codes and their fixed-coordinate branches",
        },
        "counts": counters,
        "fixtures": special,
        "random_input_sha256": hashlib.sha256(
            json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "random_cases": cases,
        "provenance": {
            "source_path": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "python_version": platform.python_version(),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "counts": result["counts"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
