"""Exact finite-population and center-control algebra; no Cohere scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path


def moments(values):
    mean = sum(values, F(0)) / len(values)
    variance = sum(((x - mean) ** 2 for x in values), F(0)) / len(values)
    return mean, variance


def check_case(q, c, signs, alpha, constant, group_size):
    dim = len(q)
    assert dim == len(c) == len(signs)
    assert dim % group_size == 0
    query_terms = [q[i] * signs[i] for i in range(dim)]
    center_terms = [c[i] * signs[i] for i in range(dim)]
    qg = [sum(query_terms[i : i + group_size]) for i in range(0, dim, group_size)]
    cg = [sum(center_terms[i : i + group_size]) for i in range(0, dim, group_size)]
    residual = [x - y for x, y in zip(qg, cg, strict=True)]
    n = len(qg)
    full = constant - alpha * sum(residual)
    base = constant + alpha * sum(cg)
    checks = 0
    for m in range(1, n + 1):
        rho = F(n, m)
        raw, controlled = [], []
        for sample in combinations(range(n), m):
            sq = sum(qg[i] for i in sample)
            sc = sum(cg[i] for i in sample)
            raw.append(base - alpha * rho * sq)
            corrected_base = base + alpha * (rho * sc - sum(cg))
            value = corrected_base - alpha * rho * sq
            assert value == constant - alpha * rho * (sq - sc)
            controlled.append(value)
            checks += 1
        for values, population in ((raw, qg), (controlled, residual)):
            mean, variance = moments(values)
            population_mean = sum(population) / n
            population_variance = sum(
                (x - population_mean) ** 2 for x in population
            ) / (n - 1)
            expected_variance = (
                alpha**2 * n**2 * (1 - F(m, n)) * population_variance / m
            )
            assert mean == full
            assert variance == expected_variance
        if m == n:
            assert raw == controlled == [full]
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rng = random.Random(8202)
    cases = checks = 0
    for group_size in (1, 4):
        dim = 8 * group_size
        for _ in range(32):
            q = [F(rng.randrange(-16, 17), 8) for _ in range(dim)]
            c = [F(rng.randrange(-16, 17), 8) for _ in range(dim)]
            signs = [rng.choice((-1, 1)) for _ in range(dim)]
            checks += check_case(q, c, signs, F(3, 7), F(2, 5), group_size)
            cases += 1

    # A fixed prefix has no sampling expectation guarantee.
    population = [F(1), F(1), F(0), F(0)]
    assert 2 * sum(population[:2]) == 4 != sum(population)

    # Every fixed size is unbiased, but data-dependent stopping need not be.
    # Randomly permute [0, 1]; stop at m=1 if the first value is 1, else m=2.
    stopped = [F(2), F(1)]
    assert sum(stopped) / 2 == F(3, 2) != 1

    # Residual control helps q=c, but can hurt q=0, c nonzero.
    q = [F(1), F(2), F(3), F(4)]
    zero = [F(0)] * 4
    raw_variance = moments([4 * x for x in q])[1]
    residual_zero_variance = moments([4 * (x - x) for x in q])[1]
    residual_hurt_variance = moments(
        [4 * (x - y) for x, y in zip(zero, q, strict=True)]
    )[1]
    assert raw_variance > residual_zero_variance == 0
    assert moments(zero)[1] == 0 < residual_hurt_variance

    source = Path(__file__).resolve()
    report = {
        "passed": True,
        "seed": 8202,
        "arithmetic": "fractions.Fraction; exact exhaustive subsets",
        "random_cases": cases,
        "population_units": ["one coordinate", "four-coordinate LUT group"],
        "subset_algebra_checks": checks,
        "fixed_size_mean_and_variance_checks": cases * 8 * 2,
        "fixed_prefix_counterexample": {"estimate": 4, "full_sum": 2},
        "optional_stopping_counterexample": {"mean_estimate": "3/2", "full_sum": 1},
        "control_variate_can_help_or_hurt": True,
        "no_original_rabitq_or_fp32_error_bound_claim": True,
        "no_cohere_scoring_search_or_timing": True,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "source_snapshot.py").write_bytes(source.read_bytes())
    print(json.dumps(report, indent=2))  # noqa: T201 - command-line evidence


if __name__ == "__main__":
    main()
