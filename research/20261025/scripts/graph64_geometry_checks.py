from __future__ import annotations

import json
import logging
from itertools import combinations
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent


def spherical(theta, phi):
    t, p = np.deg2rad([theta, phi])
    return [np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)]


x = np.array([spherical(0, 0), spherical(50, 0), spherical(80, 35), spherical(80, -35)])


def angular_prune(points, theta):
    selected = []
    bar = np.cos(np.deg2rad(theta))
    for i in range(len(points)):
        if all(np.dot(points[i], points[j]) <= bar for j in selected):
            selected.append(i)
    return selected


strict = angular_prune(x, 60)
relaxed = angular_prune(x, 45)
assert strict == [0, 2, 3]
assert relaxed == [0, 1]


def fill(distance, seeds, k):
    selected = list(seeds)
    while len(selected) < k:
        remaining = [i for i in range(len(distance)) if i not in selected]
        chosen = max(
            remaining,
            key=lambda i: (
                min(distance[i, j] for j in selected) if selected else 1.0,
                -i,
            ),
        )
        selected.append(chosen)
    return selected


rng = np.random.default_rng(64026)
maxratio = 0.0
cases = 0
for _ in range(500):
    v = rng.normal(size=(8, 5))
    v /= np.linalg.norm(v, axis=1)[:, None]
    distance = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * (v @ v.T)))
    np.fill_diagonal(distance, 0)
    for seeds in [[], [0], [0, 1]]:
        k = 3
        greedy = fill(distance, seeds, k)
        assert len(set(greedy)) == k
        assert set(seeds) <= set(greedy)
        radius = float(np.max(np.min(distance[:, greedy], axis=1)))
        remaining = [i for i in range(8) if i not in seeds]
        optimal = min(
            float(np.max(np.min(distance[:, [*seeds, *extra]], axis=1)))
            for extra in combinations(remaining, k - len(seeds))
        )
        assert radius <= 2 * optimal + 1e-12
        maxratio = max(maxratio, radius / optimal)
        cases += 1
summary = {
    "counterexample_unit_directions": x.tolist(),
    "greedy_angle60_selected": strict,
    "greedy_angle45_selected": relaxed,
    "greedy_threshold_cardinality_is_not_monotone": True,
    "fixed_seed_farthest_fill_exhaustive_cases": cases,
    "maximum_observed_radius_ratio": maxratio,
    "guarantee_scope": "2-approximation of finite nonzero-residual angular chord covering radius with prescribed seeds; not graph navigation, recall, or QPS",
    "no_novelty_claim": True,
}
(R / "results/graph64_geometry_checks.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
logging.info("%s", json.dumps(summary, indent=2))
logging.info("%s", "GEOMETRY64 COMPLETE")
