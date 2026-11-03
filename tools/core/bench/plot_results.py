#!/usr/bin/env python3
# Copyright 2025-present the zvec project
# Licensed under the Apache License, Version 2.0.
"""Plot measured SIFT/GIST/GloVe points from a benchmark results JSON file."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

ALGORITHMS = {"Zvec": "#2563eb", "Huawei qsgngt": "#d97706", "Descartes": "#0f766e"}
DATASETS = {"sift": "SIFT-128", "gist": "GIST-960", "glove": "GloVe-100"}


def frontier(points, metric):
    result, best = [], -math.inf
    for point in sorted(points, key=lambda p: (-p[metric], -p["qps"])):
        if point["qps"] > best:
            result.append(point)
            best = point["qps"]
    return sorted(result, key=lambda p: p[metric])


def load_points(path):
    points = json.loads(path.read_text())
    for point in points:
        point["dataset"] = point["dataset"].lower()
        point["algorithm"] = {"Zvec FP32": "Zvec"}.get(
            point["algorithm"], point["algorithm"]
        )
        if "strict_recall" not in point:
            point["strict_recall"] = point["recall"]
        if point["dataset"] not in DATASETS or point["algorithm"] not in ALGORITHMS:
            raise ValueError("Unknown dataset or algorithm in points JSON")
        if not math.isfinite(point["qps"]) or point["qps"] <= 0:
            raise ValueError("QPS must be finite and positive")
        for metric in ["ann_recall", "strict_recall"]:
            if not 0 <= point[metric] <= 1:
                raise ValueError("Recall must be in [0, 1]")
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("points", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-recall", type=float, default=0.88)
    args = parser.parse_args()
    if not 0 <= args.min_recall < 1:
        parser.error("--min-recall must be in [0, 1)")
    points = load_points(args.points)
    plt.switch_backend("Agg")

    output = args.output or args.points.parent
    output.mkdir(parents=True, exist_ok=True)

    def save(fig, path):
        fig.savefig(path, dpi=180)
        if path.suffix == ".svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines())
                + "\n"
            )

    def draw(ax, dataset, metric):
        found = False
        for algorithm, color in ALGORITHMS.items():
            rows = frontier(
                [
                    p
                    for p in points
                    if p["dataset"] == dataset and p["algorithm"] == algorithm
                ],
                metric,
            )
            if rows:
                found = True
                label = algorithm
                if algorithm == "Huawei qsgngt":
                    quantizers = {
                        p.get("index_metadata", {}).get("construction", {}).get("Q")
                        for p in rows
                    }
                    if len(quantizers) == 1 and None not in quantizers:
                        label += f" (Q={next(iter(quantizers))})"
                ax.plot(
                    [100 * p[metric] for p in rows],
                    [p["qps"] for p in rows],
                    marker="o",
                    markersize=4,
                    linewidth=2,
                    color=color,
                    label=label,
                )
        if not found:
            ax.text(0.5, 0.5, "No measured points", transform=ax.transAxes, ha="center")
        else:
            ax.legend(fontsize=9)
        ax.set(
            title=DATASETS[dataset],
            xlabel=(
                "ANN Recall@10 (%)"
                if metric == "ann_recall"
                else "Strict ID Recall@10 (%)"
            ),
            ylabel="Queries per second",
            xlim=(100 * args.min_recall, 100),
            yscale="log",
        )
        ax.grid(True, which="both", alpha=0.2)

    for metric, suffix in [("ann_recall", "ann"), ("strict_recall", "strict")]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
        for ax, dataset in zip(axes, DATASETS, strict=True):
            draw(ax, dataset, metric)
        fig.tight_layout()
        for extension in ["png", "svg"]:
            save(fig, output / f"recall-qps-{suffix}.{extension}")
        plt.close(fig)
        for dataset in DATASETS:
            fig, ax = plt.subplots(figsize=(6.4, 4.8))
            draw(ax, dataset, metric)
            fig.tight_layout()
            for extension in ["png", "svg"]:
                save(fig, output / f"{dataset}-{suffix}.{extension}")
            plt.close(fig)


if __name__ == "__main__":
    main()
