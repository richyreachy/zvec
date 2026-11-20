#!/usr/bin/env python3
# Copyright 2025-present the zvec project
# Licensed under the Apache License, Version 2.0.
"""Export row-preserving dense HDF5 text inputs for recall and txt2vecs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np


def export(dataset, output, integer=False):
    with output.open("x") as stream:
        for start in range(0, len(dataset), 4096):
            values = dataset[start : start + 4096]
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite values in {dataset.name}")
            rows = np.column_stack((np.arange(start, start + len(values)), values))
            fmt = "%d;" + " ".join(["%d" if integer else "%.9g"] * values.shape[1])
            np.savetxt(stream, rows, fmt=fmt)


def main(expected_dimension=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--include-train",
        action="store_true",
        help="Also export train.txt for txt2vecs",
    )
    args = parser.parse_args()
    names = ["queries.txt", "neighbors.txt"]
    if args.include_train:
        names.append("train.txt")
    if any((args.output / name).exists() for name in names):
        raise FileExistsError("Output files already exist; use a new output directory")
    with h5py.File(args.input, "r") as source:
        train, test, neighbors = (
            source[name] for name in ("train", "test", "neighbors")
        )
        if (
            train.ndim != 2
            or test.ndim != 2
            or train.shape[1] <= 0
            or test.shape[1] != train.shape[1]
            or (expected_dimension is not None and train.shape[1] != expected_dimension)
            or not len(train)
            or not len(test)
            or neighbors.ndim != 2
            or len(neighbors) != len(test)
            or neighbors.shape[1] < 10
            or not np.issubdtype(neighbors.dtype, np.integer)
        ):
            raise ValueError(
                "Expected matching dense vectors and at least 10 neighbor IDs per query"
                + (
                    f"; dimension must be {expected_dimension}"
                    if expected_dimension is not None
                    else ""
                )
            )
        for start in range(0, len(neighbors), 4096):
            ids = neighbors[start : start + 4096]
            if ids.min() < 0 or ids.max() >= len(train):
                raise ValueError("Neighbor IDs outside the training row range")
        args.output.mkdir(parents=True, exist_ok=True)
        export(test, args.output / "queries.txt")
        export(neighbors, args.output / "neighbors.txt", integer=True)
        if args.include_train:
            export(train, args.output / "train.txt")
        sys.stdout.write(
            json.dumps(
                {
                    "train_rows": len(train),
                    "queries": len(test),
                    "dimension": train.shape[1],
                    "files": names,
                    "ids": "zero-based HDF5 row IDs; no normalization",
                }
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
