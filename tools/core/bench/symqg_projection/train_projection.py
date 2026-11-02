#!/usr/bin/env python3
# Copyright 2025-present the zvec project
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Train the experimental projected codec from database residuals only.

Input residuals must use the same rotated coordinate system and decoded affine /
low-rank anchors as the index being rebuilt. The metric is a frozen RAD1 model
of the covariance of database proxy offsets, not ANN test queries.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


def train(residuals, metric, projected, seed):
    residuals = np.asarray(residuals, dtype=np.float64)
    if residuals.ndim != 2 or not residuals.shape[0]:
        raise ValueError("residuals must be a nonempty N x D matrix")
    full = residuals.shape[1]
    if not 0 < projected <= full <= 4096 or full % 64 or projected % 64:
        raise ValueError("dimensions must be multiples of 64, with 0 < p <= D <= 4096")
    if not np.isfinite(residuals).all():
        raise ValueError("nonfinite database residuals")
    if len(metric) < 32:
        raise ValueError("truncated RAD1 metric")
    magic, dimension, rank, sweeps, sigma, guard = struct.unpack_from("<4I2d", metric)
    if (
        magic != 0x52414431
        or dimension != full
        or rank > 64
        or sweeps > 8
        or not np.isfinite(sigma)
        or sigma < 0
        or not np.isfinite(guard)
        or guard < 1
        or len(metric) != 32 + full * rank * 8
    ):
        raise ValueError("invalid RAD1 metric")
    basis = np.frombuffer(metric, dtype="<f8", offset=32).reshape(full, rank)
    if not np.isfinite(basis).all():
        raise ValueError("nonfinite metric basis")
    covariance = residuals.T @ residuals / len(residuals)
    _, vectors = np.linalg.eigh(covariance)
    rng = np.random.default_rng(seed + projected)
    mixing = np.linalg.qr(rng.normal(size=(projected, projected)))[0]
    projection = vectors[:, ::-1][:, :projected] @ mixing
    matrix = np.ascontiguousarray(projection.T, dtype="<f4")
    extension = (
        struct.pack("<5I2d", 0x50524A31, full, projected, rank, sweeps, sigma, guard)
        + matrix.tobytes()
        + basis.astype("<f8").tobytes()
    )
    if len(extension) > 4 << 20:
        raise ValueError("projection extension exceeds the codec size limit")
    return extension


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--residuals",
        type=Path,
        required=True,
        help="database-only anchored, rotated N x D .npy array",
    )
    parser.add_argument(
        "--metric",
        type=Path,
        required=True,
        help="frozen database-only RAD1 covariance metric",
    )
    parser.add_argument("--dimension", type=int, default=192)
    parser.add_argument("--seed", type=int, default=2026110208)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = train(
        np.load(args.residuals, allow_pickle=False),
        args.metric.read_bytes(),
        args.dimension,
        args.seed,
    )
    with args.output.open("xb") as stream:
        stream.write(data)


if __name__ == "__main__":
    main()
