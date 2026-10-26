from __future__ import annotations

import hashlib
import json
import logging
import os
import struct
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)
R = Path(__file__).resolve().parent.parent
D = R / "models59"
start = time.monotonic()
source = D / "anchors.bin"
header = np.fromfile(source, dtype="<u4", count=5)
assert list(header[:2]) == [1363624281, 1]
n, dim, docs = map(int, header[2:])
assert (n, dim, docs) == (65536, 768, 1000000)
dtype = np.dtype([("id", "<u4"), ("key", "<u8"), ("vector", "<f4", (dim,))])
assert source.stat().st_size == 20 + n * dtype.itemsize
records = np.memmap(source, mode="r", dtype=dtype, offset=20, shape=(n,))
ids = records["id"].copy()
keys = records["key"].copy()
x = records["vector"].copy()
assert len(np.unique(ids)) == n
assert len(np.unique(keys)) == n
expected = (np.arange(n, dtype=np.uint64) * 2654435761 + 7193) % docs
assert np.array_equal(ids, expected)
assert np.isfinite(x).all()
norms = np.sqrt(np.sum(x.astype(np.float64) ** 2, axis=1))
assert np.max(np.abs(norms - 1)) < 0.0001
mean = np.fromfile(R / "mean54/mean.f32", dtype="<f4")
assert mean.shape == (dim,)
base = np.memmap(
    "/home/xingrui/bench/symphony_qg/cohere/original/data/base.f32",
    mode="r",
    dtype="<f4",
    shape=(docs, dim),
)
assert keys.max() < docs
maxdiff = 0.0
for i in range(0, n, 1024):
    maxdiff = max(
        maxdiff, float(np.max(np.abs(x[i : i + 1024] - base[keys[i : i + 1024]])))
    )
assert maxdiff < 2e-06
rng = np.random.default_rng(202610261259)
centers = []
targets = []
counts = []
depths = []
splits = []
fallbacks = [0]


def normalized(v):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, 1e-30)


def make_group(rows, depth):
    assert depth < 16
    g = len(centers)
    centers.append(np.zeros((16, dim), dtype=np.float32))
    targets.append(np.full(16, -2147483648, dtype=np.int32))
    counts.append(0)
    depths.append(depth)
    if len(rows) <= 16:
        counts[g] = len(rows)
        centers[g][: len(rows)] = x[rows]
        targets[g][: len(rows)] = -ids[rows].astype(np.int64) - 1
        return g
    data = x[rows]
    c = data[rng.choice(len(rows), 16, replace=False)].copy()
    labels = np.zeros(len(rows), dtype=np.int32)
    for _iteration in range(10):
        scores = data @ c.T
        labels = np.argmax(scores, axis=1)
        sizes = np.bincount(labels, minlength=len(c))
        updated = []
        for j in range(len(c)):
            if sizes[j]:
                updated.append(
                    np.mean(data[labels == j], axis=0, dtype=np.float64).astype(
                        np.float32
                    )
                )
        c = normalized(np.asarray(updated, dtype=np.float32))
    labels = np.argmax(data @ c.T, axis=1)
    parts = [rows[labels == j] for j in range(len(c)) if np.any(labels == j)]
    if len(parts) < 2:
        fallbacks[0] += 1
        parts = [a for a in np.array_split(rows, 16) if len(a)]
    counts[g] = len(parts)
    splits.append([len(a) for a in parts])
    for j, part in enumerate(parts):
        centers[g][j] = normalized(
            np.mean(x[part], axis=0, dtype=np.float64).astype(np.float32)
        )
        if len(part) == 1:
            centers[g][j] = x[part[0]]
            targets[g][j] = -int(ids[part[0]]) - 1
        else:
            targets[g][j] = make_group(part, depth + 1)
    return g


assert make_group(np.arange(n), 0) == 0
centers = np.asarray(centers, dtype="<f4")
targets = np.asarray(targets, dtype="<i4")
counts = np.asarray(counts, dtype="<u4")
ng = len(centers)
centers.tofile(D / "centers.f32")
targets.tofile(D / "targets.i32")
counts.tofile(D / "counts.u32")
leaves = []
with (D / "router.bin").open("xb") as f:
    f.write(struct.pack("<5I", 1363628633, 1, dim, ng, docs))
    f.write(mean.tobytes())
    for c, t, count in zip(centers, targets, counts, strict=True):
        residual = c - mean
        offset = residual.min(axis=1)
        scale = (residual.max(axis=1) - offset) / np.float32(255.0)
        inv = np.divide(
            np.float32(1.0), scale, out=np.zeros_like(scale), where=scale > 0
        )
        codes = np.clip(
            np.rint((residual - offset[:, None]) * inv[:, None]), 0, 255
        ).astype("u1")
        md = (c.astype(np.float64) @ mean.astype(np.float64)).astype("<f4")
        f.write(struct.pack("<I", int(count)))
        f.write(t.tobytes())
        f.write(scale.astype("<f4").tobytes())
        f.write(offset.astype("<f4").tobytes())
        f.write(md.tobytes())
        f.write(codes.tobytes())
        leaves.extend(-int(v) - 1 for v in t[:count] if v < 0)
assert sorted(leaves) == sorted(ids.tolist())
files = {
    str(p.relative_to(R)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in [
        source,
        D / "router.bin",
        D / "centers.f32",
        D / "targets.i32",
        D / "counts.u32",
        R / "mean54/mean.f32",
        R / "scripts/train_router59.py",
    ]
}
meta = {
    "seed": 202610261259,
    "iterations_per_split": 10,
    "fanout": 16,
    "anchors": n,
    "dimension": dim,
    "docs": docs,
    "groups": ng,
    "max_group_depth": max(depths),
    "degenerate_split_fallbacks": fallbacks[0],
    "leaf_count": len(leaves),
    "all_anchor_ids_once": True,
    "normalized_error_max": float(np.max(np.abs(norms - 1))),
    "external_key_base_max_abs_error": maxdiff,
    "internal_id_equals_external_key_fraction": float(np.mean(ids == keys)),
    "training_seconds": time.monotonic() - start,
    "model_bytes": (D / "router.bin").stat().st_size,
    "query_or_gt_inputs": [],
    "hashes": files,
    "threads": {
        k: os.environ.get(k) for k in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"]
    },
}
(D / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
(R / "results/router59_training.json").write_text(json.dumps(meta, indent=2) + "\n")
logging.info("%s", json.dumps(meta, indent=2))
logging.info("%s", "TRAIN59 COMPLETE")
