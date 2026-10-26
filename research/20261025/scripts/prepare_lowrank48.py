from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")


def emit(*items, **_kwargs):
    logging.info(" ".join(map(str, items)))


R = Path(__file__).resolve().parent.parent
out = R / "lowrank48"
out.mkdir(exist_ok=False)
source = Path("/home/xingrui/bench/symphony_qg/cohere/original/data/base.f32")
seed = 2026102648
rng = np.random.default_rng(seed)
ids = np.sort(rng.choice(1000000, 65536, replace=False))
x = np.asarray(
    np.memmap(source, dtype=np.float32, mode="r", shape=(1000000, 768))[ids]
).copy()
start = time.time()
norms = np.linalg.norm(x, axis=1)
x /= np.maximum(norms[:, None], 1e-30)
mean = x.mean(axis=0, dtype=np.float64)
x -= mean.astype(np.float32)
cov = (x.T @ x) / len(x)
e, u = np.linalg.eigh(cov.astype(np.float64))
e = e[::-1]
basis = u[:, ::-1][:, :16].T.astype(np.float32)
basis.tofile(out / "basis.f32")
np.save(out / "sample_ids.npy", ids)
err = float(np.max(np.abs(basis @ basis.T - np.eye(16))))
assert err < 2e-6
metadata = {
    "seed": seed,
    "source": str(source),
    "sample_count": len(ids),
    "dimension": 768,
    "basis_count": 16,
    "queries_or_ground_truth_read": False,
    "basis_sha256": hashlib.sha256((out / "basis.f32").read_bytes()).hexdigest(),
    "basis_orthogonality_max_error": err,
    "sample_original_norm_range": [float(norms.min()), float(norms.max())],
    "covariance_eigenvalues": e.tolist(),
    "variance_fraction": {str(k): float(e[:k].sum() / e.sum()) for k in [4, 8, 16]},
    "elapsed_seconds": time.time() - start,
}
(out / "metadata.json").write_text(json.dumps(metadata, indent=2))
emit(
    json.dumps({k: v for k, v in metadata.items() if k != "covariance_eigenvalues"}),
    flush=True,
)
