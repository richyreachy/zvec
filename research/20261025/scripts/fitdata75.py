from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
M = R / "models75"
assert "DATA75 COMPLETE" in (R / "logs/validate_data75.log").read_text()
x = np.memmap(M / "rotated.f32", mode="r", dtype="<f4", shape=(2048, 65, 768))
sel = np.fromfile(M / "selection_positions.u32", dtype="<u4")
val = np.fromfile(M / "validation_positions.u32", dtype="<u4")
assert not set(sel) & set(val)
indices = np.r_[np.arange(33), 33 + sel]
assert len(indices) == 49
with (M / "fit.f32").open("xb") as f:
    for i in range(2048):
        np.asarray(x[i, indices], dtype="<f4").tofile(f)
a = {
    "fit_shape": [2048, 49, 768],
    "source_positions": indices.tolist(),
    "validation_positions_excluded": (33 + val).tolist(),
    "optimizer_reads_fit_file_only": True,
    "fit_sha256": hashlib.sha256((M / "fit.f32").read_bytes()).hexdigest(),
}
(R / "results/coupled75_fit_data.json").write_text(json.dumps(a, indent=2) + "\n")
logging.info("%s", json.dumps(a, indent=2))
logging.info("%s", "FITDATA75 COMPLETE")
