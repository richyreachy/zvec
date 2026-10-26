from __future__ import annotations

import hashlib
import json
import logging
import struct
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
M = R / "models77"
N = 1000000
assert "PREPARE77 COMPLETE" in (R / "logs/prepare77.log").read_text()
assert "ROTATE75 COMPLETE vectors=1000000" in (R / "logs/rotate77.log").read_text()


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        while block := f.read(16 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


a = json.loads((M / "config.json").read_text())
for name, h in a["input_hashes_verified"].items():
    assert sha(R / name) == h
for name, h in a["files"].items():
    assert sha(M / name) == h
proxy = np.memmap(M / "proxies.u32", mode="r", dtype="<u4", shape=(N, 16))
pos = np.memmap(M / "positions.u32", mode="r", dtype="<u4", shape=(N, 16))
g = np.memmap(
    R / "models64/full_graph.bin",
    mode="r",
    dtype=np.dtype([("key", "<u8"), ("degree", "<u4"), ("ids", "<u4", (32,))]),
    offset=32,
    shape=(N,),
)
c = np.memmap(
    R / "models64/full_candidates.bin",
    mode="r",
    dtype=np.dtype(
        [("query", "<u4"), ("ids", "<u4", (200,)), ("distances", "<f4", (200,))]
    ),
    offset=32,
    shape=(N,),
)
inv = np.empty(N, dtype=np.uint32)
inv[c["query"]] = np.arange(N, dtype=np.uint32)
for start in range(0, N, 2048):
    rows = np.arange(start, min(start + 2048, N))
    full = rows[g["degree"][rows] == 32]
    short = rows[g["degree"][rows] < 32]
    assert np.all(proxy[short] == np.iinfo(np.uint32).max)
    assert np.all(pos[short] == np.iinfo(np.uint32).max)
    p = np.asarray(proxy[full])
    z = np.asarray(pos[full])
    assert np.all(p < N)
    assert np.all(np.diff(np.sort(p, axis=1), axis=1) > 0)
    assert not np.any(p == full[:, None])
    assert not np.any(p[:, :, None] == g["ids"][full][:, None, :])
    assert np.array_equal(p, c["ids"][inv[full, None], z])
    assert np.all(z // 50 == np.repeat(np.arange(4), 4)[None, :])
rot = np.memmap(M / "rotated.f32", mode="r", dtype="<f4", shape=(N, 768))
oldids = np.fromfile(R / "models75/ids.u32", dtype="<u4").reshape(2048, 65)
old = np.memmap(
    R / "models75/rotated.f32", mode="r", dtype="<f4", shape=(2048, 65, 768)
)
for start in range(0, 2048, 32):
    assert np.array_equal(rot[oldids[start : start + 32]], old[start : start + 32])
assert (M / "rotated.f32").stat().st_size == N * 768 * 4
rotsha = sha(M / "rotated.f32")
algsha = sha(R / "scripts/independent76.h")
# Stage32 validated both source spellings; retain the actual hash in metadata.
raw_algorithm_hash = json.loads(
    (R / "results/independent76_manifest.json").read_text()
)["files"]["scripts/independent76.h"]
archived_algorithm_hash = json.loads(
    (R / "results/stage32_postformat_validation.json").read_text()
)["formatted_script_hashes"]["independent76.h"]
assert algsha in {raw_algorithm_hash, archived_algorithm_hash}
meta = {
    "database_count": N,
    "dimension": 768,
    "max_neighbors": 32,
    "proxies_per_full_center": 16,
    "codec_bytes": 3328,
    "record_bytes": 3600,
    "objective": "independent sum(e.T@(Q.T@Q+lambda*I)@e)",
    "lambda": "trace(Q.T@Q)/768",
    "proxy_scale": 0.25,
    "sweeps": 2,
    "flips_per_edge_per_sweep": 4,
    "alpha_cap_ratio": 1.02,
    "minimum_delta_relative": 1e-13,
    "proxy_seed": 7602,
    "input_sha256": a["input_hashes_verified"],
    "proxy_sha256": a["files"]["proxies.u32"],
    "rotated_sha256": rotsha,
    "algorithm_header_sha256": algsha,
    "builder_source_sha256": sha(R / "scripts/fullbuild77.cc"),
    "builder_binary_sha256": sha(R / "fullbuild77"),
    "rotation_binary_sha256": sha(R / "rotate75"),
    "short_blocks": "native codec",
    "center_and_neighbor_keys": "exact original graph order;unused UINT64_MAX",
    "audit_record_bytes": 64,
    "audit_layout": "u32 updates,u32 flags,7xf64(lambda,initial,final,maxscale,radial,FP32radial,degree)",
    "history_layout": "257xf64 per center;unused NaN;short allNaN",
    "moves_layout": "256x3xu16(sweep,lane,bit);unused UINT16_MAX",
    "compact_layout": "32x96bytes;big-endian sign bits;unused lanes0",
    "requires_independent_validation_before_loading": True,
}
for count in [8192, N]:
    payload = json.dumps(
        dict(meta, record_count=count), sort_keys=True, separators=(",", ":")
    ).encode()
    assert len(payload) <= 4064
    with (M / f"header{count}.bin").open("xb") as f:
        f.write(
            struct.pack("<8sIIQQ", b"QGIC77\x00\x00", 1, len(payload), count, 3600)
            + payload
            + b"\x00" * (4064 - len(payload))
        )
result = {
    "all_proxy_rows_structurally_validated": N,
    "full_degree_rows": 997722,
    "native_short_rows": 2278,
    "phase75_rotation_vectors_byte_identical": 2048 * 65,
    "rotated_sha256": rotsha,
    "full_proxy_rng_replay_performed": False,
    "input_hashes_rechecked": True,
    "metadata": meta,
    "headers": {str(count): sha(M / f"header{count}.bin") for count in [8192, N]},
    "no_query_or_gt_input": True,
}
(R / "results/independent77_preflight.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
logging.info(json.dumps(result, indent=2))
logging.info("PREFLIGHT77 COMPLETE")
