from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
M = R / "models77"
M.mkdir(exist_ok=False)
started = time.monotonic()


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        while block := f.read(16 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


expected = json.loads((R / "results/stage31_postformat_validation.json").read_text())[
    "models64_input_hashes_verified"
]
for name, h in expected.items():
    assert sha(R / name) == h, name
N = 1000000
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
assert np.array_equal(np.sort(g["key"]), np.arange(N))
assert np.array_equal(np.sort(c["query"]), np.arange(N))
assert np.all(g["degree"] <= 32)
inverse = np.empty(N, dtype=np.uint32)
inverse[c["query"]] = np.arange(N, dtype=np.uint32)
proxy = np.memmap(M / "proxies.u32", mode="w+", dtype="<u4", shape=(N, 16))
pos = np.memmap(M / "positions.u32", mode="w+", dtype="<u4", shape=(N, 16))
proxy[:] = np.iinfo(np.uint32).max
pos[:] = np.iinfo(np.uint32).max
rng = np.random.default_rng(7602)
full = 0
short = 0
minimum = 50
for row in range(N):
    degree = int(g["degree"][row])
    if degree != 32:
        short += 1
        continue
    excluded = {row, *g["ids"][row].tolist()}
    assert len(excluded) == 33
    pool = c["ids"][inverse[row]]
    assert len(set(pool.tolist())) == 200
    assert np.all(pool < N)
    for layer in range(4):
        legal = [
            p
            for p in range(layer * 50, (layer + 1) * 50)
            if int(pool[p]) not in excluded
        ]
        minimum = min(minimum, len(legal))
        assert len(legal) >= 8
        chosen = rng.choice(legal, 8, replace=False)[:4]
        pos[row, layer * 4 : layer * 4 + 4] = chosen
        proxy[row, layer * 4 : layer * 4 + 4] = pool[chosen]
    assert len(set(proxy[row].tolist())) == 16
    assert not set(proxy[row].tolist()) & excluded
    full += 1
    if (row + 1) % 32768 == 0:
        logging.info(
            "%s %s %s %s", "PREPARE77", row + 1, "seconds", time.monotonic() - started
        )
assert full == 997722
assert short == 2278
proxy.flush()
pos.flush()
np.arange(N, dtype="<u4").tofile(M / "all_ids.u32")
eligible = np.flatnonzero(g["degree"] == 32)
np.sort(np.random.default_rng(7701).choice(eligible, 2048, replace=False)).astype(
    "<u4"
).tofile(M / "validation_ids.u32")
a = {
    "centers": N,
    "full_degree_centers": full,
    "native_short_blocks": short,
    "proxy_seed": 7602,
    "validation_seed": 7701,
    "validation_centers": 2048,
    "sampling": "internal ID ascending;short skips RNG;four50 bins;choose8withoutreplacement then first4",
    "minimum_eligible_pool": minimum,
    "primary_or_extra_queries_or_gt_read": False,
    "input_hashes_verified": expected,
    "files": {
        name: sha(M / name)
        for name in [
            "proxies.u32",
            "positions.u32",
            "all_ids.u32",
            "validation_ids.u32",
        ]
    },
    "elapsed_seconds": time.monotonic() - started,
}
(M / "config.json").write_text(json.dumps(a, indent=2) + "\n")
(R / "results/independent77_data.json").write_text(json.dumps(a, indent=2) + "\n")
logging.info(json.dumps(a, indent=2))
logging.info("PREPARE77 COMPLETE")
