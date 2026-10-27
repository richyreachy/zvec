from __future__ import annotations

import importlib
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
np = importlib.import_module("numpy")

R = Path(__file__).resolve().parent.parent
prefix = R / "models65" / sys.argv[1]
n = 1000000
G = np.memmap(
    R / "models64/full_graph.bin",
    dtype=np.dtype([("key", "<u8"), ("degree", "<u4"), ("ids", "<u4", (32,))]),
    mode="r",
    offset=32,
    shape=(n,),
)
V = np.memmap(
    R / "models64/full_vectors.f32", dtype="<f4", mode="r", offset=16, shape=(n, 768)
)
C = np.memmap(
    R / "models64/full_candidates.bin",
    dtype=np.dtype([("q", "<u4"), ("ids", "<u4", (200,)), ("dist", "<f4", (200,))]),
    mode="r",
    offset=32,
    shape=(n,),
)
inv = np.empty(n, dtype=np.int64)
inv[C["q"]] = np.arange(n)
first = np.fromfile(str(prefix) + "_first.u32", dtype="<u4").reshape(-1, 33)
nearest = np.fromfile(str(prefix) + "_nearest.u32", dtype="<u4").reshape(-1, 33)
angular = np.fromfile(str(prefix) + "_angular.u32", dtype="<u4").reshape(-1, 33)
stats = np.fromfile(str(prefix) + "_stats.f64", dtype="<f8").reshape(-1, 11)
count = len(first)
assert nearest.shape == angular.shape == first.shape
for rows in (first, nearest, angular):
    assert ((rows[:, 0] > 0) & (rows[:, 0] <= 32)).all()
    for lo in range(0, count, 8192):
        a = rows[lo : lo + 8192]
        valid = np.arange(32)[None, :] < a[:, 0, None]
        assert (a[:, 1:][valid] < n).all()
        assert (a[:, 1:][~valid] == 2**32 - 1).all()
        assert not ((a[:, 1:] == np.arange(lo, lo + len(a))[:, None]) & valid).any()
        sortedids = np.sort(a[:, 1:], axis=1)
        assert not (
            (sortedids[:, 1:] == sortedids[:, :-1]) & (sortedids[:, 1:] != 2**32 - 1)
        ).any()
assert (nearest[:, 0] == 32).all()
assert (angular[:, 0] == 32).all()
mask = np.arange(32)[None, :] < first[:, 0, None]
sources = np.broadcast_to(np.arange(count, dtype=np.uint32)[:, None], (count, 32))[mask]
dests = first[:, 1:][mask]
order = np.argsort(dests, kind="stable")
incoming = sources[order]
off = np.concatenate(([0], np.cumsum(np.bincount(dests, minlength=n))))
assert np.array_equal(stats[:, 3], np.diff(off)[:count])
rng = np.random.default_rng(65)
sample = np.unique(
    np.concatenate(
        (np.arange(min(16, count)), rng.choice(count, min(48, count), replace=False))
    )
)
maxcoverage = 0.0
checked = 0
for final in (False, True):
    for q in sample:
        ids = np.concatenate(
            (
                C[inv[q]]["ids"],
                G[q]["ids"][: G[q]["degree"]],
                incoming[off[q] : off[q + 1]]
                if final
                else np.array([], dtype=np.uint32),
            )
        )
        ids = np.unique(ids)
        ids = ids[ids != q]
        residual = V[ids].astype(np.float64) - V[q].astype(np.float64)
        r = np.einsum("ij,ij->i", residual, residual)
        order = np.lexsort((ids, r))[:750]
        ids = ids[order]
        r = r[order]
        residual = residual[order]
        pair = r[:, None] + r[None, :] - 2 * (residual @ residual.T)
        np.fill_diagonal(pair, 0)
        seeds = []
        for i in range(len(ids)):
            if all(pair[i, j] >= r[i] for j in seeds):
                seeds.append(i)
            if len(seeds) == 32:
                break
        if not final:
            assert first[q, 0] == len(seeds), ("first", q)
            assert np.array_equal(first[q, 1 : len(seeds) + 1], ids[seeds]), (
                "first",
                q,
            )
            continue
        assert stats[q, 2] == len(seeds)
        near = seeds + [i for i in range(len(ids)) if i not in seeds][: 32 - len(seeds)]
        assert np.array_equal(nearest[q, 1:], ids[near]), ("nearest", q)
        nonzero = r > 0
        chord = np.full(pair.shape, 4.0)
        valid = nonzero[:, None] & nonzero[None, :]
        denom = 2 * np.sqrt(r[:, None] * r[None, :])
        chord[valid] = np.clip(
            2 - 2 * (r[:, None] + r[None, :] - pair)[valid] / denom[valid], 0, 4
        )
        np.fill_diagonal(chord, 0)
        chosen = seeds.copy()
        while len(chosen) < 32:
            remain = [i for i in range(len(ids)) if i not in chosen and nonzero[i]]
            if not remain:
                chosen.append(next(i for i in range(len(ids)) if i not in chosen))
                continue
            validchosen = [i for i in chosen if nonzero[i]]
            radius = (
                chord[np.ix_(remain, validchosen)].min(axis=1)
                if validchosen
                else np.full(len(remain), 4.0)
            )
            chosen.append(remain[int(np.argmax(radius))])
        assert np.array_equal(angular[q, 1:], ids[chosen]), ("angular", q)
        if np.isfinite(stats[q, 5]):
            for col, selected in ((5, near), (6, chosen)):
                cov = (
                    np.sqrt(
                        chord[
                            np.ix_(
                                np.flatnonzero(nonzero),
                                [i for i in selected if nonzero[i]],
                            )
                        ]
                        .min(axis=1)
                        .max()
                    )
                    if nonzero.any()
                    else 0
                )
                maxcoverage = max(maxcoverage, abs(cov - stats[q, col]))
                assert abs(cov - stats[q, col]) < 1e-07
        checked += len(ids) ** 2
report = {
    "prefix": str(prefix),
    "nodes": count,
    "independently_reconstructed_first_and_final_rows": len(sample),
    "independent_pairwise_geometry_values": checked,
    "max_coverage_error": maxcoverage,
    "first_degree_mean": float(first[:, 0].mean()),
    "final_seed_degree_mean": float(stats[:, 2].mean()),
    "reverse_edges": len(incoming),
    "max_reverse_indegree": int(np.diff(off).max()),
    "truncated_nodes": int((stats[:, 0] > 750).sum()),
    "zero_residual_candidates": int(stats[:, 4].sum()),
    "coverage_sampled_nodes": int(np.isfinite(stats[:, 5]).sum()),
    "nearest_coverage_mean": float(np.nanmean(stats[:, 5])),
    "angular_coverage_mean": float(np.nanmean(stats[:, 6])),
    "nearest_mean_edge_l2": float(stats[:, 7].mean()),
    "angular_mean_edge_l2": float(stats[:, 8].mean()),
    "all_row_invariants": True,
    "pilot_reverse_edges_partial": count != n,
}
if count == n:
    for mode in ("nearest", "angular"):
        path = Path(str(prefix) + "_" + mode + ".bin")
        assert path.stat().st_size == 140000032
        h = np.fromfile(path, dtype="<u4", count=8)
        baseh = np.fromfile(R / "models64/full_graph.bin", dtype="<u4", count=8)
        assert np.array_equal(h, baseh)
        graph = np.memmap(path, dtype=G.dtype, mode="r", offset=32, shape=(n,))
        assert np.array_equal(graph["key"], G["key"])
        seen = np.zeros(n, dtype=bool)
        frontier = np.array([h[5]], dtype=np.uint32)
        seen[frontier] = True
        depth = 0
        while len(frontier):
            nxt = np.unique(graph["ids"][frontier].reshape(-1))
            frontier = nxt[~seen[nxt]]
            seen[frontier] = True
            depth += 1
        report[mode + "_reachable"] = int(seen.sum())
        report[mode + "_bfs_layers"] = depth
        report[mode + "_sha256"] = (
            __import__("hashlib").sha256(path.read_bytes()).hexdigest()
        )
(R / "results" / ("graph65_" + sys.argv[1] + "_validation.json")).write_text(
    json.dumps(report, indent=2) + "\n"
)
logging.info("%s", json.dumps(report, indent=2))
