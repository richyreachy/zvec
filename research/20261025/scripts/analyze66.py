from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
S = Path("/home/xingrui/workspace/quantized_graph/symphony_research_20261025")
assert "DIAGNOSTIC66 COMPLETE" in (R / "logs/diagnostic66.log").read_text()
plan = json.loads((R / "results/graph66_plan.json").read_text())
labels = plan["labels"]
rows = []
selected = {}
per_query = {}
checks = []
unique_scores = {}
actual_result_rows = 0


def ids(name, label):
    a = [
        tuple(x[1:])
        for x in csv.reader((R / f"experiments/{name}/search_k10_r0_ids.csv").open())
        if int(x[0]) == label
    ]
    assert len(a) == 10000
    return sorted(a)


for graph in plan["graphs"]:
    for score in plan["scores"]:
        name = f"diag66_{graph}_{score}"
        recall = list(
            csv.DictReader((R / f"experiments/{name}/search_k10_r0_recall.csv").open())
        )
        stats = np.loadtxt(R / f"results/{name}_stats.csv", delimiter=",")
        diag = np.loadtxt(R / f"results/{name}_diag.csv", delimiter=",")
        all_ids = list(
            csv.reader((R / f"experiments/{name}/search_k10_r0_ids.csv").open())
        )
        assert len(all_ids) == len(labels) * 10000
        grouped = {}
        for label_text, q_text, key_text, hex_score in all_ids:
            label_value = int(label_text)
            qid = int(q_text)
            key = int(key_text)
            value = float.fromhex(hex_score)
            assert label_value in labels
            assert 0 <= qid < 1000
            assert 0 <= key < 1000000
            assert np.isfinite(value)
            group = grouped.setdefault((label_value, qid), set())
            assert key not in group
            group.add(key)
            pair = (qid, key)
            if pair in unique_scores:
                assert unique_scores[pair] == value
            else:
                unique_scores[pair] = value
        assert len(grouped) == len(labels) * 1000
        assert all(len(group) == 10 for group in grouped.values())
        actual_result_rows += len(all_ids)
        assert np.all(np.isfinite(stats))
        assert np.all(np.isfinite(diag))
        assert stats.shape == (len(labels) * 1000, 7)
        assert diag.shape == (len(labels) * 1000, 11)
        assert not list(
            csv.DictReader((R / f"experiments/{name}/search_k10_r0.csv").open())
        )
        assert np.array_equal(stats[:, 3], diag[:, 3])
        assert np.array_equal(stats[:, 3], stats[:, 4])
        assert np.all(stats[:, 3] <= 64)
        assert np.all(stats[:, 6] == 6656)
        assert np.all(diag[:, 0] == (1 if score == "exact" else 0))
        candidates = []
        for i, label in enumerate(labels):
            rr = [x for x in recall if int(x["ef"]) == label]
            assert len(rr) == 1000
            assert sorted(int(x["query_id"]) for x in rr) == list(range(1000))
            assert all(int(x["returned"]) == 10 for x in rr)
            st = stats[i * 1000 : (i + 1) * 1000]
            d = diag[i * 1000 : (i + 1) * 1000]
            expanded = st[:, 3]
            assert np.all(d[:, 6] <= 32 * expanded)
            entry = {
                "graph": graph,
                "score": score,
                "label": label,
                "margin": label / 1000000,
                "recall": sum(int(x["hits"]) for x in rr) / 10000,
                "mean_expanded": float(expanded.mean()),
                "median_expanded": float(np.median(expanded)),
                "p95_expanded": float(np.quantile(expanded, 0.95)),
                "cap_fraction": float(np.mean(expanded == 64)),
                "mean_pushes": float(d[:, 5].mean()),
                "mean_scored_edges": float(d[:, 6].mean()),
                "weighted_raw_onebit_rmse": float(
                    np.sqrt(np.sum(d[:, 6] * d[:, 7] ** 2) / d[:, 6].sum())
                ),
                "weighted_raw_onebit_bias": float(
                    np.sum(d[:, 6] * d[:, 8]) / d[:, 6].sum()
                ),
                "raw_onebit_batch_top1_match": float(d[:, 9].sum() / d[:, 10].sum()),
            }
            rows.append(entry)
            candidates.append(entry)
            per_query[graph, score, label] = expanded
            reference = None
            reference_label = label
            if graph == "original" and label in [0, 10000, 20000, 40000, 60000]:
                reference = "diag47_" + (
                    "onebit16" if score == "onebit" else "oracle16"
                )
                reference_label = label // 10
            elif (
                graph != "original"
                and score == "onebit"
                and (label in [0, 20000, 40000, 60000])
            ):
                reference = "diag65_native_" + graph
            if reference:
                assert ids(name, label) == ids(reference, reference_label)
                checks.append(
                    {
                        "experiment": name,
                        "label": label,
                        "reference": reference,
                        "reference_label": reference_label,
                        "queries": 1000,
                    }
                )
        passing = [x for x in candidates if x["recall"] >= 0.9]
        assert passing
        selected[graph + "_" + score] = min(passing, key=lambda x: x["mean_expanded"])
rng = np.random.default_rng(6601)
comparisons = []
for score in plan["scores"]:
    base = selected["original_" + score]
    for graph in ["nearest", "angular"]:
        candidate = selected[graph + "_" + score]
        delta = (
            per_query[graph, score, candidate["label"]]
            - per_query["original", score, base["label"]]
        )
        bootstrap = []
        for _i in range(4000):
            bootstrap.append(float(delta[rng.integers(0, 1000, 1000)].mean()))
        comparisons.append(
            {
                "score": score,
                "graph": graph,
                "baseline_margin": base["margin"],
                "candidate_margin": candidate["margin"],
                "baseline_recall": base["recall"],
                "candidate_recall": candidate["recall"],
                "mean_expansion_change": float(delta.mean()),
                "relative_mean_expansion_change": candidate["mean_expanded"]
                / base["mean_expanded"]
                - 1,
                "paired_query_bootstrap95": np.quantile(
                    bootstrap, [0.025, 0.975]
                ).tolist(),
                "queries_already_used_for_threshold_selection": True,
                "not_a_qps_gain_or_global_lower_bound": True,
            }
        )
g = np.memmap(
    R / "models64/full_graph.bin",
    mode="r",
    dtype=np.dtype([("key", "<u8"), ("degree", "<u4"), ("ids", "<u4", (32,))]),
    offset=32,
    shape=(1000000,),
)
inverse = np.empty(1000000, dtype=np.uint32)
inverse[g["key"]] = np.arange(1000000, dtype=np.uint32)
v = np.memmap(
    R / "models64/full_vectors.f32",
    mode="r",
    dtype="<f4",
    offset=16,
    shape=(1000000, 768),
)
q = np.memmap(
    R / "models68/primary/queries.f32", mode="r", dtype="<f4", shape=(1000, 768)
)
items = list(unique_scores.items())
max_error = 0.0
for start in range(0, len(items), 256):
    block = items[start : start + 256]
    qid = np.array([x[0][0] for x in block])
    key = np.array([x[0][1] for x in block])
    actual = np.array([x[1] for x in block])
    queries = np.asarray(q[qid], dtype=np.float64)
    queries /= np.linalg.norm(queries, axis=1)[:, None]
    truth = 1 - np.einsum(
        "bd,bd->b", queries, np.asarray(v[inverse[key]], dtype=np.float64)
    )
    max_error = max(max_error, float(np.max(np.abs(truth - actual))))
assert max_error < 2e-06
source = {}
for name in ["graph66_onebit", "graph66_exact"]:
    m = json.loads((R / f"variants/{name}/source_files_sha256.json").read_text())
    for p, h in m.items():
        assert hashlib.sha256((S / p).read_bytes()).hexdigest() == h
    source[name] = len(m)
report = {
    "phase": 66,
    "executed_after_phase69": True,
    "plan": plan,
    "rows": rows,
    "selected_lowest_observed_work_at_recall90": selected,
    "graph_comparisons": comparisons,
    "historical_equivalence": checks,
    "historical_equivalent_queries": sum(x["queries"] for x in checks),
    "total_primary_query_evaluations": 42000,
    "result_id_score_rows": actual_result_rows,
    "unique_returned_scores_fp64_checked": len(items),
    "max_returned_score_fp64_error": max_error,
    "new_timing_rows": 0,
    "cumulative_timing_rows": 1310,
    "source_manifests": source,
    "source_changes": False,
    "default_regression_inherited_from_unchanged_stage69": 28,
    "extra_queries_unused": True,
    "breakthrough_verified": False,
}
(R / "results/graph66_analysis.json").write_text(json.dumps(report, indent=2) + "\n")
with (R / "results/graph66_work_recall.csv").open("w") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
logging.info(
    "%s",
    json.dumps(
        {
            "selected": selected,
            "comparisons": comparisons,
            "equivalent_queries": report["historical_equivalent_queries"],
        },
        indent=2,
    ),
)
logging.info("%s", "ANALYZE66 COMPLETE")
