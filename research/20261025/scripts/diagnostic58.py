from __future__ import annotations

import csv
import json
import logging
import statistics as st
import subprocess

from explore90 import R, run

logging.basicConfig(level=logging.INFO)
include = (
    "-I/home/xingrui/workspace/quantized_graph/symphony_research_20261025/src/core"
)
commands = []
for name, source, flags in [
    ("router58_legacy_test", "router33_test.cc", []),
    ("router58_large_test", "anchor58_test.cc", ["-DQG_ROUTER_MAX_ANCHORS=65536"]),
]:
    cmd = [
        "g++",
        "-O2",
        "-std=c++17",
        "-UNDEBUG",
        include,
        *flags,
        str(R / "scripts" / source),
        "-o",
        str(R / name),
    ]
    commands.append(cmd)
    subprocess.run(cmd, check=True)
    with (R / f"logs/{name}.log").open("w") as f:
        subprocess.run([str(R / name)], stdout=f, stderr=subprocess.STDOUT, check=True)
(R / "results/anchor58_test_build_commands.json").write_text(
    json.dumps(commands, indent=2)
)
variant = "anchor58_integer"
flags = json.loads((R / "variants/int55_original/metadata.json").read_text())["defines"]
flags += ["QG_PROJECTION_ROUTER=1", "QG_ROUTER_MAX_ANCHORS=65536", "QG_CENTER_STATS=1"]
with (R / f"logs/build_{variant}.log").open("w") as f:
    subprocess.run(
        ["python3", str(R / "scripts/compile_variant.py"), variant, *flags],
        stdout=f,
        stderr=subprocess.STDOUT,
        check=True,
    )
env = {
    "QG_CENTER_MEAN_FILE": str(R / "mean54/mean.f32"),
    "QG_CAP": "64",
    "QG_PATIENCE": "0",
    "QG_ROUTE_DIM": "768",
    "QG_ROUTE_SHORTLIST": "4",
}
labels = [0, 10000, 20000, 40000]
summary = []
for count, keep in [(0, 0), (4096, 0), (4096, 1), (65536, 0), (65536, 1)]:
    name = f"anchor58_count{count}_keep{keep}"
    chosen_labels = [20000] if count == 0 else labels
    stats_file = R / f"results/{name}_stats.csv"
    run(
        name,
        variant,
        chosen_labels,
        extra={
            **env,
            "QG_ROUTE_COUNT": str(count),
            "QG_ROUTE_KEEP_SCALE": str(keep),
            "QG_CENTER_STATS_FILE": str(stats_file),
            "BENCH_STOP_GRID": "1",
            "BENCH_MARGIN_DENOM": "1000",
        },
        duration=-1,
        threads="1",
        nav="margin",
    )
    recall = list(
        csv.DictReader((R / f"experiments/{name}/search_k10_r0_recall.csv").open())
    )
    stats = list(csv.reader(stats_file.open()))
    assert len(stats) == len(chosen_labels) * 1000
    for i, label in enumerate(chosen_labels):
        rr = [r for r in recall if int(r["ef"]) == label]
        assert len(rr) == 1000
        ss = stats[i * 1000 : (i + 1) * 1000]
        summary.append(
            {
                "anchors": count,
                "keep_original_bucket_scale": bool(keep),
                "margin": label / 1000000,
                "recall": sum(int(r["hits"]) for r in rr) / 10000,
                "mean_expanded": st.mean(int(r[3]) for r in ss),
                "mean_refined": st.mean(int(r[4]) for r in ss),
                "cap_fraction": sum(int(r[3]) == 64 for r in ss) / 1000,
            }
        )
    if count == 0:
        old = sorted(
            r[1:]
            for r in csv.reader(
                (
                    R
                    / "experiments/screen55_cohere_int55_original/search_k10_r0_ids.csv"
                ).open()
            )
            if r[0] == "20000"
        )
        current = sorted(
            r[1:]
            for r in csv.reader(
                (R / f"experiments/{name}/search_k10_r0_ids.csv").open()
            )
        )
        assert len(current) == len(old) == 10000
        assert current == old
        (R / "results/anchor58_disabled_equivalence.json").write_text(
            json.dumps(
                {"queries": 1000, "id_score_rows": 10000, "exact_match": True}, indent=2
            )
        )
    (R / "results/anchor58_summary.json").write_text(
        json.dumps(
            {"diagnostic_only": True, "no_deployable_qps_claim": True, "rows": summary},
            indent=2,
        )
    )
    logging.info(
        "%s %s %s", "RESULTS", name, json.dumps(summary[-len(chosen_labels) :])
    )
logging.info("%s", "DIAGNOSTIC58 COMPLETE")
