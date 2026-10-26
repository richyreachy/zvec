"""Independently validate raw phase78 confirmation outputs and summarize them."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import statistics
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
parser.add_argument("--output", type=Path)
args = parser.parse_args()
R = args.root
output_path = args.output or R / "results/independent78_confirmation_audit.json"
declared = json.loads((R / "results/independent78_confirmation.json").read_text())
assert declared["complete"] is True
orders = [
    ["baseline", "learned", "disabled"],
    ["disabled", "learned", "baseline"],
    ["learned", "baseline", "disabled"],
    ["disabled", "baseline", "learned"],
]
assert declared["orders"] == orders
reference = (R / "experiments/time65_control_pre/search_k10_r0_ids.csv").read_bytes()
screen = R / "experiments/screen78_independent_v1"
selected = b"".join(
    line
    for line in (screen / "search_k10_r0_ids.csv")
    .read_bytes()
    .splitlines(keepends=True)
    if line.startswith(b"24000,")
)
assert len(reference.splitlines()) == len(selected.splitlines()) == 10000


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def query_hits(path, ef):
    rows = [row for row in read_csv(path) if int(row["ef"]) == ef]
    assert len(rows) == 1000
    values = {}
    for row in rows:
        query = int(row["query_id"])
        assert 0 <= query < 1000
        assert query not in values
        assert int(row["returned"]) == 10
        hits = int(row["hits"])
        assert 0 <= hits <= 10
        values[query] = hits
    return [values[index] for index in range(1000)]


baseline_hits = query_hits(
    R / "experiments/time65_control_pre/search_k10_r0_recall.csv", 20000
)
learned_hits = query_hits(screen / "search_k10_r0_recall.csv", 24000)
learned_same_margin_hits = query_hits(screen / "search_k10_r0_recall.csv", 20000)
assert sum(baseline_hits) == 9032
assert sum(learned_same_margin_hits) == 8995
assert sum(learned_hits) == 9069
rows = []
files = {}
resources = []
for paired_round, order in enumerate(orders):
    for condition in order:
        name = f"confirm78_r{paired_round}_{condition}"
        out = R / "experiments" / name
        config = json.loads((out / "config.json").read_text())
        command, env = config["command"], config["env"]
        assert command[:3] == ["taskset", "-c", "0,2,4,6,8,10,12,14"]
        assert command[3] == str(R / "research_benchmark_nav")
        assert command[4:7] == [
            "search",
            str(R / "indexes/cohere1m_m50_efc500.index"),
            "symphony_qg",
        ]
        assert command[-2] == "3"
        assert env["BENCH_N"] == "1000000"
        assert env["BENCH_TOPK"] == "10"
        assert env["BENCH_THREADS"] == "1,8"
        assert env["BENCH_QUERY_BEGIN"] == "0"
        assert env["BENCH_QUERY_END"] == "1000"
        assert env["BENCH_DEGREE"] == "32"
        assert env["QG_CAP"] == "64"
        assert env["QG_PATIENCE"] == "0"
        assert env["BENCH_STOP_GRID"] == "1"
        assert env["BENCH_MARGIN_DENOM"] == "1000"
        is_learned = condition == "learned"
        ef = 24000 if is_learned else 20000
        assert env["BENCH_EFS"] == str(ef)
        variant = (
            "bucket90_w16" if condition == "baseline" else "independent78_native_v1"
        )
        assert env["LD_LIBRARY_PATH"] == f"{R}/variants/{variant}:{R}/lib"
        if is_learned:
            assert env["QG_INDEPENDENT_CODE_FILE"] == str(
                R / "models77/full/blocks.bin"
            )
            assert (
                env["QG_INDEPENDENT_MODEL_SHA256"]
                == "2ac03543d4a09ae655d38130419df41668e4d62bdd3231b49135d2603c39f138"
            )
        else:
            assert not any(key.startswith("QG_INDEPENDENT_") for key in env)
        ids = (out / "search_k10_r0_ids.csv").read_bytes()
        assert ids == (selected if is_learned else reference)
        hits = query_hits(out / "search_k10_r0_recall.csv", ef)
        assert hits == (learned_hits if is_learned else baseline_hits)
        measured = read_csv(out / "search_k10_r0.csv")
        assert len(measured) == 2
        assert {int(row["threads"]) for row in measured} == {1, 8}
        for row in measured:
            duration, qps = float(row["seconds"]), float(row["qps"])
            assert duration >= 3
            assert math.isfinite(qps)
            assert math.isclose(qps, int(row["queries"]) / duration, rel_tol=1e-8)
            assert float(row["recall"]) == sum(hits) / 10000
            assert int(row["major_faults"]) == 0
            rows.append(dict(row, paired_round=paired_round, condition=condition))
        log = (out / "run.log").read_text()
        ready = "INDEPENDENT78 ready nodes=1000000 copied=997722 native_short=2278 release_mapped_bytes=9744004112"
        assert (ready in log) == is_learned
        open_line = next(
            line for line in log.splitlines() if line.startswith("OPEN_RESULT ")
        )
        resources.append(
            {
                "experiment": name,
                "fields": dict(item.split("=", 1) for item in open_line.split()[1:]),
            }
        )
        for path in out.iterdir():
            if path.is_file():
                files[str(path.relative_to(R))] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()

assert rows == declared["rows"]
summary = {}
for threads in (1, 8):
    values = {
        condition: [
            float(row["qps"])
            for row in rows
            if int(row["threads"]) == threads and row["condition"] == condition
        ]
        for condition in ("baseline", "learned", "disabled")
    }
    assert all(len(data) == 4 for data in values.values())
    medians = {condition: statistics.median(data) for condition, data in values.items()}
    gains = [
        new / old - 1
        for new, old in zip(values["learned"], values["baseline"], strict=True)
    ]
    assert medians == declared["summary"][str(threads)]["median_qps"]
    summary[str(threads)] = {
        "qps_rounds": values,
        "median_qps": medians,
        "qps_range": {
            condition: [min(data), max(data)] for condition, data in values.items()
        },
        "learned_over_baseline_median_gain": medians["learned"] / medians["baseline"]
        - 1,
        "learned_over_baseline_paired_gains": gains,
        "disabled_over_baseline_median_gain": medians["disabled"] / medians["baseline"]
        - 1,
        "all_four_learned_qps_below_baseline": all(gain < 0 for gain in gains),
        "all_four_learned_gains_at_least_25pct": all(gain >= 0.25 for gain in gains),
    }

quality = {}
for name, values in [
    ("same_margin_020", learned_same_margin_hits),
    ("selected_margin_024", learned_hits),
]:
    delta = [new - old for new, old in zip(values, baseline_hits, strict=True)]
    quality[name] = {
        "queries_better": sum(value > 0 for value in delta),
        "queries_equal": sum(value == 0 for value in delta),
        "queries_worse": sum(value < 0 for value in delta),
        "total_top10_hit_delta": sum(delta),
        "hit_change_histogram": {
            str(value): delta.count(value) for value in sorted(set(delta))
        },
        "primary_queries_reused_not_blind": True,
    }
result = {
    "passed_raw_validation": True,
    "all_12_run_configs_ids_hex_scores_query_hits_validated": True,
    "timed_calls_check_return_status_and_length_only": True,
    "full_ids_and_scores_checked_in_each_runs_untimed_quality_pass": True,
    "timing_rows": len(rows),
    "cumulative_retrieval_timing_rows": 1316 + len(rows),
    "summary": summary,
    "quality": quality,
    "open_resources": resources,
    "source_files_sha256": files,
    "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "additional_query_gate_not_run": True,
    "breakthrough_verified": False,
}
with output_path.open("x") as stream:
    json.dump(result, stream, indent=2)
    stream.write("\n")
logging.info("CONFIRM78 AUDIT %s", json.dumps(summary))
