"""Frozen primary-query screening and paired confirmation of partial scans."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import statistics
from pathlib import Path

from explore90 import run

R = Path("/home/xingrui/bench/quantized_graph/research_20261025")
PROTOCOL_SHA = "319876ae1ae6a251a84c11c40f0d53347a0ef7af18e6c605671dc73982e507da"
logging.basicConfig(level=logging.INFO, format="%(message)s")


def need(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def inspect_run(name, caps, duration, threads):
    directory = R / "experiments" / name
    id_lines = (
        (directory / "search_k10_r0_ids.csv").read_bytes().splitlines(keepends=True)
    )
    recall_rows = list(csv.DictReader((directory / "search_k10_r0_recall.csv").open()))
    timed = list(csv.DictReader((directory / "search_k10_r0.csv").open()))
    points = []
    for label, cap in caps.items():
        rows = [x for x in recall_rows if int(x["ef"]) == label]
        need(
            len(rows) == 1000
            and {int(x["query_id"]) for x in rows} == set(range(1000)),
            "All primary queries",
        )
        need(all(int(x["returned"]) == 10 for x in rows), "Full Top10 results")
        recall = sum(int(x["hits"]) for x in rows) / 10000
        data = b"".join(
            x for x in id_lines if x.startswith((str(label) + ",").encode())
        )
        need(len(data.splitlines()) == 10000, "All primary ID/hex rows")
        group = [x for x in timed if int(x["ef"]) == label]
        need(
            len(group) == (0 if duration < 0 else len(threads.split(","))),
            "Expected timing row count",
        )
        for row in group:
            need(float(row["seconds"]) >= duration, "Minimum timing duration")
            need(float(row["recall"]) == recall, "CSV and per-query recall agree")
            need(
                math.isclose(
                    float(row["qps"]),
                    int(row["queries"]) / float(row["seconds"]),
                    rel_tol=3e-8,
                ),
                "QPS arithmetic",
            )
        points.append(
            {
                "label": label,
                "cap": cap,
                "margin": (label // 1000) / 1000,
                "patience": label % 1000,
                "actual_ef": 800,
                "recall": recall,
                "ids_hex_sha256": hashlib.sha256(data).hexdigest(),
                "experiment": name,
                "timing_rows": group,
            }
        )
    need(
        len(recall_rows) == len(caps) * 1000 and len(id_lines) == len(caps) * 10000,
        "No unexpected quality rows",
    )
    need(
        len(timed) == sum(len(x["timing_rows"]) for x in points),
        "No unexpected timing rows",
    )
    return points


def execute(name, variant, caps, duration=-1, threads="1"):
    extra = {
        "QG_CAP": "64",
        "QG_PATIENCE": "0",
        "BENCH_STOP_GRID": "1",
        "BENCH_MARGIN_DENOM": "1000",
        "BENCH_TOPK": "10",
        "BENCH_DEGREE": "32",
        "BENCH_QUERY_BEGIN": "0",
        "BENCH_QUERY_END": "1000",
        "BENCH_CAP_GRID": ",".join(f"{label}:{cap}" for label, cap in caps.items()),
    }
    run(
        name,
        variant,
        list(caps),
        extra=extra,
        rounds=1,
        duration=duration,
        threads=threads,
        nav="margin",
    )
    return inspect_run(name, caps, duration, threads)


def quality(protocol):
    control = execute("partial83_identity768_v1", "partial83_d768_v1", {20000: 64})[0]
    need(
        control["ids_hex_sha256"] == protocol["identity_control"]["reference_sha256"],
        "Full768 end-to-end native identity",
    )
    points = []
    for m in protocol["prefix_lengths"][:-1]:
        variant = protocol["variant_template"].format(m=m)
        measured = execute(
            f"partial83_quality_m{m}_cap64_v1",
            variant,
            dict.fromkeys(protocol["quality"]["cap64_labels"], 64),
        )
        if not any(p["recall"] >= 0.9 for p in measured):
            measured += execute(
                f"partial83_quality_m{m}_rescue_v1", variant, {1999: 128, 2999: 256}
            )
        for point in measured:
            point.update(prefix_m=m, variant=variant)
        points += measured
        logging.info(
            "QUALITY m=%s max_recall=%s", m, max(p["recall"] for p in measured)
        )
    return {
        "complete": True,
        "control": control,
        "points": points,
        "timing_rows": 0,
        "eligible_count": sum(p["recall"] >= 0.9 for p in points),
    }


def screen(protocol):
    quality_result = read(R / "partial83_quality_v1/result.json")
    measured = []
    for m in protocol["prefix_lengths"][:-1]:
        eligible = [
            p
            for p in quality_result["points"]
            if p["prefix_m"] == m and p["recall"] >= 0.9
        ]
        if not eligible:
            continue
        caps = {p["label"]: p["cap"] for p in eligible}
        points = execute(
            f"partial83_screen_m{m}_v1", eligible[0]["variant"], caps, duration=2
        )
        for point in points:
            old = next(p for p in eligible if p["label"] == point["label"])
            need(
                point["ids_hex_sha256"] == old["ids_hex_sha256"],
                "Screen matches frozen quality IDs/hex",
            )
            point.update(
                prefix_m=m, variant=old["variant"], quality_experiment=old["experiment"]
            )
        measured += points
    selected = (
        max(
            measured,
            key=lambda p: (
                float(p["timing_rows"][0]["qps"]),
                p["prefix_m"],
                -p["cap"],
                -p["label"],
            ),
        )
        if measured
        else None
    )
    return {
        "complete": True,
        "points": measured,
        "selected": selected,
        "timing_rows": len(measured),
        "selection_thread_count": 1,
        "not_a_confirmed_performance_result": True,
    }


def confirm(protocol):
    selected = read(R / "partial83_screen_v1/result.json")["selected"]
    if selected is None:
        return {
            "complete": True,
            "skipped": "No partial point reached Recall>=0.90",
            "timing_rows": 0,
            "reserve_gate": False,
        }
    runs = []
    for round_id, order in enumerate(protocol["confirmation"]["orders"]):
        for condition in order:
            if condition == "partial":
                variant, label, cap, expected = (
                    selected["variant"],
                    selected["label"],
                    selected["cap"],
                    selected["ids_hex_sha256"],
                )
            else:
                variant = (
                    "bucket90_w16" if condition == "baseline" else "partial83_d768_v1"
                )
                label, cap, expected = (
                    20000,
                    64,
                    protocol["identity_control"]["reference_sha256"],
                )
            point = execute(
                f"partial83_confirm_r{round_id}_{condition}_v1",
                variant,
                {label: cap},
                duration=3,
                threads="1,8",
            )[0]
            need(
                point["ids_hex_sha256"] == expected,
                "Confirmation matches frozen full or selected quality",
            )
            need(point["recall"] >= 0.9, "Matched target recall")
            point.update(condition=condition, paired_round=round_id, variant=variant)
            runs.append(point)
            (R / "partial83_confirm_v1/progress.json").write_text(
                json.dumps({"runs": runs, "complete": False}, indent=2) + "\n"
            )
    summaries = {}
    for threads in ["1", "8"]:
        qps = {
            condition: [
                float(
                    next(row for row in p["timing_rows"] if row["threads"] == threads)[
                        "qps"
                    ]
                )
                for p in runs
                if p["condition"] == condition
            ]
            for condition in ["baseline", "partial", "full768"]
        }
        medians = {k: statistics.median(v) for k, v in qps.items()}
        gains = {
            control: [qps["partial"][i] / qps[control][i] - 1 for i in range(3)]
            for control in ["baseline", "full768"]
        }
        summaries[threads] = {
            "qps_rounds": qps,
            "median_qps": medians,
            "paired_gains": gains,
            "median_gain": {c: medians["partial"] / medians[c] - 1 for c in gains},
            "all_gains_positive_against_both_controls": all(
                x > 0 for v in gains.values() for x in v
            ),
        }
    return {
        "complete": True,
        "runs": runs,
        "selected": selected,
        "summary": summaries,
        "timing_rows": 18,
        "reserve_gate": any(
            v["all_gains_positive_against_both_controls"]
            and all(g > 0 for g in v["median_gain"].values())
            for v in summaries.values()
        ),
        "breakthrough_verified": False,
        "three_round_pilot_only": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["quality", "screen", "confirm"])
    args = parser.parse_args()
    protocol_path = R / "partial83_protocol_v1/config.json"
    need(sha(protocol_path) == PROTOCOL_SHA, "Frozen protocol")
    need(
        read(R / "partial83_codec_probe_v1/proof.json")["passed"],
        "Actual codec interface gate",
    )
    need(read(R / "partial83_build_v1/proof.json")["passed"], "All variants compiled")
    for key in list(os.environ):
        if key.startswith(("QG_", "BENCH_", "GTEST_")) or key in {
            "LD_PRELOAD",
            "PYTHONPATH",
            "PYTHONOPTIMIZE",
        }:
            del os.environ[key]
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    output = R / f"partial83_{args.phase}_v1"
    output.mkdir(exist_ok=False)
    source = Path(__file__).resolve()
    if args.phase != "quality":
        need(
            sha(source)
            == read(R / "partial83_quality_v1/inputs.json")["source_sha256"],
            "Executed runner unchanged since quality",
        )
    (output / "source_snapshot.py").write_bytes(source.read_bytes())
    save(
        output / "inputs.json",
        {
            "protocol_sha256": PROTOCOL_SHA,
            "source_sha256": sha(source),
            "driver_sha256": sha(R / "research_benchmark_nav"),
            "explore90_sha256": sha(R / "scripts/explore90.py"),
            "codec_probe_sha256": sha(R / "partial83_codec_probe_v1/proof.json"),
        },
    )
    try:
        result = {"quality": quality, "screen": screen, "confirm": confirm}[args.phase](
            read(protocol_path)
        )
        save(output / "result.json", result)
        logging.info("PHASE83 %s COMPLETE %s", args.phase, output / "result.json")
    except BaseException as error:
        save(output / "failure.json", {"passed": False, "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
