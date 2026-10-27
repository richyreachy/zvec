"""Quality audit, frozen screening, and paired confirmation for phase 84."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import statistics
from pathlib import Path

import postmerge84 as p

logging.basicConfig(level=logging.INFO, format="%(message)s")
CAPS = {0: 64, 10000: 64, 20000: 64, 40000: 64, 80000: 64, 1999: 96, 2999: 128}


def inspect(name):
    directory = p.PHASE / "experiments" / name
    config = json.loads((directory / "config.json").read_text())
    labels = list(map(int, config["env"]["BENCH_EFS"].split(",")))
    caps = {
        int(pair.split(":")[0]): int(pair.split(":")[1])
        for pair in config["env"]["BENCH_CAP_GRID"].split(",")
    }
    recall = list(csv.DictReader((directory / "search_k10_r0_recall.csv").open()))
    lines = (directory / "search_k10_r0_ids.csv").read_bytes().splitlines(keepends=True)
    timing = list(csv.DictReader((directory / "search_k10_r0.csv").open()))
    assert len(lines) == 10000 * len(labels)
    assert len(recall) == 1000 * len(labels)
    truth = {}
    for line in (
        Path(config["env"]["BENCH_DATA"], "neighbors.txt").read_text().splitlines()
    ):
        key, values = line.split(";")
        truth[int(key)] = set(map(int, values.split()[:10]))
    result = []
    for label in labels:
        group = [line for line in lines if line.startswith((str(label) + ",").encode())]
        assert len(group) == 10000
        returned = {}
        for line in group:
            _, query, key, score = line.decode().strip().split(",")
            assert math.isfinite(float.fromhex(score))
            assert 0 <= int(key) < 1000000
            returned.setdefault(int(query), []).append(int(key))
        assert set(returned) == set(range(1000))
        assert all(len(ids) == len(set(ids)) == 10 for ids in returned.values())
        hits = {q: len(set(ids) & truth[q]) for q, ids in returned.items()}
        rows = [row for row in recall if int(row["ef"]) == label]
        assert {int(row["query_id"]) for row in rows} == set(range(1000))
        assert all(
            int(row["hits"]) == hits[int(row["query_id"])]
            and int(row["returned"]) == 10
            for row in rows
        )
        quality = sum(hits.values()) / 10000
        timed = [row for row in timing if int(row["ef"]) == label]
        duration = float(config["command"][-2])
        expected_threads = set(map(int, config["env"]["BENCH_THREADS"].split(",")))
        assert len(timed) == (len(expected_threads) if duration > 0 else 0)
        if duration > 0:
            assert {int(row["threads"]) for row in timed} == expected_threads
        for row in timed:
            assert float(row["recall"]) == quality
            assert float(row["seconds"]) >= duration
            assert math.isclose(
                float(row["qps"]),
                int(row["queries"]) / float(row["seconds"]),
                rel_tol=3e-8,
            )
        result.append(
            {
                "variant": config["env"]["BENCH_LABEL"],
                "experiment": name,
                "label": label,
                "cap": caps[label],
                "recall": quality,
                "ids_hex_sha256": hashlib.sha256(b"".join(group)).hexdigest(),
                "timing": timed,
            }
        )
    return result


def screen():
    quality = {
        v: inspect(v + "_quality")
        for v in ["native", "uncorrected512", "corrected384", "corrected512"]
    }
    p.save(p.PHASE / "quality_audit.json", quality)
    measured = []
    for variant in ["corrected384", "corrected512"]:
        eligible = [row for row in quality[variant] if row["recall"] >= 0.9]
        if not eligible:
            continue
        p.execute(
            variant + "_screen",
            variant,
            ",".join(str(row["label"]) for row in eligible),
            ",".join(f"{row['label']}:{row['cap']}" for row in eligible),
            2,
            "1",
        )
        rows = inspect(variant + "_screen")
        for row in rows:
            old = next(x for x in eligible if x["label"] == row["label"])
            assert row["ids_hex_sha256"] == old["ids_hex_sha256"]
        measured += rows
        logging.info(
            "%s screen %s",
            variant,
            [(row["label"], row["timing"][0]["qps"]) for row in rows],
        )
    selected = (
        max(measured, key=lambda row: float(row["timing"][0]["qps"]))
        if measured
        else None
    )
    p.save(p.PHASE / "screen_summary.json", {"points": measured, "selected": selected})
    logging.info("Selected: %s", selected)


def confirm():
    selected = json.loads((p.PHASE / "screen_summary.json").read_text())["selected"]
    native = next(row for row in inspect("native_quality") if row["label"] == 20000)
    if selected is None:
        raise RuntimeError(
            "No eligible corrected point: do not time a sub-90% candidate"
        )
    records = []
    for round_id, order in enumerate(
        [["native", "selected"], ["selected", "native"], ["native", "selected"]]
    ):
        for condition in order:
            point = native if condition == "native" else selected
            name = f"confirm_r{round_id}_{condition}"
            p.execute(
                name,
                point["variant"],
                str(point["label"]),
                f"{point['label']}:{point['cap']}",
                3,
                "1,8",
            )
            actual = inspect(name)[0]
            assert actual["ids_hex_sha256"] == point["ids_hex_sha256"]
            actual.update(condition=condition, round=round_id)
            records.append(actual)
            logging.info(
                "Round %s %s %s",
                round_id,
                condition,
                [(x["threads"], x["qps"]) for x in actual["timing"]],
            )
    medians = {
        condition: {
            str(threads): statistics.median(
                float(row["qps"])
                for record in records
                if record["condition"] == condition
                for row in record["timing"]
                if int(row["threads"]) == threads
            )
            for threads in [1, 8]
        }
        for condition in ["native", "selected"]
    }
    gain = {
        str(t): medians["selected"][str(t)] / medians["native"][str(t)] - 1
        for t in [1, 8]
    }
    p.save(
        p.PHASE / "confirmation_summary.json",
        {
            "records": records,
            "medians": medians,
            "gain": gain,
            "current_native_gate": all(x >= 0.25 for x in gain.values()),
            "historical_and_reserve_validation": "not yet run",
        },
    )
    logging.info("Medians %s; gains %s", medians, gain)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["screen", "confirm"])
    args = parser.parse_args()
    (screen if args.mode == "screen" else confirm)()
