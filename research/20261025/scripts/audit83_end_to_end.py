"""Independent raw-file phase83 audit; execute only outside benchmark timing.

No retrieval, GT evaluation, training, new selection rule or statistical-blindness
claim. Reconstruct the frozen protocol's selection and arithmetic from raw files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import re
import statistics
from collections import Counter
from pathlib import Path

DEFAULT_ROOT = Path("/home/xingrui/bench/quantized_graph/research_20261025")
PROTOCOL_SHA = "319876ae1ae6a251a84c11c40f0d53347a0ef7af18e6c605671dc73982e507da"
logging.basicConfig(level=logging.INFO, format="%(message)s")


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


class Audit:
    def __init__(self, root, protocol_path):
        self.root = root
        self.files = {}
        self.protocol = self.read(protocol_path)
        need(
            self.files[str(protocol_path)] == PROTOCOL_SHA,
            "Frozen phase83 protocol SHA",
        )
        self.resources = []
        self.counts = Counter()
        self.raw_points = []
        self.identity = self.protocol["identity_control"]
        self.reference = self.bytes(root / self.identity["reference"])
        need(
            digest(self.reference) == self.identity["reference_sha256"],
            "Frozen native ID/hex reference",
        )
        need(len(self.reference.splitlines()) == 10000, "Native reference row count")
        self.phase_source = None

    def bytes(self, path):
        data = path.read_bytes()
        self.files[str(path)] = digest(data)
        return data

    def read(self, path):
        return json.loads(self.bytes(path))

    def csv(self, path):
        return list(csv.DictReader(self.bytes(path).decode().splitlines()))

    def completed(self, phase):
        folder = self.root / f"partial83_{phase}_v1"
        inputs = self.read(folder / "inputs.json")
        need(inputs["protocol_sha256"] == PROTOCOL_SHA, "Phase protocol SHA")
        source = self.bytes(folder / "source_snapshot.py")
        need(digest(source) == inputs["source_sha256"], "Executed source snapshot SHA")
        if self.phase_source is None:
            self.phase_source = digest(source)
        need(digest(source) == self.phase_source, "Same frozen runner through phases")
        for key, path in [
            ("driver_sha256", self.root / "research_benchmark_nav"),
            ("explore90_sha256", self.root / "scripts/explore90.py"),
            ("codec_probe_sha256", self.root / "partial83_codec_probe_v1/proof.json"),
        ]:
            need(digest(self.bytes(path)) == inputs[key], "Phase dependency SHA " + key)
        result = self.read(folder / "result.json")
        need(result.get("complete") is True, "Completed phase required")
        return result

    def run(self, name, variant, caps, duration, threads):  # noqa: PLR0915 - independent raw checks
        folder = self.root / "experiments" / name
        config = self.read(folder / "config.json")
        command, env = config["command"], config["env"]
        expected_command = [
            "taskset",
            "-c",
            self.protocol["cpu_pool"],
            str(self.root / "research_benchmark_nav"),
            "search",
            str(self.root / "indexes/cohere1m_m50_efc500.index"),
            "symphony_qg",
            "0",
            str(duration),
            str(folder / "search"),
        ]
        need(command == expected_command, "Driver/index/affinity/duration " + name)
        expected_env = {
            "LD_LIBRARY_PATH": f"{self.root}/variants/{variant}:{self.root}/lib",
            "BENCH_N": "1000000",
            "BENCH_DATA": "/home/xingrui/bench/dataset/cohere/1m/",
            "BENCH_TOPK": "10",
            "BENCH_DEGREE": "32",
            "BENCH_ROUNDS": "1",
            "BENCH_EFS": ",".join(map(str, caps)),
            "BENCH_THREADS": threads,
            "BENCH_QUERY_BEGIN": "0",
            "BENCH_QUERY_END": "1000",
            "BENCH_LABEL": variant,
            "QG_CAP": "64",
            "QG_PATIENCE": "0",
            "BENCH_STOP_GRID": "1",
            "BENCH_MARGIN_DENOM": "1000",
            "BENCH_CAP_GRID": ",".join(f"{label}:{cap}" for label, cap in caps.items()),
        }
        need(env == expected_env, "Complete recorded config environment " + name)
        meta = self.read(self.root / "variants" / variant / "metadata.json")
        expected_defines = list(self.protocol["base_defines"])
        if variant != self.protocol["baseline_variant"]:
            match = re.fullmatch(r"partial83_d(128|256|384|512|768)_v1", variant)
            need(match is not None, "Known variant")
            expected_defines.append(
                self.protocol["partial_define"].format(m=int(match[1]))
            )
        need(meta["defines"] == expected_defines, "Variant macro identity")
        prefix = folder / "search_k10_r0"
        recall_rows = self.csv(Path(str(prefix) + "_recall.csv"))
        timed = self.csv(Path(str(prefix) + ".csv"))
        ids_bytes = self.bytes(Path(str(prefix) + "_ids.csv"))
        id_lines = ids_bytes.splitlines(keepends=True)
        decoded = []
        for line in id_lines:
            fields = line.decode().strip().split(",")
            need(len(fields) == 4, "ID/hex row schema")
            label, query, key = map(int, fields[:3])
            need(
                re.fullmatch(r"-?0x[0-9a-f]+(?:\.[0-9a-f]*)?p[+-]\d+", fields[3])
                is not None,
                "Hexadecimal result score",
            )
            need(
                math.isfinite(float.fromhex(fields[3])) and 0 <= key < 1000000,
                "Finite score and external key range",
            )
            decoded.append((label, query, key))
        need(
            len(recall_rows) == len(caps) * 1000 and len(decoded) == len(caps) * 10000,
            "Exact quality row counts",
        )
        labels = set(caps)
        need(
            {int(r["ef"]) for r in recall_rows} == labels
            and {row[0] for row in decoded} == labels,
            "Exact quality label sets",
        )
        requested_threads = {int(x) for x in threads.split(",")}
        need(
            len(requested_threads) == len(threads.split(",")),
            "Unique requested threads",
        )
        need(
            len(timed) == (0 if duration < 0 else len(caps) * len(requested_threads)),
            "Exact timing row count",
        )
        need(
            {int(r["ef"]) for r in timed} == (set() if duration < 0 else labels),
            "Exact timing label set",
        )
        points = []
        for label, cap in caps.items():
            recalls = [r for r in recall_rows if int(r["ef"]) == label]
            need(
                len(recalls) == 1000
                and Counter(int(r["query_id"]) for r in recalls)
                == Counter(range(1000)),
                "Each primary query exactly once",
            )
            need(
                all(
                    int(r["returned"]) == 10 and 0 <= int(r["hits"]) <= 10
                    for r in recalls
                ),
                "Valid Top10 per-query counts",
            )
            hits = sum(int(r["hits"]) for r in recalls)
            recall = hits / 10000
            triples = [r for r in decoded if r[0] == label]
            need(
                [r[1] for r in triples] == [q for q in range(1000) for _ in range(10)],
                "Ordered complete query result rows",
            )
            need(
                all(
                    len({r[2] for r in triples[q * 10 : (q + 1) * 10]}) == 10
                    for q in range(1000)
                ),
                "Distinct result keys within each query",
            )
            payload = b"".join(
                line
                for line, triple in zip(id_lines, decoded, strict=True)
                if triple[0] == label
            )
            rows = [r for r in timed if int(r["ef"]) == label]
            need(
                Counter(int(r["threads"]) for r in rows)
                == (Counter() if duration < 0 else Counter(requested_threads)),
                "Exactly one CSV row per requested thread count",
            )
            for row in rows:
                count, elapsed, qps = (
                    int(row["queries"]),
                    float(row["seconds"]),
                    float(row["qps"]),
                )
                need(
                    count > 0
                    and math.isfinite(elapsed)
                    and elapsed > 0
                    and elapsed >= duration
                    and math.isfinite(qps)
                    and qps > 0,
                    "Positive finite timing/count",
                )
                need(
                    row["algorithm"] == variant
                    and int(row["topk"]) == 10
                    and int(row["degree"]) == 32
                    and int(row["round"]) == 0,
                    "Timing algorithm/topk/degree/round",
                )
                need(
                    math.isclose(qps, count / elapsed, rel_tol=3e-8, abs_tol=0),
                    "QPS count/seconds",
                )
                need(
                    float(row["recall"]) == recall,
                    "Timing recall versus raw sumhits/10000",
                )
                for key in ["mean_us", "p50_us", "p95_us", "p99_us"]:
                    need(
                        math.isfinite(float(row[key])) and float(row[key]) >= 0,
                        "Finite latency",
                    )
            point = {
                "label": label,
                "cap": cap,
                "margin": (label // 1000) / 1000,
                "patience": label % 1000,
                "actual_ef": 800,
                "recall": recall,
                "ids_hex_sha256": digest(payload),
                "experiment": name,
                "timing_rows": rows,
            }
            points.append(point)
            self.raw_points.append(
                dict(
                    point,
                    variant=variant,
                    sum_hits=hits,
                    unique_queries=1000,
                    id_hex_rows=10000,
                )
            )
            self.counts.update(
                quality_points=1,
                quality_queries=1000,
                id_hex_rows=10000,
                timing_rows=len(rows),
            )
            if variant in ("bucket90_w16", "partial83_d768_v1"):
                need(
                    label == 20000 and cap == 64 and payload == self.reference,
                    "Full control exactly matches all 10k native ID/hex rows",
                )
        log = self.bytes(folder / "run.log").decode()
        open_line = next(
            line for line in log.splitlines() if line.startswith("OPEN_RESULT ")
        )
        self.resources.append(
            {
                "experiment": name,
                "open_result": open_line,
                "major_faults": [int(r["major_faults"]) for r in timed],
            }
        )
        return points

    def quality(self):
        declared = self.completed("quality")
        identity = self.run(
            "partial83_identity768_v1", "partial83_d768_v1", {20000: 64}, -1, "1"
        )[0]
        points = []
        for m in self.protocol["prefix_lengths"][:-1]:
            variant = self.protocol["variant_template"].format(m=m)
            measured = self.run(
                f"partial83_quality_m{m}_cap64_v1",
                variant,
                dict.fromkeys(self.protocol["quality"]["cap64_labels"], 64),
                -1,
                "1",
            )
            if all(
                p["recall"] < self.protocol["quality"]["target_recall"]
                for p in measured
            ):
                measured += self.run(
                    f"partial83_quality_m{m}_rescue_v1",
                    variant,
                    {
                        int(k): v
                        for k, v in self.protocol["quality"]["rescue_labels"].items()
                    },
                    -1,
                    "1",
                )
            for point in measured:
                point.update(prefix_m=m, variant=variant)
            points.extend(measured)
        expected = {
            "complete": True,
            "control": identity,
            "points": points,
            "timing_rows": 0,
            "eligible_count": sum(p["recall"] >= 0.9 for p in points),
        }
        need(
            declared == expected, "Independent quality points/counts equal root result"
        )
        return expected

    def screen(self, quality):
        declared = self.completed("screen")
        points = []
        for m in self.protocol["prefix_lengths"][:-1]:
            eligible = [
                p
                for p in quality["points"]
                if p["prefix_m"] == m and p["recall"] >= 0.9
            ]
            if not eligible:
                continue
            measured = self.run(
                f"partial83_screen_m{m}_v1",
                eligible[0]["variant"],
                {p["label"]: p["cap"] for p in eligible},
                2,
                "1",
            )
            by_label = {p["label"]: p for p in eligible}
            for point in measured:
                previous = by_label[point["label"]]
                need(
                    point["ids_hex_sha256"] == previous["ids_hex_sha256"]
                    and point["recall"] == previous["recall"],
                    "Screen quality unchanged",
                )
                point.update(
                    prefix_m=m,
                    variant=previous["variant"],
                    quality_experiment=previous["experiment"],
                )
            points.extend(measured)
        ranked = sorted(
            points,
            key=lambda p: (
                -float(p["timing_rows"][0]["qps"]),
                -p["prefix_m"],
                p["cap"],
                p["label"],
            ),
        )
        selected = ranked[0] if ranked else None
        expected = {
            "complete": True,
            "points": points,
            "selected": selected,
            "timing_rows": len(points),
            "selection_thread_count": 1,
            "not_a_confirmed_performance_result": True,
        }
        need(
            declared == expected,
            "Independent frozen screen selection equals root result",
        )
        return expected

    def confirm(self, screen):
        declared = self.completed("confirm")
        selected = screen["selected"]
        if selected is None:
            need(
                declared
                == {
                    "complete": True,
                    "skipped": "No partial point reached Recall>=0.90",
                    "timing_rows": 0,
                    "reserve_gate": False,
                },
                "No-candidate skip",
            )
            return declared
        runs = []
        for round_id, order in enumerate(self.protocol["confirmation"]["orders"]):
            for condition in order:
                partial = condition == "partial"
                variant = (
                    selected["variant"]
                    if partial
                    else (
                        self.protocol["baseline_variant"]
                        if condition == "baseline"
                        else "partial83_d768_v1"
                    )
                )
                label, cap = (
                    (selected["label"], selected["cap"]) if partial else (20000, 64)
                )
                point = self.run(
                    f"partial83_confirm_r{round_id}_{condition}_v1",
                    variant,
                    {label: cap},
                    3,
                    "1,8",
                )[0]
                expected_sha = (
                    selected["ids_hex_sha256"]
                    if partial
                    else self.identity["reference_sha256"]
                )
                need(
                    point["ids_hex_sha256"] == expected_sha and point["recall"] >= 0.9,
                    "Confirmation matches frozen candidate/control quality",
                )
                point.update(
                    condition=condition, paired_round=round_id, variant=variant
                )
                runs.append(point)
        summaries = {}
        for thread in (1, 8):
            values = {
                condition: [
                    float(
                        next(
                            r for r in run["timing_rows"] if int(r["threads"]) == thread
                        )["qps"]
                    )
                    for run in runs
                    if run["condition"] == condition
                ]
                for condition in ("baseline", "partial", "full768")
            }
            need(
                all(len(v) == 3 for v in values.values()),
                "Three rounds per condition/thread",
            )
            medians = {name: statistics.median(v) for name, v in values.items()}
            paired = {
                control: [
                    new / old - 1
                    for new, old in zip(values["partial"], values[control], strict=True)
                ]
                for control in ("baseline", "full768")
            }
            summaries[str(thread)] = {
                "qps_rounds": values,
                "median_qps": medians,
                "paired_gains": paired,
                "median_gain": {c: medians["partial"] / medians[c] - 1 for c in paired},
                "all_gains_positive_against_both_controls": all(
                    g > 0 for series in paired.values() for g in series
                ),
            }
        expected = {
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
        need(
            declared == expected,
            "Independent paired raw-CSV medians/gains equal root result",
        )
        return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--phase", choices=["screen", "all"], default="all")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    protocol_path = (
        args.protocol or root / "partial83_protocol_v1/config.json"
    ).resolve()
    if not args.execute:
        logging.info(
            "Plan only: independent %s audit, output=%s; execute after timing stops",
            args.phase,
            args.output,
        )
        return
    output = args.output.resolve()
    output.mkdir(exist_ok=False)
    source = Path(__file__).resolve()
    (output / "source_snapshot.py").write_bytes(source.read_bytes())
    try:
        audit = Audit(root, protocol_path)
        quality = audit.quality()
        screen = audit.screen(quality)
        confirmation = audit.confirm(screen) if args.phase == "all" else None
        result = {
            "passed": True,
            "audited_through": args.phase,
            "counts": dict(audit.counts),
            "quality_partial_points": len(quality["points"]),
            "quality_eligible_points": quality["eligible_count"],
            "selected": screen["selected"],
            "confirmation_summary": None
            if confirmation is None
            else confirmation.get("summary"),
            "reserve_gate": None
            if confirmation is None
            else confirmation["reserve_gate"],
            "all_root_results_equal_independent_raw_reconstruction": True,
            "all_requested_threads_unique_and_exact": True,
            "full_controls_all_10000_id_hex_rows_byte_identical": True,
            "raw_points_with_independent_hit_totals": audit.raw_points,
            "open_resources_and_page_faults": audit.resources,
            "source_sha256": digest(source.read_bytes()),
            "files_sha256": audit.files,
            "no_gt_reevaluation_or_retrieval_or_training": True,
            "timed_calls_check_status_and_count_only": True,
            "full_ids_scores_are_from_each_untimed_quality_pass": True,
            "saved_config_is_explicit_environment_not_full_process_environment": True,
            "primary_queries_reused_not_blind": True,
            "statistically_blind_audit": False,
            "new_threshold_or_effect_selection": False,
            "breakthrough_verified": False,
        }
        save(output / "report.json", result)
        logging.info("AUDIT83 PASSED %s", output / "report.json")
    except BaseException as error:
        save(output / "failure.json", {"passed": False, "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
