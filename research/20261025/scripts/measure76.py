from __future__ import annotations

import hashlib
import json
import logging
import statistics
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert json.loads((R / "results/models76_validation_checks.json").read_text())[
    "all_accepted_positions_and_order_identical"
]
codec = {}
for generated, original in [
    ("codes.bin", "codes_independent.bin"),
    ("scores.f32", "scores_independent.f32"),
]:
    a = sha(R / "models76_validation" / generated)
    b = sha(R / "models75" / original)
    assert a == b
    codec[generated] = {"sha256": a, "phase75_byte_identical": True}
(R / "results/independent76_codec_checks.json").write_text(
    json.dumps(
        {"outputs": codec, "score_count": 2097152, "scalar_lut_fma_bit_checks": 65536},
        indent=2,
    )
    + "\n"
)
rows = []
commands = []
names = [
    "signs.i8",
    "alpha.f64",
    "history.f64",
    "moves.u32",
    "counts.u32",
    "lambda.f64",
]
for round_id in range(4):
    for threads in [1, 8] if round_id % 2 == 0 else [8, 1]:
        name = f"models76_t{threads}_r{round_id}"
        cmd = [
            "taskset",
            "-c",
            "0" if threads == 1 else "0,2,4,6,8,10,12,14",
            "./builder76",
            "models75/fit.f32",
            name,
            str(threads),
        ]
        commands.append(cmd)
        logging.info("%s %s", "MEASURE76", name)
        with (R / "logs" / f"builder76_t{threads}_r{round_id}.log").open("x") as log:
            subprocess.run(cmd, cwd=R, stdout=log, stderr=subprocess.STDOUT, check=True)
        a = json.loads(
            (R / "logs" / f"builder76_t{threads}_r{round_id}.log").read_text()
        )
        a.update(
            {
                "round": round_id,
                "warmup": round_id == 0,
                "affinity": cmd[2],
                "output": name,
            }
        )
        for f in names:
            assert sha(R / name / f) == sha(R / "models76_validation" / f), (name, f)
        a["all_six_outputs_byte_identical"] = True
        rows.append(a)
summary = {
    str(t): {
        "median_compute_seconds": statistics.median(
            x["compute_seconds"]
            for x in rows
            if x["threads"] == t and (not x["warmup"])
        ),
        "median_elapsed_seconds": statistics.median(
            x["elapsed_seconds"]
            for x in rows
            if x["threads"] == t and (not x["warmup"])
        ),
        "max_peak_rss_kib": max(
            x["peak_rss_kib"] for x in rows if x["threads"] == t and (not x["warmup"])
        ),
    }
    for t in [1, 8]
}
a = {
    "scope": "2048-center offline build only;not query QPS or measured fullgraph build time",
    "warmup_rounds": 1,
    "measured_rounds": 3,
    "fit_sha256": sha(R / "models75/fit.f32"),
    "commands": commands,
    "runs": rows,
    "summary": summary,
    "binary_sha256": sha(R / "builder76"),
}
(R / "results/independent76_build_measurement.json").write_text(
    json.dumps(a, indent=2) + "\n"
)
logging.info(json.dumps(summary, indent=2))
logging.info("MEASURE76 COMPLETE")
