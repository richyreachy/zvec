from __future__ import annotations

import csv
import json
import logging
import statistics
import subprocess
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
M = R / "models70"
assert "TRAIN70 COMPLETE" in (R / "logs/train70.log").read_text()
training = json.loads((R / "results/rank70_training.json").read_text())
valid = np.load(M / "validation_rows.npy")
valid.astype("<u4").tofile(M / "validation_rows.u32")
truth = np.memmap(M / "truth.f64", mode="r", dtype="<f8", shape=(8192, 4, 32))[valid]
minimum = truth.min(axis=2)
commands = []
summary = {
    "models": [],
    "production_codec_and_lut_used": True,
    "block_proxy_not_full_search": True,
}
for name, coeff in [("baseline", "-")] + [
    (str(seed), str(M / f"coefficients_seed{seed}.f32")) for seed in [70, 71, 72]
]:
    cmd = [
        str(R / "codec70"),
        str(M / "rotated.f32"),
        coeff,
        str(M / "validation_rows.u32"),
        str(M / f"codec_scores_{name}.f32"),
    ]
    commands.append(cmd)
    with (R / f"logs/codec70_{name}.log").open("w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
    scores = np.fromfile(M / f"codec_scores_{name}.f32", dtype="<f4").reshape(
        2048, 4, 32
    )
    chosen = np.argmin(scores, axis=2)
    values = np.take_along_axis(truth, chosen[:, :, None], axis=2)[:, :, 0]
    error = values > minimum + 1e-07
    np.save(M / f"codec_errors_{name}.npy", error)
    if name == "baseline":
        base_error = error
    change = base_error.mean(axis=1) - error.mean(axis=1)
    base = base_error.mean(axis=1)
    boot = []
    rng = np.random.default_rng(7070)
    for _i in range(4000):
        sample = rng.integers(0, 2048, 2048)
        boot.append(float(change[sample].mean() / base[sample].mean()))
    gain = float(change.mean() / base.mean())
    ci = np.quantile(boot, [0.025, 0.975]).tolist()
    summary["models"].append(
        {
            "model": name,
            "queries": 8192,
            "top1_error_rate": float(error.mean()),
            "top1_errors": int(error.sum()),
            "relative_error_reduction": gain,
            "paired_center_bootstrap95": ci,
            "quality_gate_passed": gain >= 0.05 and ci[0] > 0,
        }
    )
    logging.info("%s %s", "CODEC", json.dumps(summary["models"][-1]))
cmd = [
    "c++",
    "-O3",
    "-std=c++17",
    "-march=native",
    "-UNDEBUG",
    str(R / "scripts/rotation61_kernel_test.cc"),
    "-o",
    str(R / "rotation70_kernel_test"),
]
commands.append(cmd)
subprocess.run(cmd, check=True)
micro = []
for seed in [70, 71, 72]:
    cmd = [
        "taskset",
        "-c",
        "0",
        str(R / "rotation70_kernel_test"),
        str(M / f"coefficients_seed{seed}.f32"),
    ]
    commands.append(cmd)
    with (R / f"logs/rotation70_kernel_{seed}.log").open("w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
    lines = (R / f"logs/rotation70_kernel_{seed}.log").read_text().splitlines()
    assert "scalar_coordinate_checks=768000" in lines[0]
    rows = list(csv.DictReader(lines[1:]))
    assert len(rows) == 3
    micro.append(
        {
            "seed": seed,
            "scalar_coordinate_checks": 768000,
            "identity_cases": 1000,
            "norm_cases": 1000,
            "incremental_ns_median": statistics.median(
                float(x["incremental_ns"]) for x in rows
            ),
            "not_full_search_timing": True,
        }
    )
summary["kernel_checks_and_microbenchmarks"] = micro
summary["all_seeds_actual_codec_gate_passed"] = all(
    x["quality_gate_passed"] for x in summary["models"][1:]
)
summary["integration_gate_passed"] = (
    training["all_seeds_quality_gate_passed"]
    and summary["all_seeds_actual_codec_gate_passed"]
)
summary["no_retrieval_qps_claim"] = True
summary["new_retrieval_timing_rows"] = 0
summary["cumulative_retrieval_timing_rows"] = 1310
(R / "results/rank70_codec_validation.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
(R / "results/rank70_post_commands.json").write_text(
    json.dumps(commands, indent=2) + "\n"
)
logging.info("%s", "POST70 COMPLETE")
