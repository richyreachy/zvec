from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
S = Path("/home/xingrui/workspace/quantized_graph/symphony_research_20261025")
commands = []


def run(name, cmd):
    commands.append({"name": name, "argv": cmd})
    (R / "results/coupled75_commands.json").write_text(
        json.dumps(
            {
                "cwd": str(R),
                "environment": {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
                "commands": commands,
            },
            indent=2,
        )
        + "\n"
    )
    logging.info("%s %s", "RUN", name)
    with (R / "logs" / f"{name}.log").open("x") as f:
        subprocess.run(cmd, cwd=R, stdout=f, stderr=subprocess.STDOUT, check=True)


run(
    "rotate75_build",
    [
        "c++",
        "-O3",
        "-std=c++17",
        "-march=native",
        "-pthread",
        "-DQG_ROTATION_768=1",
        "-I" + str(S / "src"),
        "scripts/rotate75.cc",
        "-o",
        "rotate75",
    ],
)
run(
    "rotate75",
    [
        "./rotate75",
        "models64/full_vectors.f32",
        "models75/ids.u32",
        "models75/rotated.f32",
    ],
)
run(
    "rotate75_cross",
    [
        "./rotate75",
        "models64/full_vectors.f32",
        "models75/cross_ids.u32",
        "models75/cross_rotated.f32",
    ],
)
for name in ["validate_data75", "crossdata75", "fitdata75"]:
    run(name, [sys.executable, "scripts/" + name + ".py"])
for variant in ["coupled", "independent"]:
    run("train75_" + variant, [sys.executable, "scripts/train75.py", variant])
for binary in ["codec75", "cross75"]:
    run(
        binary + "_build",
        [
            "c++",
            "-O3",
            "-std=c++17",
            "-march=native",
            "-ffp-contract=off",
            "-pthread",
            "-DRABITQ_SUPPORTED=1",
            "-DQG_ROTATION_768=1",
            "-I" + str(S / "src"),
            "-isystem",
            "/home/xingrui/workspace/zvec/symphony_qg/zvec/thirdparty/RaBitQ-Library/RaBitQ-Library-0.1/include",
            "scripts/" + binary + ".cc",
            "lib/libzvec_turbo.a",
            "lib/librabitqlib_runtime.a",
            "lib/libzvec_ailego.a",
            "-o",
            binary,
        ],
    )
for name in ["native", "precision_control", "coupled", "independent"]:
    signs = (
        "models75/signs_"
        + ("independent" if name == "independent" else "coupled")
        + ".i8"
    )
    run(
        "codec75_" + name,
        [
            "./codec75",
            "models75/rotated.f32",
            signs,
            name,
            "models75/scores_" + name + ".f32",
            "models75/codes_" + name + ".bin",
        ],
    )
    run(
        "cross75_" + name,
        [
            "./cross75",
            "models75/rotated.f32",
            "models75/cross_rotated.f32",
            "models75/pairs.u32",
            "models75/codes_" + name + ".bin",
            signs,
            name,
            "models75/cross_scores_" + name + ".f32",
        ],
    )
logging.info("%s", "RUN75 COMPLETE")
