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
    (R / "results/coupled74_commands.json").write_text(
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
    "rotate74_build",
    [
        "c++",
        "-O3",
        "-std=c++17",
        "-march=native",
        "-pthread",
        "-DQG_ROTATION_768=1",
        "-I" + str(S / "src"),
        "scripts/rotate74.cc",
        "-o",
        "rotate74",
    ],
)
run(
    "rotate74",
    [
        "./rotate74",
        "models64/full_vectors.f32",
        "models74/ids.u32",
        "models74/rotated.f32",
    ],
)
for name in ["validate_data74", "fitdata74", "train74"]:
    run(name, [sys.executable, "scripts/" + name + ".py"])
run(
    "codec74_build",
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
        "scripts/codec74.cc",
        "lib/libzvec_turbo.a",
        "lib/librabitqlib_runtime.a",
        "lib/libzvec_ailego.a",
        "-o",
        "codec74",
    ],
)
for name in ["native", "precision_control", "learned"]:
    run(
        "codec74_" + name,
        [
            "./codec74",
            "models74/rotated.f32",
            "models74/signs.i8",
            name,
            "models74/scores_" + name + ".f32",
            "models74/codes_" + name + ".bin",
        ],
    )
logging.info("%s", "RUN74 COMPLETE")
