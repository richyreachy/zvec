from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
M = R / "models72"
S = Path("/home/xingrui/workspace/quantized_graph/symphony_research_20261025")
commands = []


def run(name, cmd):
    commands.append({"name": name, "argv": cmd})
    (R / "results/risk72_commands.json").write_text(
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
    "rotate72_build",
    [
        "c++",
        "-O3",
        "-std=c++17",
        "-march=native",
        "-pthread",
        "-DQG_ROTATION_768=1",
        "-I" + str(S / "src"),
        "scripts/rotate72.cc",
        "-o",
        "rotate72",
    ],
)
run(
    "rotate72",
    [
        "./rotate72",
        "models64/full_vectors.f32",
        "models72/ids.u32",
        "models72/rotated.f32",
    ],
)
run("validate_data72", [sys.executable, "scripts/validate_data72.py"])
run(
    "codec72_build",
    [
        "c++",
        "-O3",
        "-std=c++17",
        "-march=native",
        "-pthread",
        "-DRABITQ_SUPPORTED=1",
        "-DQG_ROTATION_768=1",
        "-I" + str(S / "src"),
        "-isystem",
        "/home/xingrui/workspace/zvec/symphony_qg/zvec/thirdparty/RaBitQ-Library/RaBitQ-Library-0.1/include",
        "scripts/codec72.cc",
        "lib/libzvec_turbo.a",
        "lib/librabitqlib_runtime.a",
        "lib/libzvec_ailego.a",
        "-o",
        "codec72",
    ],
)
for name in ["baseline", "70", "71", "72"]:
    coeff = "-" if name == "baseline" else "models70/coefficients_seed" + name + ".f32"
    run(
        "codec72_" + name,
        [
            "./codec72",
            "models72/rotated.f32",
            coeff,
            "models72/rows.u32",
            "models72/codec_scores_" + name + ".f32",
        ],
    )
logging.info("%s", "RUN72 COMPLETE")
