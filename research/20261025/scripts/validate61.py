from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
assert "TRAIN61 COMPLETE" in (R / "logs/train61.log").read_text()
s = (
    (R / "scripts/validate_stage58_default.py")
    .read_text()
    .replace("stage58", "stage61")
    .replace("STAGE58", "STAGE61")
)
(R / "scripts/validate_stage61_default.py").write_text(s)
with (R / "logs/validate_stage61_default.log").open("w") as f:
    subprocess.run(
        ["python3", str(R / "scripts/validate_stage61_default.py")],
        stdout=f,
        stderr=subprocess.STDOUT,
        check=True,
    )
cmd = [
    "g++",
    "-O3",
    "-std=c++17",
    "-UNDEBUG",
    str(R / "scripts/rotation61_kernel_test.cc"),
    "-o",
    str(R / "rotation61_kernel_test"),
]
subprocess.run(cmd, check=True)
run = [
    "taskset",
    "-c",
    "0",
    str(R / "rotation61_kernel_test"),
    str(R / "models61/coefficients_seed61.f32"),
]
with (R / "results/rotation61_kernel_micro.txt").open("w") as f:
    subprocess.run(run, stdout=f, check=True)
(R / "results/rotation61_kernel_commands.json").write_text(
    json.dumps({"compile": cmd, "run": run}, indent=2) + "\n"
)
logging.info("%s", "VALIDATE61 COMPLETE")
