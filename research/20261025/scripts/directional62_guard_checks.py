from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
assert "SCREEN62 COMPLETE" in (R / "logs/screen62.log").read_text()
commands = json.loads((R / "results/directional62_test_commands.json").read_text())
original = commands[0]
base = []
skip = False
for x in original:
    if skip:
        skip = False
        continue
    if x == "-o":
        skip = True
        continue
    if x == "-c":
        base.append("-fsyntax-only")
        continue
    if x.startswith("-DQG_DIRECTION_BETA="):
        continue
    base.append(x)
cases = [
    (["QG_DIRECTION_BETA=.4"], "Directional beta must be in"),
    (["QG_DIRECTION_BETA=1.1"], "Directional beta must be in"),
    (
        ["QG_DIRECTION_BETA=.9", "QG_DIRECTION_MSE=1"],
        "Choose one directional calibration objective",
    ),
]
for flag in [
    "QG_RADIAL_CALIBRATION=1",
    "QG_LOWRANK=4",
    "QG_PROJECT_DIM=384",
    "QG_RESIDUAL_TWO_STAGE=1",
    "QG_GEOM_BIAS=1",
]:
    cases.append(
        (["QG_DIRECTION_BETA=.9", flag], "Directional calibration requires the native")
    )
results = []
for index, (flags, message) in enumerate(cases):
    cmd = base + ["-D" + x for x in flags]
    p = subprocess.run(cmd, check=False, text=True, capture_output=True)
    assert p.returncode != 0
    assert message in p.stderr
    (R / f"logs/directional62_guard{index}.log").write_text(p.stdout + p.stderr)
    results.append(
        {
            "defines": flags,
            "expected_error": message,
            "returncode": p.returncode,
            "command": cmd,
        }
    )
(R / "results/directional62_guard_checks.json").write_text(
    json.dumps(results, indent=2) + "\n"
)
logging.info("%s %s", "GUARD62 COMPLETE cases", len(results))
