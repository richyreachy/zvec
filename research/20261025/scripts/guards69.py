from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
S = Path("/home/xingrui/workspace/quantized_graph/symphony_research_20261025")
flags = [
    "QG_FUSION=1",
    "QG_BUCKET=1",
    "QG_BUCKET_WIDTH=.016f",
    "QG_ADAPTIVE=1",
    "QG_ROTATION_768=1",
]
source = '#include "core/algorithm/quantized_graph/quantized_graph.h"\n'
base = [
    "c++",
    "-std=c++17",
    "-E",
    "-x",
    "c++",
    "-I" + str(S / "src"),
    "-I" + str(S / "src/include"),
    "-I" + str(S / "src/core"),
    "-",
]
checks = []
for name, extra, remove in [
    ("supported", [], []),
    ("no_bucket", [], ["QG_BUCKET=1"]),
    ("no_adaptive", [], ["QG_ADAPTIVE=1"]),
    ("no_rotation", [], ["QG_ROTATION_768=1"]),
] + [
    (x, [x + "=1"], [])
    for x in [
        "QG_CENTER_SQ8",
        "QG_PROJECT_DIM",
        "QG_DIRECTION_MSE",
        "QG_LOWRANK",
        "QG_WINDOW",
        "QG_EXPAND_BATCH",
        "QG_FIRST_ESTIMATE",
        "QG_FLAT_BUCKET",
    ]
]:
    cmd = (
        base + ["-D" + x for x in flags if x not in remove] + ["-D" + x for x in extra]
    )
    p = subprocess.run(
        cmd,
        check=False,
        input=source,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    expected = name == "supported"
    assert (p.returncode == 0) == expected, (name, p.stderr)
    if not expected:
        assert "Fusion " in p.stderr
    checks.append(
        {
            "name": name,
            "command": cmd,
            "expected_success": expected,
            "exit_code": p.returncode,
            "stderr": p.stderr,
        }
    )
(R / "results/fusion69_compile_guards.json").write_text(
    json.dumps(checks, indent=2) + "\n"
)
logging.info("%s %s", "GUARDS69 PASS", len(checks))
