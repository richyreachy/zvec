from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
assert "TIME65 COMPLETE" in (R / "logs/time65.log").read_text()
commands = []


def run(cmd, log):
    commands.append(cmd)
    with (R / "logs" / log).open("w") as f:
        subprocess.run(cmd, cwd=R, stdout=f, stderr=subprocess.STDOUT, check=True)


run(
    [
        "taskset",
        "-c",
        "0",
        str(R / "graph65_refine"),
        str(R / "models64"),
        str(R / "models65/pilot_single"),
        "4096",
        "1",
        "1",
    ],
    "graph65_pilot_single.log",
)
checks = {}
for suffix in ("first.u32", "nearest.u32", "angular.u32", "stats.f64"):
    a = (R / "models65" / ("pilot_" + suffix)).read_bytes()
    b = (R / "models65" / ("pilot_single_" + suffix)).read_bytes()
    assert a == b
    checks[suffix] = {"bytes": len(a), "sha256": hashlib.sha256(a).hexdigest()}
a = (R / "models65/pilot_first.u32").read_bytes()
with (R / "models65/full_first.u32").open("rb") as f:
    assert f.read(len(a)) == a
(R / "results/graph65_thread_determinism.json").write_text(
    json.dumps(
        {
            "one_vs_eight_threads_byte_identical": checks,
            "pilot_first_equals_full_prefix": 4096,
            "full_final_not_compared_to_partial_reverse_pilot": True,
        },
        indent=2,
    )
    + "\n"
)
run(["python3", str(R / "scripts/validate65.py")], "validate65.log")
(R / "results/graph65_post_commands.json").write_text(
    json.dumps(commands, indent=2) + "\n"
)
logging.info("%s", "POST65 COMPLETE")
