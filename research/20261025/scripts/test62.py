from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
commands = json.loads((R / "results/build_commands.json").read_text())
executed = []
for mode, flag in [
    ("identity", "QG_DIRECTION_BETA=1."),
    ("trained", "QG_DIRECTION_BETA=.9359792556459329"),
    ("mse", "QG_DIRECTION_MSE=1"),
]:
    oldobj = str(R / "objects/benchmark.o")
    newobj = str(R / f"objects/directional62_{mode}_test.o")
    exe = str(R / f"directional62_{mode}_test")
    c = next(x["argv"] for x in commands if "-c" in x["argv"] and oldobj in x["argv"])
    c = [
        newobj
        if x == oldobj
        else str(R / "scripts/directional62_test.cc")
        if x == str(R / "scripts/benchmark.cc")
        else x
        for x in c
    ]
    c += ["-DQG_ROTATION_768=1", "-D" + flag, "-UNDEBUG"]
    executed.append(c)
    subprocess.run(c, check=True)
    c = next(
        x["argv"] for x in commands if "-c" not in x["argv"] and oldobj in x["argv"]
    )
    c = [newobj if x == oldobj else exe if x == str(R / "benchmark") else x for x in c]
    executed.append(c)
    subprocess.run(c, check=True)
    with (R / f"logs/directional62_{mode}_test.log").open("w") as f:
        subprocess.run(
            [exe],
            env={**os.environ, "LD_LIBRARY_PATH": str(R / "lib")},
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )
    logging.info("%s %s", mode, "PASSED")
(R / "results/directional62_test_commands.json").write_text(
    json.dumps(executed, indent=2) + "\n"
)
logging.info("%s", "TEST62 COMPLETE")
