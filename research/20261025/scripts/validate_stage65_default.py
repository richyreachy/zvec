from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path("/home/xingrui/bench/quantized_graph/research_20261025")
SRC = Path("/home/xingrui/workspace/quantized_graph/symphony_research_20261025")
name = "stage65_default"
executed = []
with (R / f"logs/build_{name}.log").open("w") as f:
    subprocess.run(
        ["python3", str(R / "scripts/compile_variant.py"), name],
        stdout=f,
        stderr=subprocess.STDOUT,
        check=True,
    )
commands = json.loads((R / "results/build_commands.json").read_text())


def call(c):
    executed.append(c)
    subprocess.run(c, cwd=R, check=True)


oldobj = str(R / "objects/symphony_qg_test.o")
newobj = str(R / f"variants/{name}/symphony_qg_test.o")
c = next(x["argv"] for x in commands if "-c" in x["argv"] and oldobj in x["argv"])
c = [newobj if x == oldobj else x for x in c]
c += ["-UNDEBUG"]
call(c)
c = next(x["argv"] for x in commands if "-c" not in x["argv"] and oldobj in x["argv"])
c = [
    newobj
    if x == oldobj
    else str(R / "symphony_qg_test_stage65")
    if x == str(R / "symphony_qg_test")
    else x
    for x in c
]
call(c)
with (R / "logs/stage65_default_tests.log").open("w") as f:
    subprocess.run(
        [
            str(R / "symphony_qg_test_stage65"),
            "--gtest_output=xml:" + str(R / "results/stage65_default_tests.xml"),
        ],
        env={
            **os.environ,
            "LD_LIBRARY_PATH": str(R / f"variants/{name}") + ":" + str(R / "lib"),
        },
        stdout=f,
        stderr=subprocess.STDOUT,
        check=True,
    )
(R / "results/stage65_default_validation_commands.json").write_text(
    json.dumps(executed, indent=2)
)
logging.info("%s", "STAGE65 DEFAULT VALIDATION COMPLETE")
