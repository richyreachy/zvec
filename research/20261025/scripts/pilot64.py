from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
S = Path("/home/xingrui/workspace/quantized_graph/symphony_research_20261025")
M = R / "models64"
M.mkdir(exist_ok=False)
executed = []
cmd = [
    "g++",
    "-O2",
    "-std=c++17",
    "-UNDEBUG",
    "-I" + str(S / "src/core"),
    str(R / "scripts/graph64_export_test.cc"),
    "-o",
    str(R / "graph64_export_test"),
]
executed.append(cmd)
subprocess.run(cmd, check=True)
cmd = [str(R / "graph64_export_test"), str(R / "results/graph64_export_unit_tmp")]
executed.append(cmd)
with (R / "logs/graph64_export_test.log").open("w") as f:
    subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
flags = [
    "QG_BUCKET=1",
    "QG_BUCKET_WIDTH=.016f",
    "QG_ROTATION_768=1",
    "QG_AUGMENT=3",
    "QG_GRAPH_EXPORT=1",
]
with (R / "logs/build_graph64_candidates.log").open("w") as f:
    subprocess.run(
        [
            "python3",
            str(R / "scripts/compile_variant.py"),
            "graph64_candidates",
            *flags,
        ],
        stdout=f,
        stderr=subprocess.STDOUT,
        check=True,
    )
commands = json.loads((R / "results/build_commands.json").read_text())
oldobj = str(R / "objects/benchmark.o")
newobj = str(R / "objects/graph64_driver.o")
exe = str(R / "graph64_driver")
c = next(x["argv"] for x in commands if "-c" in x["argv"] and oldobj in x["argv"])
c = [
    newobj
    if x == oldobj
    else str(R / "scripts/graph64_driver.cc")
    if x == str(R / "scripts/benchmark.cc")
    else x
    for x in c
]
executed.append(c)
subprocess.run(c, check=True)
c = next(x["argv"] for x in commands if "-c" not in x["argv"] and oldobj in x["argv"])
c = [newobj if x == oldobj else exe if x == str(R / "benchmark") else x for x in c]
executed.append(c)
subprocess.run(c, check=True)
env = {
    "LD_LIBRARY_PATH": str(R / "variants/graph64_candidates") + ":" + str(R / "lib"),
    "QG_GRAPH_EXPORT_FILE": str(M / "graph.bin"),
    "QG_GRAPH_VECTOR_FILE": str(M / "vectors.f32"),
    "GRAPH64_QUERY_COUNT": "4096",
    "GRAPH64_QUERY_THREADS": "8",
    "BENCH_DEGREE": "32",
}
cmd = [
    "taskset",
    "-c",
    "0,2,4,6,8,10,12,14",
    exe,
    "search",
    str(R / "indexes/cohere1m_m50_efc500.index"),
    "symphony_qg",
    "0",
    "-1",
    str(M / "pilot"),
]
(R / "results/graph64_pilot_commands.json").write_text(
    json.dumps(
        {
            "build": executed,
            "run": cmd,
            "env": env,
            "candidate_count": 200,
            "ef": 400,
            "no_adaptive_top10_stopping": True,
        },
        indent=2,
    )
    + "\n"
)
subprocess.run(cmd, env={**os.environ, **env}, check=True)
logging.info("%s", "PILOT64 COMPLETE")
