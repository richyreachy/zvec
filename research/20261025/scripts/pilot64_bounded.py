from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
M = R / "models64"
S = Path("/home/xingrui/workspace/quantized_graph/symphony_research_20261025")
flags = ["QG_ROTATION_768=1", "QG_AUGMENT=3", "QG_GRAPH_EXPORT=1"]
with (R / "logs/build_graph64_bounded.log").open("w") as f:
    subprocess.run(
        ["python3", str(R / "scripts/compile_variant.py"), "graph64_bounded", *flags],
        stdout=f,
        stderr=subprocess.STDOUT,
        check=True,
    )
env = {
    "LD_LIBRARY_PATH": str(R / "variants/graph64_bounded") + ":" + str(R / "lib"),
    "QG_GRAPH_EXPORT_FILE": str(M / "bounded_graph.bin"),
    "QG_GRAPH_VECTOR_FILE": str(M / "bounded_vectors.f32"),
    "GRAPH64_QUERY_COUNT": "4096",
    "GRAPH64_QUERY_THREADS": "8",
    "BENCH_DEGREE": "32",
}
cmd = [
    "taskset",
    "-c",
    "0,2,4,6,8,10,12,14",
    str(R / "graph64_driver"),
    "search",
    str(R / "indexes/cohere1m_m50_efc500.index"),
    "symphony_qg",
    "0",
    "-1",
    str(M / "bounded"),
]
(R / "results/graph64_bounded_commands.json").write_text(
    json.dumps(
        {
            "run": cmd,
            "env": env,
            "candidate_count": 200,
            "ef": 400,
            "queue": "bounded original beam",
            "no_adaptive_top10_stopping": True,
        },
        indent=2,
    )
    + "\n"
)
subprocess.run(cmd, env={**os.environ, **env}, check=True)
logging.info("%s", "PILOT64 BOUNDED COMPLETE")
