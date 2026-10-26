from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
M = R / "models69"
M.mkdir(exist_ok=False)
shared = {
    "TRACE68_GRAPH_FILE": str(R / "models64/full_graph.bin"),
    "TRACE68_VECTOR_FILE": str(R / "models64/full_vectors.f32"),
    "TRACE68_QUERY_COUNT": "4096",
    "QG_CAP": "64",
    "QG_PATIENCE": "0",
    "QG_MARGIN": ".02",
    "BENCH_DEGREE": "32",
}
commands = []
for mode, variant, enable in [
    ("disabled", "fusion69_native", "0"),
    ("enabled", "fusion69_native", "1"),
    ("audit", "fusion69_audit", "1"),
]:
    env = {
        **shared,
        "QG_FUSION_ENABLE": enable,
        "LD_LIBRARY_PATH": str(R / "variants" / variant) + ":" + str(R / "lib"),
    }
    cmd = [
        "taskset",
        "-c",
        "0,2,4,6,8,10,12,14",
        str(R / "trace68_driver"),
        "search",
        str(R / "indexes/cohere1m_m50_efc500.index"),
        "symphony_qg",
        "0",
        "-1",
        str(M / mode),
    ]
    commands.append({"mode": mode, "command": cmd, "env": env})
    (R / "results/fusion69_run_commands.json").write_text(
        json.dumps(commands, indent=2) + "\n"
    )
    with (R / f"logs/fusion69_{mode}.log").open("w") as f:
        subprocess.run(
            cmd,
            env={**os.environ, **env},
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )
    logging.info("%s %s", "DONE", mode)
reference = (R / "models68/reference_results.bin").read_bytes()
assert (M / "disabled_results.bin").read_bytes() == reference
assert (M / "enabled_results.bin").read_bytes() == (
    M / "audit_results.bin"
).read_bytes()
(R / "results/fusion69_equivalence.json").write_text(
    json.dumps(
        {
            "database_queries": 4096,
            "disabled_and_frozen_reference_bit_identical": True,
            "enabled_and_audit_bit_identical": True,
            "disabled_sha256": hashlib.sha256(reference).hexdigest(),
            "self_included_proxy_queries": True,
            "no_primary_or_gt_queries_used": True,
        },
        indent=2,
    )
    + "\n"
)
logging.info("%s", "RUN69 COMPLETE")
