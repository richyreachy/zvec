from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
M = R / "models64"
pilot = json.loads((R / "results/graph64_pilot_validation.json").read_text())
assert pilot["fp64_cosine_pairs_checked"] == 819200
assert pilot["initial_and_bounded_exports_identical"]
plan = json.loads((R / "results/graph64_bounded_commands.json").read_text())
env = plan["env"]
env["QG_GRAPH_EXPORT_FILE"] = str(M / "full_graph.bin")
env["QG_GRAPH_VECTOR_FILE"] = str(M / "full_vectors.f32")
env["GRAPH64_QUERY_COUNT"] = "1000000"
cmd = plan["run"]
cmd[-1] = str(M / "full")
plan["run"] = cmd
plan["env"] = env
plan["pilot_validation_passed"] = True
(R / "results/graph64_full_commands.json").write_text(json.dumps(plan, indent=2) + "\n")
subprocess.run(cmd, env={**os.environ, **env}, check=True)
logging.info("%s", "FULL64 COMPLETE")
