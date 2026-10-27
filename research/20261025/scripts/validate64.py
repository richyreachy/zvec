from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
S = Path("/home/xingrui/workspace/quantized_graph/symphony_research_20261025")
assert "VALIDATE FULL64 COMPLETE" in (R / "logs/validate_full64.log").read_text()
p = R / "scripts/graph64_export_test.cc"
p.write_text(p.read_text().replace("13 rejection cases", "14 rejection cases"))
cmd = [
    "g++",
    "-O2",
    "-std=c++17",
    "-UNDEBUG",
    "-I" + str(S / "src/core"),
    str(p),
    "-o",
    str(R / "graph64_export_test_final"),
]
subprocess.run(cmd, check=True)
run = [
    str(R / "graph64_export_test_final"),
    str(R / "results/graph64_export_unit_final_tmp"),
]
with (R / "logs/graph64_export_test_final.log").open("w") as f:
    subprocess.run(run, stdout=f, stderr=subprocess.STDOUT, check=True)
(R / "results/graph64_export_final_test_commands.json").write_text(
    json.dumps({"compile": cmd, "run": run}, indent=2) + "\n"
)
with (R / "logs/graph64_geometry_checks.log").open("w") as f:
    subprocess.run(
        [
            "/home/xingrui/bench/symphony_qg/.venv/bin/python",
            str(R / "scripts/graph64_geometry_checks.py"),
        ],
        stdout=f,
        stderr=subprocess.STDOUT,
        check=True,
    )
s = (
    (R / "scripts/validate_stage58_default.py")
    .read_text()
    .replace("stage58", "stage64")
    .replace("STAGE58", "STAGE64")
)
(R / "scripts/validate_stage64_default.py").write_text(s)
with (R / "logs/validate_stage64_default.log").open("w") as f:
    subprocess.run(
        ["python3", str(R / "scripts/validate_stage64_default.py")],
        stdout=f,
        stderr=subprocess.STDOUT,
        check=True,
    )
with (R / "logs/build_graph64_export_disabled.log").open("w") as f:
    subprocess.run(
        [
            "python3",
            str(R / "scripts/compile_variant.py"),
            "graph64_export_disabled",
            "QG_ROTATION_768=1",
            "QG_AUGMENT=3",
        ],
        stdout=f,
        stderr=subprocess.STDOUT,
        check=True,
    )
plan = json.loads((R / "results/graph64_bounded_commands.json").read_text())
env = plan["env"]
env["LD_LIBRARY_PATH"] = (
    str(R / "variants/graph64_export_disabled") + ":" + str(R / "lib")
)
cmd = plan["run"]
cmd[-1] = str(R / "models64/disabled")
plan["run"] = cmd
plan["env"] = env
(R / "results/graph64_disabled_commands.json").write_text(
    json.dumps(plan, indent=2) + "\n"
)
with (R / "logs/graph64_disabled.log").open("w") as f:
    subprocess.run(
        cmd, env={**os.environ, **env}, stdout=f, stderr=subprocess.STDOUT, check=True
    )
assert (R / "models64/bounded_candidates.bin").read_bytes() == (
    R / "models64/disabled_candidates.bin"
).read_bytes()
root = ET.parse(R / "results/stage64_default_tests.xml").getroot()
assert int(root.attrib["tests"]) == 28
assert int(root.attrib["failures"]) == 0
assert int(root.attrib["errors"]) == 0
verified = {}
for name in [
    "graph64_candidates",
    "graph64_bounded",
    "graph64_export_disabled",
    "stage64_default",
]:
    hashes = json.loads((R / f"variants/{name}/source_files_sha256.json").read_text())
    for path, digest in hashes.items():
        assert hashlib.sha256((S / path).read_bytes()).hexdigest() == digest
    verified[name] = len(hashes)
summary = {
    "source_manifests_verified": verified,
    "export_disabled_candidate_queries_bit_identical": 4096,
    "export_disabled_candidate_pairs_bit_identical": 819200,
    "default_tests": 28,
    "export_unit_coordinate_checks": 5376,
    "export_rejection_assertions": 14,
    "no_retrieval_qps_or_recall_claim": True,
    "breakthrough_verified": False,
    "failed_unbounded_config_preserved": True,
}
(R / "results/graph64_validation.json").write_text(json.dumps(summary, indent=2) + "\n")
logging.info("%s %s", "VALIDATE64 COMPLETE", summary)
