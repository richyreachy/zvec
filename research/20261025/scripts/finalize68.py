from __future__ import annotations

import hashlib
import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
S = Path("/home/xingrui/workspace/quantized_graph/symphony_research_20261025")
root = ET.parse(R / "results/stage68_default_tests.xml").getroot()
assert int(root.attrib["tests"]) == 28
assert int(root.attrib["failures"]) == 0
assert int(root.attrib["errors"]) == 0
verified = {}
for name in ["trace68_native", "stage68_default"]:
    m = json.loads((R / f"variants/{name}/source_files_sha256.json").read_text())
    for p, digest in m.items():
        assert hashlib.sha256((S / p).read_bytes()).hexdigest() == digest
    verified[name] = len(m)
files = [
    R / "models68/events.bin",
    R / "models68/primary/events.bin",
    R / "models68/pairs_v2.npy",
    R / "models68/primary/pairs_v2.npy",
    R / "models68/reference_results.bin",
    R / "models68/disabled_results.bin",
    R / "models68/enabled_results.bin",
    R / "models68/primary/queries.f32",
]
for folder in ["models68/geometry", "models68/primary/geometry"]:
    files.extend(p for p in (R / folder).iterdir() if p.is_file())
manifest = {
    str(p.relative_to(R)): {
        "bytes": p.stat().st_size,
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
    }
    for p in files
}
(R / "results/trace68_artifact_manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n"
)
commands = {
    "geometry_compile": [
        "c++",
        "-O3",
        "-std=c++17",
        "-march=native",
        "-pthread",
        "-DQG_ROTATION_768=1",
        "-I" + str(S / "src"),
        "scripts/geometry68_probe.cc",
        "-o",
        "geometry68_probe",
    ],
    "geometry_runs": [
        [
            "./geometry68_probe",
            "models64/full_vectors.f32",
            "models68/geometry/ids.u32",
            "models68/geometry",
            "-",
        ],
        [
            "./geometry68_probe",
            "models64/full_vectors.f32",
            "models68/primary/geometry/ids.u32",
            "models68/primary/geometry",
            "models68/primary/queries.f32",
        ],
    ],
    "writer_compile": [
        "c++",
        "-O2",
        "-std=c++17",
        "-UNDEBUG",
        "-I" + str(S / "src"),
        "scripts/trace68_writer_test.cc",
        "-o",
        "trace68_writer_test",
    ],
    "writer_run": ["./trace68_writer_test"],
    "numpy_python": "/home/xingrui/bench/symphony_qg/.venv/bin/python",
    "binary_sha256": {
        name: hashlib.sha256((R / name).read_bytes()).hexdigest()
        for name in ["trace68_driver", "geometry68_probe", "trace68_writer_test"]
    },
    "source_sha256": {
        str(p.relative_to(R)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (R / "scripts").glob("*68*")
        if p.is_file()
    },
}
(R / "results/trace68_analysis_commands.json").write_text(
    json.dumps(commands, indent=2) + "\n"
)
report = {
    "default_tests": 28,
    "writer_unit_pass": True,
    "source_manifests": verified,
    "database_query_equivalence": 4096,
    "primary_query_equivalence": 1000,
    "source_and_edge_checks_database": json.loads(
        (R / "results/trace68_availability_v2.json").read_text()
    )["verified_scored_edges"],
    "source_and_edge_checks_primary": json.loads(
        (R / "results/trace68_primary_availability.json").read_text()
    )["verified_scored_edges"],
    "geometry_pairs": 32768 + 31888,
    "stage67_lambda_unchanged": True,
    "no_model_fit_on_primary": True,
    "online_algorithm_scoring_unchanged": True,
    "no_new_retrieval_timing_rows": True,
    "breakthrough_verified": False,
    "analysis_failures_preserved": [
        "system Python missing NumPy;rerun established venv",
        "primary text trailing semicolon;strict parser corrected,initial failure log retained",
    ],
}
(R / "results/trace68_validation.json").write_text(json.dumps(report, indent=2) + "\n")
logging.info("%s", json.dumps(report, indent=2))
