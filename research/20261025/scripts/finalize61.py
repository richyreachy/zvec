from __future__ import annotations

import csv
import hashlib
import json
import logging
import statistics as st
import xml.etree.ElementTree as ET
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
S = Path("/home/xingrui/workspace/quantized_graph/symphony_research_20261025")
assert "TRAIN61 COMPLETE" in (R / "logs/train61.log").read_text()
assert "VALIDATE61 COMPLETE" in (R / "logs/validate61.log").read_text()
training = json.loads((R / "results/rotation61_training.json").read_text())
assert len(training["seeds"]) == 3
assert not training["all_seeds_quality_gate_passed"]
text = (R / "results/rotation61_kernel_micro.txt").read_text()
assert "scalar_coordinate_checks=768000" in text
assert "passed" in text
rows = list(csv.DictReader(text.splitlines()[1:]))
assert len(rows) == 3
root = ET.parse(R / "results/stage61_default_tests.xml").getroot()
assert int(root.attrib["tests"]) == 28
assert int(root.attrib["failures"]) == 0
assert int(root.attrib["errors"]) == 0
verified = {}
for variant in ["edge61_export", "stage61_default"]:
    hashes = json.loads(
        (R / f"variants/{variant}/source_files_sha256.json").read_text()
    )
    for path, digest in hashes.items():
        assert hashlib.sha256((S / path).read_bytes()).hexdigest() == digest
    verified[variant] = len(hashes)
summary = {
    "diagnostic_only": True,
    "no_search_qps_or_recall_claim": True,
    "breakthrough_verified": False,
    "samples": training["samples"],
    "train_edges": training["train_edges"],
    "validation_edges": training["validation_edges"],
    "all_seeds_quality_gate_passed": False,
    "kernel_scalar_coordinate_checks": 768000,
    "kernel_identity_vectors": 1000,
    "kernel_norm_vectors": 1000,
    "incremental_kernel_ns_median": st.median(float(r["incremental_ns"]) for r in rows),
    "default_tests": 28,
    "edge_export_checks": 511,
    "math_checks": json.loads((R / "results/rotation61_math_checks.json").read_text()),
    "source_manifests_verified": verified,
    "decision": "Do not integrate learned rotation:all seeds fail prespecified error-improvement gate,although transform cost is small. Next directional-factor probe is distinct and not implemented online.",
}
(R / "results/rotation61_validation.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
manifest = {}
for path in [
    *(R / "models61").glob("*.f32"),
    R / "models61/training_config.json",
    R / "models61/train_centers.npy",
    R / "models61/validation_centers.npy",
    R / "models61/directional62_pair_scores.npz",
]:
    manifest[str(path.relative_to(R))] = hashlib.sha256(path.read_bytes()).hexdigest()
(R / "results/rotation61_model_sha256.json").write_text(
    json.dumps(manifest, indent=2) + "\n"
)
logging.info("%s %s", "FINALIZE61 COMPLETE", summary)
