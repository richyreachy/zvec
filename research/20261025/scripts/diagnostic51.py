from __future__ import annotations

import csv
import hashlib
import json
import logging
import subprocess
import time
from pathlib import Path

from explore90 import R, run

while (
    "POST COMMIT VALIDATION COMPLETE"
    not in (R / "logs/validate_post_commit.log").read_text()
):
    time.sleep(3)
common = [
    "QG_BUCKET=1",
    "QG_BUCKET_WIDTH=.016f",
    "QG_ADAPTIVE=1",
    "QG_RUNTIME_ADAPTIVE=1",
    "QG_STOP_MIN=10",
    "QG_STOP_PATIENCE=0",
    "QG_STOP_MARGIN=.02f",
    "QG_ROTATION_768=1",
    "QG_TOPOLOGY_STATS=1",
]
variants = {
    "diag51_base": ["QG_AUGMENT=3"],
    "diag51_cover_true": ["QG_AUGMENT=2", "QG_COVER_PRUNE=1"],
    "diag51_cover_quantized": ["QG_AUGMENT=2", "QG_COVER_PRUNE=2"],
}
summary = {}
truth = []
with Path("/home/xingrui/bench/dataset/cohere/1m/neighbors.txt").open() as f:
    for line in f:
        truth.extend(int(x) for x in line.split(";")[1].split()[:10])
assert len(truth) == 10000
for name, flags in variants.items():
    with (R / f"logs/build_{name}.log").open("w") as f:
        subprocess.run(
            ["python3", str(R / "scripts/compile_variant.py"), name, *common, *flags],
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )
    paths = {
        "QG_TOPOLOGY_FILE": str(R / f"results/{name}_topology.json"),
        "QG_UNREACHABLE_KEYS_FILE": str(R / f"results/{name}_unreachable_keys.txt"),
        "QG_BFS_PARENT_FILE": str(R / f"results/{name}_parents.u32"),
    }
    run(
        name,
        name,
        [800],
        extra={"QG_MARGIN": ".02", "QG_PATIENCE": "0", "QG_CAP": "64", **paths},
        duration=-1,
        threads="1",
    )
    result = json.loads((R / f"results/{name}_topology.json").read_text())
    unreachable = {
        int(x) for x in (R / f"results/{name}_unreachable_keys.txt").read_text().split()
    }
    assert len(unreachable) == result["valid_nodes"] - result["reachable"]
    result["primary_gt_top10_unreachable_occurrences"] = sum(
        x in unreachable for x in truth
    )
    result["reachable_gt_fraction"] = (
        1 - result["primary_gt_top10_unreachable_occurrences"] / 10000
    )
    parent = R / f"results/{name}_parents.u32"
    result["parent_tree_file"] = str(parent)
    result["parent_tree_sha256"] = hashlib.sha256(parent.read_bytes()).hexdigest()
    assert parent.stat().st_size == result["nodes"] * 4
    recalls = list(
        csv.DictReader((R / f"experiments/{name}/search_k10_r0_recall.csv").open())
    )
    assert len(recalls) == 1000
    result["recall_at_margin_002_cap64"] = sum(int(r["hits"]) for r in recalls) / 10000
    summary[name] = result
    (R / "results/diagnostic51.json").write_text(json.dumps(summary, indent=2))
logging.warning("DIAGNOSTIC51 COMPLETE")
