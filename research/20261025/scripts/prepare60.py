from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)
R = Path(__file__).resolve().parent.parent
S = Path("/home/xingrui/workspace/quantized_graph/symphony_research_20261025")
commands = []


def call(cmd, log):
    commands.append(cmd)
    with (R / "logs" / log).open("w") as f:
        subprocess.run(cmd, cwd=R, stdout=f, stderr=subprocess.STDOUT, check=True)


call(
    [
        "g++",
        "-O3",
        "-std=c++17",
        "-UNDEBUG",
        "-I" + str(S / "src/core"),
        str(R / "scripts/integer_router59_test.cc"),
        "-o",
        str(R / "router60_legacy_test"),
    ],
    "router60_legacy_build.log",
)
call(
    [
        str(R / "router60_legacy_test"),
        str(R / "models59/router.bin"),
        str(R / "models60/invalid_legacy.tmp"),
    ],
    "router60_legacy_test.log",
)
original_targets = np.fromfile(R / "models59/targets.i32", dtype="<i4").reshape(-1, 16)
original_counts = np.fromfile(R / "models59/counts.u32", dtype="<u4")
metadata = {"fixed_tree": True, "query_or_gt_inputs": [], "variants": {}}
for dim in [768, 384, 256]:
    path = R / f"models60/router{dim}.bin"
    header = np.fromfile(path, dtype="<u4", count=5)
    assert list(header) == [1363628633, 2, dim, len(original_counts), 1000000]
    dtype = np.dtype(
        [
            ("count", "<u4"),
            ("targets", "<i4", (16,)),
            ("scale", "<f4", (16,)),
            ("offset", "<f4", (16,)),
            ("mean_dot", "<f4", (16,)),
            ("codes", "u1", (16, dim)),
        ]
    )
    assert path.stat().st_size == 6168 + len(original_counts) * dtype.itemsize
    model = np.memmap(
        path, dtype=dtype, mode="r", offset=6168, shape=(len(original_counts),)
    )
    assert np.array_equal(model["targets"], original_targets)
    assert np.array_equal(model["count"], original_counts)
    meta = {
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "groups": len(model),
        "tree_ids_and_counts_equal": True,
    }
    metadata["variants"][dim] = meta
    exe = str(R / f"projected_router60_test{dim}")
    call(
        [
            "g++",
            "-O3",
            "-std=c++17",
            "-UNDEBUG",
            "-DQG_ROTATION_768=1",
            f"-DQG_ROUTER_PROJECTED={dim}",
            "-I" + str(S / "src/core"),
            "-I" + str(S / "src"),
            str(R / "scripts/projected_router60_test.cc"),
            str(R / "lib/libzvec_turbo.a"),
            str(R / "lib/libzvec_ailego.a"),
            "-o",
            exe,
        ],
        f"projected_router60_test{dim}_build.log",
    )
    call(
        [exe, str(path), str(R / "mean54/mean.f32"), str(R / "models59/router.bin")],
        f"projected_router60_test{dim}.log",
    )
(R / "results/router60_preparation.json").write_text(
    json.dumps(metadata, indent=2) + "\n"
)
(R / "results/router60_unit_commands.json").write_text(
    json.dumps(commands, indent=2) + "\n"
)
logging.info("%s", "PREPARE60 COMPLETE")
