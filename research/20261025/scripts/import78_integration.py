"""Actual stage78 graph integration, 40-vector synthetic fixture, no full1M timing."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")

R = Path(__file__).resolve().parent.parent
SRC = Path("/home/xingrui/workspace/quantized_graph/symphony_research_20261025")
name = sys.argv[1] if len(sys.argv) > 1 else "integration78_actual_v1"
fixture = R / name
fixture.mkdir(exist_ok=False)
logpath = R / "logs" / (name + ".log")
resultpath = R / "results" / (name + ".json")
manifestpath = R / "results" / (name + "_commands.json")
binary = R / (name + "_bin")
commands = []
started = time.time()
env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
for key in list(env):
    if key.startswith("QG_"):
        del env[key]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(command):
    commands.append(command)
    manifestpath.write_text(
        json.dumps(
            {
                "cwd": str(R),
                "commands": commands,
                "fixture": str(fixture),
                "started_unix": started,
            },
            indent=2,
        )
        + "\n"
    )
    with logpath.open("a") as log:
        log.write("COMMAND " + json.dumps(command) + "\n")
        log.flush()
        result = subprocess.run(
            command,
            check=False,
            cwd=R,
            env=env,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
        )
        log.write(result.stdout)
        log.flush()
    if result.returncode:
        raise RuntimeError(f"command failed rc={result.returncode}; see {logpath}")
    return result.stdout


rng = np.random.default_rng(7802)
N, D = 40, 768
raw = rng.normal(size=(N, D)).astype("<f4")
raw /= np.linalg.norm(raw, axis=1)[:, None]
keys = (1009 + 17 * rng.permutation(N)).astype("<u8")
links = []
graph = []
for i in range(N):
    degree = {1: 7, 2: 0, 3: 1, 4: 31}.get(i, 32)
    ids = rng.permutation(np.array([j for j in range(N) if j != i], dtype="<u4"))[
        :degree
    ]
    if degree > 1:
        assert not np.array_equal(ids, np.sort(ids))
    links.append(ids)
    graph.append(
        struct.pack("<QI", int(keys[i]), degree)
        + ids.tobytes()
        + b"\xff" * ((32 - degree) * 4)
    )
(fixture / "graph.bin").write_bytes(b"".join(graph))
(fixture / "raw.f32").write_bytes(
    struct.pack("<4I", 0x51475664, 1, N, D) + raw.tobytes()
)
compile_cmd = [
    "c++",
    "-O3",
    "-std=c++17",
    "-march=native",
    "-ffp-contract=off",
    "-pthread",
    "-DRABITQ_SUPPORTED=1",
    "-DQG_ROTATION_768=1",
    "-DQG_INDEPENDENT_CODE_IMPORT=1",
    "-I",
    str(SRC / "src"),
    "-I",
    str(SRC / "src/include"),
    "-I",
    str(SRC / "src/core"),
    "-isystem",
    "/home/xingrui/workspace/zvec/symphony_qg/zvec/thirdparty/RaBitQ-Library/RaBitQ-Library-0.1/include",
    str(R / "scripts/import78_integration.cc"),
    str(R / "lib/libcore_framework.a"),
    str(R / "lib/libzvec_turbo.a"),
    str(R / "lib/librabitqlib_runtime.a"),
    str(R / "lib/libzvec_ailego.a"),
    "-o",
    str(binary),
]
run(compile_cmd)
prepared = json.loads(run([str(binary), "prepare", str(fixture)]))
meta = copy.deepcopy(
    json.loads((R / "results/independent77_preflight.json").read_text())["metadata"]
)
meta["database_count"] = meta["record_count"] = N
meta["input_sha256"]["models64/full_vectors.f32"] = sha(fixture / "raw.f32")
meta["rotated_sha256"] = sha(fixture / "rot.f32")
# Frozen-schema fields are reused only to exercise the import contract. These
# synthetic codes were NOT produced by the phase77 independent optimizer.
meta["synthetic_fixture"] = (
    "integration78;native signs with minimum absolute residual bit flipped once per full-block lane;not optimizer output"
)
payload = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode()
assert len(payload) <= 4064
header = (
    struct.pack("<8sIIQQ", b"QGIC77\0\0", 1, len(payload), N, 3600)
    + payload
    + b"\0" * (4064 - len(payload))
)
(fixture / "header.bin").write_bytes(header)
codes = (fixture / "imported.codes").read_bytes()
records = []
for i, ids in enumerate(links):
    degree = len(ids)
    records.append(
        struct.pack("<QII", int(keys[i]), degree, int(degree == 32))
        + keys[ids].tobytes()
        + b"\xff" * ((32 - degree) * 8)
        + codes[i * 3328 : (i + 1) * 3328]
    )
(fixture / "model.bin").write_bytes(header + b"".join(records))
result = json.loads(
    run([str(binary), "test", str(fixture), sha(fixture / "model.bin")])
)
result["prepare"] = prepared
result["fixture"] = str(fixture)
result["commands"] = str(manifestpath)
result["log"] = str(logpath)
result["source_files_sha256"] = {
    str(p): sha(p)
    for p in [
        Path(__file__),
        R / "scripts/import78_integration.cc",
        SRC / "src/core/algorithm/quantized_graph/quantized_graph.h",
        SRC / "src/core/algorithm/quantized_graph/independent_code_import.h",
        SRC / "src/turbo/quantizer/symphony_qg_quantizer/symphony_qg_quantizer.h",
        R / "scripts/coupled73_codec.h",
    ]
}
result["binary_sha256"] = sha(binary)
result["fixture_sha256"] = {
    p.name: sha(p) for p in sorted(fixture.iterdir()) if p.is_file()
}
result["duration_seconds_build_and_test"] = time.time() - started
resultpath.write_text(json.dumps(result, indent=2) + "\n")
logging.info("%s", json.dumps(result, indent=2))
