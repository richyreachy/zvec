from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
assert "PREFLIGHT77 COMPLETE" in (R / "logs/preflight77.log").read_text()


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        while block := f.read(16 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


meta = json.loads((R / "results/independent77_preflight.json").read_text())
assert sha(R / "fullbuild77") == meta["metadata"]["builder_binary_sha256"]
commands = []
summaries = []
for threads in [1, 8]:
    cmd = [
        "taskset",
        "-c",
        "0" if threads == 1 else "0,2,4,6,8,10,12,14",
        "./fullbuild77",
        "models64/full_graph.bin",
        "models77/rotated.f32",
        "models77/proxies.u32",
        "models77/header8192.bin",
        f"models77/pilot_t{threads}",
        str(threads),
    ]
    commands.append(cmd)
    logging.info("%s %s", "PILOT77", threads)
    with (R / "logs" / f"pilot77_t{threads}.log").open("x") as log:
        subprocess.run(cmd, cwd=R, stdout=log, stderr=subprocess.STDOUT, check=True)
    lines = (R / "logs" / f"pilot77_t{threads}.log").read_text().splitlines()
    summaries.append(json.loads(lines[-1]))
files = {}
for name in ["blocks.partial", "compact.u8", "audit.bin", "history.f64", "moves.u16"]:
    h = sha(R / "models77/pilot_t1" / name)
    assert h == sha(R / "models77/pilot_t8" / name)
    files[name] = h
(R / "results/independent77_pilot.json").write_text(
    json.dumps(
        {
            "commands": commands,
            "runs": summaries,
            "all_five_outputs_thread_byte_identical": files,
            "python_oracle_and_codec_verification_still_required": True,
        },
        indent=2,
    )
    + "\n"
)
logging.info(json.dumps(summaries, indent=2))
logging.info("PILOT77 COMPLETE")
