from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
M = R / "models77"
out = M / "full"
started = time.monotonic()


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        while block := f.read(16 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


assert "PILOT77 COMPLETE" in (R / "logs/runpilot77.log").read_text()
assert "VALIDATE77 COMPLETE" in (R / "logs/validate77_pilot.log").read_text()
pilot = json.loads((R / "results/independent77_pilot.json").read_text())
oracle = json.loads((R / "results/independent77_pilot_validation.json").read_text())
codec = json.loads((R / "logs/codec_check77_pilot.log").read_text())
meta = json.loads((R / "results/independent77_preflight.json").read_text())
assert pilot["all_five_outputs_thread_byte_identical"]
assert oracle["python_signs_and_all_moves_bit_identical"]
assert codec["centers"] == 8192
assert codec["learned_code_blocks_byte_identical"] == 8182
assert codec["native_short_code_blocks_byte_identical"] == 10
expected = dict(meta["metadata"]["input_sha256"])
expected.update(
    {
        "fullbuild77": meta["metadata"]["builder_binary_sha256"],
        "scripts/fullbuild77.cc": meta["metadata"]["builder_source_sha256"],
        "scripts/independent76.h": meta["metadata"]["algorithm_header_sha256"],
        "models77/rotated.f32": meta["metadata"]["rotated_sha256"],
        "models77/proxies.u32": meta["metadata"]["proxy_sha256"],
        "models77/header1000000.bin": meta["headers"]["1000000"],
    }
)
for name, h in expected.items():
    assert sha(R / name) == h, name
commands = []


def run(name, cmd):
    commands.append({"name": name, "argv": cmd})
    (R / "results/independent77_full_commands.json").write_text(
        json.dumps(
            {
                "cwd": str(R),
                "started_unix": time.time() if len(commands) == 1 else start_unix,
                "commands": commands,
            },
            indent=2,
        )
        + "\n"
    )
    logging.info("%s %s", "FULL77", name)
    with (R / "logs" / f"{name}.log").open("x") as f:
        subprocess.run(cmd, cwd=R, stdout=f, stderr=subprocess.STDOUT, check=True)


start_unix = time.time()
run(
    "fullbuild77_full",
    [
        "taskset",
        "-c",
        "0,2,4,6,8,10,12,14",
        "./fullbuild77",
        "models64/full_graph.bin",
        "models77/rotated.f32",
        "models77/proxies.u32",
        "models77/header1000000.bin",
        "models77/full",
        "8",
    ],
)
build = json.loads((R / "logs/fullbuild77_full.log").read_text().splitlines()[-1])
assert build["centers"] == 1000000
assert build["learned_blocks"] == 997722
assert build["native_blocks"] == 2278
run(
    "validate77_full",
    [sys.executable, "scripts/validate77.py", "models77/full", "full"],
)
run(
    "codec_check77_full",
    [
        "./codec_check77",
        "models64/full_graph.bin",
        "models77/rotated.f32",
        "models77/proxies.u32",
        "models77/full",
        "blocks.partial",
    ],
)
validation = json.loads((R / "results/independent77_full_validation.json").read_text())
actual = json.loads((R / "logs/codec_check77_full.log").read_text())
assert validation["python_oracle_sample_centers"] == 2048
assert validation["python_signs_and_all_moves_bit_identical"]
assert validation["centers_audited"] == 1000000
assert actual["centers"] == 1000000
assert actual["learned_code_blocks_byte_identical"] == 997722
assert actual["native_short_code_blocks_byte_identical"] == 2278
for name, h in expected.items():
    assert sha(R / name) == h, name
assert not (out / "blocks.bin").exists()
(out / "blocks.partial").rename(out / "blocks.bin")
files = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}
a = {
    "full_model_offline_validation_passed": True,
    "build": build,
    "audit_and_python_oracle": validation,
    "actual_codec_validation": actual,
    "input_hashes_before_and_after_verified": expected,
    "output_sha256": files,
    "orchestrator_sha256": sha(Path(__file__)),
    "validation_script_sha256": sha(R / "scripts/validate77.py"),
    "codec_checker_sha256": sha(R / "codec_check77"),
    "total_pipeline_elapsed_seconds": time.monotonic() - started,
    "production_loader_implemented": False,
    "no_retrieval_qps_or_recall_claim": True,
}
(R / "results/independent77_full_acceptance.json").write_text(
    json.dumps(a, indent=2) + "\n"
)
logging.info(json.dumps(a, indent=2))
logging.info("FULL77 COMPLETE")
