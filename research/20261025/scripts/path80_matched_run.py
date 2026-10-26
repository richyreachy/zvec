"""Compile/run matched scorer against already frozen mechanism-A tables."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

R = Path(__file__).resolve().parent.parent
table = R / (sys.argv[1] if len(sys.argv) > 1 else "path80_matched_v1")
label = sys.argv[2] if len(sys.argv) > 2 else "score_v1"
if (
    table.parent != R
    or not table.name.startswith("path80_matched_")
    or Path(label).name != label
):
    raise RuntimeError("Use direct frozen-table and new score-directory names")
out = table / label
out.mkdir(exist_ok=False)
(out / "source").mkdir()
(out / "logs").mkdir()
started = time.time()
commands = []


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(4 << 20), b""):
            h.update(data)
    return h.hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def need(yes, message):
    if not yes:
        raise RuntimeError(message)


try:
    manifest = json.loads((table / "sampling_manifest.json").read_text())
    manifest_sha = sha(table / "sampling_manifest.json")
    need(
        manifest_sha
        == "ee0adfe3092cc3e40702a20992f68798ce597ec96eba0a7389572d482a41e94c",
        "Expected frozen mechanism-A sampling manifest",
    )
    protected = {str(table / "sampling_manifest.json"): manifest_sha}
    for name, expected in manifest["table_sha256"].items():
        need(sha(table / name) == expected, "Frozen proxy/state table changed")
        protected[str(table / name)] = expected
    accepted = R / "path80_native_v1"
    old_proof = json.loads((accepted / "proof.json").read_text())
    need(
        old_proof["passed"] and old_proof["all_queries_native_gate"],
        "Full native gate required",
    )
    protected[str(accepted / "proof.json")] = sha(accepted / "proof.json")
    old_inputs = json.loads((accepted / "inputs.json").read_text())
    for name in [
        "path80_matched_score.cc",
        "path80_matched_run.py",
        "path80_score.cc",
        "independent78_import.h",
        "coupled73_codec.h",
    ]:
        source = R / "scripts" / name
        protected[str(source)] = sha(source)
        shutil.copyfile(source, out / "source" / name)
    need(
        protected[str(R / "scripts/path80_score.cc")]
        == old_inputs["source_sha256"][str(R / "scripts/path80_score.cc")],
        "Accepted scorer source changed",
    )
    frozen = R / "variants/trace68_native/build_source"
    for name, expected in old_inputs["frozen_source_sha256"].items():
        need(sha(frozen / name) == expected, "Frozen compiler inputs changed")
    for path, expected in old_inputs["libraries_sha256"].items():
        need(sha(path) == expected, "Accepted production library changed")
        protected[path] = expected
    for relative in [
        "models77/proxies.u32",
        "models77/full/compact.u8",
        "models77/full/audit.bin",
        "models68/primary/events.bin",
    ]:
        protected[str(R / relative)] = sha(R / relative)
    need(
        protected[str(R / "models77/proxies.u32")]
        == manifest["input_sha256"]["models77/proxies.u32"],
        "Training proxy table changed",
    )
    need(
        protected[str(R / "models68/primary/events.bin")]
        == old_proof["source_events_sha256"],
        "Original trace changed",
    )
    overlay = accepted / "native_path_learned_scores.bin"
    protected[str(overlay)] = sha(overlay)
    need(
        protected[str(overlay)] == old_proof["rescored_events_sha256"],
        "Accepted overlay changed",
    )
    compile_command = json.loads((accepted / "logs/compile_command.json").read_text())[
        "argv"
    ]
    compile_command = [
        str(out / "source/path80_matched_score.cc")
        if x == str(accepted / "source/path80_score.cc")
        else str(out / "score.o")
        if x == str(accepted / "path80_score.o")
        else str(out / "source")
        if x == str(accepted / "source")
        else x
        for x in compile_command
    ]
    link_command = json.loads((accepted / "logs/link_command.json").read_text())["argv"]
    link_command = [
        str(out / "score.o")
        if x == str(accepted / "path80_score.o")
        else str(out / "score")
        if x == str(accepted / "path80_score")
        else str(x).replace(str(accepted / "path80.link.d"), str(out / "score.link.d"))
        for x in link_command
    ]
    planned = [
        compile_command,
        link_command,
        [str(out / "score"), str(R), str(table), str(out)],
    ]
    save(
        out / "inputs.json",
        {
            "frozen_sampling_manifest_sha256": manifest_sha,
            "protected_input_sha256": protected,
            "commands": planned,
            "model_sha256_verified_by_import_helper": old_proof["model_sha256"],
            "raw_and_rotation_sha256_verified_by_import_helper": True,
            "native_full_proof": str(accepted / "proof.json"),
            "no_training_or_retrieval": True,
        },
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("QG_", "BENCH_", "GTEST_"))
        and k not in {"LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH", "PYTHONOPTIMIZE"}
    }
    env.update(
        LD_LIBRARY_PATH=f"{R}/variants/trace68_native:{R}/lib",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    for phase, argv in zip(["compile", "link", "score"], planned, strict=True):
        begin = time.time()
        save(out / "logs" / (phase + "_command.json"), {"cwd": str(out), "argv": argv})
        with (out / "logs" / (phase + ".log")).open("x") as stream:
            run = subprocess.run(
                argv,
                cwd=out,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        record = {
            "phase": phase,
            "argv": argv,
            "returncode": run.returncode,
            "elapsed_seconds": time.time() - begin,
        }
        commands.append(record)
        save(out / "logs" / (phase + "_result.json"), record)
        need(run.returncode == 0, f"{phase} failed; see {out}/logs/{phase}.log")
    controls = json.loads((out / "score_controls.json").read_text())
    need(controls["passed"], "Controls failed")
    for path, expected in protected.items():
        need(sha(path) == expected, f"Input changed during score: {path}")
    controls.update(
        frozen_sampling_manifest_sha256=manifest_sha,
        source_events_sha256=old_proof["source_events_sha256"],
        model_sha256=old_proof["model_sha256"],
        command_results=commands,
        source_and_inputs_unchanged=True,
        binary_sha256=sha(out / "score"),
        artifact_sha256={
            name: sha(out / name)
            for name in [
                "observations.bin",
                "training_controls.f64",
                "score_controls.json",
            ]
        },
        elapsed_seconds=time.time() - started,
    )
    save(out / "proof.json", controls)
    logging.info("MATCHED80 SCORE COMPLETE %s", out / "proof.json")
except BaseException as error:
    save(
        out / "failure.json",
        {"error": repr(error), "commands": commands, "passed": False},
    )
    raise
