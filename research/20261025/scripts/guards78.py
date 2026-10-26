from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

R = Path(__file__).resolve().parent.parent
S = Path("/home/xingrui/workspace/quantized_graph/symphony_research_20261025")
parser = argparse.ArgumentParser(
    description="Phase78 preprocessing guards and real test-TU template syntax checks; no linking or query timing."
)
parser.add_argument("--label", default="independent78_compile_guards")
parser.add_argument(
    "--preprocess-only",
    action="store_true",
    help="Omit the complete28-test TU syntax checks; result explicitly records this narrower scope.",
)
args = parser.parse_args()
result = R / "results" / f"{args.label}.json"
assert not result.exists(), f"Existing result: {result}; use a new --label"
source_paths = [
    S / "src/core/algorithm/quantized_graph/quantized_graph.h",
    S / "src/core/algorithm/quantized_graph/independent_code_import.h",
    S / "src/turbo/quantizer/symphony_qg_quantizer/symphony_qg_quantizer.h",
    S / "tests/core/algorithm/hnsw/symphony_qg_test.cc",
]
hashes = {
    str(p.relative_to(S)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in source_paths
}
base = [
    "c++",
    "-std=c++17",
    "-E",
    "-x",
    "c++",
    "-I" + str(S / "src"),
    "-I" + str(S / "src/include"),
    "-I" + str(S / "src/core"),
    "-",
]
source = '#include "core/algorithm/quantized_graph/quantized_graph.h"\n'
flags = ["QG_INDEPENDENT_CODE_IMPORT=1", "QG_ROTATION_768=1"]
checks = []
started = time.monotonic()
# Extra flags support the unrelated Fusion prerequisite guard, so its test
# also reaches the intended Independent-code incompatibility diagnostic.
cases = [
    ("supported", [], [], True),
    ("macro_off_default", [], flags, True),
    ("missing_rotation", [], ["QG_ROTATION_768=1"], False),
]
for macro in [
    "QG_PROJECT_DIM",
    "QG_RESIDUAL_TWO_STAGE",
    "QG_LOWRANK",
    "QG_DIRECTION_BETA",
    "QG_DIRECTION_MSE",
    "QG_RADIAL_CALIBRATION",
    "QG_GEOM_BIAS",
    "QG_CENTER_QUANTIZED",
    "QG_CENTER_FP16",
    "QG_CENTER_SQ8",
    "QG_REANCHOR",
    "QG_HIERARCHICAL",
    "QG_GRAPH_IMPORT",
    "QG_INTEGER_ROUTER",
    "QG_PROJECTION_ROUTER",
]:
    cases.append((macro, [macro + "=1"], [], False))
cases.append(("QG_FUSION", ["QG_FUSION=1", "QG_BUCKET=1", "QG_ADAPTIVE=1"], [], False))
for name, extra, remove, expected in cases:
    command = (
        base
        + ["-D" + flag for flag in flags if flag not in remove]
        + ["-D" + flag for flag in extra]
    )
    run = subprocess.run(
        command,
        check=False,
        input=source,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        cwd=R,
    )
    record = {
        "name": name,
        "kind": "preprocessing",
        "command": command,
        "expected_success": expected,
        "exit_code": run.returncode,
        "stderr": run.stderr,
    }
    checks.append(record)
    assert (run.returncode == 0) == expected, record
    if not expected:
        assert "Independent code import" in run.stderr, record

if not args.preprocess_only:
    commands = json.loads((R / "results/build_commands.json").read_text())
    old_object = str(R / "objects/symphony_qg_test.o")
    original = next(
        x["argv"] for x in commands if "-c" in x["argv"] and old_object in x["argv"]
    )
    # Syntax-check the actual test TU: it instantiates ScalarGraphCodec,
    # SQ8GraphCodec, PQGraphCodec, PQResidualGraphCodec and Graphs without key().
    syntax = []
    skip = False
    for item in original:
        if skip:
            skip = False
            continue
        if item == "-o":
            skip = True
            continue
        if item in ["-c", "-DNDEBUG"] or item.startswith("-DRABITQ_SUPPORTED="):
            continue
        syntax.append(item)
    syntax += ["-UNDEBUG", "-fsyntax-only", "-fdiagnostics-color=never"]
    for name, rabitq, extra in [
        ("templates_macro_off", 1, []),
        ("templates_import_on", 1, flags),
        ("templates_import_on_without_rabitq", 0, flags),
    ]:
        command = (
            syntax
            + ["-DRABITQ_SUPPORTED=" + str(rabitq)]
            + ["-D" + flag for flag in extra]
        )
        env = {
            k: v for k, v in os.environ.items() if not k.startswith("QG_INDEPENDENT_")
        }
        run = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            cwd=R,
            env=env,
        )
        record = {
            "name": name,
            "kind": "full_test_translation_unit_syntax",
            "command": command,
            "model_environment_removed": True,
            "expected_success": True,
            "exit_code": run.returncode,
            "stderr": run.stderr,
        }
        checks.append(record)
        assert run.returncode == 0, record

for path in source_paths:
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == hashes[str(path.relative_to(S))]
    ), "Source changed during checks; rerun after source freezes"
summary = {
    "source_hashes_verified_before_and_after": hashes,
    "preprocessing_cases": len(cases),
    "template_syntax_cases": 0 if args.preprocess_only else 3,
    "template_scope": "Existing28-test TU includes direct Scalar/SQ8/PQ/PQResidual graph templates; syntax checking is not execution of those tests",
    "checks": checks,
    "elapsed_seconds": time.monotonic() - started,
    "no_link_or_runtime_test_or_query_timing": True,
}
with result.open("x") as output:
    output.write(json.dumps(summary, indent=2) + "\n")
logging.info("GUARDS78 PASS %s", len(checks))
