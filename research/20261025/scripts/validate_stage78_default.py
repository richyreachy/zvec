from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

R = Path(__file__).resolve().parent.parent
S = Path("/home/xingrui/workspace/quantized_graph/symphony_research_20261025")
parser = argparse.ArgumentParser(
    description="Compile/link/run existing28 regressions against an ALREADY BUILT frozen phase78 variant. Never builds the variant or runs retrieval timing."
)
parser.add_argument("--variant", default="stage78_default")
args = parser.parse_args()
variant = R / "variants" / args.variant
assert (variant / "commands.json").exists(), (
    f"First build the variant separately: python3 scripts/compile_variant.py {args.variant}"
)
metadata = json.loads((variant / "metadata.json").read_text())
defines = metadata["defines"]
allowed = {"QG_INDEPENDENT_CODE_IMPORT=1", "QG_ROTATION_768=1"}
assert not defines or set(defines) == allowed, (
    "Default28 regressions allow either no research flags or only import+rotation; use a matching library, not mixed inline macros"
)
frozen = Path(metadata["immutable_build_source"])
assert frozen.is_dir()
folder = variant / "validation78"
folder.mkdir(exist_ok=False)
# Preserve test sources as well as the per-variant production source snapshot.
shutil.copytree(S / "tests/core/algorithm/hnsw", folder / "test_source")
test_source = folder / "test_source/symphony_qg_test.cc"
source_digest = hashlib.sha256(test_source.read_bytes()).hexdigest()
source_hashes = json.loads((variant / "build_source_sha256.json").read_text())
for relative, digest in source_hashes.items():
    assert hashlib.sha256((frozen / relative).read_bytes()).hexdigest() == digest, (
        relative
    )
all_commands = json.loads((R / "results/build_commands.json").read_text())
old_object = str(R / "objects/symphony_qg_test.o")
new_object = str(folder / "symphony_qg_test.o")
binary = folder / "symphony_qg_test"
compile_command = next(
    x["argv"] for x in all_commands if "-c" in x["argv"] and old_object in x["argv"]
)
compile_command = [
    new_object
    if x == old_object
    else str(test_source)
    if x == str(S / "tests/core/algorithm/hnsw/symphony_qg_test.cc")
    else x.replace(
        str(S / "tests/core/algorithm/hnsw"), str(folder / "test_source")
    ).replace(str(S / "src"), str(frozen / "src"))
    for x in compile_command
]
compile_command += ["-UNDEBUG", *["-D" + define for define in defines]]
link_command = next(
    x["argv"] for x in all_commands if "-c" not in x["argv"] and old_object in x["argv"]
)
link_command = [
    new_object
    if x == old_object
    else str(binary)
    if x == str(R / "symphony_qg_test")
    else str(variant / "libcore_knn_hnsw.so")
    if x == str(R / "lib/libcore_knn_hnsw.so")
    else "-Wl,--dependency-file=" + str(folder / "test_link.d")
    if x.startswith("-Wl,--dependency-file=")
    else x
    for x in link_command
]
executed = []


def run(label, command, env=None):
    executed.append({"name": label, "argv": command})
    with (folder / (label + ".log")).open("x") as log:
        subprocess.run(
            command, cwd=R, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
        )


run("compile", compile_command)
run("link", link_command)
env = {
    key: value
    for key, value in os.environ.items()
    if not key.startswith(("QG_", "GTEST_"))
}
env["LD_LIBRARY_PATH"] = str(variant) + ":" + str(R / "lib")
xml_path = folder / "tests.xml"
run("tests", [str(binary), "--gtest_output=xml:" + str(xml_path)], env)
root = ET.parse(xml_path).getroot()
counts = {
    name: int(root.attrib.get(name, "0"))
    for name in ["tests", "failures", "errors", "disabled"]
}
assert counts == {"tests": 28, "failures": 0, "errors": 0, "disabled": 0}, counts
skipped = len(root.findall(".//skipped"))
assert skipped == 0, skipped
assert hashlib.sha256(test_source.read_bytes()).hexdigest() == source_digest
summary = {
    "variant": args.variant,
    "frozen_source": str(frozen),
    "defines_used_for_both_library_and_tests": defines,
    "test_source_sha256": source_digest,
    "source_manifest_verified": True,
    "test_counts": counts,
    "skipped": skipped,
    "commands": executed,
    "runtime_environment": {
        "all_QG_variables_removed": True,
        "gtest_filter_sharding_removed": True,
        "LD_LIBRARY_PATH": env["LD_LIBRARY_PATH"],
    },
    "xml_path": str(xml_path),
    "variant_build_performed_by_this_script": False,
    "no_query_timing": True,
}
with (R / "results" / f"{args.variant}_validation78.json").open("x") as output:
    output.write(json.dumps(summary, indent=2) + "\n")
logging.info("STAGE78 DEFAULT VALIDATION COMPLETE %s", args.variant)
