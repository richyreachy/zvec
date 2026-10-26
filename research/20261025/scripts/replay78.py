"""Isolate and replay archived phase78 tests, never a benchmark or variant build.

Only --execute runs tests. All generated files are under a new --output directory.
Archived scripts and frozen production variants are read-only inputs.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

PHASE_SCRIPTS = [
    "independent78_test.py",
    "independent78_test.cc",
    "independent78_alloc_test.cc",
    "independent78_import.h",
    "guards78.py",
    "validate_stage78_default.py",
    "import78_integration.cc",
    "import78_integration.py",
    "run78_disabled.py",
    "screen78.py",
]
LIBRARIES = [
    "libcore_framework.a",
    "libzvec_turbo.a",
    "librabitqlib_runtime.a",
    "libzvec_ailego.a",
    "libcore_metric.so",
    "libcore_quantizer.so",
    "libcore_knn_hnsw.so",
    "libcore_knn_flat.so",
    "libcore_utility.so",
    "libgtest.a",
    "libgtest_main.a",
]
VARIANTS = ["stage78_default", "stage78_import_default"]
ORIGINAL_SRC = Path(
    "/home/xingrui/workspace/quantized_graph/symphony_research_20261025"
)


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def exclusive_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def tree_hashes(path):
    return {
        str(p.relative_to(path)): digest(p)
        for p in sorted(path.rglob("*"))
        if p.is_file()
    }


def read_json(path):
    return json.loads(Path(path).read_text())


def main():  # noqa: PLR0915 -- Preserve the audited sequential replay and validation flow.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archived-scripts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--bench-root",
        type=Path,
        default=Path("/home/xingrui/bench/quantized_graph/research_20261025"),
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("/home/xingrui/bench/symphony_qg/.venv/bin/python"),
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    need(sys.flags.optimize == 0, "Run without Python -O")
    root = args.bench_root.resolve(strict=True)
    archive = args.archived_scripts.resolve(strict=True)
    output = args.output.absolute()
    need(
        output.parent.resolve() == root and output.name.startswith("replay78_"),
        "--output must be a new direct child of bench-root named replay78_...",
    )
    need(
        not output.exists() and not output.is_symlink(),
        "Output already exists; choose a new directory",
    )
    need(archive.is_dir(), "Archived scripts directory required")
    for name in [*PHASE_SCRIPTS, "coupled73_codec.h"]:
        need((archive / name).is_file(), f"Missing archived input {name}")
    output.mkdir(exist_ok=False)
    for name in [
        "scripts",
        "original_scripts",
        "diffs",
        "logs",
        "results",
        "lib",
        "variants",
        "frozen_inputs",
        "original_evidence",
        "objects",
    ]:
        (output / name).mkdir(exist_ok=False)
    protected = {}
    source_records = []
    replacements = []

    def protect(path):
        path = Path(path).absolute()
        value = digest(path)
        protected[str(path)] = value
        return value

    def copy_input(source, target):
        before = protect(source)
        need(
            not target.exists() and not target.is_symlink(),
            f"Refuse to overwrite {target}",
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        need(digest(target) == before, f"Copy differs: {source}")
        return before

    # Preserve original bench and archive bytes and retain their full diff.
    for name in [*PHASE_SCRIPTS, "coupled73_codec.h"]:
        original = root / "scripts" / name
        archived = archive / name
        target = output / "scripts" / name
        before = copy_input(original, output / "original_scripts" / name)
        archived_hash = copy_input(archived, target)
        old_text, new_text = original.read_text(), archived.read_text()
        difference = "".join(
            difflib.unified_diff(
                old_text.splitlines(True),
                new_text.splitlines(True),
                fromfile=str(original),
                tofile=str(archived),
            )
        )
        (output / "diffs" / (name + ".original_to_archive.diff")).write_text(difference)
        record = {
            "name": name,
            "original_path": str(original),
            "original_sha256": before,
            "archived_path": str(archived),
            "archived_sha256": archived_hash,
        }
        if name.endswith(".py"):
            record["original_and_archive_ast_identical"] = ast.dump(
                ast.parse(old_text), include_attributes=False
            ) == ast.dump(ast.parse(new_text), include_attributes=False)
        source_records.append(record)

    # Frozen variants supply production sources and previously frozen test sources.
    frozen = None
    for name in VARIANTS:
        original = root / "variants" / name
        local = output / "variants" / name
        local.mkdir(exist_ok=False)
        for filename in ["metadata.json", "commands.json", "build_source_sha256.json"]:
            copy_input(original / filename, local / filename)
        metadata = read_json(local / "metadata.json")
        expected = (
            []
            if name == "stage78_default"
            else ["QG_INDEPENDENT_CODE_IMPORT=1", "QG_ROTATION_768=1"]
        )
        need(metadata["defines"] == expected, f"Unexpected frozen macros for {name}")
        source = Path(metadata["immutable_build_source"]).resolve(strict=True)
        for relative, expected_hash in read_json(
            local / "build_source_sha256.json"
        ).items():
            need(
                protect(source / relative) == expected_hash,
                f"Frozen source mismatch: {name}/{relative}",
            )
        protect(original / "libcore_knn_hnsw.so")
        (local / "libcore_knn_hnsw.so").symlink_to(original / "libcore_knn_hnsw.so")
        if name == "stage78_import_default":
            frozen = source
    need(frozen is not None, "Missing frozen production tree")
    (output / "frozen_inputs/src").symlink_to(frozen / "src", target_is_directory=True)
    test_source = root / "variants/stage78_import_default/validation78/test_source"
    original_default_tests = root / "variants/stage78_default/validation78/test_source"
    need(
        tree_hashes(test_source) == tree_hashes(original_default_tests),
        "Frozen test source copies differ",
    )
    for p in test_source.rglob("*"):
        if p.is_file():
            protect(p)
    test_parent = output / "frozen_inputs/tests/core/algorithm"
    test_parent.mkdir(parents=True)
    (test_parent / "hnsw").symlink_to(test_source, target_is_directory=True)

    # Only these three AST constant values are changed in sandbox script copies.
    for name, variable in [
        ("guards78.py", "S"),
        ("validate_stage78_default.py", "S"),
        ("import78_integration.py", "SRC"),
    ]:
        target = output / "scripts" / name
        before = target.read_text()
        tree = ast.parse(before)
        assignments = [
            n
            for n in tree.body
            if isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == variable
        ]
        need(len(assignments) == 1, f"Expected one {variable} assignment in {name}")
        node = assignments[0]
        need(
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "Path"
            and len(node.value.args) == 1
            and isinstance(node.value.args[0], ast.Constant)
            and node.value.args[0].value == str(ORIGINAL_SRC),
            f"Unexpected root expression in {name}",
        )
        replacement = str(output / "frozen_inputs")
        node.value.args[0].value = replacement
        lines = before.splitlines(True)
        after = "".join(
            [
                *lines[: node.lineno - 1],
                f"{variable} = Path({json.dumps(replacement)})\n",
                *lines[node.end_lineno :],
            ]
        )
        need(
            ast.dump(tree, include_attributes=False)
            == ast.dump(ast.parse(after), include_attributes=False),
            f"Unexpected AST mutation while relocating {name}",
        )
        target.write_text(after)
        diff_name = name + ".sandbox_path.diff"
        (output / "diffs" / diff_name).write_text(
            "".join(
                difflib.unified_diff(
                    before.splitlines(True),
                    after.splitlines(True),
                    fromfile=str(archive / name),
                    tofile=str(target),
                )
            )
        )
        replacements.append(
            {
                "file": name,
                "variable": variable,
                "from": str(ORIGINAL_SRC),
                "to": replacement,
                "archive_sha256": hashlib.sha256(before.encode()).hexdigest(),
                "sandbox_sha256": digest(target),
                "ast_only_path_change_verified": True,
                "diff": str(output / "diffs" / diff_name),
            }
        )

    # Keep writable directories local and link only read-only individual libraries.
    for name in LIBRARIES:
        protect(root / "lib" / name)
        (output / "lib" / name).symlink_to(root / "lib" / name)
    copy_input(
        root / "results/independent77_preflight.json",
        output / "results/independent77_preflight.json",
    )
    raw_commands = root / "results/build_commands.json"
    copy_input(raw_commands, output / "original_evidence/build_commands.json")
    commands = read_json(raw_commands)
    mappings = [
        (str(root), str(output)),
        (str(ORIGINAL_SRC), str(output / "frozen_inputs")),
    ]
    changes = []
    for record in commands:
        if "cwd" in record:
            record["cwd"] = str(output)
        for index, argument in enumerate(record["argv"]):
            rewritten = argument
            for old, new in mappings:
                rewritten = rewritten.replace(old, new)
            if rewritten != argument:
                changes.append({"before": argument, "after": rewritten})
            record["argv"][index] = rewritten
    exclusive_json(output / "results/build_commands.json", commands)
    exclusive_json(output / "diffs/build_commands_path_changes.json", changes)

    evidence = [
        "results/independent78_import_checks_loader78_unit_final.json",
        "results/independent78_compile_guards.json",
        "results/integration78_actual_v2.json",
        "results/stage78_default_validation78.json",
        "results/stage78_import_default_validation78.json",
        "variants/stage78_default/validation78/tests.xml",
        "variants/stage78_import_default/validation78/tests.xml",
    ]
    for relative in evidence:
        copy_input(root / relative, output / "original_evidence" / relative)
    for dirname in ["loader78_unit_final", "integration78_actual_v2"]:
        need((root / dirname).is_dir(), f"Original fixture missing: {dirname}")
        for p in (root / dirname).rglob("*"):
            if p.is_file():
                protect(p)
    manifest = {
        "scope": "Archived phase78 small tests only; no benchmark, full1M retrieval, or variant build",
        "bench_root": str(root),
        "archived_scripts": str(archive),
        "sandbox": str(output),
        "python": str(args.python),
        "source_records": source_records,
        "archive_contains_lint_changes_not_only_layout": True,
        "path_replacements": replacements,
        "command_path_mapping": mappings,
        "protected_input_sha256": protected,
        "sandbox_script_sha256": tree_hashes(output / "scripts"),
        "runner_sha256": digest(Path(__file__)),
        "runner_path": str(Path(__file__).resolve()),
    }
    exclusive_json(output / "input_manifest.json", manifest)
    if not args.execute:
        logging.info("PREPARED78 %s", output)
        return

    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("QG_", "GTEST_"))
        and k not in ["PYTHONOPTIMIZE", "LD_LIBRARY_PATH", "PYTHONPATH"]
    }
    env.update(
        OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1"
    )
    executed = []

    def run(name, command):
        logpath = output / "logs" / (name + ".log")
        need(not logpath.exists(), f"Log already exists: {logpath}")
        logging.info("REPLAY78 START %s", name)
        started = time.time()
        with logpath.open("x") as log:
            result = subprocess.run(
                command,
                cwd=output,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        record = {
            "name": name,
            "argv": list(map(str, command)),
            "cwd": str(output),
            "log": str(logpath),
            "returncode": result.returncode,
            "elapsed_seconds": time.time() - started,
        }
        executed.append(record)
        exclusive_json(output / "logs" / (name + ".command.json"), record)
        need(result.returncode == 0, f"Replay failed: {name}; see {logpath}")
        logging.info("REPLAY78 PASS %s", name)

    try:
        run(
            "versions",
            [
                str(args.python),
                "-c",
                "import sys,numpy,subprocess;print(sys.version);print('numpy',numpy.__version__);subprocess.run(['c++','--version'],check=True)",
            ],
        )
        run(
            "helper_shared_compile",
            [
                "c++",
                "-O3",
                "-std=c++17",
                "-fPIC",
                "-shared",
                "scripts/independent78_test.cc",
                "-o",
                "lib/independent78_test.so",
            ],
        )
        run(
            "helper_alloc_compile",
            [
                "c++",
                "-O3",
                "-std=c++17",
                "scripts/independent78_alloc_test.cc",
                "-o",
                "independent78_alloc_test",
            ],
        )
        run(
            "helper",
            [str(args.python), "scripts/independent78_test.py", "replay78_helper_v1"],
        )
        run(
            "integration",
            [
                str(args.python),
                "scripts/import78_integration.py",
                "replay78_integration_v1",
            ],
        )
        run(
            "guards",
            [str(args.python), "scripts/guards78.py", "--label", "replay78_guards_v1"],
        )
        for name in VARIANTS:
            run(
                name,
                [
                    str(args.python),
                    "scripts/validate_stage78_default.py",
                    "--variant",
                    name,
                ],
            )

        comparison = {}
        old = read_json(
            root / "results/independent78_import_checks_loader78_unit_final.json"
        )
        new = read_json(
            output / "results/independent78_import_checks_replay78_helper_v1.json"
        )
        ignored = {"script_sha256", "helper_sha256", "binding_sha256", "binary_sha256"}
        need(
            {k: v for k, v in old.items() if k not in ignored}
            == {k: v for k, v in new.items() if k not in ignored},
            "Helper structured results differ",
        )
        need(
            tree_hashes(root / "loader78_unit_final")
            == tree_hashes(output / "replay78_helper_v1"),
            "Helper fixture bytes differ",
        )
        comparison["helper"] = {
            "semantic_result_equal": True,
            "all_fixture_bytes_equal": True,
            "fixture_files": len(tree_hashes(output / "replay78_helper_v1")),
            "rejection_cases": len(new["malformed_and_compatibility_rejections"]),
            "excluded_provenance_fields": sorted(ignored),
        }
        old = read_json(root / "results/integration78_actual_v2.json")
        new = read_json(output / "results/replay78_integration_v1.json")
        ignored = {
            "fixture",
            "commands",
            "log",
            "source_files_sha256",
            "binary_sha256",
            "duration_seconds_build_and_test",
        }
        need(
            {k: v for k, v in old.items() if k not in ignored}
            == {k: v for k, v in new.items() if k not in ignored},
            "Real integration structured results differ",
        )
        need(
            tree_hashes(root / "integration78_actual_v2")
            == tree_hashes(output / "replay78_integration_v1"),
            "Real integration fixture bytes differ",
        )
        comparison["integration"] = {
            "semantic_result_equal": True,
            "all_fixture_bytes_equal": True,
            "actual_fastscan_calls": new["actual_fastscan_calls"],
            "scalar_lane_checks": new["bit_identical_scalar_lut_fma_lane_checks"],
            "excluded_provenance_fields": sorted(ignored),
        }
        old = read_json(root / "results/independent78_compile_guards.json")
        new = read_json(output / "results/replay78_guards_v1.json")
        fields = ["name", "kind", "expected_success", "exit_code"]
        pick = lambda value: [{k: item[k] for k in fields} for item in value["checks"]]
        need(pick(old) == pick(new), "Guard cases differ")
        need(
            old["source_hashes_verified_before_and_after"]
            == new["source_hashes_verified_before_and_after"],
            "Guard frozen source hashes differ",
        )
        comparison["guards"] = {
            "all_case_results_equal": True,
            "production_and_test_hashes_equal": True,
            "preprocessing_cases": new["preprocessing_cases"],
            "template_syntax_cases": new["template_syntax_cases"],
        }
        for name in VARIANTS:
            old = read_json(root / "results" / (name + "_validation78.json"))
            new = read_json(output / "results" / (name + "_validation78.json"))
            for key in [
                "defines_used_for_both_library_and_tests",
                "test_source_sha256",
                "test_counts",
                "skipped",
            ]:
                need(old[key] == new[key], f"Regression differs: {name}/{key}")

            def cases(path):
                document = ET.parse(path).getroot()
                return sorted(
                    (
                        x.attrib.get("classname"),
                        x.attrib.get("name"),
                        x.attrib.get("status"),
                        x.attrib.get("result"),
                    )
                    for x in document.findall(".//testcase")
                )

            need(
                cases(old["xml_path"]) == cases(new["xml_path"]),
                f"Regression XML cases differ: {name}",
            )
            comparison[name] = {
                "test_counts": new["test_counts"],
                "skipped": new["skipped"],
                "xml_case_results_equal": True,
            }
        for path, expected in protected.items():
            need(
                digest(path) == expected,
                f"Original input mutated during replay: {path}",
            )
        comparison.update(
            passed=True,
            original_inputs_unchanged=True,
            protected_input_count=len(protected),
            no_benchmark_or_full1m_or_variant_build=True,
            input_manifest=str(output / "input_manifest.json"),
            commands=executed,
        )
        exclusive_json(output / "results/replay78_comparison.json", comparison)
        logging.info(
            "REPLAY78 COMPLETE %s", output / "results/replay78_comparison.json"
        )
    except BaseException as error:
        exclusive_json(
            output / "results/replay78_failure.json",
            {"error": repr(error), "commands": executed},
        )
        raise


if __name__ == "__main__":
    main()
