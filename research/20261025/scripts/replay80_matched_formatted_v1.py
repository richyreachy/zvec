"""Replay archived mechanism A against frozen raw evidence without retrieval."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


class NormalizeArchive(ast.NodeTransformer):
    """Allow only terminal logging and equivalent conjunctive assertions."""

    def visit_Import(self, node):
        node.names = [name for name in node.names if name.name != "logging"]
        return node if node.names else None

    def visit_Expr(self, node):
        value = node.value
        if isinstance(value, ast.Call):
            function = value.func
            if isinstance(function, ast.Name) and function.id == "print":
                return ast.Expr(ast.Constant("TERMINAL_STATUS_OUTPUT"))
            if (
                isinstance(function, ast.Attribute)
                and isinstance(function.value, ast.Name)
                and function.value.id == "logging"
            ):
                if function.attr == "basicConfig":
                    return None
                if function.attr == "info":
                    return ast.Expr(ast.Constant("TERMINAL_STATUS_OUTPUT"))
        return self.generic_visit(node)

    def visit_Assert(self, node):
        if isinstance(node.test, ast.BoolOp) and isinstance(node.test.op, ast.And):
            return [ast.Assert(part, node.msg) for part in node.test.values]
        return self.generic_visit(node)


def verify_ast(raw, archive):
    result = {}
    for suffix in ["sample", "run", "analyze"]:
        name = f"path80_matched_{suffix}.py"
        trees = [
            ast.parse((base / "scripts" / name).read_text()) for base in [raw, archive]
        ]
        counts = {
            "raw_print_calls": sum(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
                for node in ast.walk(trees[0])
            ),
            "raw_conjunctive_asserts": sum(
                isinstance(node, ast.Assert)
                and isinstance(node.test, ast.BoolOp)
                and isinstance(node.test.op, ast.And)
                for node in ast.walk(trees[0])
            ),
        }
        dumps = [ast.dump(NormalizeArchive().visit(tree)) for tree in trees]
        need(dumps[0] == dumps[1], f"Unexpected AST change in {name}")
        result[name] = {"restricted_ast_equivalent": True, **counts}
    return result


def run_phase(out, phase, argv, env):
    log = out / "logs" / (phase + ".log")
    save(
        out / "logs" / (phase + "_command.json"),
        {
            "argv": argv,
            "cwd": str(out),
            "environment": {
                name: env[name]
                for name in [
                    "LD_LIBRARY_PATH",
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                ]
            },
        },
    )
    begin = time.time()
    with log.open("x") as stream:
        run = subprocess.run(
            argv, cwd=out, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False
        )
    result = {
        "phase": phase,
        "argv": argv,
        "returncode": run.returncode,
        "elapsed_seconds": time.time() - begin,
        "log": str(log),
    }
    save(out / "logs" / (phase + "_result.json"), result)
    need(run.returncode == 0, f"{phase} failed: {log}")
    logging.info("Completed %s", phase)
    return result


def prepare(raw, archive, out):
    original = raw / "path80_matched_v1"
    original_inputs = json.loads((original / "score_v1/inputs.json").read_text())
    protected = dict(original_inputs["protected_input_sha256"])
    for source in ["path80_matched_sample.py", "path80_matched_analyze.py"]:
        protected[str(raw / "scripts" / source)] = sha(raw / "scripts" / source)
    for path, expected in protected.items():
        need(sha(path) == expected, f"Frozen original input changed: {path}")
    source_names = [
        "path80_matched_sample.py",
        "path80_matched_analyze.py",
        "path80_matched_run.py",
        "path80_matched_score.cc",
        "path80_score.cc",
        "independent78_import.h",
        "coupled73_codec.h",
    ]
    source_hashes = {}
    (out / "scripts").mkdir()
    for name in source_names:
        source = archive / "scripts" / name
        source_hashes[str(source)] = sha(source)
        shutil.copyfile(source, out / "scripts" / name)
    for name in ["models64", "models77", "models68"]:
        (out / name).symlink_to(raw / name, target_is_directory=True)
    (out / "notes").mkdir()
    (out / "notes/path80_diagnostic_review.md").symlink_to(
        raw / "notes/path80_diagnostic_review.md"
    )
    return original, protected, source_hashes


def verify_sampling(original, generated):
    manifest = json.loads((original / "sampling_manifest.json").read_text())
    replayed = json.loads((generated / "sampling_manifest.json").read_text())
    binary_hashes = {}
    for name, expected in manifest["table_sha256"].items():
        need(sha(generated / name) == expected, f"Sampled table mismatch: {name}")
        binary_hashes[name] = expected
    for name in ["missing_states.json", "unavailable_proxy_layers.json"]:
        need(
            (original / name).read_bytes() == (generated / name).read_bytes(),
            f"Missing-state mismatch: {name}",
        )
    metadata = ["created_unix", "elapsed_sampling_seconds", "sampling_script_sha256"]
    need(
        {key: value for key, value in manifest.items() if key not in metadata}
        == {key: value for key, value in replayed.items() if key not in metadata},
        "Sampling protocol/value mismatch",
    )
    return {
        "all_four_tables_byte_identical": True,
        "table_sha256": binary_hashes,
        "manifest_semantic_identity": True,
        "permitted_manifest_metadata_differences": metadata,
    }


def score_commands(original, out, raw, table):
    old = original / "score_v1"
    score = table / "score_v1"
    score.mkdir()
    compile_argv = json.loads((old / "logs/compile_command.json").read_text())["argv"]
    replacements = {
        str(old / "score.o"): str(score / "score.o"),
        str(old / "source/path80_matched_score.cc"): str(
            out / "scripts/path80_matched_score.cc"
        ),
        str(old / "source"): str(out / "scripts"),
    }
    compile_argv = [replacements.get(value, value) for value in compile_argv]
    link_argv = json.loads((old / "logs/link_command.json").read_text())["argv"]
    replacements.update(
        {
            str(old / "score"): str(score / "score"),
            f"-Wl,--dependency-file={old}/score.link.d": f"-Wl,--dependency-file={score}/score.link.d",
        }
    )
    link_argv = [replacements.get(value, value) for value in link_argv]
    return score, [
        ("compile", compile_argv),
        ("link", link_argv),
        ("score", [str(score / "score"), str(raw), str(table), str(score)]),
    ]


def verify_score(original, score, table, source_hashes):
    names = ["observations.bin", "training_controls.f64", "score_controls.json"]
    hashes = {name: sha(score / name) for name in names}
    original_proof = json.loads((original / "score_v1/proof.json").read_text())
    for name in names:
        need(
            hashes[name] == original_proof["artifact_sha256"][name],
            f"Complete scorer output differs: {name}",
        )
    controls = json.loads((score / "score_controls.json").read_text())
    need(
        controls["passed"] and controls["observations"] == 63297,
        "Complete 63297-observation gate",
    )
    proof = {
        **controls,
        "frozen_sampling_manifest_sha256": sha(table / "sampling_manifest.json"),
        "artifact_sha256": hashes,
        "source_and_inputs_unchanged": True,
        "archived_source_sha256": source_hashes,
        "replay_kind": "full formatted scorer byte comparison",
        "original_score_proof": str(original / "score_v1/proof.json"),
        "original_score_proof_sha256": sha(original / "score_v1/proof.json"),
        "model_sha256": original_proof["model_sha256"],
        "source_events_sha256": original_proof["source_events_sha256"],
    }
    save(score / "proof.json", proof)
    return {
        "complete_scorer_outputs_byte_identical": True,
        "artifact_sha256": hashes,
        "observations": controls["observations"],
        "controls": controls,
    }


def verify_analysis(original, table):
    old, new = original / "analysis_v1", table / "analysis_v1"
    names = ["per_query.json", "per_state.csv", "bootstrap.f64"]
    for name in names:
        need(sha(old / name) == sha(new / name), f"Analysis artifact differs: {name}")
    with (
        np.load(old / "signed_gap_diagnostics.npz") as left,
        np.load(new / "signed_gap_diagnostics.npz") as right,
    ):
        need(left.files == right.files, "Diagnostic array names")
        for name in left.files:
            need(
                np.array_equal(left[name], right[name]),
                f"Diagnostic array differs: {name}",
            )
    reports = [
        json.loads((folder / "report.json").read_text()) for folder in [old, new]
    ]
    metadata = [
        "sampling_manifest",
        "score_proof",
        "score_proof_sha256",
        "analysis_script_sha256",
        "artifact_sha256",
    ]
    need(
        {key: value for key, value in reports[0].items() if key not in metadata}
        == {key: value for key, value in reports[1].items() if key not in metadata},
        "Analysis numerical or policy report differs",
    )
    return {
        "report_numerical_and_policy_identity": True,
        "per_query_per_state_bootstrap_byte_identical": True,
        "diagnostic_arrays_exactly_equal": True,
        "allowed_report_metadata_differences": metadata,
        "decision": reports[1]["decision"],
        "effects_query_equal_weight": reports[1]["effects_query_equal_weight"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw, archive, out = (
        args.bench_root.resolve(),
        args.archive_root.resolve(),
        args.out.resolve(),
    )
    need(
        out.parent == raw and out.name.startswith("replay80_matched_formatted_"),
        "Exclusive direct replay directory required",
    )
    out.mkdir(exist_ok=False)
    (out / "logs").mkdir()
    started = time.time()
    commands = []
    try:
        ast_result = verify_ast(raw, archive)
        original, protected, source_hashes = prepare(raw, archive, out)
        save(
            out / "plan.json",
            {
                "argv": sys.argv,
                "raw_root": str(raw),
                "archive_root": str(archive),
                "source_sha256": source_hashes,
                "protected_original_input_sha256": protected,
                "restricted_ast": ast_result,
                "retrieval_or_training": False,
            },
        )
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("QG_", "BENCH_", "GTEST_"))
            and key
            not in {"LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH", "PYTHONOPTIMIZE"}
        }
        env.update(
            LD_LIBRARY_PATH=f"{raw}/variants/trace68_native:{raw}/lib",
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
        )
        commands.append(
            run_phase(
                out,
                "sample",
                [
                    sys.executable,
                    str(out / "scripts/path80_matched_sample.py"),
                    "path80_matched_generated_v1",
                ],
                env,
            )
        )
        sampling = verify_sampling(original, out / "path80_matched_generated_v1")
        table = out / "path80_matched_v1"
        table.mkdir()
        manifest = json.loads((original / "sampling_manifest.json").read_text())
        for name in ["sampling_manifest.json", *manifest["table_sha256"]]:
            shutil.copyfile(original / name, table / name)
        score, phases = score_commands(original, out, raw, table)
        for phase, argv in phases:
            commands.append(run_phase(out, phase, argv, env))
        for path, expected in protected.items():
            need(sha(path) == expected, f"Original input changed during replay: {path}")
        for path, expected in source_hashes.items():
            need(sha(path) == expected, f"Archive source changed during replay: {path}")
        scored = verify_score(original, score, table, source_hashes)
        commands.append(
            run_phase(
                out,
                "analyze",
                [
                    sys.executable,
                    str(out / "scripts/path80_matched_analyze.py"),
                    table.name,
                    "score_v1",
                    "analysis_v1",
                ],
                env,
            )
        )
        analysis = verify_analysis(original, table)
        result = {
            "passed": True,
            "scope": "mechanism A archived formatting replay; no retrieval or fitting",
            "out": str(out),
            "restricted_ast": ast_result,
            "sampling": sampling,
            "score": scored,
            "analysis": analysis,
            "source_sha256": source_hashes,
            "raw_inputs_unchanged": True,
            "commands": commands,
            "elapsed_seconds": time.time() - started,
            "bootstrap_scope": "query resampling conditional on fixed heldout proxy table; proxies shared across queries are not independent observations",
        }
        save(out / "stage37_mechanism_a_replay.json", result)
        logging.info(
            "MECHANISM A REPLAY PASSED %s", out / "stage37_mechanism_a_replay.json"
        )
    except BaseException as error:
        save(
            out / "failure.json",
            {"passed": False, "error": repr(error), "commands": commands},
        )
        raise


if __name__ == "__main__":
    main()
