"""Phase81 actual-code export. Default plans only; training and validation are gated.

No search is opened. Validation accepts exactly two selected models only after
training_frozen_models and all its artifacts have been verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import struct
import subprocess
import time
from pathlib import Path

PROTOCOL_SHA = "ced04e30f808f016156bacdabfd89ad27ddd238e8388a4ba798457b4eb849f29"
TRUSTED = {
    "models64/full_graph.bin": "c04308b5445a666026bd460100ef0a649fd3dfd7e75a1b57561ab40a1e7dc53b",
    "models64/full_vectors.f32": "5c9d8196d5c46d617c5cb3d9eedae5f9b8bf1677e370f210c5b84b08cdd66d24",
    "models77/rotated.f32": "69787cd3b97ebc844a5547d76c10fdaca14a103f62d6fd3fe62a67caccf18f22",
    "models68/events.bin": "03ed33a889c106748c87d870bdccc7dbfd367d44b8a6b3047dfe8b8ecc01f51b",
    "models68/reference_results.bin": "0038285c11c0881a111f1330f705e993bae2012343b3503ee9ac629ca6e7c8a9",
}
logging.basicConfig(level=logging.INFO, format="%(message)s")


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def verify_artifacts(root, hashes):
    need(isinstance(hashes, dict) and hashes, "Nonempty artifact hash mapping required")
    for name, digest in hashes.items():
        path = Path(name)
        need(
            not path.is_absolute() and ".." not in path.parts,
            "Artifacts must stay within their frozen directory",
        )
        need(sha(root / path) == digest, "Artifact SHA mismatch: " + str(root / path))


def compile_plan(root, output):
    variant = root / "variants/trace68_native"
    metadata = read(variant / "metadata.json")
    frozen = Path(metadata["immutable_build_source"])
    commands = read(variant / "commands.json")
    original = next(
        c
        for c in commands
        if "-c" in c and any(x.endswith("/hnsw_quantized_graph.cc") for x in c)
    )
    old_source = next(x for x in original if x.endswith("/hnsw_quantized_graph.cc"))
    old_object = original[original.index("-o") + 1]
    obj, binary = output / "score.o", output / "path81_actual_score"
    compile_command = [
        str(output / "source/path81_actual_score.cc")
        if x == old_source
        else str(obj)
        if x == old_object
        else x
        for x in original
    ]
    compile_command += ["-I", str(output / "source"), "-pthread"]
    original_link = next(
        c["argv"]
        for c in read(root / "results/build_commands.json")
        if "-c" not in c["argv"] and str(root / "objects/benchmark.o") in c["argv"]
    )
    old_binary = original_link[original_link.index("-o") + 1]
    link_command = [
        str(obj)
        if x == str(root / "objects/benchmark.o")
        else str(binary)
        if x == old_binary
        else "-Wl,--dependency-file=" + str(output / "score.link.d")
        if x.startswith("-Wl,--dependency-file=")
        else x
        for x in original_link
    ]
    link_command.append("-pthread")
    libraries = {Path(x) for x in link_command if x.endswith((".a", ".so"))}
    libraries.add(variant / "libcore_knn_hnsw.so")
    return variant, frozen, compile_command, link_command, libraries, binary


def main():  # noqa: PLR0915 - preserve explicit partition/freeze gates
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent.parent
    )
    parser.add_argument("--protocol", type=Path)
    parser.add_argument(
        "--mode", choices=["training", "validation"], default="training"
    )
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--frozen-models", type=Path)
    parser.add_argument("--training-export-proof", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    protocol_path = (args.protocol or root / "path81_protocol_v1/config.json").resolve()
    need(sha(protocol_path) == PROTOCOL_SHA, "Frozen root protocol SHA")
    protocol = read(protocol_path)
    need(
        protocol["frozen"] and protocol["phase"] == 81,
        "Frozen phase81 protocol required",
    )
    need(
        not output.exists() and not output.is_symlink(),
        "Exclusive fresh output required",
    )
    need(not (args.compile_only and args.execute), "Choose compile-only or execute")
    need(
        args.mode == "validation"
        or (args.frozen_models is None and args.training_export_proof is None),
        "Training export cannot accept validation model artifacts",
    )
    variant, frozen, compiler, linker, libraries, binary = compile_plan(root, output)
    plan = {
        "phase": 81,
        "partition": args.mode,
        "compile": compiler,
        "link": linker,
        "output": str(output),
        "protocol_sha256": PROTOCOL_SHA,
        "compile_only": args.compile_only,
        "execute_requested": args.execute,
        "candidate_manifest": None
        if args.candidate_manifest is None
        else str(args.candidate_manifest.resolve()),
        "validation_requires_frozen_training": True,
        "no_search_or_training": True,
    }
    if not args.execute and not args.compile_only:
        logging.info("%s", json.dumps(plan, indent=2))
        return
    output.mkdir()
    (output / "source").mkdir()
    (output / "logs").mkdir()
    save(output / "plan.json", plan)
    protected = {str(protocol_path): PROTOCOL_SHA}
    commands = []
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("QG_", "BENCH_", "GTEST_"))
        and k not in {"LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH", "PYTHONOPTIMIZE"}
    }
    env.update(
        LD_LIBRARY_PATH=str(variant) + ":" + str(root / "lib"),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )

    def run(name, command):
        begin = time.time()
        log_path = output / "logs" / (name + ".log")
        with log_path.open("x") as stream:
            completed = subprocess.run(
                command,
                cwd=output,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        record = {
            "name": name,
            "argv": command,
            "cwd": str(output),
            "returncode": completed.returncode,
            "elapsed_seconds": time.time() - begin,
            "log": str(log_path),
        }
        commands.append(record)
        save(output / "logs" / (name + "_command_result.json"), record)
        need(completed.returncode == 0, name + " failed; preserve this directory")

    try:
        source_hashes = {}
        # Preserve accepted path80 source exactly; compile the copied phase81 TU.
        names = [
            "path81_actual_score.cc",
            "path81_actual_score.py",
            "path80_score.cc",
            "independent78_import.h",
            "candidate81_codec.h",
        ]
        for name in names:
            source = Path(__file__).resolve().parent / name
            source_hashes[name] = sha(source)
            protected[str(source)] = source_hashes[name]
            shutil.copyfile(source, output / "source" / name)
        frozen_hashes = read(variant / "build_source_sha256.json")
        for name, expected in frozen_hashes.items():
            need(
                sha(frozen / name) == expected, "Frozen native source mismatch: " + name
            )
        library_hashes = {str(path): sha(path) for path in libraries}
        protected.update(library_hashes)
        save(
            output / "build_inputs.json",
            {
                "source_sha256": source_hashes,
                "frozen_source_root": str(frozen),
                "frozen_source_sha256": frozen_hashes,
                "libraries_sha256": library_hashes,
                "compiler_flags_and_link_order_from_accepted_trace68_variant": True,
            },
        )
        run("compile", compiler)
        run("link", linker)
        if args.compile_only:
            save(
                output / "build_proof.json",
                {
                    "passed": True,
                    "compiled_only": True,
                    "scores_read": False,
                    "source_sha256": source_hashes,
                    "commands": commands,
                    "binary_sha256": sha(binary),
                },
            )
            return
        need(
            args.candidate_manifest is not None,
            "Scoring requires a successful geometry candidate manifest",
        )
        candidate_manifest_path = args.candidate_manifest.resolve()
        candidate = read(candidate_manifest_path)
        candidate_dir = candidate_manifest_path.parent
        need(
            candidate.get("passed") is True
            and candidate.get("geometry_only") is True
            and candidate.get("native_blocks_bit_identical") is True,
            "Candidate geometry/native-byte gate required",
        )
        need(
            candidate["candidate_count"] == 137 and candidate["center_count"] == 20,
            "Full candidate scope",
        )
        need(
            candidate["protocol_sha256"] == PROTOCOL_SHA, "Candidate protocol mismatch"
        )
        need(
            candidate["centers_source_sha256"]
            == protocol["preflight"]["files_sha256"]["centers.json"],
            "Candidate frozen center list mismatch",
        )
        verify_artifacts(candidate_dir, candidate["artifacts_sha256"])
        protected[str(candidate_manifest_path)] = sha(candidate_manifest_path)
        for name, digest in candidate["artifacts_sha256"].items():
            protected[str(candidate_dir / name)] = digest
        preflight = Path(protocol["preflight"]["root"])
        for name, digest in protocol["preflight"]["files_sha256"].items():
            need(sha(preflight / name) == digest, "Frozen preflight metadata mismatch")
            protected[str(preflight / name)] = digest
        need(
            (candidate_dir / "centers.u32").read_bytes()
            == struct.pack("<20I", *protocol["preflight"]["center_order"]),
            "Candidate center order",
        )
        selected_dir = "-"
        frozen_model_sha = None
        if args.mode == "validation":
            need(
                args.frozen_models is not None
                and args.training_export_proof is not None,
                "Validation requires frozen training manifest and exact training export proof",
            )
            model_path = args.frozen_models.resolve()
            model = read(model_path)
            frozen_model_sha = sha(model_path)
            need(
                model.get("validation_authorized_after_freeze") is True,
                "Validation not authorized by frozen training manifest",
            )
            need(
                model["protocol_sha256"] == PROTOCOL_SHA
                and model["candidate_manifest_sha256"] == sha(candidate_manifest_path),
                "Model protocol/candidate mismatch",
            )
            training_path = args.training_export_proof.resolve()
            need(
                sha(training_path) == model["training_export_proof_sha256"],
                "Training export proof SHA mismatch",
            )
            training = read(training_path)
            need(
                training.get("passed") is True
                and training["partition"] == "training"
                and training["protocol_sha256"] == PROTOCOL_SHA
                and training["candidate_manifest_sha256"]
                == sha(candidate_manifest_path),
                "Training export proof gate",
            )
            required = {
                "selected_indices.json",
                "objectives.npz",
                "selected_models.bin",
                "training_result.json",
            }
            need(
                required <= set(model["artifacts_sha256"]),
                "Incomplete training frozen artifacts",
            )
            verify_artifacts(model_path.parent, model["artifacts_sha256"])
            protected[str(model_path)] = frozen_model_sha
            protected[str(training_path)] = sha(training_path)
            for name, digest in model["artifacts_sha256"].items():
                protected[str(model_path.parent / name)] = digest
            indices = read(model_path.parent / "selected_indices.json")
            need(
                indices["method_order"] == ["second_moment", "hinge"]
                and indices["protocol_sha256"] == PROTOCOL_SHA
                and indices["candidate_manifest_sha256"] == sha(candidate_manifest_path)
                and indices["training_export_proof_sha256"] == sha(training_path),
                "Selected index provenance",
            )
            values = indices["indices"]
            need(
                len(values) == 2
                and all(len(v) == 20 for v in values)
                and all(len(row) == 32 for v in values for row in v),
                "Selected indices [2,20,32]",
            )
            flat = [value for v in values for row in v for value in row]
            need(
                all(type(value) is int and 0 <= value < 137 for value in flat),
                "Selected candidate indices",
            )
            selected_dir = output / "selected_inputs"
            selected_dir.mkdir()
            shutil.copyfile(
                model_path.parent / "selected_models.bin",
                selected_dir / "selected_models.bin",
            )
            with (selected_dir / "selected_indices.u32").open("xb") as stream:
                stream.write(struct.pack("<1280I", *flat))
        # Only this branch-specific state metadata is decoded. Training never
        # prepares or scores validation queries and never reads validation labels.
        state_name = (
            "validation_selected_states.json"
            if args.mode == "validation"
            else "training_selected_states.json"
        )
        states = read(preflight / state_name)
        expected_count = protocol["preflight"][
            "validation_states" if args.mode == "validation" else "train_states"
        ]
        need(len(states) == expected_count, "All frozen selected states required")
        split = read(preflight / "split.json")
        partition_key = (
            "validation_query_serials"
            if args.mode == "validation"
            else "train_query_serials"
        )
        partition_ids = set(split[partition_key])
        need(
            all(state["query_serial"] in partition_ids for state in states),
            "Wrong partition query",
        )
        with (output / "input_states.bin").open("xb") as stream:
            stream.write(
                struct.pack(
                    "<8I",
                    0x51475381,
                    1,
                    16,
                    len(states),
                    int(args.mode == "validation"),
                    4096,
                    768,
                    0,
                )
            )
            for state in states:
                stream.write(
                    struct.pack(
                        "<4I",
                        state["query_serial"],
                        state["query_internal_id"],
                        state["center"],
                        state["step"],
                    )
                )
        for relative, expected in TRUSTED.items():
            need(sha(root / relative) == expected, "Frozen input mismatch: " + relative)
            protected[str(root / relative)] = expected
        score_command = [
            str(binary),
            args.mode,
            str(root / "models68/events.bin"),
            str(root / "models64/full_graph.bin"),
            str(root / "models64/full_vectors.f32"),
            str(root / "models77/rotated.f32"),
            str(candidate_dir),
            str(output / "input_states.bin"),
            str(root / "models68/reference_results.bin"),
            str(selected_dir),
            str(output),
        ]
        save(
            output / "inputs.json",
            {
                "protocol_path": str(protocol_path),
                "protocol_sha256": PROTOCOL_SHA,
                "candidate_manifest_path": str(candidate_manifest_path),
                "candidate_manifest_sha256": sha(candidate_manifest_path),
                "frozen_training_models_sha256": frozen_model_sha,
                "protected_sha256": protected,
                "input_states_sha256": sha(output / "input_states.bin"),
                "scores_command": score_command,
            },
        )
        run("score", score_command)
        result = read(output / "scorer_result.json")
        need(
            result["passed"]
            and result["state_count"] == expected_count
            and result["partition"] == args.mode
            and result["native_gate_completed_before_candidate_scoring"]
            and result["native_edges_bit_identical"] == expected_count * 32
            and result["models_scored_per_state"]
            == (137 if args.mode == "training" else 3),
            "Actual native score gate and scope",
        )
        for path, expected in protected.items():
            need(sha(path) == expected, "Input/source changed during export: " + path)
        for name, expected in frozen_hashes.items():
            need(sha(frozen / name) == expected, "Frozen source changed during export")
        proof = {
            **result,
            "protocol_sha256": PROTOCOL_SHA,
            "candidate_manifest_sha256": sha(candidate_manifest_path),
            "candidate_manifest_path": str(candidate_manifest_path),
            "training_frozen_models_sha256": frozen_model_sha,
            "source_sha256": source_hashes,
            "binary_sha256": sha(binary),
            "commands": commands,
            "artifacts_sha256": {
                name: sha(output / name)
                for name in [
                    "states.bin",
                    "queries.bin",
                    "scores.f32",
                    "scorer_result.json",
                ]
            },
            "full_frozen_partition_scope": True,
            "only_native_path_states_not_actual_learned_path": True,
            "inputs_manifest_sha256": sha(output / "inputs.json"),
            "no_search_timing_or_optimization": True,
        }
        save(output / "proof.json", proof)
        logging.info("PHASE81 %s export passed: %s", args.mode, output / "proof.json")
    except BaseException as error:
        save(
            output / "failure.json",
            {"passed": False, "error": repr(error), "commands": commands},
        )
        raise


if __name__ == "__main__":
    main()
