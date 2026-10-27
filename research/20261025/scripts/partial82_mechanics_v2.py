"""Fixed mechanical checks on accepted training exports; no validation or search."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

DEFAULT_ROOT = Path("/home/xingrui/bench/quantized_graph/research_20261025")
DIMS = [128, 256, 384, 512, 768]
logging.basicConfig(level=logging.INFO, format="%(message)s")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():  # noqa: PLR0915 - keep fixed mechanics provenance sequence explicit
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--source-dir", type=Path, default=Path(__file__).resolve().parent
    )
    args = parser.parse_args()
    R = args.root.resolve()
    A = (args.output or R / "partial82_mechanics_v2").resolve()
    source_dir = args.source_dir.resolve()
    A.mkdir(exist_ok=False)
    for name in ["source", "logs"]:
        (A / name).mkdir()
    spec = importlib.util.spec_from_file_location(
        "accepted81_compile_plan", R / "scripts/path81_actual_score.py"
    )
    accepted = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(accepted)
    variant, frozen, compile_argv, link_argv, libraries, old_binary = (
        accepted.compile_plan(R, A)
    )
    binary = A / "partial82_mechanics"
    compile_argv = [
        x.replace(
            str(A / "source/path81_actual_score.cc"),
            str(A / "source/partial82_mechanics_v2.cc"),
        )
        for x in compile_argv
    ]
    link_argv = [str(binary) if x == str(old_binary) else x for x in link_argv]
    export = R / "path81_training_export_v1"
    bank = R / "path81_candidates_v1"
    proof_path, bank_proof_path = export / "proof.json", bank / "manifest.json"
    proof, bank_proof = read(proof_path), read(bank_proof_path)
    need(
        proof["passed"] and proof["partition"] == "training",
        "Successful training export only",
    )
    need(
        bank_proof["passed"] and bank_proof["native_blocks_bit_identical"],
        "Actual native code bank proof",
    )
    need(
        proof["candidate_manifest_sha256"] == sha(bank_proof_path),
        "Linked candidate provenance",
    )
    inputs = {
        str(proof_path): sha(proof_path),
        str(bank_proof_path): sha(bank_proof_path),
        str(R / "scripts/path81_actual_score.py"): sha(
            R / "scripts/path81_actual_score.py"
        ),
    }
    for name in ["states.bin", "queries.bin", "scores.f32"]:
        need(
            sha(export / name) == proof["artifacts_sha256"][name],
            "Training binary changed",
        )
        inputs[str(export / name)] = proof["artifacts_sha256"][name]
    for name in ["native_blocks.bin", "candidate_compact.u8"]:
        need(
            sha(bank / name) == bank_proof["artifacts_sha256"][name],
            "Code bank input changed",
        )
        inputs[str(bank / name)] = bank_proof["artifacts_sha256"][name]
    sources = {}
    source_paths = {}
    for name in [
        "partial82_mechanics_v2.py",
        "partial82_mechanics_v2.cc",
        "path80_score.cc",
        "independent78_import.h",
    ]:
        source = (source_dir if name.startswith("partial82_") else R / "scripts") / name
        source_paths[name] = source
        sources[name] = sha(source)
        shutil.copyfile(source, A / "source" / name)
    build_manifest_path = export / "build_inputs.json"
    previous_build = read(build_manifest_path)
    for path in libraries:
        need(
            sha(path) == previous_build["libraries_sha256"][str(path)],
            "Accepted linked library changed",
        )
    frozen_relative = [
        "src/turbo/distance/common/symphony_qg_distance.h",
        "src/turbo/distance/avx512/symphony_qg/symphony_qg_kernels_avx512.cc",
        "src/turbo/preprocessor/symphony_qg_rotator/symphony_qg_rotator.h",
    ]
    for name in frozen_relative:
        need(
            sha(frozen / name) == previous_build["frozen_source_sha256"][name],
            "Accepted kernel/rotation source changed",
        )
        inputs[str(frozen / name)] = sha(frozen / name)
    kernel_commands = read(R / "results/all_compile_commands.json")
    kernel_matches = [
        entry
        for entry in kernel_commands
        if any(
            str(arg).endswith("symphony_qg_kernels_avx512.cc") for arg in entry["argv"]
        )
    ]
    need(len(kernel_matches) == 1, "Exactly one frozen kernel compile recipe")
    kernel_command = kernel_matches[0]
    kernel_argv = kernel_command["argv"]
    need(
        "-O3" in kernel_argv and "-march=icelake-server" in kernel_argv,
        "Frozen native optimization recipe",
    )
    need(
        "-ffp-contract=off" not in kernel_argv,
        "Existing bench recipe permits contraction; do not alter baseline",
    )
    kernel_object = Path(kernel_argv[kernel_argv.index("-o") + 1])
    turbo_library = R / "lib/libzvec_turbo.a"
    need(
        sha(kernel_object)
        == "d2bb34bf7dfcbb1abfe17a213d6f532d90e42f1b487c4a436603e3144c9acb69",
        "Audited actual kernel object",
    )
    need(
        sha(turbo_library)
        == "31224cffd76fa576f89bf640288659368cf495728e0435776632c4008997d308",
        "Audited actual turbo library",
    )
    evidence_paths = [
        R / "scripts/build.py",
        R / "results/all_compile_commands.json",
        R / "partial82_mechanics_v1/failure.json",
        R / "partial82_diagnose_v1/first_difference.json",
        R / "partial82_diagnose_v1/native_scan_disassembly.txt",
        kernel_object,
        turbo_library,
    ]
    for path in evidence_paths:
        inputs[str(path)] = sha(path)
    shutil.copyfile(
        R / "partial82_diagnose_v1/first_difference.json",
        A / "v1_first_difference.json",
    )
    shutil.copyfile(
        R / "partial82_diagnose_v1/native_scan_disassembly.txt",
        A / "v1_native_scan_disassembly.txt",
    )
    save(
        A / "native_build_provenance.json",
        {
            "existing_baseline_condition_not_algorithm_change": True,
            "build_script_path": str(R / "scripts/build.py"),
            "build_script_sha256": sha(R / "scripts/build.py"),
            "build_script_lines": [40, 46],
            "recipe_origin": "copied fp32 squared_euclidean compile recipe",
            "kernel_compile_command": kernel_command,
            "kernel_object_sha256": sha(kernel_object),
            "turbo_library_sha256": sha(turbo_library),
            "CMake_ffp_contract_off_not_in_bench_command": True,
            "v1_binary_epilogue_addresses": {
                "fma_delta_N_vl": ["0x1811f4", "0x18120b"],
                "add_k1": ["0x181211", "0x181217"],
                "add_A_g": ["0x181200", "0x181229"],
                "fma_B_dot_base": ["0x18121d", "0x181230"],
            },
            "oracle": "fma(delta,float(N),vl);fp32_add(k1);fp32_add(A,g);fma(B,dot,base)",
            "v1_failure_preserved": True,
        },
    )
    rng = np.random.default_rng(8201)
    permutation = rng.permutation(192)
    subsets = np.full((2, 5, 192), np.iinfo(np.uint32).max, dtype="<u4")
    for row, m in enumerate(DIMS):
        subsets[0, row, : m // 4] = np.arange(m // 4)
        subsets[1, row, : m // 4] = permutation[: m // 4]
    with (A / "subset_indices.u32").open("xb") as stream:
        stream.write(subsets.tobytes())
    save(
        A / "plan.json",
        {
            "scope": "mechanics only;baseline partial q-dot with A_full;not Split residual estimator",
            "states": 6625,
            "training_queries": 3072,
            "lanes": 32,
            "dimensions": DIMS,
            "random_subset_seed": 8201,
            "rng": "numpy.default_rng PCG64",
            "numpy_version": np.__version__,
            "one_global_permutation_shared_by_every_state_lane": permutation.tolist(),
            "subset_sha256": sha(A / "subset_indices.u32"),
            "source_sha256": sources,
            "source_paths": {name: str(path) for name, path in source_paths.items()},
            "native_build_provenance_sha256": sha(A / "native_build_provenance.json"),
            "input_sha256": inputs,
            "accepted_build_manifest_path": str(build_manifest_path),
            "accepted_build_manifest_sha256": sha(build_manifest_path),
            "full_dimension_required_bit_exact": True,
            "lookup_tail_bound_target": "actual native score only",
            "no_effect_gate_or_tuning": True,
            "no_validation_or_retrieval_or_QPS": True,
        },
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("QG_", "BENCH_", "GTEST_"))
        and k not in {"LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH", "PYTHONOPTIMIZE"}
    }
    env.update(
        LD_LIBRARY_PATH=str(variant) + ":" + str(R / "lib"),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    commands = [
        ("compile", compile_argv),
        ("link", link_argv),
        (
            "mechanics",
            [
                str(binary),
                str(export / "states.bin"),
                str(export / "queries.bin"),
                str(export / "scores.f32"),
                str(bank),
                str(A / "subset_indices.u32"),
                str(A),
            ],
        ),
    ]
    completed = []
    try:
        for name, command in commands:
            logging.info("PARTIAL82 %s", name)
            with (A / "logs" / (name + ".log")).open("x") as log:
                process = subprocess.run(
                    command,
                    cwd=A,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            record = {"name": name, "argv": command, "returncode": process.returncode}
            completed.append(record)
            save(A / "logs" / (name + "_command.json"), record)
            need(
                process.returncode == 0,
                "Failed " + name + "; preserve exclusive evidence",
            )
        result = read(A / "mechanics_result.json")
        need(
            result["passed"]
            and result["states"] == 6625
            and result["mD_bitexact_and_bound_collapses"] == 2 * 6625 * 32,
            "Full mechanics gates required",
        )
        for path, expected in inputs.items():
            need(sha(path) == expected, "Frozen input changed")
        for name, expected in sources.items():
            need(sha(source_paths[name]) == expected, "Source changed during mechanics")
        for path in libraries:
            need(
                sha(path) == previous_build["libraries_sha256"][str(path)],
                "Library changed during mechanics",
            )
        result.update(
            source_sha256=sources,
            input_manifest_sha256=sha(A / "plan.json"),
            binary_sha256=sha(binary),
            native_build_provenance_sha256=sha(A / "native_build_provenance.json"),
            original_v1_failure_preserved=True,
            explicit_two_FMA_oracle_matches_frozen_binary=True,
            commands=completed,
            diagnostics_sha256=sha(A / "first_state_diagnostic_samples.bin"),
            no_validation_or_retrieval_or_QPS=True,
        )
        save(A / "proof.json", result)
        logging.info("PARTIAL82 PASSED %s", A / "proof.json")
    except BaseException as error:
        save(
            A / "failure.json",
            {"passed": False, "error": repr(error), "commands": completed},
        )
        raise


if __name__ == "__main__":
    main()
