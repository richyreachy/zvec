"""Check actual partial codecs against frozen training data, without search or QPS."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

DEFAULT_ROOT = Path("/home/xingrui/bench/quantized_graph/research_20261025")
HEADER = "src/turbo/quantizer/symphony_qg_quantizer/symphony_qg_quantizer.h"
HEADERS = [
    HEADER,
    "src/turbo/distance/common/symphony_qg_distance.h",
    "src/turbo/preprocessor/symphony_qg_rotator/symphony_qg_rotator.h",
]
logging.basicConfig(level=logging.INFO, format="%(message)s")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    with Path(path).open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def variant_plan(root, output, name):
    need(Path(name).name == name, "Simple variant directory name")
    variant = root / "variants" / name
    meta = read(variant / "metadata.json")
    frozen = Path(meta["immutable_build_source"]).resolve()
    need(frozen == (variant / "build_source").resolve(), "Variant immutable source")
    macros = [v for v in meta["defines"] if v.startswith("QG_PARTIAL_SCAN_DIM=")]
    need(len(macros) <= 1, "Unambiguous prefix dimension")
    m = int(macros[0].split("=")[1]) if macros else 0
    commands = read(variant / "commands.json")
    original = next(
        c
        for c in commands
        if "-c" in c and any(x.endswith("/hnsw_quantized_graph.cc") for x in c)
    )
    need(
        [x[2:] for x in original if x.startswith("-DQG_")] == meta["defines"],
        "Compiler and frozen metadata macros agree",
    )
    old_source = original[original.index("-c") + 1]
    old_object = original[original.index("-o") + 1]
    need(Path(old_source).is_relative_to(frozen), "Compiler source is frozen")
    target = output / name
    obj, binary = target / "probe.o", target / "probe"
    compile_argv = [
        str(output / "source/partial83_codec_probe.cc")
        if x == old_source
        else str(obj)
        if x == old_object
        else x
        for x in original
    ]
    compile_argv += ["-pthread", "-MMD", "-MF", str(target / "probe.d")]
    libraries = [
        root / "lib" / name
        for name in ["libzvec_turbo.a", "librabitqlib_runtime.a", "libzvec_ailego.a"]
    ]
    link_argv = [
        original[0],
        "-march=x86-64",
        "-O3",
        "-DNDEBUG",
        str(obj),
        "-o",
        str(binary),
        *map(str, libraries),
        "-ldl",
        "-pthread",
    ]
    return {
        "name": name,
        "macro_dimension": m,
        "metadata": meta,
        "frozen": str(frozen),
        "compile": compile_argv,
        "link": link_argv,
        "binary": str(binary),
        "output": str(target),
        "libraries": list(map(str, libraries)),
    }


def main():  # noqa: PLR0915 - explicit frozen inputs and sequential gates
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-dir", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root, output, source_dir = (
        args.root.resolve(),
        args.output.resolve(),
        args.source_dir.resolve(),
    )
    plans = [variant_plan(root, output, name) for name in args.variants]
    need(
        sorted(p["macro_dimension"] for p in plans) == [0, 128, 256, 384, 512, 768],
        "Exactly undefined and all five frozen dimensions required",
    )
    if not args.execute:
        logging.info("%s", json.dumps({"plans": plans, "execute": False}, indent=2))
        return
    output.mkdir(exist_ok=False)
    (output / "source").mkdir()
    protected = {}
    for name in ["partial83_codec_probe.cc", "partial83_codec_probe.py"]:
        path = source_dir / name
        protected[str(path)] = sha(path)
        shutil.copyfile(path, output / "source" / name)
    export, bank = root / "path81_training_export_v1", root / "path81_candidates_v1"
    proof_path, candidate_path = export / "proof.json", bank / "manifest.json"
    proof, candidate = read(proof_path), read(candidate_path)
    need(proof["passed"] and proof["partition"] == "training", "Frozen training export")
    need(
        candidate["passed"] and candidate["native_blocks_bit_identical"],
        "Native block bank",
    )
    need(
        proof["candidate_manifest_sha256"] == sha(candidate_path),
        "Linked candidate proof",
    )
    need(
        sha(proof_path)
        == "1e10be6bec5d3faf4bce52a0b02a0b08b5887ff36fa7807840572ad4703fa1d0",
        "Accepted training export proof",
    )
    protected[str(proof_path)], protected[str(candidate_path)] = (
        sha(proof_path),
        sha(candidate_path),
    )
    for name in ["states.bin", "queries.bin", "scores.f32"]:
        need(
            sha(export / name) == proof["artifacts_sha256"][name], "Training input SHA"
        )
        protected[str(export / name)] = proof["artifacts_sha256"][name]
    need(
        sha(bank / "native_blocks.bin")
        == candidate["artifacts_sha256"]["native_blocks.bin"],
        "Native block SHA",
    )
    protected[str(bank / "native_blocks.bin")] = candidate["artifacts_sha256"][
        "native_blocks.bin"
    ]
    accepted_build = read(export / "build_inputs.json")
    accepted_mechanics = root / "partial82_mechanics_v2/proof.json"
    need(read(accepted_mechanics)["passed"], "Phase82 actual low-level gate")
    protected[str(accepted_mechanics)] = sha(accepted_mechanics)
    for plan in plans:
        variant = root / "variants" / plan["name"]
        frozen = Path(plan["frozen"])
        manifest_path = variant / "build_source_sha256.json"
        manifest = read(manifest_path)
        for name in ["metadata.json", "commands.json", "build_source_sha256.json"]:
            protected[str(variant / name)] = sha(variant / name)
        for name in HEADERS:
            need(sha(frozen / name) == manifest[name], "Frozen codec dependency SHA")
            protected[str(frozen / name)] = manifest[name]
        for library in plan["libraries"]:
            need(
                sha(library) == accepted_build["libraries_sha256"][library],
                "Accepted actual kernel/library SHA",
            )
            protected[library] = sha(library)
    save(
        output / "plan.json",
        {
            "plans": plans,
            "protected_sha256": protected,
            "queries": 3072,
            "states": 6625,
            "training_only": True,
            "baseline_reference_is_phase82_verified_actual_kernel": True,
            "minimal_direct_header_probe_no_HNSW_TU": True,
            "no_retrieval_or_QPS": True,
        },
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("QG_", "BENCH_", "GTEST_"))
        and k not in {"LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONOPTIMIZE"}
    }
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    results, completed = [], []
    try:
        for plan in plans:
            target = Path(plan["output"])
            target.mkdir()
            commands = [
                ("compile", plan["compile"]),
                ("link", plan["link"]),
                (
                    "probe",
                    [
                        plan["binary"],
                        str(export / "states.bin"),
                        str(export / "queries.bin"),
                        str(export / "scores.f32"),
                        str(bank / "native_blocks.bin"),
                        str(target),
                    ],
                ),
            ]
            for label, argv in commands:
                logging.info("PARTIAL83 %s %s", plan["name"], label)
                with (target / (label + ".log")).open("x") as stream:
                    process = subprocess.run(
                        argv,
                        cwd=target,
                        env=env,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                record = {
                    "variant": plan["name"],
                    "step": label,
                    "argv": argv,
                    "returncode": process.returncode,
                }
                completed.append(record)
                save(target / (label + "_command.json"), record)
                need(process.returncode == 0, "Failed " + plan["name"] + "/" + label)
            result = read(target / "result.json")
            need(
                result["passed"]
                and result["queries"] == 3072
                and result["states"] == 6625
                and result["actual_codec_scan_bit_checks"] == 212000
                and result["full_native_reference_bit_checks"] == 212000,
                "Complete codec probe gates",
            )
            result.update(
                variant=plan["name"],
                macro_dimension=plan["macro_dimension"],
                scores_sha256=sha(target / "actual_scores.u32"),
                binary_sha256=sha(plan["binary"]),
            )
            results.append(result)
        full = [r for r in results if r["macro_dimension"] in (0, 768)]
        need(
            len(full) == 2
            and full[0]["scores_sha256"] == full[1]["scores_sha256"]
            and full[0]["query_size_bytes"] == full[1]["query_size_bytes"],
            "Undefined/m768 exact score bytes and query object size",
        )
        for path, expected in protected.items():
            need(sha(path) == expected, "Protected input/source/library changed")
        save(
            output / "proof.json",
            {
                "passed": True,
                "results": results,
                "commands": completed,
                "plan_sha256": sha(output / "plan.json"),
                "all_protected_inputs_unchanged": True,
                "undefined_m768_score_bytes_and_query_size_equal": True,
                "truth_labels_or_validation_used": False,
                "new_retrieval_or_QPS": False,
            },
        )
        logging.info("PARTIAL83 PASSED %s", output / "proof.json")
    except BaseException as error:
        save(
            output / "failure.json",
            {"passed": False, "error": repr(error), "completed": completed},
        )
        raise


if __name__ == "__main__":
    main()
