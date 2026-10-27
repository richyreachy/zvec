"""Assemble frozen selected phase81 candidate lanes without queries or scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import struct
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(4 << 20), b""):
            digest.update(data)
    return digest.hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--indices", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="New exclusive output directory"
    )
    args = parser.parse_args()
    manifest_path, indices_path, out = (
        args.candidate_manifest.resolve(),
        args.indices.resolve(),
        args.output.resolve(),
    )
    bank = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    indices = json.loads(indices_path.read_text())
    need(
        manifest["passed"]
        and manifest["native_blocks_bit_identical"]
        and manifest["geometry_only"],
        "Accepted candidate bank required",
    )
    need(indices["method_order"] == ["second_moment", "hinge"], "Frozen model order")
    need(
        indices["candidate_manifest_sha256"] == sha(manifest_path),
        "Candidate manifest identity",
    )
    need(
        indices["protocol_sha256"] == manifest["protocol_sha256"],
        "Frozen protocol identity",
    )
    need(
        isinstance(indices["training_export_proof_sha256"], str)
        and len(indices["training_export_proof_sha256"]) == 64,
        "Training export proof reference required",
    )
    for name, expected in manifest["artifacts_sha256"].items():
        need(sha(bank / name) == expected, "Candidate artifact changed")
    binary = Path(manifest["binary_path"])
    need(sha(binary) == manifest["binary_sha256"], "Frozen assembler binary identity")
    mask = (bank / "valid.u8").read_bytes()
    values = indices["indices"]
    need(len(values) == 2, "Two model index matrices required")
    flattened = []
    for model in values:
        need(len(model) == 20, "20 centers per model")
        for ci, row in enumerate(model):
            need(len(row) == 32, "32 selected lanes per center")
            for lane, value in enumerate(row):
                need(
                    type(value) is int
                    and 0 <= value < 137
                    and mask[(ci * 137 + value) * 32 + lane] == 1,
                    "Invalid selected candidate",
                )
                flattened.append(value)
    out.mkdir(exist_ok=False)
    try:
        shutil.copyfile(indices_path, out / "selected_indices.json")
        with (out / "selected_indices.u32").open("xb") as stream:
            stream.write(struct.pack("<1280I", *flattened))
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("QG_", "BENCH_", "GTEST_"))
            and key
            not in {"LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH", "PYTHONOPTIMIZE"}
        }
        root = bank.parent
        env["LD_LIBRARY_PATH"] = f"{root}/variants/trace68_native:{root}/lib"
        command = [
            str(binary),
            "--assemble",
            str(bank),
            str(out / "selected_indices.u32"),
            str(out),
        ]
        save(out / "command.json", {"argv": command, "cwd": str(out)})
        with (out / "assemble.log").open("x") as stream:
            result = subprocess.run(
                command,
                cwd=out,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        need(result.returncode == 0, "Frozen native assembly failed")
        proof = json.loads((out / "assembly_result.json").read_text())
        need(
            proof["passed"] and proof["assembled_blocks_verified"] == 40,
            "Full selected model assembly proof",
        )
        need(
            sha(indices_path) == sha(out / "selected_indices.json"),
            "Selected indices changed during assembly",
        )
        need(
            sha(manifest_path) == indices["candidate_manifest_sha256"],
            "Candidate manifest changed during assembly",
        )
        for name, expected in manifest["artifacts_sha256"].items():
            need(
                sha(bank / name) == expected,
                "Candidate artifact changed during assembly",
            )
        save(
            out / "assembly_proof.json",
            {
                **proof,
                "method_order": indices["method_order"],
                "protocol_sha256": manifest["protocol_sha256"],
                "candidate_manifest_sha256": sha(manifest_path),
                "training_export_proof_sha256": indices["training_export_proof_sha256"],
                "indices_source_sha256": sha(indices_path),
                "binary_sha256": manifest["binary_sha256"],
                "assembler_script_sha256": sha(Path(__file__)),
                "artifacts_sha256": {
                    name: sha(out / name)
                    for name in [
                        "selected_indices.json",
                        "selected_indices.u32",
                        "selected_models.bin",
                        "assembly_result.json",
                    ]
                },
                "selected_models_header_u32": [
                    0x51474D81,
                    1,
                    64,
                    768,
                    20,
                    2,
                    32,
                    3328,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                "selected_models_payload_shape": [2, 20, 3328],
                "no_query_label_or_score_files_read": True,
                "note": "geometry assembly only; parent must freeze all training artifacts before validation",
            },
        )
        logging.info("ASSEMBLE81 COMPLETE %s", out / "assembly_proof.json")
    except BaseException as error:
        save(out / "failure.json", {"passed": False, "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
