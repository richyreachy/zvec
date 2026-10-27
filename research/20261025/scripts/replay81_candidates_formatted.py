"""Replay candidate geometry and lane assembly after archive formatting only."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(4 << 20), b""):
            digest.update(data)
    return digest.hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


class Normalize(ast.NodeTransformer):
    """Only flatten equivalent context managers and a fixed dictionary loop."""

    def visit_With(self, node):
        node = self.generic_visit(node)
        while len(node.body) == 1 and isinstance(node.body[0], ast.With):
            node.items.extend(node.body[0].items)
            node.body = node.body[0].body
        return node

    def visit_For(self, node):
        node = self.generic_visit(node)
        if isinstance(node.iter, ast.Name) and node.iter.id == "GEOMETRY_SHA":
            need(
                isinstance(node.target, ast.Name) and node.target.id == "relative",
                "Unexpected raw dictionary loop",
            )
            assignment = ast.parse("expected = GEOMETRY_SHA[relative]").body[0]
            matches = [
                index
                for index, item in enumerate(node.body)
                if ast.dump(item) == ast.dump(assignment)
            ]
            need(len(matches) == 1, "Exact dictionary lookup required")
            del node.body[matches[0]]
            node.target = ast.Tuple(
                [ast.Name("relative", ast.Store()), ast.Name("expected", ast.Store())],
                ast.Store(),
            )
            node.iter = ast.Call(
                ast.Attribute(
                    ast.Name("GEOMETRY_SHA", ast.Load()), "items", ast.Load()
                ),
                [],
                [],
            )
        return node


def ast_checks(raw, archive):
    result = {}
    for name in ["prepare81_candidate_codes.py", "assemble81_models.py"]:
        before = ast.parse((raw / "scripts" / name).read_text())
        after = ast.parse((archive / "scripts" / name).read_text())
        need(
            ast.dump(Normalize().visit(before)) == ast.dump(Normalize().visit(after)),
            "Unexpected archive semantics: " + name,
        )
        result[name] = {"equivalent_after_narrow_normalization": True}
    return result


def execute(out, name, argv, env):
    save(out / "logs" / (name + "_command.json"), {"argv": argv, "cwd": str(out)})
    started = time.time()
    with (out / "logs" / (name + ".log")).open("x") as stream:
        result = subprocess.run(
            argv, cwd=out, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False
        )
    record = {
        "argv": argv,
        "returncode": result.returncode,
        "elapsed_seconds": time.time() - started,
    }
    save(out / "logs" / (name + "_result.json"), record)
    need(result.returncode == 0, "Replay phase failed: " + name)
    logging.info("Candidate replay %s passed", name)
    return record


def prepare(raw, archive, out):
    original = raw / "path81_candidates_v1"
    manifest = json.loads((original / "manifest.json").read_text())
    need(
        sha(original / "manifest.json")
        == "4b72ae11c27a6dac17642771c0753a684467896f94bb77d37e34afb1ee965868",
        "Frozen successful candidate manifest",
    )
    need(
        manifest["passed"] and manifest["geometry_only"],
        "Successful geometry-only candidate proof",
    )
    protected = {str(original / "manifest.json"): sha(original / "manifest.json")}
    for name, digest in manifest["artifacts_sha256"].items():
        need(sha(original / name) == digest, "Original artifact changed")
        protected[str(original / name)] = digest
    for name, digest in manifest["source_sha256"].items():
        need(
            sha(raw / "scripts" / name) == digest, "Original successful source changed"
        )
        protected[str(raw / "scripts" / name)] = digest
    source_names = list(manifest["source_sha256"])
    (out / "source").mkdir()
    sources = {}
    for name in source_names:
        source = archive / "scripts" / name
        sources[str(source)] = sha(source)
        shutil.copyfile(source, out / "source" / name)
    (out / "variants").symlink_to(raw / "variants", target_is_directory=True)
    (out / "lib").symlink_to(raw / "lib", target_is_directory=True)
    return original, manifest, protected, sources


def main():  # noqa: PLR0915 -- Ordered archive evidence gates prevent downstream use before comparison.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw, archive, out = args.root.resolve(), args.archive.resolve(), args.out.resolve()
    need(
        out.parent == raw and out.name.startswith("replay81_candidates_formatted_"),
        "Exclusive replay output",
    )
    out.mkdir(exist_ok=False)
    (out / "logs").mkdir()
    commands = []
    started = time.time()
    try:
        ast_result = ast_checks(raw, archive)
        original, manifest, protected, sources = prepare(raw, archive, out)
        spec = importlib.util.spec_from_file_location(
            "formatted_candidate81", out / "source/prepare81_candidate_codes.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        protocol = Path(manifest["protocol_path"])
        centers_path = Path(manifest["centers_source_path"])
        for path, digest in [
            (protocol, manifest["protocol_sha256"]),
            (centers_path, manifest["centers_source_sha256"]),
        ]:
            need(sha(path) == digest, "Frozen protocol/center IDs changed")
            protected[str(path)] = digest
        for relative, digest in module.GEOMETRY_SHA.items():
            need(sha(raw / relative) == digest, "Frozen geometry hash changed")
            protected[str(raw / relative)] = digest
        old_inputs = json.loads((raw / "path80_native_v1/inputs.json").read_text())
        for path, digest in old_inputs["libraries_sha256"].items():
            need(sha(path) == digest, "Native library changed")
            protected[path] = digest
        frozen = raw / "variants/trace68_native/build_source"
        for path, digest in old_inputs["frozen_source_sha256"].items():
            need(sha(frozen / path) == digest, "Frozen production source changed")
        save(
            out / "plan.json",
            {
                "source_sha256": sources,
                "protected_inputs": [
                    {"path": path, "sha256": digest}
                    for path, digest in protected.items()
                ],
                "restricted_ast": ast_result,
                "query_label_or_validation_data_read": False,
            },
        )
        bank = out / "candidate_bank"
        bank.mkdir()
        geometry = module.geometry_tables(raw, manifest["center_order"], bank)
        need(
            geometry == manifest["exact_geometry"],
            "Full exact geometry summary changed",
        )
        for path in bank.iterdir():
            need(
                sha(path) == manifest["artifacts_sha256"][path.name],
                "Geometry table differs: " + path.name,
            )
        logging.info("All exact geometry tables reproduced")
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
        for phase in ["compile", "link"]:
            argv = json.loads(
                (original / "logs" / (phase + "_command.json")).read_text()
            )["argv"]
            replacements = {
                str(original / "source/prepare81_candidate_codes.cc"): str(
                    out / "source/prepare81_candidate_codes.cc"
                ),
                str(original / "source"): str(out / "source"),
                str(original / "prepare81.o"): str(out / "prepare81.o"),
                str(original / "prepare81_candidate_codes"): str(
                    out / "prepare81_candidate_codes"
                ),
                f"-Wl,--dependency-file={original}/prepare81.link.d": f"-Wl,--dependency-file={out}/prepare81.link.d",
            }
            argv = [replacements.get(value, value) for value in argv]
            commands.append(execute(out, phase, argv, env))
        commands.append(
            execute(
                out,
                "encode",
                [str(out / "prepare81_candidate_codes"), str(raw), str(bank)],
                env,
            )
        )
        for name, digest in manifest["artifacts_sha256"].items():
            need(
                sha(bank / name) == digest,
                "Formatted candidate output differs: " + name,
            )
        new_manifest = {
            **manifest,
            "binary_path": str(out / "prepare81_candidate_codes"),
            "binary_sha256": sha(out / "prepare81_candidate_codes"),
            "source_sha256": {
                Path(path).name: digest for path, digest in sources.items()
            },
            "commands": commands,
            "elapsed_seconds": time.time() - started,
            "formatted_replay": True,
            "original_manifest_sha256": sha(original / "manifest.json"),
        }
        save(bank / "manifest.json", new_manifest)
        control = raw / "path81_assembly_control_v1"
        indices = json.loads((control / "indices.json").read_text())
        indices["candidate_manifest_sha256"] = sha(bank / "manifest.json")
        save(out / "control_indices.json", indices)
        commands.append(
            execute(
                out,
                "assemble",
                [
                    "python3",
                    str(out / "source/assemble81_models.py"),
                    "--candidate-manifest",
                    str(bank / "manifest.json"),
                    "--indices",
                    str(out / "control_indices.json"),
                    "--output",
                    str(out / "assembled"),
                ],
                env,
            )
        )
        assembly_hashes = {}
        for name in [
            "selected_models.bin",
            "selected_indices.u32",
            "assembly_result.json",
        ]:
            assembly_hashes[name] = sha(out / "assembled" / name)
            need(
                assembly_hashes[name] == sha(control / "assembled" / name),
                "Formatted assembler output differs: " + name,
            )
        for path, digest in protected.items():
            need(sha(path) == digest, "Original input changed during replay")
        for path, digest in sources.items():
            need(sha(path) == digest, "Archive source changed during replay")
        save(
            out / "stage39_candidates_replay.json",
            {
                "passed": True,
                "scope": "full geometry and encoding replay only; no queries, labels, validation or training",
                "restricted_ast": ast_result,
                "candidate_artifact_count": len(manifest["artifacts_sha256"]),
                "all_candidate_binary_and_json_artifacts_byte_identical": True,
                "artifacts_sha256": manifest["artifacts_sha256"],
                "assemblies_byte_identical": assembly_hashes,
                "exact_geometry_identical": True,
                "native_and_candidate_assemblies_verified": new_manifest[
                    "native_and_candidate_assemblies_verified"
                ],
                "source_sha256": sources,
                "raw_sources_and_artifacts_unchanged": True,
                "commands": commands,
                "elapsed_seconds": time.time() - started,
            },
        )
        logging.info(
            "CANDIDATE REPLAY COMPLETE %s", out / "stage39_candidates_replay.json"
        )
    except BaseException as error:
        save(
            out / "failure.json",
            {"passed": False, "error": repr(error), "commands": commands},
        )
        raise


if __name__ == "__main__":
    main()
