"""Generate frozen phase81 geometry candidates; never read queries or labels."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import math
import mmap
import os
import shutil
import struct
import subprocess
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
D, K, NC, NQ, N = 768, 32, 20, 137, 1000000
PROTOCOL_SHA = "ced04e30f808f016156bacdabfd89ad27ddd238e8388a4ba798457b4eb849f29"
GEOMETRY_SHA = {
    "models64/full_graph.bin": "c04308b5445a666026bd460100ef0a649fd3dfd7e75a1b57561ab40a1e7dc53b",
    "models64/full_vectors.f32": "5c9d8196d5c46d617c5cb3d9eedae5f9b8bf1677e370f210c5b84b08cdd66d24",
    "models77/rotated.f32": "69787cd3b97ebc844a5547d76c10fdaca14a103f62d6fd3fe62a67caccf18f22",
}


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


def binary(path, values, code):
    with path.open("xb") as stream:
        stream.write(struct.pack("<" + str(len(values)) + code, *values))


def lattice(bits):
    """Exact finite FP32 coordinate in integer units of 2**-149."""
    exponent, mantissa = (bits >> 23) & 255, bits & ((1 << 23) - 1)
    need(exponent != 255, "Nonfinite rotated coordinate")
    value = mantissa if exponent == 0 else ((1 << 23) | mantissa) << (exponent - 1)
    return -value if bits >> 31 else value


def geometry_tables(root, centers, out):  # noqa: PLR0915 -- Preserve deterministic geometry and audit output order.
    combos = (
        [()] + [(i,) for i in range(16)] + list(itertools.combinations(range(16), 2))
    )
    need(len(combos) == NQ, "137 candidate enumeration")
    ids, degrees, neighbors, coords, ranked, signs, exact = [], [], [], [], [], [], []
    valid = bytearray(NC * NQ * K)
    masses = [0.0] * (NC * NQ * K)
    center_keys, neighbor_keys = [], []
    short, zero = 0, 0
    with (
        (root / "models64/full_graph.bin").open("rb") as gf,
        (root / "models77/rotated.f32").open("rb") as vf,
        mmap.mmap(gf.fileno(), 0, access=mmap.ACCESS_READ) as graph,
        mmap.mmap(vf.fileno(), 0, access=mmap.ACCESS_READ) as vectors,
    ):
        header = struct.unpack_from("<8I", graph)
        need(header[:5] == (0x51474764, 1, N, D, K), "Frozen graph header")
        need(
            len(graph) == 32 + N * 140 and len(vectors) == N * D * 4,
            "Geometry file size",
        )
        for ci, center in enumerate(centers):
            key, degree = struct.unpack_from("<QI", graph, 32 + center * 140)
            need(degree <= K, "Degree range")
            ids.append(center)
            degrees.append(degree)
            center_keys.append(key)
            short += degree != K
            target_ids = struct.unpack_from("<32I", graph, 32 + center * 140 + 12)
            source = [
                lattice(x) for x in struct.unpack_from("<768I", vectors, center * D * 4)
            ]
            for lane in range(K):
                id_ = target_ids[lane] if lane < degree else 0xFFFFFFFF
                neighbors.append(id_)
                if lane < degree:
                    need(id_ < N and id_ != center, "Neighbor identity")
                    neighbor_keys.append(
                        struct.unpack_from("<Q", graph, 32 + id_ * 140)[0]
                    )
                    target = [
                        lattice(x)
                        for x in struct.unpack_from("<768I", vectors, id_ * D * 4)
                    ]
                    residual = [v - c for v, c in zip(target, source, strict=True)]
                else:
                    neighbor_keys.append(0xFFFFFFFFFFFFFFFF)
                    residual = [0] * D
                absolute = list(map(abs, residual))
                total = sum(absolute)
                selection = sorted(
                    range(D), key=lambda index: (absolute[index], index)
                )[:16]
                enumeration = sorted(selection)
                ranked.extend(selection)
                coords.extend(enumeration)
                signs.extend(1 if value > 0 else -1 for value in residual)
                if lane < degree and total == 0:
                    zero += 1
                for choice, combination in enumerate(combos):
                    flipped = sum(absolute[enumeration[index]] for index in combination)
                    index = (ci * NQ + choice) * K + lane
                    masses[index] = math.ldexp(float(flipped), -149)
                    valid[index] = int(
                        choice == 0
                        or (degree == K and total > 0 and 102 * flipped <= total)
                    )
                exact.append(
                    {
                        "center_row": ci,
                        "center": center,
                        "lane": lane,
                        "target": id_,
                        "absolute_residual_unit": "2**-149",
                        "sum_abs_integer": str(total),
                        "selected_abs_in_coordinate_order": [
                            str(absolute[index]) for index in enumeration
                        ],
                        "zero_residual": total == 0,
                        "degree": degree,
                    }
                )
    need(
        short == 0, "Frozen protocol requires all 20 centers degree32; no substitution"
    )
    binary(out / "centers.u32", ids, "I")
    binary(out / "degrees.u32", degrees, "I")
    binary(out / "neighbors.u32", neighbors, "I")
    binary(out / "center_keys.u64", center_keys, "Q")
    binary(out / "neighbor_keys.u64", neighbor_keys, "Q")
    binary(out / "coords.u16", coords, "H")
    binary(out / "ranked_coords.u16", ranked, "H")
    binary(out / "native_signs.i8", signs, "b")
    binary(
        out / "flip_local.u16",
        [
            value
            for combo in combos
            for value in [*combo, *([65535] * (2 - len(combo)))]
        ],
        "H",
    )
    binary(out / "flipped_mass.f64", masses, "d")
    with (out / "valid.u8").open("xb") as stream:
        stream.write(valid)
    save(
        out / "exact_budget.json",
        {
            "arithmetic": "integers on exact FP32 coordinate lattice 2**-149; no tolerance",
            "test": "102*flipped_mass<=sum_abs",
            "rows": exact,
        },
    )
    return {
        "short_blocks": short,
        "zero_residual_edges": zero,
        "valid_lane_candidates": sum(valid),
        "invalid_lane_candidates": len(valid) - sum(valid),
        "native_candidates_valid": NC * K,
        "candidate_order": [list(combo) for combo in combos],
        "coordinates_use_only_frozen_geometry": True,
    }


def prepare_sources(root, out):
    old = root / "path80_native_v1"
    inputs = json.loads((old / "inputs.json").read_text())
    proof = json.loads((old / "proof.json").read_text())
    need(
        proof["passed"] and proof["all_queries_native_gate"],
        "Accepted phase80 native codec proof required",
    )
    source_names = [
        "prepare81_candidate_codes.py",
        "prepare81_candidate_codes.cc",
        "candidate81_codec.h",
        "assemble81_models.py",
        "path80_score.cc",
        "coupled73_codec.h",
        "independent78_import.h",
    ]
    copied = {}
    (out / "source").mkdir()
    for name in source_names:
        source = root / "scripts" / name
        copied[name] = sha(source)
        shutil.copyfile(source, out / "source" / name)
    need(
        copied["path80_score.cc"]
        == inputs["source_sha256"][str(root / "scripts/path80_score.cc")],
        "Accepted source codec changed",
    )
    need(
        copied["coupled73_codec.h"]
        == "0028835018024073f6844db55b4dfa7a7f425f67cf3ff48399df2fa124dbc5aa",
        "Accepted phase77 sign encoder changed",
    )
    frozen = root / "variants/trace68_native/build_source"
    for name, expected in inputs["frozen_source_sha256"].items():
        need(
            sha(frozen / name) == expected,
            "Frozen production compilation source changed",
        )
    for name, expected in inputs["libraries_sha256"].items():
        need(sha(name) == expected, "Accepted production link library changed")
    return copied, inputs


def commands(root, out):
    old = root / "path80_native_v1"
    compile_argv = json.loads((old / "logs/compile_command.json").read_text())["argv"]
    replacements = {
        str(old / "source/path80_score.cc"): str(
            out / "source/prepare81_candidate_codes.cc"
        ),
        str(old / "path80_score.o"): str(out / "prepare81.o"),
        str(old / "source"): str(out / "source"),
    }
    compile_argv = [replacements.get(value, value) for value in compile_argv]
    link_argv = json.loads((old / "logs/link_command.json").read_text())["argv"]
    replacements.update(
        {
            str(old / "path80_score"): str(out / "prepare81_candidate_codes"),
            f"-Wl,--dependency-file={old}/path80.link.d": f"-Wl,--dependency-file={out}/prepare81.link.d",
        }
    )
    link_argv = [replacements.get(value, value) for value in link_argv]
    return [
        ("compile", compile_argv),
        ("link", link_argv),
        ("encode", [str(out / "prepare81_candidate_codes"), str(root), str(out)]),
    ]


def execute(out, phase, argv, env):
    save(out / "logs" / (phase + "_command.json"), {"argv": argv, "cwd": str(out)})
    begin = time.time()
    with (out / "logs" / (phase + ".log")).open("x") as stream:
        result = subprocess.run(
            argv, cwd=out, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False
        )
    record = {
        "argv": argv,
        "returncode": result.returncode,
        "elapsed_seconds": time.time() - begin,
    }
    save(out / "logs" / (phase + "_result.json"), record)
    need(result.returncode == 0, f"{phase} failed; inspect {out}/logs/{phase}.log")
    logging.info("Prepared81 %s passed", phase)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent.parent
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root, protocol_path, out = (
        args.root.resolve(),
        args.protocol.resolve(),
        args.out.resolve(),
    )
    need(sha(protocol_path) == PROTOCOL_SHA, "Expected frozen phase81 protocol")
    config = json.loads(protocol_path.read_text())
    need(
        config["frozen"]
        and config["candidate"]["tau_numerator"] == 51
        and config["candidate"]["tau_denominator"] == 50,
        "Frozen candidate protocol",
    )
    centers_path = Path(config["preflight"]["root"]) / "centers.json"
    need(
        sha(centers_path) == config["preflight"]["files_sha256"]["centers.json"],
        "Frozen center order",
    )
    centers = [row["center"] for row in json.loads(centers_path.read_text())["centers"]]
    need(
        centers == config["preflight"]["center_order"]
        and len(centers) == NC
        and len(set(centers)) == NC,
        "20 distinct frozen centers",
    )
    need(
        out.parent == root and out.name.startswith("path81_candidates_"),
        "Exclusive candidate output under bench root",
    )
    out.mkdir(exist_ok=False)
    (out / "logs").mkdir()
    started = time.time()
    try:
        protected = {
            str(protocol_path): PROTOCOL_SHA,
            str(centers_path): sha(centers_path),
        }
        # The accepted phase80 proof identifies input vector and graph files without reading its trace.
        accepted_inputs = json.loads(
            (root / "path80_native_v1/inputs.json").read_text()
        )["inputs_sha256"]
        for relative, expected in GEOMETRY_SHA.items():
            path = root / relative
            need(
                accepted_inputs[relative] == expected,
                "Accepted native proof input differs",
            )
            need(sha(path) == expected, "Frozen raw/rotation/graph hash mismatch")
            protected[str(path)] = expected
        source_hashes, native_inputs = prepare_sources(root, out)
        save(
            out / "plan.json",
            {
                "protocol_sha256": PROTOCOL_SHA,
                "centers_source_sha256": sha(centers_path),
                "protected_input_sha256": [
                    {"path": path, "sha256": digest}
                    for path, digest in protected.items()
                ],
                "source_sha256": source_hashes,
                "geometry_only": True,
                "query_or_label_files_opened": False,
            },
        )
        geometry = geometry_tables(root, centers, out)
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("QG_", "BENCH_", "GTEST_"))
            and key
            not in {"LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH", "PYTHONOPTIMIZE"}
        }
        env.update(
            LD_LIBRARY_PATH=f"{root}/variants/trace68_native:{root}/lib",
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
        )
        execution = [
            execute(out, phase, argv, env) for phase, argv in commands(root, out)
        ]
        proof = json.loads((out / "encoding_proof.json").read_text())
        need(
            proof["passed"]
            and proof["native_blocks_bit_identical"]
            and proof["short_blocks_native_only"] == 0,
            "Actual codec bit proof",
        )
        need(
            proof["invalid_non_native_lane_candidates"]
            == geometry["invalid_lane_candidates"],
            "Exact mask/encoder fallback count",
        )
        for path, expected in protected.items():
            need(
                sha(path) == expected, "Geometry or protocol changed during generation"
            )
        for name, expected in source_hashes.items():
            need(
                sha(root / "scripts" / name) == expected,
                "Generation source changed during run",
            )
        for name, expected in native_inputs["libraries_sha256"].items():
            need(sha(name) == expected, "Linked library changed during generation")
        names = [
            "centers.u32",
            "degrees.u32",
            "neighbors.u32",
            "center_keys.u64",
            "neighbor_keys.u64",
            "coords.u16",
            "ranked_coords.u16",
            "native_signs.i8",
            "flip_local.u16",
            "flipped_mass.f64",
            "valid.u8",
            "exact_budget.json",
            "candidates.bin",
            "native_blocks.bin",
            "candidate_compact.u8",
            "geometry.f64",
            "residuals.f64",
            "native_norms.f32",
            "encoding_proof.json",
        ]
        save(
            out / "manifest.json",
            {
                **proof,
                "protocol_path": str(protocol_path),
                "protocol_sha256": PROTOCOL_SHA,
                "centers_source_path": str(centers_path),
                "centers_source_sha256": sha(centers_path),
                "center_order": centers,
                "artifacts_sha256": {name: sha(out / name) for name in names},
                "source_sha256": source_hashes,
                "binary_sha256": sha(out / "prepare81_candidate_codes"),
                "binary_path": str(out / "prepare81_candidate_codes"),
                "exact_geometry": geometry,
                "commands": execution,
                "elapsed_seconds": time.time() - started,
                "schema": {
                    "endianness": "little",
                    "candidates.bin": {
                        "header_u32": [
                            0x51474381,
                            1,
                            64,
                            D,
                            NC,
                            NQ,
                            K,
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
                        "payload_shape": [NC, NQ, 3328],
                    },
                    "centers.u32": [NC],
                    "degrees.u32": [NC],
                    "neighbors.u32": [NC, K],
                    "center_keys.u64": [NC],
                    "neighbor_keys.u64": [NC, K],
                    "coords.u16": [NC, K, 16],
                    "ranked_coords.u16": [NC, K, 16],
                    "native_signs.i8": [NC, K, D],
                    "flip_local.u16": [NQ, 2],
                    "flipped_mass.f64": [NC, NQ, K],
                    "valid.u8": [NC, NQ, K],
                    "native_blocks.bin": [NC, 3328],
                    "candidate_compact.u8": [NC, NQ, K, 96],
                    "geometry.f64": {
                        "shape": [NC, K, 8],
                        "fields": [
                            "residual_norm2",
                            "residual_l1",
                            "center_norm64",
                            "neighbor_norm64",
                            "native_ideal_alpha",
                            "native_f_add_as_double",
                            "native_f_rescale_as_double",
                            "budget_l1_over102",
                        ],
                    },
                    "residuals.f64": [NC, K, D],
                    "native_norms.f32": {
                        "shape": [NC, K, 2],
                        "fields": [
                            "actual_codec_FP32_center_norm",
                            "actual_codec_FP32_neighbor_norm",
                        ],
                    },
                },
                "invalid_lane_representation": "native symbols and factors exactly; valid mask still false",
                "native0_representation": "original production codec.encode block; factors never approximated",
                "nonzero_candidates_encoder": "unchanged accepted coupled73::encode",
                "query_or_label_files_opened": False,
            },
        )
        logging.info("PREPARE81 COMPLETE %s", out / "manifest.json")
    except BaseException as error:
        save(out / "failure.json", {"passed": False, "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
