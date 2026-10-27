"""Build isolated post-merge variants and run the frozen Cohere protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[3]
ROOT = Path("/home/xingrui/bench/quantized_graph/research_20261025")
BUILD = Path("/home/xingrui/bench/quantized_graph/merge_main_20261027/build")
PHASE = ROOT / "postmerge84_v1"
DEFINES = [
    "QG_BUCKET=1",
    "QG_BUCKET_WIDTH=.016f",
    "QG_ADAPTIVE=1",
    "QG_RUNTIME_ADAPTIVE=1",
    "QG_STOP_MIN=10",
    "QG_STOP_PATIENCE=0",
    "QG_STOP_MARGIN=.04f",
    "QG_ROTATION_768=1",
    "QG_AUGMENT=3",
]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def build(name, extra):
    out = PHASE / "variants" / name
    out.mkdir(parents=True, exist_ok=False)
    frozen = out / "source"
    shutil.copytree(SOURCE / "src", frozen / "src")
    shutil.copytree(BUILD / "lib", out / "lib")
    driver = frozen / "postmerge84_benchmark.cc"
    shutil.copyfile(Path(__file__).with_name(driver.name), driver)
    # The driver includes the repo's dataset reader, outside src/.
    shutil.copytree(SOURCE / "tools/core", frozen / "tools/core")
    before = {
        str(p.relative_to(frozen)): sha(p) for p in frozen.rglob("*") if p.is_file()
    }
    save(out / "source_sha256.json", before)
    save(
        out / "library_inputs_sha256.json",
        {p.name: sha(p) for p in (out / "lib").iterdir() if p.is_file()},
    )
    entries = json.loads((BUILD / "compile_commands.json").read_text())
    commands = []

    def run(cmd, cwd):
        commands.append({"argv": cmd, "cwd": str(cwd)})
        subprocess.run(cmd, cwd=cwd, check=True)

    def compile_one(suffix, source, obj, defines):
        entry = next(e for e in entries if e["file"].endswith(suffix))
        cmd = shlex.split(entry["command"])
        cmd[cmd.index("-c") + 1] = str(source)
        cmd[cmd.index("-o") + 1] = str(obj)
        cmd = [x.replace(str(SOURCE / "src"), str(frozen / "src")) for x in cmd]
        cmd += ["-I" + str(frozen / "tools/core")]
        cmd += ["-D" + d for d in defines]
        cmd += ["-MMD", "-MF", str(obj) + ".d"]
        run(cmd, out)

    def link(relative, obj, output, old_name):
        recipe = BUILD / relative
        cwd = recipe.parent.parent.parent
        cmd = shlex.split(recipe.read_text())
        for i, item in enumerate(cmd):
            if item.startswith("-Wl,--dependency-file="):
                cmd[i] = "-Wl,--dependency-file=" + str(output) + ".d"
            elif item.startswith("-Wl,-rpath,"):
                cmd[i] = "-Wl,-rpath," + str(out / "lib")
            elif item.endswith(".o"):
                cmd[i] = (
                    str(obj)
                    if item.endswith(old_name + ".cc.o")
                    else str((cwd / item).resolve())
                )
            elif item.endswith((".a", ".so")) and not item.startswith("-"):
                path = (cwd / item).resolve()
                cmd[i] = (
                    str(out / "lib" / path.name)
                    if path.parent == BUILD / "lib"
                    else str(path)
                )
        cmd[cmd.index("-o") + 1] = str(output)
        if old_name == "index_interface_test":
            cmd = [
                x for x in cmd if not x.endswith(("/libgtest.a", "/libgtest_main.a"))
            ]
        run(cmd, cwd)

    compile_one(
        "/hnsw_quantized_graph.cc",
        frozen / "src/core/algorithm/hnsw/hnsw_quantized_graph.cc",
        out / "graph.o",
        DEFINES + extra,
    )
    link(
        "src/core/algorithm/hnsw/CMakeFiles/core_knn_hnsw.dir/link.txt",
        out / "graph.o",
        out / "lib/libcore_knn_hnsw.so",
        "hnsw_quantized_graph",
    )
    compile_one("/index_interface_test.cc", driver, out / "driver.o", [])
    link(
        "tests/core/interface/CMakeFiles/index_interface_test.dir/link.txt",
        out / "driver.o",
        out / "benchmark",
        "index_interface_test",
    )
    assert before == {
        str(p.relative_to(frozen)): sha(p) for p in frozen.rglob("*") if p.is_file()
    }
    save(out / "commands.json", commands)
    save(
        out / "manifest.json",
        {
            "head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True
            ).strip(),
            "defines": DEFINES + extra,
            "thirdparty": subprocess.check_output(
                ["git", "submodule", "status"], cwd=SOURCE, text=True
            ),
            "binary_sha256": sha(out / "benchmark"),
            "libraries_sha256": {
                p.name: sha(p) for p in (out / "lib").iterdir() if p.is_file()
            },
        },
    )


def execute(name, variant, labels, caps, duration, threads):
    out = PHASE / "experiments" / name
    out.mkdir(parents=True, exist_ok=False)
    directory = PHASE / "variants" / variant
    manifest = json.loads((directory / "manifest.json").read_text())
    assert sha(directory / "benchmark") == manifest["binary_sha256"]
    assert all(
        sha(directory / "lib" / name) == digest
        for name, digest in manifest["libraries_sha256"].items()
    )
    env = {
        "LD_LIBRARY_PATH": str(directory / "lib"),
        "BENCH_N": "1000000",
        "BENCH_DATA": "/home/xingrui/bench/dataset/cohere/1m/",
        "BENCH_TOPK": "10",
        "BENCH_DEGREE": "32",
        "BENCH_ROUNDS": "1",
        "BENCH_EFS": labels,
        "BENCH_THREADS": threads,
        "BENCH_QUERY_BEGIN": "0",
        "BENCH_QUERY_END": "1000",
        "BENCH_LABEL": variant,
        "QG_CAP": "64",
        "QG_PATIENCE": "0",
        "BENCH_STOP_GRID": "1",
        "BENCH_MARGIN_DENOM": "1000",
        "BENCH_CAP_GRID": caps,
    }
    command = [
        "taskset",
        "-c",
        "0,2,4,6,8,10,12,14",
        str(directory / "benchmark"),
        "search",
        str(ROOT / "indexes/cohere1m_m50_efc500.index"),
        "symphony_qg",
        "0",
        str(duration),
        str(out / "search"),
    ]
    save(out / "config.json", {"command": command, "env": env})
    with (out / "run.log").open("x") as log:
        subprocess.run(
            command,
            env={**os.environ, **env},
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["build", "run"])
    parser.add_argument("name")
    parser.add_argument("--defines", nargs="*", default=[])
    parser.add_argument("--variant", default="native")
    parser.add_argument("--labels", default="20000")
    parser.add_argument("--caps", default="20000:64")
    parser.add_argument("--duration", type=float, default=-1)
    parser.add_argument("--threads", default="1")
    args = parser.parse_args()
    if args.mode == "build":
        build(args.name, args.defines)
    else:
        execute(
            args.name, args.variant, args.labels, args.caps, args.duration, args.threads
        )


if __name__ == "__main__":
    main()
