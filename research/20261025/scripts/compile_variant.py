from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")


def emit(*items, **_kwargs):
    logging.info(" ".join(map(str, items)))


R = Path(__file__).resolve().parent.parent
SRC = Path("/home/xingrui/workspace/quantized_graph/symphony_research_20261025")
BASELINE = "5a1c8a9852acd34c16114940b4cfc41e157dd9b7"
name = sys.argv[1]
defines = sys.argv[2:]
out = R / "variants" / name
out.mkdir(parents=True, exist_ok=True)
if (out / "commands.json").exists():
    raise SystemExit("Variant already frozen; use a new name: " + name)
# Compile from an immutable per-variant tree. Later research edits must not
# race with queued compiler reads or change recorded source provenance.
patch = subprocess.check_output(["git", "diff", BASELINE, "--", "src"], cwd=SRC)
changed = subprocess.check_output(
    ["git", "diff", "--name-only", BASELINE, "--", "src"], cwd=SRC, text=True
).splitlines()
untracked = subprocess.check_output(
    ["git", "ls-files", "--others", "--exclude-standard"], cwd=SRC, text=True
).splitlines()
frozen = out / "build_source"
shutil.copytree(SRC / "src", frozen / "src", dirs_exist_ok=True)
frozen_hashes = {
    str(f.relative_to(frozen)): hashlib.sha256(f.read_bytes()).hexdigest()
    for f in (frozen / "src").rglob("*")
    if f.is_file()
}
for relative, digest in frozen_hashes.items():
    if hashlib.sha256((SRC / relative).read_bytes()).hexdigest() != digest:
        raise SystemExit("Source changed while freezing; retry with new variant name")
if subprocess.check_output(["git", "diff", BASELINE, "--", "src"], cwd=SRC) != patch:
    raise SystemExit("Patch changed while freezing; retry with new variant name")
(out / "build_source_sha256.json").write_text(json.dumps(frozen_hashes, indent=2))

allcmd = json.loads((R / "results/build_commands.json").read_text())
plan = json.loads((R / "results/all_compile_commands.json").read_text())
executed = []


def run(a):
    if "-c" in a:
        a = [x.replace(str(SRC / "src"), str(frozen / "src")) for x in a]
        for i, x in enumerate(a):
            if x.startswith(str(R / "scripts") + "/") and x.endswith(".cc"):
                target = frozen / Path(x).name
                shutil.copy2(x, target)
                a[i] = str(target)
    executed.append(a)
    subprocess.run(a, cwd=R, check=True)


if name in [
    "driver",
    "driver_gist",
    "driver_codec",
    "driver_nav",
    "driver_router",
    "driver_gap",
]:
    oldobj = R / "objects/benchmark.o"
    newobj = out / "benchmark.o"
    c = next(
        x["argv"]
        for x in allcmd
        if "-c" in x["argv"] and str(R / "scripts/benchmark.cc") in x["argv"]
    )
    c = [
        str(newobj)
        if x == str(oldobj)
        else str(R / "scripts/benchmark_gist.cc")
        if name == "driver_gist" and x == str(R / "scripts/benchmark.cc")
        else str(R / "scripts/benchmark_gap.cc")
        if name == "driver_gap" and x == str(R / "scripts/benchmark.cc")
        else str(R / "scripts/benchmark_router.cc")
        if name == "driver_router" and x == str(R / "scripts/benchmark.cc")
        else str(R / "scripts/benchmark_nav.cc")
        if name == "driver_nav" and x == str(R / "scripts/benchmark.cc")
        else str(R / "scripts/codec_geometry_test.cc")
        if name == "driver_codec" and x == str(R / "scripts/benchmark.cc")
        else x
        for x in c
    ]
    c += ["-D" + x for x in defines]
    run(c)
    c = next(
        x["argv"] for x in allcmd if "-c" not in x["argv"] and str(oldobj) in x["argv"]
    )
    c = [
        str(newobj)
        if x == str(oldobj)
        else str(
            R
            / (
                "research_benchmark_gap"
                if name == "driver_gap"
                else "research_benchmark_router"
                if name == "driver_router"
                else "research_benchmark_nav"
                if name == "driver_nav"
                else "research_benchmark_gist"
                if name == "driver_gist"
                else "codec_geometry_test"
                if name == "driver_codec"
                else "research_benchmark"
            )
        )
        if x == str(R / "benchmark")
        else x
        for x in c
    ]
    run(c)
else:
    c = next(
        x["argv"]
        for x in plan
        if any(z.endswith("/hnsw_quantized_graph.cc") for z in x["argv"])
    )
    c = list(c)
    oldobj = c[c.index("-o") + 1]
    newobj = str(out / "hnsw_quantized_graph.o")
    c[c.index("-o") + 1] = newobj
    c += ["-D" + x for x in defines]
    run(c)
    c = next(x["argv"] for x in allcmd if "-c" not in x["argv"] and oldobj in x["argv"])
    c = [
        newobj
        if x == oldobj
        else str(out / "libcore_knn_hnsw.so")
        if x == str(R / "lib/libcore_knn_hnsw.so")
        else x
        for x in c
    ]
    run(c)
(out / "commands.json").write_text(json.dumps(executed, indent=2))
(out / "source.patch").write_bytes(patch)
(out / "metadata.json").write_text(
    json.dumps(
        {
            "name": name,
            "defines": defines,
            "baseline": "5a1c8a9852acd34c16114940b4cfc41e157dd9b7",
            "patch_sha256": hashlib.sha256(patch).hexdigest(),
            "immutable_build_source": str(frozen),
            "research_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=SRC, text=True
            ).strip(),
        },
        indent=2,
    )
)
emit("COMPILED", name, flush=True)
# Include modified/untracked source files from the same frozen tree.
hashes = {}
for relative in sorted(set(changed + untracked)):
    srcfile = frozen / relative
    if not srcfile.is_file():
        continue
    target = out / "source_files" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(srcfile, target)
    hashes[relative] = hashlib.sha256(srcfile.read_bytes()).hexdigest()
(out / "source_files_sha256.json").write_text(json.dumps(hashes, indent=2))
