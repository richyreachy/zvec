from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
out = R / "models61"
out.mkdir(exist_ok=False)
s = (R / "scripts/benchmark_nav.cc").read_text()
begin = s.index("void search(")
end = s.index("\nint main(", begin)
s = (
    s[:begin]
    + 'void search(const std::string &path, const std::string &method, int, int, const std::string &) {\n  auto index = make_index(method); check(bool(index), "create export index");\n  ci::StorageOptions storage; storage.type = ci::StorageOptions::StorageType::kMMAP; storage.read_only = true;\n  check(index->open(path, storage) == 0, "open export index");\n  check(index->get_doc_count() == N, "export index count");\n  check(index->close() == 0, "close export index");\n  std::cout << "EXPORT61 COMPLETE; no query or GT objects constructed" << std::endl;\n}\n'
    + s[end:]
)
source = R / "scripts/edge61_export_driver.cc"
source.write_text(s)
flags = json.loads((R / "variants/int55_original/metadata.json").read_text())[
    "defines"
] + ["QG_EDGE_TRAIN_EXPORT=1"]
subprocess.run(
    ["python3", str(R / "scripts/compile_variant.py"), "edge61_export", *flags],
    check=True,
)
commands = json.loads((R / "results/build_commands.json").read_text())
oldobj = str(R / "objects/benchmark.o")
newobj = str(R / "objects/edge61_export.o")
exe = str(R / "edge61_export")
executed = []
c = next(x["argv"] for x in commands if "-c" in x["argv"] and oldobj in x["argv"])
c = [
    newobj
    if x == oldobj
    else str(source)
    if x == str(R / "scripts/benchmark.cc")
    else x
    for x in c
]
executed.append(c)
subprocess.run(c, check=True)
c = next(x["argv"] for x in commands if "-c" not in x["argv"] and oldobj in x["argv"])
c = [newobj if x == oldobj else exe if x == str(R / "benchmark") else x for x in c]
executed.append(c)
subprocess.run(c, check=True)
env = {
    "LD_LIBRARY_PATH": str(R / "variants/edge61_export") + ":" + str(R / "lib"),
    "QG_CENTER_MEAN_FILE": str(R / "mean54/mean.f32"),
    "QG_EDGE_TRAIN_EXPORT_FILE": str(out / "edges.bin"),
    "BENCH_DEGREE": "32",
}
cmd = [
    "taskset",
    "-c",
    "0,2,4,6,8,10,12,14",
    exe,
    "search",
    str(R / "indexes/cohere1m_m50_efc500.index"),
    "symphony_qg",
    "0",
    "-1",
    str(out / "unused"),
]
(R / "results/edge61_export_commands.json").write_text(
    json.dumps({"build_commands": executed, "run": cmd, "env": env}, indent=2)
)
subprocess.run(cmd, env={**os.environ, **env}, check=True)
assert (out / "edges.bin").stat().st_size > 200000000
logging.info("%s", "EXPORT61 DRIVER COMPLETE")
