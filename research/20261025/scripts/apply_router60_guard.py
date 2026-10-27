from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO)
R = Path(__file__).resolve().parent.parent
S = Path("/home/xingrui/workspace/quantized_graph/symphony_research_20261025")
assert "SCREEN60 COMPLETE" in (R / "logs/screen60.log").read_text()
p = S / "src/core/algorithm/quantized_graph/quantized_graph.h"
before = p.read_text()
needle = "int QuantizedGraph<Codec>::prebuild(const Graph &entity, size_t doc_cnt,\n                                    size_t threads) {\n"
assert before.count(needle) == 1
added = "#if defined(QG_ROUTER_PROJECTED)\n  // Reject incompatible layouts before touching the projected mean or graph.\n  if (!compressed_center_ || dimension_ != 768) return IndexError_Mismatch;\n#endif\n"
assert added not in before
p.write_text(before.replace(needle, needle + added))
subprocess.run(
    [
        "/home/xingrui/.cache/pre-commit/repo9itypczp/py_env-default/bin/clang-format",
        "-i",
        str(p),
    ],
    check=True,
)
meta = {
    "file": str(p.relative_to(S)),
    "before_sha256": hashlib.sha256(before.encode()).hexdigest(),
    "after_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
    "added_code": added,
    "scope": "prebuild only; timed performance snapshots remain original; subsequent audit/full/default builds include this guard; no online search code changes",
}
(R / "results/router60_prebuild_guard_change.json").write_text(
    json.dumps(meta, indent=2) + "\n"
)
commands = json.loads((R / "results/build_commands.json").read_text())
oldobj = str(R / "objects/benchmark.o")
newobj = str(R / "objects/router60_prebuild_guard_test.o")
executed = []
flags = json.loads((R / "variants/proj60_256/metadata.json").read_text())["defines"]
c = next(x["argv"] for x in commands if "-c" in x["argv"] and oldobj in x["argv"])
c = [
    newobj
    if x == oldobj
    else str(R / "scripts/router60_prebuild_guard_test.cc")
    if x == str(R / "scripts/benchmark.cc")
    else x
    for x in c
]
c += ["-UNDEBUG", *["-D" + f for f in flags]]
executed.append(c)
subprocess.run(c, check=True)
c = next(x["argv"] for x in commands if "-c" not in x["argv"] and oldobj in x["argv"])
c = [
    newobj
    if x == oldobj
    else str(R / "router60_prebuild_guard_test")
    if x == str(R / "benchmark")
    else x
    for x in c
]
executed.append(c)
subprocess.run(c, check=True)
subprocess.run([str(R / "router60_prebuild_guard_test")], check=True)
(R / "results/router60_prebuild_guard_commands.json").write_text(
    json.dumps(executed, indent=2)
)
logging.info("%s", "ROUTER60 PREBUILD GUARD COMPLETE")
