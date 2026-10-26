from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
S = Path("/home/xingrui/workspace/quantized_graph/symphony_research_20261025")
a = json.loads((R / "results/fusion69_analysis.json").read_text())
assert a["new_timing_rows"] == 22
assert "GUARDS69 PASS 12" in (R / "logs/guards69.log").read_text()
rows = []
for key, label in [
    ("baseline", "冻结基线 bucket90_w16"),
    ("disabled", "相同元数据/缓存\uff0c融合关闭"),
    ("enabled", "两估计融合开启"),
]:
    t = a["timings"][key]
    rows.append(
        f"| {label} | {t['1']['recall'][0] * 100:.2f}% | {t['1']['qps_median']:,.2f} | {t['8']['qps_median']:,.2f} | {t['1']['rss_gib_median']:.4f} |"
    )
table = "\n".join(rows)
report = (
    (Path(__file__).resolve().parent.parent / "notes/fusion69_report_template_1.md")
    .read_text()
    .replace("TABLE", table)
)
(R / "notes/multianchor69.md").write_text(report)
p = R / "cohere90_breakthrough_research.md"
s = p.read_text()
assert "## 阶段69\uff1a" not in s
s += (
    Path(__file__).resolve().parent.parent / "notes/fusion69_report_template_2.md"
).read_text()
p.write_text(s)
inc = "-I" + str(S / "src")
third = "-isystem"
lib = "/home/xingrui/workspace/zvec/symphony_qg/zvec/thirdparty/RaBitQ-Library/RaBitQ-Library-0.1/include"
commands = {
    "unit_compile": [
        "c++",
        "-std=c++17",
        "-O2",
        "-UNDEBUG",
        inc,
        str(R / "scripts/fusion69_test.cc"),
        "-o",
        str(R / "fusion69_test"),
    ],
    "unit_run": [str(R / "fusion69_test")],
    "geometry_compile": [
        "c++",
        "-O3",
        "-std=c++17",
        "-march=native",
        "-pthread",
        "-DRABITQ_SUPPORTED=1",
        "-DQG_ROTATION_768=1",
        inc,
        third,
        lib,
        "scripts/fusion69_geometry.cc",
        "-o",
        "fusion69_geometry",
    ],
    "geometry_run": [
        "./fusion69_geometry",
        "models64/full_vectors.f32",
        "models68/geometry/ids.u32",
        "models68/geometry/moments.f64",
        "models69/geometry.f64",
    ],
    "cwd": str(R),
    "numpy_python": "/home/xingrui/bench/symphony_qg/.venv/bin/python",
    "scripts_run": [
        "setup69.py",
        "run69.py",
        "screen69.py",
        "time69.py",
        "guards69.py",
        "analyze69.py",
        "finalize69.py",
    ],
}
(R / "results/fusion69_validation_commands.json").write_text(
    json.dumps(commands, indent=2) + "\n"
)
paths = [
    *list((R / "scripts").glob("*69*")),
    *list((R / "models69").glob("*")),
    *list((R / "results").glob("fusion69*.json")),
    *list((R / "results").glob("stage69*")),
    R / "fusion69_test",
    R / "fusion69_geometry",
    R / "trace68_driver",
]
for name in ["fusion69_native", "fusion69_audit", "stage69_default"]:
    paths += [
        R / f"variants/{name}" / x
        for x in [
            "metadata.json",
            "commands.json",
            "source_files_sha256.json",
            "libcore_knn_hnsw.so",
        ]
    ]
manifest = {
    str(p.relative_to(R)): {
        "bytes": p.stat().st_size,
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
    }
    for p in paths
    if p.is_file() and p.name != "fusion69_artifact_manifest.json"
}
(R / "results/fusion69_artifact_manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n"
)
logging.info("%s %s", "FINALIZE69 COMPLETE", len(manifest))
