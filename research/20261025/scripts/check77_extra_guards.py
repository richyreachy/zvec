from __future__ import annotations

import hashlib
import json
import logging
import shutil
import struct
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
P = R / "models77/pilot_t8"
base = R / "models77/extra_negative"
base.mkdir(exist_ok=False)
names = ["blocks.partial", "compact.u8", "audit.bin", "history.f64", "moves.u16"]
checks = []
for case, target, offset, fmt, mutate in [
    ("compact", "compact.u8", 0, "B", lambda x: x ^ 1),
    ("moves", "moves.u16", 4, "H", lambda x: (x + 1) % 768),
    ("scale_audit", "audit.bin", 32, "d", lambda x: x + 0.1),
    ("scale_factor", "blocks.partial", 4096 + 272 + 3200, "f", lambda x: x * 0.5),
    ("history", "history.f64", 8, "d", lambda x: x + 1.0),
]:
    out = base / case
    out.mkdir()
    for name in names:
        if name == target:
            shutil.copy2(P / name, out / name)
        else:
            (out / name).symlink_to(P / name)
    with (out / target).open("r+b") as f:
        f.seek(offset)
        v = struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))[0]
        f.seek(offset)
        f.write(struct.pack("<" + fmt, mutate(v)))
    cmd = [
        "./audit77_extra",
        "models64/full_graph.bin",
        "models77/rotated.f32",
        "models77/proxies.u32",
        str(out),
        "blocks.partial",
        "none",
    ]
    result = subprocess.run(cmd, check=False, cwd=R, capture_output=True, text=True)
    assert result.returncode != 0, case
    checks.append(
        {"case": case, "exit_code": result.returncode, "stderr": result.stderr.strip()}
    )
for case, offset, value in [("magic", 0, b"X"), ("zero_padding", 4095, b"X")]:
    p = base / (case + ".bin")
    header = bytearray((P / "blocks.partial").read_bytes()[:4096])
    header[offset : offset + 1] = value
    with p.open("xb") as f:
        f.write(header)
        f.truncate((P / "blocks.partial").stat().st_size)
    result = subprocess.run(
        [sys.executable, "scripts/header77_check.py", str(p)],
        check=False,
        cwd=R,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, case
    checks.append(
        {
            "case": "header_" + case,
            "exit_code": result.returncode,
            "stderr_last_line": result.stderr.strip().splitlines()[-1],
        }
    )
expected = json.loads((R / "results/independent77_pilot.json").read_text())[
    "all_five_outputs_thread_byte_identical"
]
for name in names:
    assert hashlib.sha256((P / name).read_bytes()).hexdigest() == expected[name]
a = {
    "corrupted_cases_rejected": checks,
    "original_pilot_files_unchanged": True,
    "scope": "guard checks for supplemental auditors,not production loader tests",
}
(R / "results/independent77_extra_guard_checks.json").write_text(
    json.dumps(a, indent=2) + "\n"
)
logging.info(json.dumps(a, indent=2))
logging.info("EXTRA77 GUARDS COMPLETE")
