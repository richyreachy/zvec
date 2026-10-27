from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
flags = json.loads((R / "variants/bucket90_w16/metadata.json").read_text())[
    "defines"
] + ["QG_FUSION=1"]
for name, extra in [
    ("fusion69_native", []),
    ("fusion69_audit", ["QG_FUSION_AUDIT=1", "QG_DIAG=1"]),
]:
    with (R / f"logs/build_{name}.log").open("w") as f:
        subprocess.run(
            ["python3", str(R / "scripts/compile_variant.py"), name, *flags, *extra],
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )
    logging.info("%s %s", "BUILT", name)
subprocess.run(["python3", str(R / "scripts/validate_stage69_default.py")], check=True)
logging.info("%s", "SETUP69 COMPLETE")
