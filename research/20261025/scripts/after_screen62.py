from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
assert "SCREEN62 COMPLETE" in (R / "logs/screen62.log").read_text()
for script in ["recover62", "directional62_guard_checks", "validate62"]:
    with (R / f"logs/{script}.log").open("w") as f:
        subprocess.run(
            ["python3", str(R / f"scripts/{script}.py")],
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )
    logging.info("%s %s", script, "COMPLETE")
logging.info("%s", "AFTER_SCREEN62 COMPLETE")
