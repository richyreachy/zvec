from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")


def emit(*items, **_kwargs):
    logging.info(" ".join(map(str, items)))


R = Path(__file__).resolve().parent.parent


def run(name, variant, efs, extra=None, rounds=1, duration=1, threads="1", nav=False):
    out = R / "experiments" / name
    out.mkdir(exist_ok=False)
    config = json.loads(
        (R / "experiments/confirm90_cohere_native/config.json").read_text()
    )
    env = config["env"]
    env.update(
        LD_LIBRARY_PATH=f"{R}/variants/{variant}:{R}/lib",
        BENCH_EFS=",".join(map(str, efs)),
        BENCH_ROUNDS=str(rounds),
        BENCH_THREADS=threads,
        BENCH_LABEL=variant,
    )
    if nav in [True, "local"]:
        env["BENCH_LOCAL_GRID" if nav == "local" else "BENCH_NAV_GRID"] = "1"
    env.update(extra or {})
    cmd = config["command"]
    cmd[-2] = str(duration)
    cmd[-1] = str(out / "search")
    if nav:
        cmd[3] = str(
            R
            / (
                "research_benchmark_gap"
                if nav == "gap"
                else "research_benchmark_router"
                if nav == "route"
                else "research_benchmark_nav"
            )
        )
    (out / "config.json").write_text(json.dumps({"command": cmd, "env": env}, indent=2))
    emit("START", name, flush=True)
    with (out / "run.log").open("w") as f:
        subprocess.run(
            cmd,
            env={**os.environ, **env},
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )
    emit("DONE", name, flush=True)


if __name__ == "__main__":
    run(
        "screen21_cohere_tail",
        "tail90",
        [
            64 * 1000000 + cap * 1000 + tail
            for cap in [12, 16, 20, 24, 28]
            for tail in [0, 8, 16, 32]
        ],
        nav=True,
    )
