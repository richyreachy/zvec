"""Collect actual native/learned traces at frozen margin .020; no QPS timing."""

from __future__ import annotations

import hashlib
import json
import logging
import os

from explore90 import R, run

logging.basicConfig(level=logging.INFO, format="%(message)s")
for key in list(os.environ):
    if key.startswith("QG_INDEPENDENT_") or key == "QG_MULTI_TRACE_FILE":
        del os.environ[key]
out = R / "path80_actual_v1"
out.mkdir(exist_ok=False)
protocol = R / "notes/path80_diagnostic_review.md"
assert protocol.is_file()
variant = "trace80_learned_v1"
meta = json.loads((R / "variants" / variant / "metadata.json").read_text())
expected = json.loads(
    (R / "variants/independent78_native_v1/metadata.json").read_text()
)["defines"]
assert meta["defines"] == [*expected, "QG_MULTI_TRACE=1"]
native_ids = (R / "experiments/time65_control_pre/search_k10_r0_ids.csv").read_bytes()
learned_ids = b"".join(
    line
    for line in (R / "experiments/screen78_independent_v1/search_k10_r0_ids.csv")
    .read_bytes()
    .splitlines(keepends=True)
    if line.startswith(b"20000,")
)
assert len(native_ids.splitlines()) == len(learned_ids.splitlines()) == 10000
model = {
    "QG_INDEPENDENT_CODE_FILE": str(R / "models77/full/blocks.bin"),
    "QG_INDEPENDENT_RAW_FILE": str(R / "models64/full_vectors.f32"),
    "QG_INDEPENDENT_ROTATED_FILE": str(R / "models77/rotated.f32"),
    "QG_INDEPENDENT_HEADER_FILE": str(R / "models77/header1000000.bin"),
    "QG_INDEPENDENT_MODEL_SHA256": "2ac03543d4a09ae655d38130419df41668e4d62bdd3231b49135d2603c39f138",
}
configuration = {
    "variant": meta,
    "protocol_sha256_before_trace": hashlib.sha256(protocol.read_bytes()).hexdigest(),
    "same_margin": 0.020,
    "queries": 1000,
    "no_timing": True,
    "no_training": True,
    "runner_sha256": hashlib.sha256(
        __import__("pathlib").Path(__file__).read_bytes()
    ).hexdigest(),
}
(out / "config.json").write_text(json.dumps(configuration, indent=2) + "\n")
proof = []
for condition in ("native", "learned"):
    folder = out / condition
    folder.mkdir(exist_ok=False)
    extra = {
        "QG_CAP": "64",
        "QG_PATIENCE": "0",
        "BENCH_STOP_GRID": "1",
        "BENCH_MARGIN_DENOM": "1000",
        "QG_MULTI_TRACE_FILE": str(folder / "events.bin"),
    }
    if condition == "learned":
        extra.update(model)
    name = f"diag80_actual_v1_{condition}"
    run(
        name,
        variant,
        [20000],
        extra=extra,
        rounds=1,
        duration=-1,
        threads="1",
        nav="margin",
    )
    experiment = R / "experiments" / name
    actual_ids = (experiment / "search_k10_r0_ids.csv").read_bytes()
    assert actual_ids == (native_ids if condition == "native" else learned_ids)
    events = (folder / "events.bin").read_bytes()
    assert len(events) >= 16
    assert (len(events) - 16) % 32 == 0
    if condition == "native":
        assert events == (R / "models68/primary/events.bin").read_bytes()
    measured = (experiment / "search_k10_r0.csv").read_text().splitlines()
    assert len(measured) == 1
    proof.append(
        {
            "condition": condition,
            "events_file": str(folder / "events.bin"),
            "events_sha256": hashlib.sha256(events).hexdigest(),
            "event_records": (len(events) - 16) // 32,
            "ids_file": str(experiment / "search_k10_r0_ids.csv"),
            "ids_sha256": hashlib.sha256(actual_ids).hexdigest(),
            "top10_id_hex_rows_byte_identical": 10000,
            "native_entire_trace_byte_identical_to_phase68": condition == "native",
        }
    )
    (out / f"{condition}_proof.json").write_text(json.dumps(proof[-1], indent=2) + "\n")
assert (
    configuration["protocol_sha256_before_trace"]
    == hashlib.sha256(protocol.read_bytes()).hexdigest()
)
result = {
    "passed": True,
    "runs": proof,
    "configuration": configuration,
    "new_timing_rows": 0,
    "no_new_model_or_parameter_selection": True,
}
with (R / "results/path80_actual_trace_equivalence_v1.json").open("x") as stream:
    json.dump(result, stream, indent=2)
    stream.write("\n")
logging.info("TRACE80 ACTUAL COMPLETE %s", json.dumps(proof))
