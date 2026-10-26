"""Nontrivial fixed-rule score perturbation preserving every FP32 bucket ID."""

from __future__ import annotations

import hashlib
import json
import logging
import struct
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")

R = Path(__file__).resolve().parent.parent
source = R / "models68/primary/events.bin"
out = R / "path80_same_bin_control_v1"
out.mkdir(exist_ok=False)
record = struct.Struct("<5I2fI")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def locate(value, width):
    if value < 0:
        return 0
    ratio = np.float32(value / width)
    return 2047 if ratio >= np.float32(2046) else int(ratio) + 1


source_hash = sha(source)
assert source_hash == "4787aa0f822bb2ddf35f4af28c04579283fac5d557fbea74c9f1802047d3d04c"
counts = {
    "events": 0,
    "queries": 0,
    "kind2": 0,
    "changed": 0,
    "toward_positive_infinity": 0,
    "toward_negative_infinity": 0,
    "unchanged": 0,
    "changed_bucket_ids": 0,
}
destination = out / "native_path_same_bin_scores.bin"
width = None
with source.open("rb") as incoming, destination.open("xb") as outgoing:
    header = incoming.read(16)
    assert header == struct.pack("<4I", 0x51474D68, 1, 32, 768)
    outgoing.write(header)
    while raw := incoming.read(32):
        assert len(raw) == 32
        row = record.unpack(raw)
        query, kind, step = row[:3]
        if kind == 1 and step == 1:
            assert query == counts["queries"]
            width = np.float32(
                max(np.float32(row[6]), np.float32(1e-6)) * np.float32(0.016)
            )
            counts["queries"] += 1
        if kind == 2:
            counts["kind2"] += 1
            old = np.float32(row[5])
            assert np.isfinite(old)
            assert width is not None
            bucket = locate(old, width)
            chosen = old
            direction = None
            for name, toward in [
                ("toward_positive_infinity", np.float32(np.inf)),
                ("toward_negative_infinity", np.float32(-np.inf)),
            ]:
                candidate = np.nextafter(old, toward, dtype=np.float32)
                if np.isfinite(candidate) and locate(candidate, width) == bucket:
                    chosen, direction = candidate, name
                    break
            if direction:
                counts["changed"] += 1
                counts[direction] += 1
                raw = raw[:20] + struct.pack("<f", chosen) + raw[24:]
            else:
                counts["unchanged"] += 1
            assert locate(chosen, width) == bucket
        outgoing.write(raw)
        counts["events"] += 1
assert counts["queries"] == 1000
assert counts["changed"] > 0
assert counts["changed"] + counts["unchanged"] == counts["kind2"]
assert sha(source) == source_hash
proof = {
    "proof_kind": "same_bin_nextafter_control",
    "control_native_scores_from_original_trace": True,
    "off_policy_rescored_native_events": True,
    "actual_learned_path": False,
    "query_count": 1000,
    "source_events_sha256": source_hash,
    "rescored_events_sha256": sha(destination),
    "model_sha256": None,
    "model_path": None,
    "no_codec_rescore_claim": True,
    "no_GT_or_hits_input": True,
    "fixed_rule": "try nextafter(+inf), then nextafter(-inf), adopt only finite same-FP32-bucket value; else unchanged",
    "counts": counts,
    "script_sha256": sha(Path(__file__)),
}
with (out / "proof.json").open("x") as output:
    json.dump(proof, output, indent=2)
    output.write("\n")
logging.info("%s", json.dumps(proof, indent=2))
