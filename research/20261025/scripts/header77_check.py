"""Read-only phase77 model-header and frozen-provenance audit.

Run only after the writer exits. Both blocks.partial and blocks.bin are accepted.
This script hashes recorded source files; it does not train or rewrite the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import struct
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
ROOT = Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def forbidden_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON constant: {value}")


def verify_metadata_and_sources(metadata, data_manifest, count):
    exact_integers = {
        "database_count": 1000000,
        "record_count": count,
        "dimension": 768,
        "max_neighbors": 32,
        "proxies_per_full_center": 16,
        "codec_bytes": 3328,
        "record_bytes": 3600,
        "sweeps": 2,
        "flips_per_edge_per_sweep": 4,
        "proxy_seed": 7602,
        "audit_record_bytes": 64,
    }
    for key, expected in exact_integers.items():
        if type(metadata[key]) is not int or metadata[key] != expected:
            raise ValueError(f"Invalid integer metadata: {key}")
    exact_values = {
        "objective": "independent sum(e.T@(Q.T@Q+lambda*I)@e)",
        "lambda": "trace(Q.T@Q)/768",
        "proxy_scale": 0.25,
        "alpha_cap_ratio": 1.02,
        "minimum_delta_relative": 1e-13,
        "short_blocks": "native codec",
        "center_and_neighbor_keys": "exact original graph order;unused UINT64_MAX",
        "audit_layout": "u32 updates,u32 flags,7xf64(lambda,initial,final,maxscale,radial,FP32radial,degree)",
        "history_layout": "257xf64 per center;unused NaN;short allNaN",
        "moves_layout": "256x3xu16(sweep,lane,bit);unused UINT16_MAX",
        "compact_layout": "32x96bytes;big-endian sign bits;unused lanes0",
    }
    for key, expected in exact_values.items():
        if metadata[key] != expected:
            raise ValueError(f"Invalid algorithm/layout metadata: {key}")
    if metadata["requires_independent_validation_before_loading"] is not True:
        raise ValueError("Offline validation requirement missing")
    # Verify all input/source hashes that the header claims. The model body hash
    # is covered by the full acceptance manifest, separately from this audit.
    hashes = dict(metadata["input_sha256"])
    if hashes != data_manifest["input_hashes_verified"]:
        raise ValueError("Source input hashes differ from frozen data preparation")
    hashes.update(
        {
            "models77/proxies.u32": metadata["proxy_sha256"],
            "models77/rotated.f32": metadata["rotated_sha256"],
            "scripts/independent76.h": metadata["algorithm_header_sha256"],
            "scripts/fullbuild77.cc": metadata["builder_source_sha256"],
            "fullbuild77": metadata["builder_binary_sha256"],
            "rotate75": metadata["rotation_binary_sha256"],
        }
    )
    # These files fix the sampling policy and Python-oracle selection. Their
    # hashes were recorded before model training, although not in the model header.
    for name, expected in data_manifest["files"].items():
        key = "models77/" + name
        if key in hashes and hashes[key] != expected:
            raise ValueError(f"Inconsistent data-manifest hash: {key}")
        hashes[key] = expected
    for relative, expected in hashes.items():
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError(f"Malformed SHA256: {relative}")
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise ValueError(f"Recorded source hash mismatch: {relative}")
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model", type=Path, help="Complete blocks.partial or blocks.bin"
    )
    parser.add_argument("--header", type=Path, help="Frozen 4096-byte source header")
    parser.add_argument(
        "--preflight", type=Path, default=ROOT / "results/independent77_preflight.json"
    )
    parser.add_argument(
        "--data-manifest", type=Path, default=ROOT / "results/independent77_data.json"
    )
    parser.add_argument(
        "--report", type=Path, help="Optional new JSON report, never overwritten"
    )
    args = parser.parse_args()
    model = args.model if args.model.is_absolute() else ROOT / args.model
    with model.open("rb") as stream:
        raw = stream.read(4096)
    if len(raw) != 4096:
        raise ValueError("Truncated model header")
    magic, version, json_size, count, row_bytes = struct.unpack("<8sIIQQ", raw[:32])
    if magic != b"QGIC77\0\0" or version != 1 or row_bytes != 3600:
        raise ValueError("Unsupported model magic/version/record length")
    if count not in (8192, 1000000) or not 0 < json_size <= 4064:
        raise ValueError("Unexpected record count or JSON length")
    if model.stat().st_size != 4096 + count * row_bytes:
        raise ValueError("Model length does not match record count")
    if any(raw[32 + json_size :]):
        raise ValueError("Nonzero unused header bytes")
    payload = raw[32 : 32 + json_size].decode("utf-8", errors="strict")
    metadata = json.loads(
        payload, object_pairs_hook=unique_object, parse_constant=forbidden_constant
    )
    if not isinstance(metadata, dict):
        raise ValueError("Metadata must be a JSON object")
    preflight_path = (
        args.preflight if args.preflight.is_absolute() else ROOT / args.preflight
    )
    data_path = (
        args.data_manifest
        if args.data_manifest.is_absolute()
        else ROOT / args.data_manifest
    )
    preflight = json.loads(preflight_path.read_text())
    data_manifest = json.loads(data_path.read_text())
    frozen_header = args.header or ROOT / f"models77/header{count}.bin"
    if not frozen_header.is_absolute():
        frozen_header = ROOT / frozen_header
    frozen = frozen_header.read_bytes()
    if len(frozen) != 4096 or raw != frozen:
        raise ValueError("Output header differs from frozen source header")
    header_hash = hashlib.sha256(raw).hexdigest()
    if header_hash != preflight["headers"][str(count)]:
        raise ValueError("Header hash differs from preflight")
    expected_metadata = dict(preflight["metadata"], record_count=count)
    if metadata != expected_metadata:
        raise ValueError("Parsed metadata differs from frozen preflight")
    hashes = verify_metadata_and_sources(metadata, data_manifest, count)
    result = {
        "model": str(model),
        "record_count": count,
        "record_bytes": row_bytes,
        "model_bytes": model.stat().st_size,
        "complete_4096_byte_header_identical_to_frozen_input": True,
        "strict_json_semantics_and_zero_padding_checked": True,
        "header_sha256": header_hash,
        "parsed_metadata_matches_frozen_preflight": True,
        "all_recorded_source_and_sampling_hashes_verified": hashes,
        "validation_selection_fixed_before_training_hash_rechecked": True,
        "model_body_verified_by_this_tool": False,
        "loader_runtime_compatibility_verified_by_this_tool": False,
        "no_retrieval_qps_or_recall_claim": True,
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.report:
        report = args.report if args.report.is_absolute() else ROOT / args.report
        with report.open("x") as stream:
            stream.write(text)
    logging.info(text.rstrip())
    logging.info("HEADER77 EXTRA COMPLETE")


if __name__ == "__main__":
    main()
