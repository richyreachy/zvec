from __future__ import annotations

import copy
import ctypes as ct
import hashlib
import json
import locale
import logging
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")

R = Path(__file__).resolve().parent.parent
out = R / (sys.argv[1] if len(sys.argv) > 1 else "loader78_unit")
out.mkdir(exist_ok=False)
lib = ct.CDLL(str(R / "lib/independent78_test.so"))
P = ct.c_void_p
S = ct.c_char_p
Z = ct.c_size_t
U = ct.c_uint32
L = ct.c_uint64
lib.create78.restype = P
lib.destroy78.argtypes = [P]
lib.loaded78.argtypes = [P]
lib.clear78.argtypes = [P]
lib.move78.argtypes = [P]
lib.load78.argtypes = [P, S, S, S, S, S, Z, Z, Z, ct.c_int]
lib.verify78.argtypes = [P, U, L, P, P, Z]
lib.copy78.argtypes = [P, U, L, P, Z, P, Z]
lib.sha78.argtypes = [S, Z, P]
lib.sha_extend78.argtypes = [S, Z, P]
lib.number78.argtypes = [S, ct.POINTER(ct.c_double)]
lib.error78.restype = S
lib.error_code78.restype = ct.c_int
lib.error_row78.restype = L
err = lambda: (lib.error_code78(), lib.error78().decode(), lib.error_row78())
numeric_checks = 0
tested_locales = []
old_locale = locale.setlocale(locale.LC_NUMERIC)
for candidate in ["C", "C.utf8", "en_US.utf8", "de_DE.utf8", "fr_FR.utf8"]:
    try:
        locale.setlocale(locale.LC_NUMERIC, candidate)
    except locale.Error:
        continue
    tested_locales.append(candidate)
    for text, expected in [
        ("0.25", 0.25),
        ("1.02", 1.02),
        ("1e-13", 1e-13),
        ("-0", 0.0),
        ("1E+2", 100.0),
        ("-2.5e-3", -0.0025),
    ]:
        value = ct.c_double()
        assert lib.number78(text.encode(), ct.byref(value)) == 0, err()
        assert value.value == expected
        numeric_checks += 1
    for text in [
        "0.25junk",
        "01",
        "+1",
        "1.",
        ".1",
        "--1",
        "1e",
        "1e999",
        "null",
        "true",
    ]:
        value = ct.c_double()
        assert lib.number78(text.encode(), ct.byref(value)) == -1
        assert lib.error_code78() == 2
        numeric_checks += 1
locale.setlocale(locale.LC_NUMERIC, old_locale)
sha_checks = 0
sha_continue_checks = 0
messages = [
    b"",
    b"abc",
    b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
    b"a" * 1000000,
]
assert (
    hashlib.sha256(messages[0]).hexdigest()
    == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
)
assert (
    hashlib.sha256(messages[1]).hexdigest()
    == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
)
assert (
    hashlib.sha256(messages[3]).hexdigest()
    == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
)
rng = np.random.default_rng(7801)
for n in [
    1,
    55,
    56,
    57,
    63,
    64,
    65,
    127,
    128,
    129,
    4095,
    4096,
    4097,
    65535,
    65536,
    65537,
    1048589,
]:
    messages.append(rng.integers(0, 256, size=n, dtype=np.uint8).tobytes())
for i, data in enumerate(messages):
    p = out / f"sha_{i}.bin"
    p.write_bytes(data)
    digest = ct.create_string_buffer(65)
    assert lib.sha_extend78(data, len(data), digest) == 0, err()
    assert digest.value.decode() == hashlib.sha256(data).hexdigest()
    sha_continue_checks += 1
    for chunk in [1, 7, 64, 65, 4096]:
        digest = ct.create_string_buffer(65)
        assert lib.sha78(str(p).encode(), chunk, digest) == 0, err()
        assert digest.value.decode() == hashlib.sha256(data).hexdigest()
        sha_checks += 1
N = 40
D = 768
raw = rng.normal(size=(N, D)).astype("<f4")
raw /= np.linalg.norm(raw, axis=1)[:, None]
rot = np.ascontiguousarray(np.roll(raw, 7, axis=1))
keys = (1000 + 17 * rng.permutation(N)).astype("<u8")
rawbytes = struct.pack("<4I", 0x51475664, 1, N, D) + raw.tobytes()
rotbytes = rot.tobytes()
meta = copy.deepcopy(
    json.loads((R / "results/independent77_preflight.json").read_text())["metadata"]
)
meta["database_count"] = meta["record_count"] = N
meta["input_sha256"]["models64/full_vectors.f32"] = hashlib.sha256(rawbytes).hexdigest()
meta["rotated_sha256"] = hashlib.sha256(rotbytes).hexdigest()


def header(metadata, payload=None):
    payload = (
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        if payload is None
        else payload
    )
    assert len(payload) <= 4064
    return (
        struct.pack("<8sIIQQ", b"QGIC77\0\0", 1, len(payload), N, 3600)
        + payload
        + b"\0" * (4064 - len(payload))
    )


h = header(meta)
records = []
neighbors = []
codes = []
for i in range(N):
    degree = 7 if i == 1 else 0 if i == 2 else 1 if i == 3 else 31 if i == 4 else 32
    ns = keys[(np.arange(degree) + i + 1) % N]
    neighbors.append(np.ascontiguousarray(ns, dtype="<u8"))
    code = (
        rng.integers(0, 256, size=3072, dtype=np.uint8).tobytes()
        + rng.normal(size=32).astype("<f4").tobytes()
        + (-np.abs(rng.normal(size=32))).astype("<f4").tobytes()
    )
    codes.append(code)
    records.append(
        struct.pack("<QII", int(keys[i]), degree, int(degree == 32))
        + ns.tobytes()
        + b"\xff" * (8 * (32 - degree))
        + code
    )
model = h + b"".join(records)
orig_digest = hashlib.sha256(model).hexdigest()
paths = {name: out / name for name in ["model.bin", "raw.f32", "rot.f32", "header.bin"]}
for name, data in [
    ("model.bin", model),
    ("raw.f32", rawbytes),
    ("rot.f32", rotbytes),
    ("header.bin", h),
]:
    paths[name].write_bytes(data)
handle = lib.create78()
assert handle
args = [
    str(paths[n]).encode() for n in ["model.bin", "raw.f32", "rot.f32", "header.bin"]
]
assert lib.load78(handle, *args, orig_digest.encode(), N, D, 32, 1) == 0, err()
assert lib.loaded78(handle) == 1
vector_checks = 0
copied = 0
short = 0
for i in range(N):
    assert (
        lib.verify78(handle, i, int(keys[i]), raw[i].ctypes.data, rot[i].ctypes.data, D)
        == 0
    ), err()
    vector_checks += 1
    dst = ct.create_string_buffer(b"\x5a" * 3328)
    kind = lib.copy78(
        handle, i, int(keys[i]), neighbors[i].ctypes.data, len(neighbors[i]), dst, 3328
    )
    if len(neighbors[i]) == 32:
        assert kind == 1
        assert dst.raw[:3328] == codes[i]
        copied += 1
    else:
        assert kind == 0
        assert dst.raw[:3328] == b"\x5a" * 3328
        short += 1
assert lib.copy78(handle, 2, int(keys[2]), None, 0, None, 0) == 0
assert lib.move78(handle) == 0
assert lib.loaded78(handle) == 1


def parallel_check(_worker):
    checks = 0
    for _repeat in range(4):
        for i in range(N):
            assert (
                lib.verify78(
                    handle, i, int(keys[i]), raw[i].ctypes.data, rot[i].ctypes.data, D
                )
                == 0
            )
            dst = ct.create_string_buffer(b"\x5a" * 3328)
            kind = lib.copy78(
                handle,
                i,
                int(keys[i]),
                neighbors[i].ctypes.data,
                len(neighbors[i]),
                dst,
                3328,
            )
            assert kind == (1 if len(neighbors[i]) == 32 else 0)
            assert dst.raw[:3328] == (codes[i] if kind == 1 else b"\x5a" * 3328)
            checks += 1
    return checks


with ThreadPoolExecutor(max_workers=8) as pool:
    concurrent_checks = sum(pool.map(parallel_check, range(8)))
failures = []


def rejected(name, call, expected_code=None):
    result = call()
    assert result == -1, (name, result, err())
    code, message, row = err()
    assert code != 100, (name, message)
    if expected_code is not None:
        assert code == expected_code, (name, code, message)
    failures.append({"name": name, "code": code, "message": message, "row": row})


raw_bad = raw[0].copy()
raw_bad[0] = np.nextafter(raw_bad[0], np.float32(np.inf))
rot_bad = rot[0].copy()
rot_bad[0] = np.nextafter(rot_bad[0], np.float32(np.inf))
swapped = neighbors[0].copy()
swapped[[0, 1]] = swapped[[1, 0]]
dst = ct.create_string_buffer(3328)
rejected(
    "wrong center key",
    lambda: lib.verify78(
        handle, 0, int(keys[1]), raw[0].ctypes.data, rot[0].ctypes.data, D
    ),
    5,
)
rejected(
    "wrong source vector one ULP",
    lambda: lib.verify78(
        handle, 0, int(keys[0]), raw_bad.ctypes.data, rot[0].ctypes.data, D
    ),
    5,
)
rejected(
    "wrong rotation one ULP",
    lambda: lib.verify78(
        handle, 0, int(keys[0]), raw[0].ctypes.data, rot_bad.ctypes.data, D
    ),
    5,
)
rejected(
    "wrong vector dimension",
    lambda: lib.verify78(
        handle, 0, int(keys[0]), raw[0].ctypes.data, rot[0].ctypes.data, D - 1
    ),
    5,
)
rejected(
    "out of range internal ID",
    lambda: lib.verify78(
        handle, N, int(keys[0]), raw[0].ctypes.data, rot[0].ctypes.data, D
    ),
    5,
)
rejected(
    "wrong neighbor order",
    lambda: lib.copy78(handle, 0, int(keys[0]), swapped.ctypes.data, 32, dst, 3328),
    5,
)
rejected(
    "wrong degree",
    lambda: lib.copy78(
        handle, 0, int(keys[0]), neighbors[0].ctypes.data, 31, dst, 3328
    ),
    5,
)
rejected(
    "too short destination",
    lambda: lib.copy78(
        handle, 0, int(keys[0]), neighbors[0].ctypes.data, 32, dst, 3327
    ),
    5,
)
lib.clear78(handle)
assert lib.loaded78(handle) == 0
rejected(
    "use after clear",
    lambda: lib.verify78(
        handle, 0, int(keys[0]), raw[0].ctypes.data, rot[0].ctypes.data, D
    ),
    6,
)
# Each malformed fixture deliberately has a recomputed trust digest/header,
# so schema/record tests exercise the parser beyond the digest gate.
case_id = 0


def bad_file(
    name,
    model_data=model,
    raw_data=rawbytes,
    rot_data=rotbytes,
    header_data=None,
    digest=None,
    docs=N,
    dims=D,
    degree=32,
    cosine=1,
    expected_code=None,
):
    global case_id  # noqa: PLW0603 - Sequential negative-case artifact numbering.
    case_id += 1
    folder = out / f"bad_{case_id}"
    folder.mkdir()
    hd = model_data[:4096] if header_data is None else header_data
    files = [folder / "model", folder / "raw", folder / "rot", folder / "header"]
    for p, data in zip(files, [model_data, raw_data, rot_data, hd], strict=True):
        p.write_bytes(data)
    expected = hashlib.sha256(model_data).hexdigest() if digest is None else digest
    rejected(
        name,
        lambda: lib.load78(
            handle,
            *[str(p).encode() for p in files],
            expected.encode(),
            docs,
            dims,
            degree,
            cosine,
        ),
        expected_code,
    )
    assert lib.loaded78(handle) == 0


bad_file("truncated header", model_data=model[:4095], header_data=h, expected_code=2)
for name, offset, data in [
    ("bad magic", 0, b"X"),
    ("bad version", 8, struct.pack("<I", 2)),
    ("oversized JSON", 12, struct.pack("<I", 4065)),
    ("binary count mismatch", 16, struct.pack("<Q", N + 1)),
    ("count overflow", 16, struct.pack("<Q", 2**64 - 1)),
    ("bad record bytes", 24, struct.pack("<Q", 3599)),
    ("nonzero header padding", 4095, b"X"),
]:
    b = bytearray(model)
    b[offset : offset + len(data)] = data
    bad_file(name, model_data=bytes(b), expected_code=2)
b = bytearray(h)
b[-1] = 1
bad_file("trusted header mismatch", header_data=bytes(b), expected_code=2)
bad_file("extra model byte", model_data=model + b"X", header_data=h, expected_code=2)
bad_file("truncated record", model_data=model[:-1], header_data=h, expected_code=2)
bad_file("bad model SHA", digest="0" * 64, expected_code=3)
bad_file("missing trusted SHA", digest="", expected_code=3)
b = bytearray(model)
b[4096 + 272] ^= 1
bad_file(
    "code bit corruption against trusted SHA",
    model_data=bytes(b),
    digest=orig_digest,
    expected_code=3,
)
bad_file(
    "raw hash mismatch",
    raw_data=rawbytes[:-1] + bytes([rawbytes[-1] ^ 1]),
    expected_code=3,
)
bad_file(
    "rotated hash mismatch",
    rot_data=rotbytes[:-1] + bytes([rotbytes[-1] ^ 1]),
    expected_code=3,
)
bad_file("raw length mismatch", raw_data=rawbytes[:-4], expected_code=2)
b = bytearray(rawbytes)
b[12:16] = struct.pack("<I", D - 1)
bad_file("raw header dimension", raw_data=bytes(b), expected_code=2)
for name, field, value in [
    ("wrong metadata metric size", "dimension", 767),
    ("wrong objective", "objective", "other"),
    ("wrong metadata type", "dimension", "768"),
    ("missing metadata field", "sweeps", None),
    (
        "false validation prerequisite",
        "requires_independent_validation_before_loading",
        False,
    ),
    ("bad hash syntax", "rotated_sha256", "X" * 64),
    ("wrong scale cap", "alpha_cap_ratio", 1.1),
    ("wrong metadata record count", "record_count", N - 1),
]:
    m = copy.deepcopy(meta)
    if value is None:
        del m[field]
    else:
        m[field] = value
    bad_file(name, model_data=header(m) + model[4096:], expected_code=2)
base_payload = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode()
for name, payload in [
    ("duplicate JSON key", b'{"dimension":768,' + base_payload[1:]),
    ("escaped duplicate JSON key", b'{"dimens\\u0069on":768,' + base_payload[1:]),
    ("trailing JSON comma", base_payload[:-1] + b",}"),
    (
        "integer overflow",
        base_payload.replace(
            b'"record_count":40', b'"record_count":18446744073709551616'
        ),
    ),
    ("invalid JSON trailing content", base_payload + b" false"),
    ("invalid JSON syntax", base_payload[:-1]),
    (
        "number overflow",
        base_payload.replace(b'"proxy_scale":0.25', b'"proxy_scale":1e999'),
    ),
    ("unpaired surrogate", b'{"extra":"\\ud800",' + base_payload[1:]),
    ("invalid raw UTF8", b'{"extra":"\xc0\x80",' + base_payload[1:]),
    ("JSON array unsupported schema", b'{"extra":[], ' + base_payload[1:]),
]:
    bad_file(name, model_data=header(meta, payload) + model[4096:], expected_code=2)
# Positive escaped/non-ASCII JSON strings are parsed correctly in unknown fields.
m = copy.deepcopy(meta)
m["extra"] = "测试😀"
valid_model = header(m) + model[4096:]
folder = out / "unicode_valid"
folder.mkdir()
(folder / "model").write_bytes(valid_model)
(folder / "header").write_bytes(valid_model[:4096])
assert (
    lib.load78(
        handle,
        str(folder / "model").encode(),
        args[1],
        args[2],
        str(folder / "header").encode(),
        hashlib.sha256(valid_model).hexdigest().encode(),
        N,
        D,
        32,
        1,
    )
    == 0
), err()
for name, offset, data in [
    ("duplicate center key", 4096 + 3600, struct.pack("<Q", int(keys[0]))),
    ("invalid center sentinel", 4096, struct.pack("<Q", 2**64 - 1)),
    ("degree too large", 4096 + 8, struct.pack("<I", 33)),
    ("wrong full flag", 4096 + 12, struct.pack("<I", 0)),
    ("wrong short flag", 4096 + 3600 + 12, struct.pack("<I", 1)),
    ("unknown neighbor key", 4096 + 16, struct.pack("<Q", 999999999)),
    ("self neighbor key", 4096 + 16, struct.pack("<Q", int(keys[0]))),
    ("duplicate neighbor key", 4096 + 24, struct.pack("<Q", int(neighbors[0][0]))),
    ("short padding key", 4096 + 3600 + 16 + 7 * 8, struct.pack("<Q", int(keys[0]))),
    ("nonfinite add factor", 4096 + 272 + 3072, struct.pack("<f", float("nan"))),
    ("nonfinite scale factor", 4096 + 272 + 3200, struct.pack("<f", float("inf"))),
]:
    b = bytearray(model)
    b[offset : offset + len(data)] = data
    bad_file(name, model_data=bytes(b), expected_code=4)
for kwargs in [
    {"docs": N + 1},
    {"docs": 0},
    {"dims": 767},
    {"degree": 64},
    {"cosine": 0},
]:
    bad_file("unsupported caller " + str(kwargs), **kwargs)
# Success followed by failure cannot leave a previously loaded snapshot active.
assert lib.load78(handle, *args, orig_digest.encode(), N, D, 32, 1) == 0
rejected(
    "successful then failed reload",
    lambda: lib.load78(handle, *args, b"0" * 64, N, D, 32, 1),
    3,
)
assert lib.loaded78(handle) == 0
fd_before = len(list(Path("/proc/self/fd").iterdir()))
for _ in range(10):
    assert lib.load78(handle, *args, orig_digest.encode(), N, D, 32, 1) == 0
    lib.clear78(handle)
    assert lib.load78(handle, *args, b"0" * 64, N, D, 32, 1) == -1
assert len(list(Path("/proc/self/fd").iterdir())) == fd_before
lib.destroy78(handle)
allocation_result = subprocess.run(
    [
        str(R / "independent78_alloc_test"),
        *[str(paths[n]) for n in ["model.bin", "raw.f32", "rot.f32", "header.bin"]],
        orig_digest,
        str(N),
    ],
    capture_output=True,
    text=True,
    check=True,
)
allocation = json.loads(allocation_result.stdout)
assert allocation["heap_allocations_during_successful_verification_and_copy"] == 0
a = {
    "locale_independent_numeric_checks": numeric_checks,
    "numeric_locales_tested": tested_locales,
    "comma_decimal_locale_available": any(
        x.startswith(("de_", "fr_")) for x in tested_locales
    ),
    "sha256_hex_then_continue_checks": sha_continue_checks,
    "concurrent_const_vector_and_copy_checks": concurrent_checks,
    "successful_verification_allocation_test": allocation,
    "sha256_streaming_hashlib_cases": sha_checks,
    "sha256_standard_vectors": 4,
    "synthetic_centers": N,
    "source_and_rotation_rows_verified": vector_checks,
    "full_blocks_copied_byte_identically": copied,
    "short_or_empty_blocks_not_overwritten": short,
    "malformed_and_compatibility_rejections": failures,
    "unicode_and_surrogate_pair_json_accepted": True,
    "moved_from_state_safe": True,
    "failed_reload_invalidates_previous_snapshot": True,
    "repeat_load_clear_failure_file_descriptors_stable": True,
    "scope": "standalone helper;no production integration or full1M helper load performed",
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "helper_sha256": hashlib.sha256(
        (R / "scripts/independent78_import.h").read_bytes()
    ).hexdigest(),
    "binding_sha256": hashlib.sha256(
        (R / "scripts/independent78_test.cc").read_bytes()
    ).hexdigest(),
    "binary_sha256": hashlib.sha256(
        (R / "lib/independent78_test.so").read_bytes()
    ).hexdigest(),
}
(R / "results" / f"independent78_import_checks_{out.name}.json").write_text(
    json.dumps(a, indent=2) + "\n"
)
logging.info("%s", json.dumps(a, indent=2))
logging.info("IMPORT78 UNIT COMPLETE")
