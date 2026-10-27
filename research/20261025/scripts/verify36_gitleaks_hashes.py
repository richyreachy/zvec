"""Verify exact staged gitleaks fingerprints against actual frozen file digests."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
PREFIX = "research/20261025/results/path80_artifacts/"
ALLOWED = [
    PREFIX + "path80_native_v1/inputs.json",
    PREFIX + "replay80_formatted_v1/scorer_inputs.json",
]


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")


def execute(argv, cwd, out, name, input_text=None):
    result = subprocess.run(
        argv, cwd=cwd, input=input_text, text=True, capture_output=True, check=False
    )
    with (out / (name + ".log")).open("x") as stream:
        stream.write(result.stdout)
        stream.write(result.stderr)
    return {"argv": argv, "cwd": str(cwd), "returncode": result.returncode}


def validate_findings(repo, frozen, findings, seal):
    need(len(findings) == 50, "Expected exactly 50 findings; review any changed scan")
    rows = []
    expected_sha = {item["path"]: item["sha256"] for item in seal["files"]}
    for finding in findings:
        filename, line = finding["File"], finding["StartLine"]
        need(
            filename in ALLOWED and finding["RuleID"] == "generic-api-key",
            "Unexpected finding scope",
        )
        need(finding["EndLine"] == line, "Only one-line SHA entries accepted")
        path = repo / filename
        need(sha(path) == expected_sha[str(path)], "Sealed artifact digest changed")
        staged = subprocess.run(
            ["git", "show", ":" + filename], cwd=repo, capture_output=True, check=True
        ).stdout
        need(
            staged == path.read_bytes(),
            "Working evidence must exactly equal staged evidence",
        )
        entry = json.loads(
            "{" + path.read_text().splitlines()[line - 1].strip().rstrip(",") + "}"
        )
        need(len(entry) == 1, "Single JSON mapping entry required")
        relative, digest = next(iter(entry.items()))
        need(
            isinstance(digest, str) and re.fullmatch("[0-9a-f]{64}", digest),
            "Expected SHA256 hex",
        )
        need(
            finding["Secret"] == digest,
            "Detected value is not the complete recorded SHA256",
        )
        need(
            json.loads(path.read_text())["frozen_source_sha256"][relative] == digest,
            "Digest must belong to frozen_source_sha256",
        )
        actual = (frozen / relative).resolve()
        need(
            actual.is_relative_to(frozen) and actual.is_file(),
            "Source must exist inside the frozen source tree",
        )
        need(
            sha(actual) == digest,
            "Recorded value differs from actual frozen file SHA256",
        )
        fingerprint = f"{filename}:generic-api-key:{line}"
        need(
            finding["Fingerprint"] == fingerprint,
            "Only exact staged fingerprints accepted",
        )
        rows.append(
            {
                "fingerprint": fingerprint,
                "artifact": filename,
                "line": line,
                "relative_frozen_source": relative,
                "recorded_sha256": digest,
                "actual_file_sha256": sha(actual),
                "frozen_file_bytes": actual.stat().st_size,
                "staged_worktree_seal_identical": True,
            }
        )
    need(len({row["fingerprint"] for row in rows}) == 50, "No duplicate fingerprints")
    need(
        len({row["relative_frozen_source"] for row in rows}) == 25,
        "Expected 25 sources repeated in two artifacts",
    )
    need(
        all(sum(row["artifact"] == name for row in rows) == 25 for name in ALLOWED),
        "Expected 25 findings per artifact",
    )
    return sorted(rows, key=lambda row: row["fingerprint"])


def main():  # noqa: PLR0915 -- Baseline, digest checks, narrow suppression, and positive control are ordered gates.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--bench-root", type=Path, required=True)
    parser.add_argument("--gitleaks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--write-ignore",
        action="store_true",
        help="Create the exact verified fingerprint file if absent",
    )
    args = parser.parse_args()
    repo, bench, binary, out = [
        value.resolve()
        for value in [args.repo, args.bench_root, args.gitleaks, args.out]
    ]
    need(
        not os.environ.get("GITLEAKS_CONFIG"),
        "Unexpected environment config; inspect before proceeding",
    )
    need(not (repo / ".gitleaks.toml").exists(), "Unexpected repository rule config")
    out.mkdir(exist_ok=False)
    commands = []
    try:
        seal_path = repo / "research/20261025/results/stage36_archive_seal.json"
        seal = json.loads(seal_path.read_text())
        need(seal["passed"], "Existing stage36 seal must pass")
        before = {str(repo / name): sha(repo / name) for name in ALLOWED}
        before[str(seal_path)] = sha(seal_path)
        before[str(repo / ".pre-commit-config.yaml")] = sha(
            repo / ".pre-commit-config.yaml"
        )
        frozen = (bench / "variants/trace68_native/build_source").resolve()
        empty_ignore = out / "empty.ignore"
        empty_ignore.write_text("")
        base = [
            str(binary),
            "git",
            "--pre-commit",
            "--staged",
            "--no-banner",
            "--no-color",
            "--report-format",
            "json",
        ]
        # Gitleaks always also reads source/.gitleaksignore, even with -i.
        # Scan byte-identical index copies in an isolated directory to recover
        # all 50 baseline findings without temporarily changing the live ignore.
        fixture = out / "isolated_staged_evidence"
        fixture.mkdir()
        for name in ALLOWED:
            data = subprocess.run(
                ["git", "show", ":" + name], cwd=repo, capture_output=True, check=True
            ).stdout
            need(
                data == (repo / name).read_bytes(),
                "Fixture must equal index and original evidence",
            )
            copy = fixture / name
            copy.parent.mkdir(parents=True, exist_ok=True)
            copy.write_bytes(data)
        command = execute(
            [
                str(binary),
                "dir",
                ".",
                "--no-banner",
                "--no-color",
                "--report-format",
                "json",
                "--gitleaks-ignore-path",
                str(empty_ignore),
                "--report-path",
                str(out / "baseline_findings.json"),
            ],
            fixture,
            out,
            "baseline",
        )
        commands.append(command)
        need(command["returncode"] == 1, "Expected baseline findings exit code")
        findings = json.loads((out / "baseline_findings.json").read_text())
        rows = validate_findings(repo, frozen, findings, seal)
        ignore = repo / ".gitleaksignore"
        text = "# Exact stage36 archived source-file SHA256 false positives; no rule/path wildcard exclusions.\n"
        text += "# Each value is machine-verified against the actual frozen source file; see research/20261025/notes/stage36_gitleaks_hashes.md.\n"
        text += "\n".join(row["fingerprint"] for row in rows) + "\n"
        if args.write_ignore and not ignore.exists():
            with ignore.open("x") as stream:
                stream.write(text)
        need(
            ignore.read_text() == text,
            "Ignore file must contain exactly the verified 50 fingerprints",
        )
        command = execute(
            [
                *base,
                "--gitleaks-ignore-path",
                str(ignore),
                "--report-path",
                str(out / "after_findings.json"),
            ],
            repo,
            out,
            "after",
        )
        commands.append(command)
        need(
            command["returncode"] == 0
            and (
                not (out / "after_findings.json").exists()
                or json.loads((out / "after_findings.json").read_text()) == []
            ),
            "Narrow ignore must clear exactly the staged false positives",
        )
        synthetic = hashlib.sha256(
            b"public synthetic noncredential gitleaks regression control"
        ).hexdigest()
        command = execute(
            [
                str(binary),
                "stdin",
                "--no-banner",
                "--no-color",
                "--gitleaks-ignore-path",
                str(ignore),
                "--report-format",
                "json",
                "--report-path",
                str(out / "positive_control.json"),
            ],
            repo,
            out,
            "positive_control",
            input_text='api_key = "' + synthetic + '"\n',
        )
        commands.append(command)
        controls = json.loads((out / "positive_control.json").read_text())
        need(
            command["returncode"] == 1
            and any(
                row["RuleID"] == "generic-api-key" and row["Secret"] == synthetic
                for row in controls
            ),
            "Unrelated generic API key positive control must remain detectable",
        )
        hook = execute(
            [
                "/home/xingrui/anaconda3/bin/python",
                "-m",
                "pre_commit",
                "run",
                "gitleaks",
                "--files",
                *ALLOWED,
            ],
            repo,
            out,
            "precommit",
        )
        commands.append(hook)
        need(
            hook["returncode"] == 0,
            "Existing unchanged gitleaks precommit hook must pass",
        )
        for name, expected in before.items():
            need(
                sha(Path(name)) == expected,
                "Evidence, seal or hook changed during verification",
            )
        save(
            out / "stage36_gitleaks_hash_verification.json",
            {
                "passed": True,
                "scope": "exact staged file:rule:line fingerprints only",
                "baseline_findings": 50,
                "baseline_scope": "isolated byte-identical index copies; existing source ignore is never removed or changed",
                "verified_actual_source_files": 25,
                "after_findings": 0,
                "generic_api_key_positive_control_detected": True,
                "existing_precommit_hook_passed": True,
                "ignore_sha256": sha(ignore),
                "verifier_sha256": sha(Path(__file__)),
                "gitleaks_binary_sha256": sha(binary),
                "frozen_source_root": str(frozen),
                "unchanged_evidence_seal_hook_sha256": before,
                "verified_rows": rows,
                "commands": commands,
                "limitation": "Gitleaks fingerprints bind file/rule/line, not value. Re-run this verifier before changing these sealed artifacts; a changed hash or row must fail digest validation.",
            },
        )
        logging.info(
            "VERIFIED 50 SHA256 false positives; unchanged hook passed; unrelated positive control detected"
        )
    except BaseException as error:
        save(
            out / "failure.json",
            {"passed": False, "error": repr(error), "commands": commands},
        )
        raise


if __name__ == "__main__":
    main()
