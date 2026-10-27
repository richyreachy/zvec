#!/usr/bin/env python3
# ruff: noqa: T201 -- preserve frozen zip semantics and explicit CLI summaries
"""Independent synthetic checks of phase81 transitions against production locate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

PREFIX = r"""
#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#include "core/algorithm/quantized_graph/quantized_graph_bucket.h"
using Bucket = zvec::core::QuantizedGraphBucket;
uint32_t bits(float value) { uint32_t out; std::memcpy(&out, &value, 4); return out; }
void need(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
struct Boundary {
  uint32_t at_bits, previous_bits, next_bits;
  uint32_t previous_bucket, at_bucket, next_bucket;
  uint32_t flags, reserved;
};
static_assert(sizeof(Boundary) == 32, "exact boundary schema");
"""

SUFFIX = r"""

// Independent oracle uses only nonnegative bit patterns, with -0 explicitly
// chosen for right bucket 1. It does not reuse the exported ordered_float map.
bool reference(const Bucket& bucket, uint32_t right, float& at) {
  if (bucket.locate(std::numeric_limits<float>::max()) < right) return false;
  if (right == 1) { at = -0.0f; return true; }
  uint32_t low = 0, high = 0x7f7fffffU;
  while (low < high) {
    const uint32_t middle = low + (high - low) / 2;
    if (bucket.locate(unbits(middle)) >= right) high = middle;
    else low = middle + 1;
  }
  at = unbits(low);
  return true;
}

int main() {
  try {
    need(std::fegetround() == FE_TONEAREST, "rounding mode must be nearest");
    need(std::numeric_limits<float>::is_iec559 && sizeof(float) == 4, "IEEE binary32");
    std::vector<uint32_t> entries = {
      0, 0x80000000U, 1, 0x80000001U, 0x00800000U,
      bits(-1.0f), bits(1e-6f), bits(1e-3f), bits(.1234567f),
      bits(.5f), bits(1.0f), bits(2.0f), bits(1000.0f),
      bits(1e20f), 0x7f7fffffU
    };
    // Adjacent representable entry scales near extreme finite transitions.
    const float maximum = std::numeric_limits<float>::max();
    for (float denominator : {64.0f, 128.0f, 1024.0f, 2046.0f}) {
      const float entry = float(double(maximum) / (double(denominator) * double(.016f)));
      const uint32_t center = bits(entry);
      for (int offset = -2; offset <= 2; ++offset) entries.push_back(center + offset);
    }
    std::sort(entries.begin(), entries.end());
    entries.erase(std::unique(entries.begin(), entries.end()), entries.end());
    uint64_t finite = 0, absent = 0, comparisons = 0, overflow = 0;
    uint64_t maxfinite_at = 0, invalid_rejections = 0, zero_checks = 0;
    uint32_t random_state = 8104;
    std::vector<std::array<uint64_t, 5>> summary;
    for (uint32_t entry_bits : entries) {
      const float entry = unbits(entry_bits);
      need(std::isfinite(entry), "synthetic entry finite");
      Bucket bucket; bucket.reset(0); bucket.set_scale(entry);
      need(std::isfinite(bucket.width()) && bucket.width() > 0, "synthetic width finite positive");
      need(bucket.locate(-0.0f) == 1 && bucket.locate(+0.0f) == 1, "zero right inclusive");
      need(bucket.locate(-std::numeric_limits<float>::denorm_min()) == 0, "negative subnormal bin");
      need(bucket.locate(std::numeric_limits<float>::denorm_min()) == 1, "positive subnormal bin");
      zero_checks += 4;
      const float ratio = maximum / bucket.width();
      if (std::isinf(ratio)) {
        need(bucket.locate(maximum) == Bucket::N - 1, "finite score division overflow saturates");
        ++overflow;
      }
      uint64_t local_finite = 0, local_absent = 0, local_max = 0;
      for (uint32_t right = 1; right < Bucket::N; ++right) {
        float expected = 0;
        const bool exists = reference(bucket, right, expected);
        const Boundary got = transition(bucket, right);
        if (!exists) {
          const Boundary empty{};
          need(std::memcmp(&got, &empty, sizeof(got)) == 0, "absent transition must be all zero");
          ++absent; ++local_absent;
          continue;
        }
        ++finite; ++local_finite;
        const float previous = std::nextafter(expected, -std::numeric_limits<float>::infinity());
        const float next = std::nextafter(expected, std::numeric_limits<float>::infinity());
        need(got.flags == 7 && got.reserved == 0, "finite transition flags");
        need(got.at_bits == bits(expected) && got.previous_bits == bits(previous)
             && got.next_bits == bits(next), "independent boundary bits");
        need(got.previous_bucket == bucket.locate(previous)
             && got.at_bucket == bucket.locate(expected)
             && got.next_bucket == bucket.locate(next), "boundary bucket fields");
        need(got.previous_bucket < right && got.at_bucket == right
             && got.next_bucket >= right, "exact right-bucket transition");
        if (bits(expected) == 0x7f7fffffU) { ++maxfinite_at; ++local_max; }
        const std::array<float, 7> probes = {
          previous, expected, next, -0.0f, +0.0f, -maximum, maximum
        };
        for (float score : probes) {
          need((bucket.locate(score) >= right) == (score >= expected), "right inclusive equivalence");
          ++comparisons;
        }
        // A finite nonnegative score well away from the adjacent points.
        random_state = random_state * 1664525U + 1013904223U;
        const float probe = unbits(random_state % 0x7f800000U);
        need((bucket.locate(probe) >= right) == (probe >= expected), "random score equivalence");
        ++comparisons;
      }
      summary.push_back({entry_bits, bits(bucket.width()), local_finite, local_absent, local_max});
      for (uint32_t right : {0U, uint32_t(Bucket::N)}) {
        bool rejected = false;
        try { (void)transition(bucket, right); } catch (const std::runtime_error&) { rejected = true; }
        need(rejected, "invalid right ID must fail closed"); ++invalid_rejections;
      }
    }
    for (float entry : {std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN()}) {
      Bucket bucket; bucket.reset(0); bucket.set_scale(entry);
      bool rejected = false;
      try { (void)transition(bucket, 1); } catch (const std::runtime_error&) { rejected = true; }
      need(rejected, "nonfinite width must fail closed"); ++invalid_rejections;
    }
    need(absent > 0 && overflow > 0, "extreme fixtures must be nonvacuous");
    need(maxfinite_at > 0, "must exercise finite transition with infinite successor");
    std::cout << "{\"status\":\"passed\",\"synthetic_entry_widths\":" << entries.size()
              << ",\"finite_transitions\":" << finite << ",\"absent_transitions\":" << absent
              << ",\"right_inclusive_checks\":" << comparisons
              << ",\"finite_score_division_overflow_widths\":" << overflow
              << ",\"maximum_finite_transition_cases\":" << maxfinite_at
              << ",\"zero_subnormal_checks\":" << zero_checks
              << ",\"invalid_input_rejections\":" << invalid_rejections
              << ",\"per_width_columns\":[\"entry_bits\",\"width_bits\",\"finite\",\"absent\",\"at_maxfinite\"]"
              << ",\"per_width\":[";
    for (size_t i = 0; i < summary.size(); ++i) {
      if (i) std::cout << ',';
      const auto& row = summary[i];
      std::cout << '[' << row[0] << ',' << row[1] << ',' << row[2] << ',' << row[3] << ',' << row[4] << ']';
    }
    std::cout << "]}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "transition fixture failure: " << error.what() << '\n';
    return 1;
  }
}
"""


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    source_text = args.source.read_text()
    start = source_text.index("float unbits(uint32_t value)")
    end = source_text.index("\nstruct Output", start)
    extracted = source_text[start:end]
    fixture = args.out_dir / "transition_fixture.cc"
    fixture.write_text(PREFIX + extracted + SUFFIX)
    binary = args.out_dir / "transition_fixture"
    header = args.repo / "src/core/algorithm/quantized_graph/quantized_graph_bucket.h"
    command = [
        "g++",
        "-std=c++17",
        "-O2",
        "-fno-fast-math",
        "-DQG_BUCKET_WIDTH=.016f",
        "-DQG_BUCKET_COUNT=2048",
        "-I",
        str(args.repo / "src"),
        str(fixture),
        "-o",
        str(binary),
    ]
    compile_run = subprocess.run(command, text=True, capture_output=True, check=False)
    record = {
        "compile": {
            "argv": command,
            "returncode": compile_run.returncode,
            "stdout": compile_run.stdout,
            "stderr": compile_run.stderr,
        }
    }
    if compile_run.returncode == 0:
        run = subprocess.run([str(binary)], text=True, capture_output=True, check=False)
        record["run"] = {
            "argv": [str(binary)],
            "returncode": run.returncode,
            "stdout": run.stdout,
            "stderr": run.stderr,
        }
        if run.returncode == 0:
            record["result"] = json.loads(run.stdout)
    record["status"] = (
        "passed" if record.get("result", {}).get("status") == "passed" else "failed"
    )
    record["scope"] = (
        "synthetic widths only; actual production Bucket and exact extracted transition source"
    )
    record["nonfinite_scores"] = (
        "Actual exported score/truth must be finite; +inf used only as nextafter(maxfinite,+inf) fixture. Finite-input division overflow tested separately."
    )
    record["provenance"] = {
        "checker_sha256": sha(Path(__file__)),
        "source_path": str(args.source),
        "source_sha256": sha(args.source),
        "transition_sha256": hashlib.sha256(extracted.encode()).hexdigest(),
        "bucket_header_path": str(header),
        "bucket_header_sha256": sha(header),
        "fixture_sha256": sha(fixture),
    }
    output = args.out_dir / "report.json"
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": record["status"],
                "report": str(output),
                "result": record.get("result"),
            },
            sort_keys=True,
        )
    )
    if record["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
