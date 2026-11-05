// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
#pragma once
#include <immintrin.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <vector>
namespace zvec::core {
// Expanded-node membership. UINT32_MAX is reserved; batches contain at most
// 32 IDs. Use a bounded bitmap for dense domains, clearing only touched words.
// Otherwise retain linear probing and overflow storage. Both paths preserve
// lane order and accept IDs outside the optional bitmap domain.
class QuantizedGraphVisitSet {
 public:
  using NodeId = uint32_t;
  void reset(size_t expect, size_t domain = 0) {
    const size_t words = domain / 32 + (domain % 32 != 0);
    // Bound bitmap storage relative to the hash table, and leave the reserved
    // ID outside the bitmap even when the domain ends in a partial word.
    if (words && words <= size_t{kEmpty} / 32 &&
        words / 32 + (words % 32 != 0) <= expect) {
      if (dense_.size() != words)
        dense_.assign(words, 0);
      else
        for (uint32_t word : touched_) dense_[word] = 0;
      touched_.clear();
      overflow_.clear();
      return;
    }
    dense_.clear();
    touched_.clear();
    size_t cap = 64;
    while (cap < expect * 8) cap *= 2;
    if (table_.size() != cap)
      table_.assign(cap, kEmpty);
    else
      std::fill(table_.begin(), table_.end(), kEmpty);
    mask_ = cap - 1;
    overflow_.clear();
  }

  bool visited(NodeId id) const {
    if (!dense_.empty()) {
      if (id / 32 < dense_.size())
        return (dense_[id / 32] & (uint32_t{1} << (id % 32))) != 0;
      return id == kEmpty || overflow_.find(id) != overflow_.end();
    }
    size_t i = id & mask_;
    for (size_t n = 0; n <= mask_; ++n) {
      const NodeId v = table_[i];
      if (v == id) return true;
      if (v == kEmpty) return false;
      i = (i + 1) & mask_;
    }
    // Table completely full: entries beyond it live in the overflow set.
    return id == kEmpty || overflow_.find(id) != overflow_.end();
  }

  __attribute__((target("avx512f"))) uint32_t unvisited_batch(
      const NodeId *ids, const float *distances, size_t count) const {
    assert(count <= 32);
    uint32_t accepted = 0;
    if (!dense_.empty()) {
      for (size_t first = 0; first < count; first += 16) {
        const size_t lanes = std::min(size_t{16}, count - first);
        const __mmask16 active = __mmask16((uint32_t{1} << lanes) - 1);
        const __m512i keys = _mm512_maskz_loadu_epi32(active, ids + first);
        const __m512i offsets = _mm512_srli_epi32(keys, 5);
        const __mmask16 valid =
            active & _mm512_cmplt_epu32_mask(
                         offsets, _mm512_set1_epi32(int(dense_.size())));
        const __m512i values = _mm512_mask_i32gather_epi32(
            _mm512_setzero_si512(), valid, offsets, dense_.data(), 4);
        const __m512i masks =
            _mm512_sllv_epi32(_mm512_set1_epi32(1),
                              _mm512_and_si512(keys, _mm512_set1_epi32(31)));
        const __m512i bits =
            _mm512_maskz_loadu_epi32(active, distances + first);
        const __mmask16 finite = _mm512_cmplt_epi32_mask(
            _mm512_and_si512(bits, _mm512_set1_epi32(0x7fffffff)),
            _mm512_set1_epi32(0x7f800000));
        uint32_t keep = valid & finite & ~_mm512_test_epi32_mask(values, masks);
        uint32_t outside = active & finite & ~valid;
        while (outside) {
          const unsigned lane = __builtin_ctz(outside);
          outside &= outside - 1;
          if (!visited(ids[first + lane])) keep |= uint32_t{1} << lane;
        }
        accepted |= keep << first;
      }
      return accepted;
    }
    if (mask_ > size_t{INT32_MAX}) {
      for (size_t i = 0; i < count; ++i)
        if (std::isfinite(distances[i]) && !visited(ids[i]))
          accepted |= uint32_t{1} << i;
      return accepted;
    }
    const __m512i table_mask = _mm512_set1_epi32(int(mask_));
    for (size_t first = 0; first < count; first += 16) {
      const size_t lanes = std::min(size_t{16}, count - first);
      const __mmask16 active = (__mmask16)((uint32_t{1} << lanes) - 1);
      const __m512i keys = _mm512_maskz_loadu_epi32(active, ids + first);
      const __m512i offsets = _mm512_and_si512(keys, table_mask);
      const __m512i values = _mm512_mask_i32gather_epi32(
          _mm512_setzero_si512(), active, offsets, table_.data(), 4);
      const __m512i bits = _mm512_maskz_loadu_epi32(active, distances + first);
      const __mmask16 finite = _mm512_cmplt_epi32_mask(
          _mm512_and_si512(bits, _mm512_set1_epi32(0x7fffffff)),
          _mm512_set1_epi32(0x7f800000));
      const __mmask16 empty =
          _mm512_cmpeq_epi32_mask(values, _mm512_set1_epi32(-1));
      const __mmask16 equal = _mm512_cmpeq_epi32_mask(values, keys);
      uint32_t keep = active & finite & empty & ~equal;
      uint32_t collisions = active & finite & ~empty & ~equal;
      while (collisions) {
        const unsigned lane = __builtin_ctz(collisions);
        collisions &= collisions - 1;
        if (!visited(ids[first + lane])) keep |= uint32_t{1} << lane;
      }
      accepted |= keep << first;
    }
    return accepted;
  }

  // Return true only for the first insertion in this query.
  bool set_visited(NodeId id) {
    if (id == kEmpty) return false;
    if (!dense_.empty()) {
      if (id / 32 < dense_.size()) {
        uint32_t &word = dense_[id / 32];
        const uint32_t mask = uint32_t{1} << (id % 32);
        if (word & mask) return false;
        if (word == 0) touched_.push_back(id / 32);
        word |= mask;
        return true;
      }
      return overflow_.insert(id).second;
    }
    size_t i = id & mask_;
    for (size_t n = 0; n <= mask_; ++n) {
      const NodeId v = table_[i];
      if (v == id) return false;
      if (v == kEmpty) {
        table_[i] = id;
        return true;
      }
      i = (i + 1) & mask_;
    }
    return overflow_.insert(id).second;
  }

 private:
  static constexpr NodeId kEmpty = 0xFFFFFFFFu;
  std::vector<uint32_t> dense_, touched_;
  std::vector<NodeId> table_;
  std::unordered_set<NodeId> overflow_;
  size_t mask_{0};
};

}  // namespace zvec::core
