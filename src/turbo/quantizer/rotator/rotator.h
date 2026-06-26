// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include "quantizer/preprocessor/preprocessor.h"

namespace zvec {
namespace turbo {

//! Kind of vector rotation applied as a quantizer preprocessing stage.
enum class RotatorType : uint16_t {
  kMatrix = 0,  // dense dim x dim orthogonal matrix (OPQ-style)
  kFht = 1,     // fast Walsh-Hadamard rotation (RaBitQ-style)
};

//! Magic number ('ROTR') stamped at the start of a serialized rotator blob.
constexpr uint32_t kRotatorMagic = 0x52544F52u;
//! Current rotator serialization format version.
constexpr uint16_t kRotatorSerVersion = 1;

//! Self-describing, fixed-size header that prefixes every serialized rotator.
//! The type-specific payload (matrix floats, seed, ...) follows immediately
//! after this header.
struct RotatorSerHeader {
  uint32_t magic;         // kRotatorMagic
  uint16_t version;       // kRotatorSerVersion
  uint16_t rotator_type;  // RotatorType
  uint32_t in_dim;        // input dimensionality
  uint32_t out_dim;       // output dimensionality
  uint32_t payload_size;  // bytes following the header
  uint32_t reserved;      // 0, for future use / alignment
};
static_assert(sizeof(RotatorSerHeader) == 24,
              "RotatorSerHeader must be 24 bytes");

//! A pluggable, dimension-aware vector rotation stage. A rotation is an
//! orthogonal (distance-preserving) transform applied to fp32 vectors before
//! quantization, with the inverse used to recover the original space.
//!
//! Inherits the general preprocessing contract from Preprocessor and adds
//! the rotator-specific type() accessor.
class Rotator : public Preprocessor {
 public:
  using Pointer = std::shared_ptr<Rotator>;

  //! Kind of rotator.
  virtual RotatorType type() const = 0;

  // in_dim(), out_dim(), apply(), apply_inverse(), train(), serialize(),
  // and deserialize() are inherited from Preprocessor.
};

//! Create an untrained rotator of the given type for in_dim dimensions.
Rotator::Pointer CreateRotator(RotatorType type, int in_dim);

//! Create and restore a rotator from a serialized blob (reads the type from
//! the embedded RotatorSerHeader). Returns nullptr on malformed input.
Rotator::Pointer CreateRotatorFromBlob(const void *data, size_t len);

}  // namespace turbo
}  // namespace zvec
