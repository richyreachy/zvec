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

#include "quantizer/rotator/rotator.h"
#include "quantizer/rotator/fht_rotator.h"
#include "quantizer/rotator/matrix_rotator.h"

namespace zvec {
namespace turbo {

Rotator::Pointer CreateRotator(RotatorType type, int in_dim) {
  switch (type) {
    case RotatorType::kMatrix:
      return std::make_shared<MatrixRotator>(in_dim);
    case RotatorType::kFht:
      return std::make_shared<FhtRotator>(in_dim);
    default:
      return nullptr;
  }
}

Rotator::Pointer CreateRotatorFromBlob(const void *data, size_t len) {
  if (!data || len < sizeof(RotatorSerHeader)) {
    return nullptr;
  }
  const RotatorSerHeader *header =
      reinterpret_cast<const RotatorSerHeader *>(data);
  if (header->magic != kRotatorMagic || header->version != kRotatorSerVersion) {
    return nullptr;
  }

  Rotator::Pointer rotator =
      CreateRotator(static_cast<RotatorType>(header->rotator_type), 0);
  if (!rotator) {
    return nullptr;
  }
  if (rotator->deserialize(data, len) != 0) {
    return nullptr;
  }
  return rotator;
}

}  // namespace turbo
}  // namespace zvec
