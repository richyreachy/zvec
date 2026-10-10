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

#include <cstddef>
#include <cstdint>
#include <memory>

namespace zvec::core {
class HnswEntity;
class HnswContext;

// Adapts ordinary HNSW level zero to QuantizedGraph<SymphonyQGCodec>.
// The streamer owns synchronization and invalidates the cache on mutation.
class HnswSymphonyQG {
 public:
  // dimension counts coordinates only; stored_dimension additionally includes
  // any trailing norm. The streamer resolves legacy and Turbo metadata.
  HnswSymphonyQG(size_t dimension, size_t max_neighbors = 32,
                 bool cosine = false, size_t stored_dimension = 0);
  ~HnswSymphonyQG();
  int search(uint32_t entry, HnswContext &ctx) const;
  void clear();
  int prebuild(const HnswEntity &entity, size_t doc_cnt, size_t threads);
  uint32_t entry() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace zvec::core
