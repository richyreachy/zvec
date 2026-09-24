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

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "hnsw_context.h"
#include "symphony_qg_rotation.h"

namespace zvec::core {

// SymphonyQG's node-centered, one-bit neighbor blocks over HNSW level zero.
// The caller excludes graph mutations during search and clears this derived
// cache before inserting. Blocks are immutable and shared between queries.
class HnswSymphonyQG {
 public:
  explicit HnswSymphonyQG(size_t dimension)
      : dimension_(dimension), rotation_(dimension) {}

  int search(node_id_t entry, HnswContext &ctx) const;
  void clear();
  int prebuild(const HnswEntity &entity, size_t doc_cnt, size_t threads);

 private:
  struct Block {
    const float *center;
    std::vector<node_id_t> neighbors;
    std::shared_ptr<const char> codes;
  };

  int get_block(const HnswEntity &entity, node_id_t id,
                std::shared_ptr<const Block> &block) const;

  size_t dimension_;
  SymphonyQGRotation rotation_;
  mutable std::mutex mutex_;
  mutable std::unordered_map<node_id_t, std::shared_ptr<const Block>> blocks_;
};

}  // namespace zvec::core
