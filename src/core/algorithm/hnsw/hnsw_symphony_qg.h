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
  // max_neighbors is the per-block degree bound, rounded down to a whole
  // number of FastScan batches (the .cc enforces the multiple). 32 trades
  // recall for one-batch scans; 64 matches upstream SymphonyQG's deg64.
  HnswSymphonyQG(size_t dimension, size_t max_neighbors = 32);

  int search(node_id_t entry, HnswContext &ctx) const;
  void clear();
  int prebuild(const HnswEntity &entity, size_t doc_cnt, size_t threads);

  // Level-0 node nearest the corpus centroid, as upstream SymphonyQG's entry.
  // Invalid until prebuild ran.
  node_id_t entry() const {
    return entry_;
  }

 private:
  // One fixed-stride slot per node inside a single arena allocation, like
  // upstream SymphonyQG's row layout: header, center vector, neighbor ids,
  // then FastScan codes. The fixed stride makes every block address pure
  // arithmetic (base + id * stride) and keeps the index in one contiguous
  // mapping. Trivially destructible; the arena is freed wholesale.
  struct Block {
    static constexpr size_t kCenterOffset = 64;

    uint32_t neighbor_cnt;
    uint32_t center_floats;
    uint32_t code_bytes;
    // Rounded-up capacity of the neighbor region so the codes offset does
    // not depend on the actual degree.
    uint32_t neighbor_bytes;

    const float *center() const {
      return reinterpret_cast<const float *>(
          reinterpret_cast<const char *>(this) + kCenterOffset);
    }
    const node_id_t *neighbors() const {
      return reinterpret_cast<const node_id_t *>(
          reinterpret_cast<const char *>(this) + kCenterOffset +
          center_floats * sizeof(float));
    }
    const char *codes() const {
      return reinterpret_cast<const char *>(neighbors()) + neighbor_bytes;
    }
  };

  struct BlockDeleter {
    void operator()(const Block *block) const {
      ::operator delete(const_cast<Block *>(block), std::align_val_t{64});
    }
  };

  struct ArenaDeleter {
    static constexpr size_t kAlignment = 2 * 1024 * 1024;
    void operator()(char *ptr) const {
      ::operator delete(ptr, std::align_val_t{kAlignment});
    }
  };

  struct Scratch {
    std::vector<float> rotated_center;
    std::vector<float> rotated;
    std::vector<float> vector;
    std::vector<node_id_t> neighbors;
    std::vector<float> selection_cache;
  };

  const Block *slot(node_id_t id) const {
    return reinterpret_cast<const Block *>(arena_.get() +
                                           static_cast<size_t>(id) * stride_);
  }

  // The returned pointer is owned by the arena (prebuilt) or lazy_blocks_
  // (post-clear rebuild); the streamer's exclusive lock keeps clear() from
  // destroying blocks during a search.
  const Block *get_block(const HnswEntity &entity, node_id_t id) const;
  // Writes one fixed-stride block into slot; false on error, with no cleanup
  // (the caller owns the memory).
  bool build_block(const HnswEntity &entity, node_id_t id, const void *center,
                   Block *slot, Scratch &scratch, int &error) const;

  size_t dimension_;
  size_t max_neighbors_{32};
  size_t neighbor_bytes_{0};
  SymphonyQGRotation rotation_;
  // Immutable after prebuild and before the next clear(); read lock-free.
  std::unique_ptr<char[], ArenaDeleter> arena_;
  node_id_t arena_count_{0};
  size_t stride_{0};
  node_id_t entry_{kInvalidNodeId};
  mutable std::mutex mutex_;
  mutable std::unordered_map<node_id_t, std::shared_ptr<const Block>>
      lazy_blocks_;
};

}  // namespace zvec::core
