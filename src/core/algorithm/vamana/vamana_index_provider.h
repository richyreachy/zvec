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

#include <zvec/core/framework/index_provider.h>
#include <zvec/core/framework/index_searcher.h>
#include <zvec/core/framework/index_streamer.h>
#include "vamana_entity.h"

namespace zvec {
namespace core {

class VamanaIndexProvider : public IndexProvider {
 public:
  VamanaIndexProvider(
      const IndexMeta &meta, const VamanaEntity::Pointer &entity,
      const std::string &owner,
      std::shared_ptr<const std::vector<node_id_t>> order = nullptr)
      : meta_(meta),
        entity_(entity),
        owner_class_(owner),
        order_(std::move(order)) {}

  VamanaIndexProvider(const VamanaIndexProvider &) = delete;
  VamanaIndexProvider &operator=(const VamanaIndexProvider &) = delete;

 public:
  IndexProvider::Iterator::Pointer create_iterator() override {
    return VamanaIndexProvider::Iterator::Pointer(
        new (std::nothrow) Iterator(entity_, order_));
  }

  size_t count() const override {
    return entity_->doc_cnt();
  }

  size_t dimension() const override {
    return meta_.dimension();
  }

  IndexMeta::DataType data_type() const override {
    return meta_.data_type();
  }

  size_t element_size() const override {
    return meta_.element_size();
  }

  const void *get_vector(uint64_t key) const override {
    return entity_->get_vector_by_key(key);
  }

  int get_vector(const uint64_t key,
                 IndexStorage::MemoryBlock &block) const override {
    return entity_->get_vector_by_key(key, block);
  }

  const std::string &owner_class() const override {
    return owner_class_;
  }

 private:
  class Iterator : public IndexProvider::Iterator {
   public:
    Iterator(const VamanaEntity::Pointer &entity,
             const std::shared_ptr<const std::vector<node_id_t>> &order)
        : entity_(entity), order_(order), cur_id_(0U) {
      cur_id_ = get_next_valid_id(0);
    }

    const void *data() const override {
      return entity_->get_vector(node_at(cur_id_), block_) == 0 ? block_.data()
                                                                : nullptr;
    }

    bool is_valid() const override {
      return cur_id_ < entity_->doc_cnt();
    }

    uint64_t key() const override {
      return entity_->get_key(node_at(cur_id_));
    }

    void next() override {
      cur_id_ = get_next_valid_id(cur_id_ + 1);
    }

    void reset() {
      cur_id_ = get_next_valid_id(0);
    }

   private:
    node_id_t node_at(node_id_t position) const {
      return order_ ? (*order_)[position] : position;
    }

    node_id_t get_next_valid_id(node_id_t start_id) {
      for (node_id_t i = start_id; i < entity_->doc_cnt(); i++) {
        if (entity_->get_key(node_at(i)) != kInvalidKey) {
          return i;
        }
      }
      return kInvalidNodeId;
    }

    const VamanaEntity::Pointer entity_;
    const std::shared_ptr<const std::vector<node_id_t>> order_;
    node_id_t cur_id_;
    mutable IndexStorage::MemoryBlock block_;
  };

  const IndexMeta &meta_;
  const VamanaEntity::Pointer entity_;
  const std::string owner_class_;
  const std::shared_ptr<const std::vector<node_id_t>> order_;
};

}  // namespace core
}  // namespace zvec
