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
#include <string>
#include <zvec/ailego/container/params.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_holder.h>
#include <zvec/core/framework/index_meta.h>
#include <zvec/turbo/turbo.h>

using namespace zvec::core;

namespace zvec {
namespace turbo {

class Quantizer {
 public:
  typedef std::shared_ptr<Quantizer> Pointer;

  Quantizer() {}
  virtual ~Quantizer() {}

  virtual QuantizeType type() const {
    return type_;
  }

  //! Initialize quantizer with index metadata and parameters
  virtual int init(const IndexMeta &meta, const ailego::Params &params) = 0;

  //! Get the output metadata after initialization
  virtual const IndexMeta &meta() const = 0;

  //! Train the quantizer with data from an IndexHolder
  virtual int train(IndexHolder::Pointer /*holder*/) {
    return IndexError_NotImplemented;
  }

  //! Quantize a query vector for search
  virtual int quantize(const void * /*query*/, const IndexQueryMeta & /*qmeta*/,
                       std::string * /*out*/,
                       IndexQueryMeta * /*ometa*/) const {
    return IndexError_NotImplemented;
  }

  //! Dequantize a result vector back to original format
  virtual int dequantize(const void * /*in*/, const IndexQueryMeta & /*qmeta*/,
                         std::string * /*out*/) const {
    return IndexError_NotImplemented;
  }

  //! Dequantize a result vector back to original format
  virtual int serialize(std::string * /*out*/) const {
    return IndexError_NotImplemented;
  }

  //! Deserialize
  virtual int deserialize(std::string & /*in*/) {
    return IndexError_NotImplemented;
  }

 protected:
  QuantizeType type_{QuantizeType::kDefault};
  uint32_t extra_meta_size_{0};
};

}  // namespace turbo
}  // namespace zvec
