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
#include <zvec/core/framework/index_meta.h>
#include <zvec/turbo/turbo.h>

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
  virtual int init(const core::IndexMeta &meta,
                   const ailego::Params &params) = 0;

  //! Get the output metadata after initialization
  virtual const core::IndexMeta &meta() const = 0;

  //! Convert a record for indexing (quantize a stored vector)
  virtual int convert(const void *record, const core::IndexQueryMeta &rmeta,
                      std::string *out, core::IndexQueryMeta *ometa) const = 0;

  //! Revert a quantized record back to original format
  virtual int revert(const void *in, const core::IndexQueryMeta &qmeta,
                     std::string *out) const = 0;

  //! Quantize a query vector for search
  virtual int quantize(const void *query, const core::IndexQueryMeta &qmeta,
                       std::string *out, core::IndexQueryMeta *ometa) const = 0;

  //! Dequantize a result vector back to original format
  virtual int dequantize(const void *in, const core::IndexQueryMeta &qmeta,
                         std::string *out) const = 0;

 protected:
  QuantizeType type_{QuantizeType::kDefault};
};

}  // namespace turbo
}  // namespace zvec
