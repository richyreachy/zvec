// Copyright 2026-present the zvec project
// Licensed under the Apache License, Version 2.0.
#pragma once

#include "vector_column_indexer.h"

namespace zvec {

class QuantizedGraphColumn {
 public:
  static Status validate_options(const QuantizedGraphIndexParams &);
  Status open(const FieldSchema &, core_interface::Index &, bool create,
              bool read_only);
  Status build(const FieldSchema &, core_interface::Index &);
  Result<IndexResults::Ptr> search(
      core_interface::Index &, const void *,
      const vector_column_params::QueryParams &) const;
  Status search_ids(const void *, const QueryParams::Ptr &, uint32_t,
                    const IndexFilter *, int64_t *, float *) const;

 private:
  core::IndexRunner::Pointer runner_;
  core::IndexQueryMeta query_meta_;
};

}  // namespace zvec
