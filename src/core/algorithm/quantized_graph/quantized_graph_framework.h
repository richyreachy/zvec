// Copyright 2026-present the zvec project
// Licensed under the Apache License, Version 2.0.
#pragma once

#include <zvec/core/framework/index_factory.h>

namespace zvec::core {
// Collection embeds the builder's snapshot alongside its original FP32 rows in
// an already-open storage. Standalone consumers use IndexBuilder::dump instead.
int AppendQuantizedGraph(IndexBuilder &builder, IndexStorage &storage);
}  // namespace zvec::core
