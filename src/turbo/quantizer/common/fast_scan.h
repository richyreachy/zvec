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
#include <turbo/fast_scan.h>
namespace zvec::turbo {
using ::turbo::fast_scan_even_chunk;
using ::turbo::fast_scan_nibble_slot;
using ::turbo::fast_scan_pack_codes;
using ::turbo::fast_scan_pack_lut;
using ::turbo::fast_scan_packed_block_size;
using ::turbo::fast_scan_packed_lut_size;
using ::turbo::kFastScanBlockSize;
using ::turbo::kFastScanMapper;
}  // namespace zvec::turbo
