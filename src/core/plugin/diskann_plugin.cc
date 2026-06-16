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

// DiskAnn is now compiled directly into the library (no separate plugin).
// These functions are retained as no-op stubs for API compatibility with
// callers that still reference them (e.g. Python bindings, schema validation).

#include <zvec/plugin/diskann_plugin.h>

namespace zvec {

bool IsLibAioAvailable() {
  // AIO dependency removed; always report "available" to avoid false negatives
  // in legacy call-sites that gate on this.
  return true;
}

bool IsDiskAnnPluginLoaded() {
  // DiskAnn is always linked in; conceptually always "loaded".
  return true;
}

int LoadDiskAnnPlugin(const std::string & /*path*/) {
  // No-op: DiskAnn code is statically linked; nothing to dlopen.
  return kDiskAnnPluginOk;
}

bool UnloadDiskAnnPlugin() {
  // No-op: cannot unload statically linked code.
  return false;
}

}  // namespace zvec
