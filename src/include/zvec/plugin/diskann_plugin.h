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

#include <string>

// NOTE: The APIs declared in this header are INTERNAL to zvec. They are
// invoked implicitly by ``DiskAnnIndex`` on first use so that DiskAnn works
// out of the box, without users ever calling a ``load_diskann_plugin()`` /
// ``is_libaio_available()`` entry point. External callers should not depend
// on these symbols; they may change or be removed in future releases.
//
// DiskAnn is compiled directly into the hosting binary (_zvec.so for the
// Python wheel, the test executable for gtest, or the tool binary for C++
// tools). ``LoadDiskAnnPlugin`` no longer dlopens a separate .so — it simply
// loads libaio eagerly via dlopen()/dlsym() and logs the result. On hosts
// missing libaio, DiskAnn falls back to synchronous pread() (with a warning)
// while other index types (HNSW / IVF / Flat / Vamana) keep working.

#if defined(_WIN32) || defined(__CYGWIN__)
#ifdef ZVEC_BUILD_SHARED
#define ZVEC_PLUGIN_EXPORT __declspec(dllexport)
#elif defined(ZVEC_USE_SHARED)
#define ZVEC_PLUGIN_EXPORT __declspec(dllimport)
#else
#define ZVEC_PLUGIN_EXPORT
#endif
#else
#if __GNUC__ >= 4
#define ZVEC_PLUGIN_EXPORT __attribute__((visibility("default")))
#else
#define ZVEC_PLUGIN_EXPORT
#endif
#endif

namespace zvec {

// Return codes for LoadDiskAnnPlugin().
enum DiskAnnPluginStatus {
  kDiskAnnPluginOk = 0,
  kDiskAnnPluginUnsupportedPlatform = -1,
  kDiskAnnPluginLibAioMissing = -2,
  kDiskAnnPluginDlopenFailed = -3,
};

// Returns true if libaio is present on the host and the minimum set of symbols
// required by the DiskAnn runtime (io_setup / io_submit / io_getevents /
// io_destroy) can be resolved at runtime.
//
// Internal probe used by ``LoadDiskAnnPlugin`` before attempting dlopen. Not
// part of the user-facing API.
ZVEC_PLUGIN_EXPORT bool IsLibAioAvailable();

// Load libaio at runtime via dlopen()/dlsym() and mark the DiskAnn runtime
// as ready. DiskAnn code is compiled directly into the hosting binary, so
// no separate .so is loaded. Invoked implicitly by
// ``DiskAnnIndex::CreateAndInitStreamer`` on first use; callers should not
// invoke it directly. The call is idempotent and returns
// ``kDiskAnnPluginOk`` when the runtime is already active.
//
// The ``path`` parameter is retained for API compatibility but is ignored.
ZVEC_PLUGIN_EXPORT int LoadDiskAnnPlugin(const std::string &path = "");

// Returns true if the DiskAnn runtime is currently loaded in this process.
// Internal diagnostic; not a user-facing API.
ZVEC_PLUGIN_EXPORT bool IsDiskAnnPluginLoaded();

// Unload the DiskAnn runtime. Since DiskAnn is statically linked, this is
// a no-op that always returns false. Retained for API compatibility.
ZVEC_PLUGIN_EXPORT bool UnloadDiskAnnPlugin();

}  // namespace zvec
