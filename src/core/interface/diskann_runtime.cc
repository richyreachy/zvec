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

#include <atomic>
#include <mutex>
#include <string>
#include <zvec/core/framework/index_logger.h>
#include <zvec/core/interface/diskann_runtime.h>

#if defined(__linux__) || defined(__linux) || defined(__APPLE__)
#include <dlfcn.h>
#include <unistd.h>
#endif

#if defined(__linux__) || defined(__linux)
#include <limits.h>
#endif

// Include the libaio dlopen wrapper so we can load libaio eagerly at DiskAnn
// bring-up time (instead of waiting for the first I/O operation).
#if defined(__linux__) || defined(__linux)
#include "algorithm/diskann/libaio_loader.h"
#endif

namespace zvec {

namespace {

#if defined(__linux__) || defined(__linux)
constexpr bool kPlatformSupportsDiskAnn = true;
#elif defined(__APPLE__)
constexpr bool kPlatformSupportsDiskAnn = false;
#else
constexpr bool kPlatformSupportsDiskAnn = false;
#endif

// Tracks whether the DiskAnn runtime has been initialised.  Since DiskAnn is
// now compiled directly into the hosting binary (_zvec.so for Python, the test
// executable for gtest, etc.) there is no separate .so to dlopen; the "loaded"
// flag simply means libaio has been probed and the result cached.
std::atomic<bool> g_runtime_ready{false};
std::mutex g_runtime_mutex;

}  // namespace

bool IsLibAioAvailable() {
#if defined(__linux__) || defined(__linux)
  // Use the LibAioLoader singleton so we share the cached dlopen handle with
  // the DiskAnn file reader.  Load() is idempotent and thread-safe.
  return LibAioLoader::Instance().Load();
#else
  return false;
#endif
}

bool IsDiskAnnRuntimeReady() {
  return g_runtime_ready.load(std::memory_order_acquire);
}

int InitDiskAnnRuntime(const std::string &path) {
  (void)path;  // No external path needed — DiskAnn is linked in statically.

  if (!kPlatformSupportsDiskAnn) {
    LOG_ERROR(
        "DiskAnn is not supported on this platform; it is only "
        "available on Linux x86_64 with libaio.");
    return kDiskAnnRuntimeUnsupportedPlatform;
  }

#if defined(__linux__) || defined(__linux)
  // Fast path: already initialised.
  if (g_runtime_ready.load(std::memory_order_acquire)) {
    return kDiskAnnRuntimeOk;
  }

  std::lock_guard<std::mutex> lock(g_runtime_mutex);
  if (g_runtime_ready.load(std::memory_order_relaxed)) {
    return kDiskAnnRuntimeOk;
  }

  // Eagerly load libaio at DiskAnn bring-up time so the user gets immediate
  // feedback (success or failure) rather than a delayed error on the first
  // async-I/O operation.
  LOG_INFO("DiskAnn: initializing runtime — loading libaio ...");
  if (!LibAioLoader::Instance().Load()) {
    LOG_WARN(
        "DiskAnn: libaio could not be loaded (tried libaio.so.1 and "
        "libaio.so.1t64). Install it (e.g. 'apt-get install libaio1', or "
        "'libaio1t64' on Ubuntu 24.04+) and retry. DiskAnn will fall back "
        "to synchronous pread() — performance will be degraded. Other "
        "index types (HNSW/IVF/Flat/Vamana) are unaffected.");
    // We still mark the runtime as "ready" — DiskAnn code is linked in and
    // the file reader will gracefully fall back to pread().  The user gets
    // a clear warning now rather than a hard failure later.
    g_runtime_ready.store(true, std::memory_order_release);
    return kDiskAnnRuntimeLibAioMissing;
  }

  LOG_INFO("DiskAnn: libaio loaded successfully — async I/O enabled.");
  g_runtime_ready.store(true, std::memory_order_release);
  return kDiskAnnRuntimeOk;
#else
  return kDiskAnnRuntimeUnsupportedPlatform;
#endif
}

}  // namespace zvec
