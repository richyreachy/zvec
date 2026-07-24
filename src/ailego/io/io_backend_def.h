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

// Abstract I/O backend selector.
//
// Wraps the low-level loaders (LibAioLoader for libaio) and provides a uniform
// way to initialize, query, and report the active I/O backend.  The actual I/O
// operations are still performed by the underlying loaders; this class is
// responsible only for backend initialization and reporting.
//
// When no asynchronous backend is available, the caller should fall back to
// synchronous pread().
//
// Usage:
//   auto& backend = ailego::IOBackend::Instance();
//   if (!backend.is_pread()) { ... }
//   LOG_INFO("I/O backend: %s", backend.name());

#pragma once

#if defined(__APPLE__) && defined(__MACH__)
#include <TargetConditionals.h>
#endif

#include <ailego/io/libaio_loader.h>
#include <zvec/ailego/io/io_backend.h>

namespace zvec {
namespace ailego {

// Returns a human-readable name for the given backend type.
inline const char *IOBackendTypeName(IOBackendType type) {
  switch (type) {
    case IOBackendType::kLibAio:
      return "libaio";
    case IOBackendType::kThreadPoolPread:
      return "thread_pool_pread";
    case IOBackendType::kPread:
      return "pread";
  }
  return "unknown";
}

// Returns a human-readable description for the given backend type.
// When the backend is kPread, includes installation guidance for libaio.
inline const char *IOBackendDescription(IOBackendType type) {
  switch (type) {
    case IOBackendType::kLibAio:
      return "libaio async I/O backend loaded at runtime via dlopen().";
    case IOBackendType::kThreadPoolPread:
      return "Blocking pread() calls executed by a shared worker pool, with "
             "completion delivered through kqueue.";
    case IOBackendType::kPread:
      return "No async I/O backend available. Install libaio (e.g. "
             "'apt-get install libaio1', or 'libaio1t64' on Ubuntu 24.04+) "
             "and retry. DiskAnn will fall back to synchronous pread() \u2014 "
             "performance will be degraded.";
  }
  return "Unknown I/O backend.";
}

// Singleton that loads and queries an I/O backend on demand.
//
// available() (no arg) selects libaio on Linux when available, worker-pool
// pread() on macOS, and synchronous pread() otherwise.
// available(IOBackendType) tries a specific backend.
// Use type() / name() to query the loaded backend without triggering a load.
class IOBackend {
 public:
  static IOBackend &Instance() {
    static IOBackend instance;
    return instance;
  }

  // Try to load the best available backend for the current platform.
  // Returns the loaded backend type.
  // Idempotent — if already loaded, returns immediately.
  IOBackendType available() {
    if (type_ != IOBackendType::kPread) {
      return type_;
    }
#if defined(__APPLE__) && defined(__MACH__)
#if TARGET_OS_OSX
    return available(IOBackendType::kThreadPoolPread);
#endif
#endif
    return available(IOBackendType::kLibAio);
  }

  // Try to load the requested backend.  Returns the loaded backend type
  // (may differ from requested if the load failed — falls back to kPread).
  // Idempotent — if the same backend is already loaded, returns immediately.
  IOBackendType available(IOBackendType requested) {
    if (type_ == requested && type_ != IOBackendType::kPread) {
      return type_;
    }
#if defined(__APPLE__) && defined(__MACH__)
#if TARGET_OS_OSX
    if (requested == IOBackendType::kThreadPoolPread) {
      type_ = IOBackendType::kThreadPoolPread;
      return type_;
    }
#endif
#endif
#if defined(__linux) || defined(__linux__)
    if (requested == IOBackendType::kLibAio) {
      if (LibAioLoader::Instance().load() &&
          LibAioLoader::Instance().is_available()) {
        type_ = IOBackendType::kLibAio;
        return type_;
      }
    }
#endif
    type_ = IOBackendType::kPread;
    return type_;
  }

  bool is_pread() {
    return available() == IOBackendType::kPread;
  }

  bool is_libaio() {
    return available() == IOBackendType::kLibAio;
  }

  bool is_thread_pool_pread() {
    return available() == IOBackendType::kThreadPoolPread;
  }

  // Returns the loaded backend type.
  IOBackendType type() const {
    return type_;
  }

  // Human-readable name for the selected backend.
  const char *name() const {
    return IOBackendTypeName(type_);
  }

  // Human-readable description for the selected backend.
  const char *description() const {
    return IOBackendDescription(type_);
  }

 private:
  IOBackend() = default;

  IOBackendType type_{IOBackendType::kPread};
};

}  // namespace ailego
}  // namespace zvec
