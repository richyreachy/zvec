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

// I/O backend abstraction — public, dependency-free header.
//
// This is the single io_backend.h in the project.  It defines the
// IOBackendType enum, the IOBackendTypeName() helper, and the convenience
// helpers current_io_backend_type() / current_io_backend_description() so that
// public headers can reference IOBackendType without pulling in the internal
// IOBackend singleton, libaio_loader, or the io_uring kernel ABI headers.
//
// The IOBackend singleton (which probes io_uring / libaio at runtime) lives in
// the internal header ailego/io/io_backend_def.h.

#pragma once

#include <string>
#include <zvec/ailego/internal/platform.h>

namespace zvec {
namespace ailego {

// Supported I/O backend types.
//
// Numeric values are part of the C ABI (see zvec_io_backend_type_t in c_api.h):
//   kPread = 0, kLibAio = 1, kIoUring = 2.
enum class IOBackendType {
  kPread = 0,    // Synchronous pread() — no async I/O
  kLibAio = 1,   // libaio loaded at runtime via dlopen()
  kIoUring = 2,  // io_uring via raw kernel syscalls (zero dependency)
};

// Returns a human-readable name for the given backend type
// ("pread", "libaio", "io_uring", or "unknown").
inline const char *IOBackendTypeName(IOBackendType type) {
  switch (type) {
    case IOBackendType::kPread:
      return "pread";
    case IOBackendType::kLibAio:
      return "libaio";
    case IOBackendType::kIoUring:
      return "io_uring";
  }
  return "unknown";
}

// Returns the currently active I/O backend type.
// Triggers backend initialization on first call (io_uring > libaio > pread).
IOBackendType current_io_backend_type();

// Returns a human-readable description of the currently active I/O backend.
// When only pread is available, includes installation guidance for libaio.
std::string current_io_backend_description();

}  // namespace ailego
}  // namespace zvec
