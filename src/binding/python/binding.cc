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

#include <pybind11/pybind11.h>
#include <zvec/core/interface/diskann_runtime.h>
#include "python_collection.h"
#include "python_config.h"
#include "python_doc.h"
#include "python_param.h"
#include "python_reranker.h"
#include "python_schema.h"
#include "python_type.h"

namespace zvec {

namespace {

// Expose DiskAnn runtime management to Python. DiskAnn is compiled directly
// into _zvec.so, so "initializing" just means eagerly dlopen()-ing libaio and
// caching the result. Tests (and diagnostic tooling) use these entry points
// to force the init up-front and get actionable warnings when libaio is
// missing.
void InitializeDiskAnnRuntimeBindings(pybind11::module_ &m) {
  m.def(
      "init_diskann_runtime",
      [](const std::string &path) { return ::zvec::InitDiskAnnRuntime(path); },
      pybind11::arg("path") = std::string(),
      "Initialize the DiskAnn runtime by loading libaio via dlopen(). "
      "Returns DISKANN_RUNTIME_OK (0) on success, or "
      "DISKANN_RUNTIME_LIBAIO_MISSING if libaio is not available (DiskAnn "
      "falls back to synchronous pread in that case). Returns a negative "
      "code for unsupported platforms.");
  m.def("is_diskann_runtime_ready", &::zvec::IsDiskAnnRuntimeReady,
        "Return True if the DiskAnn runtime has been initialized.");
  m.def("is_libaio_available", &::zvec::IsLibAioAvailable,
        "Return True if libaio is resolvable on this host (required by the "
        "DiskAnn runtime).");

  // Status constants so callers can compare against well-known codes without
  // hard-coding integers.
  m.attr("DISKANN_RUNTIME_OK") = static_cast<int>(::zvec::kDiskAnnRuntimeOk);
  m.attr("DISKANN_RUNTIME_UNSUPPORTED_PLATFORM") =
      static_cast<int>(::zvec::kDiskAnnRuntimeUnsupportedPlatform);
  m.attr("DISKANN_RUNTIME_LIBAIO_MISSING") =
      static_cast<int>(::zvec::kDiskAnnRuntimeLibAioMissing);
  m.attr("DISKANN_RUNTIME_DLOPEN_FAILED") =
      static_cast<int>(::zvec::kDiskAnnRuntimeDlopenFailed);
}

}  // namespace

PYBIND11_MODULE(_zvec, m) {
  m.doc() = "Zvec core module";

  ZVecPyTyping::Initialize(m);
  ZVecPyParams::Initialize(m);
  ZVecPySchemas::Initialize(m);
  ZVecPyReranker::Initialize(m);
  ZVecPyConfig::Initialize(m);
  ZVecPyDoc::Initialize(m);
  ZVecPyCollection::Initialize(m);
  InitializeDiskAnnRuntimeBindings(m);
}
}  // namespace zvec
