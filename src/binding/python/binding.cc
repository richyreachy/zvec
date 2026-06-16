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
#include <zvec/plugin/diskann_plugin.h>
#include "python_collection.h"
#include "python_config.h"
#include "python_doc.h"
#include "python_param.h"
#include "python_reranker.h"
#include "python_schema.h"
#include "python_type.h"

namespace zvec {

namespace {

// DiskAnn is now statically linked. These bindings are kept for backward
// compatibility with Python code that calls load_diskann_plugin() etc.
void InitializeDiskAnnPluginBindings(pybind11::module_ &m) {
  m.def(
      "load_diskann_plugin",
      [](const std::string &path) { return ::zvec::LoadDiskAnnPlugin(path); },
      pybind11::arg("path") = std::string(),
      "No-op (DiskAnn is always available). Returns 0.");
  m.def("is_diskann_plugin_loaded", &::zvec::IsDiskAnnPluginLoaded,
        "Always returns True (DiskAnn is statically linked).");
  m.def("is_libaio_available", &::zvec::IsLibAioAvailable,
        "Always returns True (AIO dependency removed).");

  m.attr("DISKANN_PLUGIN_OK") = static_cast<int>(::zvec::kDiskAnnPluginOk);
  m.attr("DISKANN_PLUGIN_UNSUPPORTED_PLATFORM") =
      static_cast<int>(::zvec::kDiskAnnPluginUnsupportedPlatform);
  m.attr("DISKANN_PLUGIN_LIBAIO_MISSING") =
      static_cast<int>(::zvec::kDiskAnnPluginLibAioMissing);
  m.attr("DISKANN_PLUGIN_DLOPEN_FAILED") =
      static_cast<int>(::zvec::kDiskAnnPluginDlopenFailed);
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
  InitializeDiskAnnPluginBindings(m);
}
}  // namespace zvec
