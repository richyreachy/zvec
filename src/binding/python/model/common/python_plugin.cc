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

#include "python_plugin.h"
#include <string>
#include <zvec/plugin/diskann_plugin.h>

namespace zvec {

void ZVecPyPlugin::Initialize(py::module_ &m) {
  py::enum_<DiskAnnPluginStatus>(m, "DiskAnnPluginStatus")
      .value("OK", kDiskAnnPluginOk)
      // kDiskAnnPluginAlreadyLoaded shares the value 0 with OK; exposing it as
      // an alias would collide in the Python enum, so we only surface OK.
      .value("UNSUPPORTED_PLATFORM", kDiskAnnPluginUnsupportedPlatform)
      .value("LIBAIO_MISSING", kDiskAnnPluginLibAioMissing)
      .value("DLOPEN_FAILED", kDiskAnnPluginDlopenFailed)
      .export_values();

  m.def("is_libaio_available", &IsLibAioAvailable,
        R"pbdoc(Return True if libaio is present on the host and exposes the
symbols required by the DiskAnn plugin (io_setup/io_submit/io_getevents/
io_destroy). Non-destructive probe: safe to call multiple times.)pbdoc");

  m.def(
      "load_diskann_plugin",
      [](const std::string &path) { return LoadDiskAnnPlugin(path); },
      py::arg("path") = std::string(),
      R"pbdoc(Load the DiskAnn plugin shared library (libzvec_diskann_plugin.so)
via dlopen. Returns a DiskAnnPluginStatus value. When ``path`` is empty the
plugin is searched next to the running executable first, then on the platform
default dynamic-linker path.)pbdoc");

  m.def("is_diskann_plugin_loaded", &IsDiskAnnPluginLoaded,
        "Return True if the DiskAnn plugin is currently loaded.");

  m.def("unload_diskann_plugin", &UnloadDiskAnnPlugin,
        R"pbdoc(Unload the DiskAnn plugin. Caller must guarantee that no
DiskAnn objects are still alive and no background threads are executing
DiskAnn code. Returns True when a live handle was released.)pbdoc");
}

}  // namespace zvec
