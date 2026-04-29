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

#ifndef ZVEC_PLUGIN_DISKANN_PLUGIN_H
#define ZVEC_PLUGIN_DISKANN_PLUGIN_H

#include <string>

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
  kDiskAnnPluginAlreadyLoaded = 0,
  kDiskAnnPluginUnsupportedPlatform = -1,
  kDiskAnnPluginLibAioMissing = -2,
  kDiskAnnPluginDlopenFailed = -3,
};

// Returns true if libaio is present on the host and the minimum set of symbols
// required by the DiskAnn plugin (io_setup / io_submit / io_getevents /
// io_destroy) can be resolved at runtime.
//
// The probe is non-destructive: it does not permanently load libaio; it is
// safe to call multiple times.
ZVEC_PLUGIN_EXPORT bool IsLibAioAvailable();

// Load the DiskAnn plugin shared library (libzvec_diskann_plugin.so) via
// dlopen(). Upon success the DiskAnn builder / searcher / streamer factory
// entries are registered as a side effect of the plugin's static
// initializers, and IndexFactory::CreateBuilder("DiskAnnBuilder") starts
// returning a valid object.
//
// Parameters:
//   path - optional explicit path to the plugin. When empty, the following
//          locations are tried in order:
//            1. next to the currently running executable (resolved via
//               /proc/self/exe on Linux);
//            2. the platform default dynamic-linker search path (RPATH,
//               LD_LIBRARY_PATH, /etc/ld.so.conf, ...).
//
// Returns kDiskAnnPluginOk on success (also when the plugin is already
// loaded).
ZVEC_PLUGIN_EXPORT int LoadDiskAnnPlugin(const std::string &path = "");

// Returns true if the DiskAnn plugin is currently loaded in this process.
ZVEC_PLUGIN_EXPORT bool IsDiskAnnPluginLoaded();

// Unload the DiskAnn plugin. Note: unloading a library that has registered
// itself into global factory singletons is inherently racy; callers must
// guarantee that no DiskAnn objects are still alive and no background
// threads are executing DiskAnn code before calling this function. Returns
// true when a live handle was released.
ZVEC_PLUGIN_EXPORT bool UnloadDiskAnnPlugin();

}  // namespace zvec

#endif  // ZVEC_PLUGIN_DISKANN_PLUGIN_H
