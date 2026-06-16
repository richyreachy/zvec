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

// NOTE: DiskAnn is now compiled directly into zvec (no separate runtime
// plugin). The APIs below are retained as no-op stubs for backward
// compatibility with existing call-sites (Python bindings, schema validation,
// etc.). They will always succeed.

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

// Return codes for LoadDiskAnnPlugin() (kept for API compat).
enum DiskAnnPluginStatus {
  kDiskAnnPluginOk = 0,
  kDiskAnnPluginUnsupportedPlatform = -1,
  kDiskAnnPluginLibAioMissing = -2,
  kDiskAnnPluginDlopenFailed = -3,
};

// Always returns true (AIO dependency removed).
ZVEC_PLUGIN_EXPORT bool IsLibAioAvailable();

// Always returns kDiskAnnPluginOk (DiskAnn is statically linked).
ZVEC_PLUGIN_EXPORT int LoadDiskAnnPlugin(const std::string &path = "");

// Always returns true (DiskAnn is always available).
ZVEC_PLUGIN_EXPORT bool IsDiskAnnPluginLoaded();

// No-op; returns false.
ZVEC_PLUGIN_EXPORT bool UnloadDiskAnnPlugin();

}  // namespace zvec
