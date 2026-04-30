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
#include <vector>
#include <zvec/core/framework/index_logger.h>
#include <zvec/plugin/diskann_plugin.h>

#if defined(__linux__) || defined(__linux) || defined(__APPLE__)
#include <dlfcn.h>
#include <unistd.h>
#endif

#if defined(__linux__) || defined(__linux)
#include <limits.h>
#endif

namespace zvec {

namespace {

#if defined(__linux__) || defined(__linux)
constexpr const char *kPluginFileName = "libzvec_diskann_plugin.so";
// Candidate soname list. On Ubuntu 24.04 the libaio package was renamed with
// the t64 suffix (64-bit time_t transition), so we probe both spellings.
constexpr const char *kLibAioSoNames[] = {
    "libaio.so.1",
    "libaio.so.1t64",
};
constexpr bool kPlatformSupportsDiskAnnPlugin = true;
#elif defined(__APPLE__)
constexpr const char *kPluginFileName = "libzvec_diskann_plugin.dylib";
constexpr bool kPlatformSupportsDiskAnnPlugin = false;
#else
constexpr const char *kPluginFileName = "zvec_diskann_plugin.dll";
constexpr bool kPlatformSupportsDiskAnnPlugin = false;
#endif

// Global plugin handle. Nullptr means "not loaded".
std::atomic<void *> g_plugin_handle{nullptr};
std::mutex g_plugin_mutex;

#if defined(__linux__) || defined(__linux) || defined(__APPLE__)

// Resolve the directory containing the currently running executable, so we
// can look for the plugin next to it regardless of the working directory.
std::string GetExecutableDir() {
#if defined(__linux__) || defined(__linux)
  char buf[PATH_MAX];
  ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (n <= 0) {
    return {};
  }
  buf[n] = '\0';
  std::string path(buf);
  auto slash = path.find_last_of('/');
  if (slash == std::string::npos) {
    return {};
  }
  return path.substr(0, slash);
#else
  return {};
#endif
}

// Resolve the directory containing the shared object that hosts this
// function. For Python wheels this is the directory of
// ``_zvec.cpython-*.so``; for regular C++ binaries it is the directory of
// ``libzvec_core.so``. In either case the DiskAnn plugin is shipped
// alongside, so this is the most reliable lookup location.
//
// NOTE: we pass the address of an *exported* function (LoadDiskAnnPlugin)
// rather than one in this anonymous namespace, because dladdr() on a symbol
// with internal linkage can report the main executable instead of the
// hosting shared object when the translation unit is whole-archived into
// another .so.
std::string ResolveHostingSoDir() {
  ::Dl_info info{};
  if (::dladdr(reinterpret_cast<void *>(&::zvec::LoadDiskAnnPlugin), &info) ==
          0 ||
      info.dli_fname == nullptr) {
    return {};
  }
  std::string path(info.dli_fname);
  auto slash = path.find_last_of('/');
  if (slash == std::string::npos) {
    return {};
  }
  return path.substr(0, slash);
}

// Full path of the shared object that hosts LoadDiskAnnPlugin, or empty
// string on failure.
std::string ResolveHostingSoPath() {
  ::Dl_info info{};
  if (::dladdr(reinterpret_cast<void *>(&::zvec::LoadDiskAnnPlugin), &info) ==
          0 ||
      info.dli_fname == nullptr) {
    return {};
  }
  return std::string(info.dli_fname);
}

// Promote the hosting shared object (e.g. the Python extension module
// ``_zvec.cpython-*.so``) to the global symbol scope. Python loads C
// extensions with RTLD_LOCAL by default, which means their C++ symbols are
// invisible to subsequently dlopen(RTLD_GLOBAL)ed libraries. Without this
// promotion, the DiskAnn plugin's undefined references to zvec:: symbols
// cannot be resolved against the already-loaded host module and dlopen
// fails with messages like:
//
//   undefined symbol: _ZN4zvec6ailego6Logger10LEVEL_INFOE
//
// Using RTLD_NOLOAD re-opens the existing image without loading it again,
// while RTLD_GLOBAL merges its symbols into the global scope. This is a
// no-op for hosts that were already loaded with RTLD_GLOBAL.
void PromoteHostingSoToGlobal() {
  const std::string host = ResolveHostingSoPath();
  if (host.empty()) {
    return;
  }
  void *h = ::dlopen(host.c_str(), RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
  if (h == nullptr) {
    const char *err = ::dlerror();
    LOG_WARN("Could not promote host '%s' to RTLD_GLOBAL: %s", host.c_str(),
             err ? err : "unknown");
    return;
  }
  // We purposely keep the handle refcount incremented for the life of the
  // process: there is no safe point at which to dlclose() it, and doing so
  // would only decrement the count anyway (the image remains mapped because
  // Python still holds its own reference).
  (void)h;
}

// Build the list of candidate paths for the plugin.
std::vector<std::string> BuildCandidatePaths(const std::string &explicit_path) {
  std::vector<std::string> candidates;
  if (!explicit_path.empty()) {
    candidates.push_back(explicit_path);
    return candidates;
  }

  // 1. Directory of the library that hosts LoadDiskAnnPlugin (e.g. the
  //    Python extension module or libzvec_core). This works for Python,
  //    C++ embedding, and most packaging layouts.
  const std::string own_dir = ResolveHostingSoDir();
  if (!own_dir.empty()) {
    candidates.push_back(own_dir + "/" + kPluginFileName);
  }
  // 2. Directory of the running executable. Useful for self-contained C++
  //    tools that drop the plugin next to their binary.
  const std::string exe_dir = GetExecutableDir();
  if (!exe_dir.empty() && exe_dir != own_dir) {
    candidates.push_back(exe_dir + "/" + kPluginFileName);
  }
  // 3. Fallback: rely on the dynamic linker's default search path
  //    (RPATH / LD_LIBRARY_PATH / /etc/ld.so.conf).
  candidates.emplace_back(kPluginFileName);
  return candidates;
}

#endif  // linux || apple

}  // namespace

bool IsLibAioAvailable() {
#if defined(__linux__) || defined(__linux)
  const char *kRequiredSymbols[] = {"io_setup", "io_submit", "io_getevents",
                                    "io_destroy"};
  for (const char *soname : kLibAioSoNames) {
    // RTLD_LAZY keeps the cost low; we only need to know whether the library
    // is resolvable and exposes the symbols DiskAnn actually calls.
    void *handle = ::dlopen(soname, RTLD_LAZY);
    if (handle == nullptr) {
      continue;
    }
    bool ok = true;
    for (const char *sym : kRequiredSymbols) {
      if (::dlsym(handle, sym) == nullptr) {
        ok = false;
        break;
      }
    }
    ::dlclose(handle);
    if (ok) {
      return true;
    }
  }
  return false;
#else
  return false;
#endif
}

bool IsDiskAnnPluginLoaded() {
  return g_plugin_handle.load(std::memory_order_acquire) != nullptr;
}

int LoadDiskAnnPlugin(const std::string &path) {
  if (!kPlatformSupportsDiskAnnPlugin) {
    LOG_ERROR(
        "DiskAnn plugin is not supported on this platform; it is only "
        "available on Linux x86_64 with libaio.");
    return kDiskAnnPluginUnsupportedPlatform;
  }

#if defined(__linux__) || defined(__linux)
  // Fast path: already loaded.
  if (g_plugin_handle.load(std::memory_order_acquire) != nullptr) {
    return kDiskAnnPluginAlreadyLoaded;
  }

  std::lock_guard<std::mutex> lock(g_plugin_mutex);
  if (g_plugin_handle.load(std::memory_order_relaxed) != nullptr) {
    return kDiskAnnPluginAlreadyLoaded;
  }

  if (!IsLibAioAvailable()) {
    LOG_ERROR(
        "libaio is not available on this host; DiskAnn plugin cannot be "
        "loaded. Install libaio1 (or equivalent) before calling "
        "zvec::LoadDiskAnnPlugin().");
    return kDiskAnnPluginLibAioMissing;
  }

  const std::vector<std::string> candidates = BuildCandidatePaths(path);
  // Ensure the hosting module's C++ symbols (zvec::*) are visible to the
  // plugin at dlopen time. See PromoteHostingSoToGlobal() for the rationale.
  PromoteHostingSoToGlobal();
  void *handle = nullptr;
  std::string last_error;
  for (const std::string &candidate : candidates) {
    // RTLD_GLOBAL so the plugin's factory registrations (which live in the
    // plugin's own static-init code) can reference symbols from the main
    // library, and any callers that later dlsym against the process can see
    // the plugin's symbols.
    handle = ::dlopen(candidate.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (handle != nullptr) {
      LOG_INFO("Loaded DiskAnn plugin from: %s", candidate.c_str());
      break;
    }
    const char *err = ::dlerror();
    last_error = err ? err : "unknown dlopen error";
    LOG_WARN("dlopen(%s) failed: %s", candidate.c_str(), last_error.c_str());
  }

  if (handle == nullptr) {
    LOG_ERROR("Failed to load DiskAnn plugin; last error: %s",
              last_error.c_str());
    return kDiskAnnPluginDlopenFailed;
  }

  g_plugin_handle.store(handle, std::memory_order_release);
  return kDiskAnnPluginOk;
#else
  (void)path;
  return kDiskAnnPluginUnsupportedPlatform;
#endif
}

bool UnloadDiskAnnPlugin() {
#if defined(__linux__) || defined(__linux)
  std::lock_guard<std::mutex> lock(g_plugin_mutex);
  void *handle = g_plugin_handle.exchange(nullptr, std::memory_order_acq_rel);
  if (handle == nullptr) {
    return false;
  }
  if (::dlclose(handle) != 0) {
    const char *err = ::dlerror();
    LOG_WARN("dlclose for DiskAnn plugin returned non-zero: %s",
             err ? err : "unknown");
  }
  return true;
#else
  return false;
#endif
}

}  // namespace zvec
