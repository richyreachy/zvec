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

#include "diskann_file_reader.h"
#include <cassert>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <zvec/core/framework/index_logger.h>

#define MAX_EVENTS 1024

namespace zvec {
namespace core {

#if (defined(__linux) || defined(__linux__))
typedef struct io_event io_event_t;
typedef struct iocb iocb_t;
#endif

int setup_io_ctx(IOContext &ctx, const std::string &backend) {
#if (defined(__linux) || defined(__linux__))
  ctx.initialized = false;
  ctx.aio_ctx = nullptr;

  if (backend == "pread") {
    ctx.backend = IOBackend::kPRead;
    ctx.initialized = true;
    return 0;
  }
  if (backend == "io_uring") {
    int ret = io_uring_queue_init(MAX_EVENTS, &ctx.uring, 0);
    if (ret == 0) {
      ctx.backend = IOBackend::kIOUring;
      ctx.initialized = true;
      return 0;
    }
    LOG_WARN("io_uring_queue_init failed; ret=%d, falling back to aio", ret);
  }
  // Default: aio
  ctx.backend = IOBackend::kAIO;
  int ret = io_setup(MAX_EVENTS, &ctx.aio_ctx);
  if (ret != 0) {
    LOG_WARN("io_setup failed; ret=%d, falling back to pread", ret);
    ctx.backend = IOBackend::kPRead;
    ctx.initialized = true;
    return 0;
  }
  ctx.initialized = true;
  return 0;
#else
  return 0;
#endif
}

int destroy_io_ctx(IOContext &ctx) {
#if (defined(__linux) || defined(__linux__))
  if (!ctx.initialized) return 0;
  if (ctx.backend == IOBackend::kIOUring) {
    io_uring_queue_exit(&ctx.uring);
  } else if (ctx.backend == IOBackend::kAIO) {
    io_destroy(ctx.aio_ctx);
  }
  ctx.initialized = false;
  return 0;
#else
  return 0;
#endif
}

static int execute_io_pread(int fd, std::vector<AlignedRead> &read_reqs) {
  for (auto &req : read_reqs) {
    ssize_t bytes_read = ::pread(fd, req.buf, req.len, req.offset);
    if (bytes_read < 0) {
      LOG_ERROR("pread failed; errno=%d, %s, offset=%lu, len=%lu", errno,
                ::strerror(errno), (unsigned long)req.offset,
                (unsigned long)req.len);
      return IndexError_Runtime;
    }
    if ((size_t)bytes_read != req.len) {
      LOG_ERROR("pread short read; got=%zd, expected=%lu", bytes_read,
                (unsigned long)req.len);
      return IndexError_Runtime;
    }
  }
  return 0;
}

static int execute_io_uring(struct io_uring &ring, int fd,
                            std::vector<AlignedRead> &read_reqs) {
  if (read_reqs.empty()) return 0;
  uint64_t n_ops = read_reqs.size();

  for (uint64_t j = 0; j < n_ops; j++) {
    struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
    if (sqe == nullptr) {
      io_uring_submit(&ring);
      sqe = io_uring_get_sqe(&ring);
      if (sqe == nullptr) {
        LOG_WARN("io_uring SQ full; falling back to pread");
        return execute_io_pread(fd, read_reqs);
      }
    }
    io_uring_prep_read(sqe, fd, read_reqs[j].buf, (uint32_t)read_reqs[j].len,
                       read_reqs[j].offset);
    sqe->user_data = j;
  }

  int ret = io_uring_submit_and_wait(&ring, (unsigned)n_ops);
  if (ret < 0) {
    LOG_WARN("io_uring_submit_and_wait failed; ret=%d, falling back to pread",
             ret);
    return execute_io_pread(fd, read_reqs);
  }

  std::vector<struct io_uring_cqe *> cqes(n_ops);
  unsigned reaped = io_uring_peek_batch_cqe(&ring, cqes.data(), (unsigned)n_ops);
  bool all_ok = true;
  for (unsigned i = 0; i < reaped; i++) {
    uint64_t idx = cqes[i]->user_data;
    if (cqes[i]->res < 0 || (uint64_t)cqes[i]->res != read_reqs[idx].len) {
      LOG_WARN("io_uring read failed: req=%lu, res=%d, expected=%lu",
               (unsigned long)idx, cqes[i]->res,
               (unsigned long)read_reqs[idx].len);
      all_ok = false;
    }
  }
  io_uring_cq_advance(&ring, reaped);
  if (!all_ok) return execute_io_pread(fd, read_reqs);
  return 0;
}

int execute_io(IOContext &ctx, int fd,
               std::vector<AlignedRead> &read_reqs, uint64_t n_retries = 0) {
#if (defined(__linux) || defined(__linux__))
  if (!ctx.initialized || ctx.backend == IOBackend::kPRead) {
    return execute_io_pread(fd, read_reqs);
  }
  if (ctx.backend == IOBackend::kIOUring) {
    return execute_io_uring(ctx.uring, fd, read_reqs);
  }
  // kAIO — libaio path
  uint64_t iters = DiskAnnUtil::div_round_up(read_reqs.size(), MAX_EVENTS);

  for (uint64_t iter = 0; iter < iters; iter++) {
    uint64_t n_ops = std::min((uint64_t)read_reqs.size() - (iter * MAX_EVENTS),
                              (uint64_t)MAX_EVENTS);

    std::vector<iocb_t *> cbs(n_ops, nullptr);
    std::vector<io_event_t> evts(n_ops);
    std::vector<struct iocb> cb(n_ops);
    for (uint64_t j = 0; j < n_ops; j++) {
      io_prep_pread(cb.data() + j, fd, read_reqs[j + iter * MAX_EVENTS].buf,
                    read_reqs[j + iter * MAX_EVENTS].len,
                    read_reqs[j + iter * MAX_EVENTS].offset);
    }

    for (uint64_t i = 0; i < n_ops; i++) {
      cbs[i] = cb.data() + i;
    }

    size_t n_tries = 0;
    // Phase 1: io_submit with retry.
    while (true) {
      int ret = io_submit(ctx.aio_ctx, (int64_t)n_ops, cbs.data());
      if (ret == (int)n_ops) {
        break;
      }
      if ((ret == -EAGAIN || ret == -EINTR) && n_tries < n_retries) {
        n_tries++;
        continue;
      }
      LOG_WARN(
          "io_submit failed; returned: %d, expected=%lu. falling back to "
          "pread",
          ret, n_ops);
      return execute_io_pread(fd, read_reqs);
    }

    // Phase 2: io_getevents with retry (never re-submits).
    n_tries = 0;
    while (true) {
      int ret = io_getevents(ctx.aio_ctx, (int64_t)n_ops, (int64_t)n_ops,
                             evts.data(), nullptr);
      if (ret == (int)n_ops) {
        break;
      }
      if (ret == -EINTR && n_tries < n_retries) {
        n_tries++;
        continue;
      }
      LOG_WARN(
          "io_getevents failed; returned: %d, expected=%lu, errno=%d, %s, "
          "falling back to pread",
          ret, n_ops, errno, ::strerror(-ret));
      return execute_io_pread(fd, read_reqs);
    }

    // Phase 3: verify each completed read (res must equal requested length).
    bool all_ok = true;
    for (uint64_t i = 0; i < n_ops; i++) {
      int64_t expected_len = read_reqs[i + iter * MAX_EVENTS].len;
      if ((int64_t)evts[i].res != expected_len) {
        LOG_WARN("aio request %zu failed: res=%ld, expected=%ld, offset=%zu",
                 (size_t)i, (long)evts[i].res, (long)expected_len,
                 (size_t)read_reqs[i + iter * MAX_EVENTS].offset);
        all_ok = false;
      }
    }
    if (!all_ok) {
      return execute_io_pread(fd, read_reqs);
    }
  }

  return 0;
#else
  return execute_io_pread(fd, read_reqs);
#endif
}

LinuxAlignedFileReader::LinuxAlignedFileReader(int file_desc) {
  this->file_desc = file_desc;
}

LinuxAlignedFileReader::LinuxAlignedFileReader() {
  this->file_desc = -1;
}

LinuxAlignedFileReader::~LinuxAlignedFileReader() {
  deregister_all_threads();
  if (file_desc >= 0) {
    ::close(file_desc);
    file_desc = -1;
  }
}

IOContext &LinuxAlignedFileReader::get_ctx() {
  std::unique_lock<std::mutex> lk(ctx_mut);
  auto it = ctx_map.find(std::this_thread::get_id());
  if (it == ctx_map.end()) {
    LOG_ERROR("bad thread access; returning -1 as io_context_t");
    return this->bad_ctx;
  } else {
    return it->second;
  }
}

void LinuxAlignedFileReader::register_thread() {
#if (defined(__linux) || defined(__linux__))
  auto thread_id = std::this_thread::get_id();
  std::unique_lock<std::mutex> lk(ctx_mut);
  if (ctx_map.find(thread_id) != ctx_map.end()) {
    LOG_ERROR("multiple calls to register_thread from the same thread");

    return;
  }

  IOContext ctx{};

  int ret = setup_io_ctx(ctx);
  if (ret != 0) {
    lk.unlock();
    if (ret == -EAGAIN) {
      LOG_ERROR(
          "io_setup failed with EAGAIN: Consider increasing "
          "/proc/sys/fs/aio-max-nr");
    } else {
      LOG_ERROR("io_setup failed; returned: %d, %s", ret, ::strerror(-ret));
      ;
    }
  } else {
    LOG_INFO("allocating ctx: %lu", (uint64_t)ctx.aio_ctx);

    ctx_map[thread_id] = ctx;
  }

  lk.unlock();
#endif
}

void LinuxAlignedFileReader::deregister_thread() {
#if (defined(__linux) || defined(__linux__))
  auto thread_id = std::this_thread::get_id();
  IOContext ctx{};

  {
    std::lock_guard<std::mutex> lk(ctx_mut);
    auto it = ctx_map.find(thread_id);
    if (it == ctx_map.end()) {
      LOG_ERROR("deregister_thread: thread not registered");
      return;
    }
    ctx = it->second;
    ctx_map.erase(it);
  }

  // io_destroy is a syscall; keep it outside the lock to avoid blocking others
  destroy_io_ctx(ctx);
  LOG_INFO("returned ctx from thread");
#endif
}

void LinuxAlignedFileReader::deregister_all_threads() {
#if (defined(__linux) || defined(__linux__))
  std::unique_lock<std::mutex> lk(ctx_mut);
  for (auto x = ctx_map.begin(); x != ctx_map.end(); x++) {
    IOContext ctx = x->second;
    destroy_io_ctx(ctx);
  }
  ctx_map.clear();
#endif
}

void LinuxAlignedFileReader::open(const std::string &fname) {
  int flags = O_RDONLY;

#if defined(__linux__) || defined(__linux)
  flags |= O_DIRECT | O_LARGEFILE;
#endif

  this->file_desc = ::open(fname.c_str(), flags);

#if defined(__linux__) || defined(__linux)
  // O_DIRECT may not be supported on all filesystems (e.g. tmpfs, overlay).
  // Fall back to regular buffered I/O when it fails.
  if (this->file_desc == -1) {
    LOG_WARN(
        "open with O_DIRECT failed for %s (errno=%d: %s), "
        "falling back to buffered I/O",
        fname.c_str(), errno, ::strerror(errno));
    this->file_desc = ::open(fname.c_str(), O_RDONLY | O_LARGEFILE);
  }
#endif

  if (this->file_desc == -1) {
    LOG_ERROR("Failed to open file: %s (errno=%d: %s)", fname.c_str(), errno,
              ::strerror(errno));
  }

  LOG_INFO("Opened file : %s", fname.c_str());
}

void LinuxAlignedFileReader::close() {
  if (file_desc >= 0) {
    ::close(file_desc);
    file_desc = -1;
  }
}

int LinuxAlignedFileReader::read(std::vector<AlignedRead> &read_reqs,
                                 IOContext &ctx, bool async) {
  if (async == true) {
    LOG_WARN("Async currently not supported");
  }

  if (this->file_desc == -1) {
    LOG_ERROR("Attempt to read from invalid file descriptor");
    return IndexError_Runtime;
  }

  int ret = execute_io(ctx, this->file_desc, read_reqs);

  return ret;
}

#if (defined(__linux) || defined(__linux__))
int LinuxAlignedFileReader::submit(PendingBatch &batch,
                                   std::vector<AlignedRead> &read_reqs,
                                   IOContext &ctx) {
  batch.n_submitted = 0;
  batch.n_reaped = 0;
  batch.used_pread = false;
  batch.cbs.clear();
  batch.cb_ptrs.clear();

  if (this->file_desc == -1) {
    LOG_ERROR("submit: invalid file descriptor");
    return IndexError_Runtime;
  }

  if (read_reqs.empty()) {
    return 0;
  }

  uint32_t n_ops = (uint32_t)read_reqs.size();

  // --- PRead backend ---
  if (!ctx.initialized || ctx.backend == IOBackend::kPRead) {
    int ret = execute_io_pread(this->file_desc, read_reqs);
    if (ret != 0) return ret;
    batch.used_pread = true;
    batch.n_submitted = n_ops;
    return 0;
  }

  // --- IO Uring backend ---
  if (ctx.backend == IOBackend::kIOUring) {
    for (uint32_t j = 0; j < n_ops; j++) {
      struct io_uring_sqe *sqe = io_uring_get_sqe(&ctx.uring);
      if (sqe == nullptr) {
        io_uring_submit(&ctx.uring);
        sqe = io_uring_get_sqe(&ctx.uring);
        if (sqe == nullptr) {
          LOG_WARN("submit: io_uring SQ full, falling back to pread");
          int ret = execute_io_pread(this->file_desc, read_reqs);
          if (ret != 0) return ret;
          batch.used_pread = true;
          batch.n_submitted = n_ops;
          return 0;
        }
      }
      io_uring_prep_read(sqe, this->file_desc, read_reqs[j].buf,
                         (uint32_t)read_reqs[j].len, read_reqs[j].offset);
      sqe->user_data = (uint64_t)j;
    }
    int ret = io_uring_submit(&ctx.uring);
    if (ret < 0 || ret != (int)n_ops) {
      LOG_WARN("submit: io_uring_submit returned %d (expected %u), falling back to pread",
               ret, n_ops);
      int pread_ret = execute_io_pread(this->file_desc, read_reqs);
      if (pread_ret != 0) return pread_ret;
      batch.used_pread = true;
      batch.n_submitted = n_ops;
      return 0;
    }
    batch.n_submitted = n_ops;
    return 0;
  }

  // --- AIO backend (libaio) ---
  batch.cbs.resize(n_ops);
  batch.cb_ptrs.resize(n_ops);

  for (uint32_t j = 0; j < n_ops; j++) {
    io_prep_pread(&batch.cbs[j], this->file_desc, read_reqs[j].buf,
                  read_reqs[j].len, read_reqs[j].offset);
    batch.cbs[j].data = (void *)(uintptr_t)j;
    batch.cb_ptrs[j] = &batch.cbs[j];
  }

  int ret = io_submit(ctx.aio_ctx, (int64_t)n_ops, batch.cb_ptrs.data());
  if (ret == (int)n_ops) {
    batch.n_submitted = n_ops;
    return 0;
  }

  LOG_WARN("submit: io_submit returned %d (expected %u), falling back to pread",
           ret, n_ops);
  int pread_ret = execute_io_pread(this->file_desc, read_reqs);
  if (pread_ret != 0) {
    return pread_ret;
  }
  batch.used_pread = true;
  batch.n_submitted = n_ops;
  return 0;
}

int LinuxAlignedFileReader::get_completed(PendingBatch &batch, IOContext &ctx,
                                          int min_completed,
                                          std::vector<uint32_t> &completed_indices) {
  completed_indices.clear();

  if (batch.n_reaped >= batch.n_submitted) {
    return 0;
  }

  if (batch.used_pread) {
    for (uint32_t i = batch.n_reaped; i < batch.n_submitted; i++) {
      completed_indices.push_back(i);
    }
    batch.n_reaped = batch.n_submitted;
    return (int)completed_indices.size();
  }

  uint32_t n_remaining = batch.n_submitted - batch.n_reaped;
  int min_req = std::min((int)n_remaining, min_completed);
  if (min_req < 1) min_req = 1;

  // --- IO Uring backend ---
  if (ctx.backend == IOBackend::kIOUring) {
    struct io_uring_cqe *first;
    int ret = io_uring_wait_cqe(&ctx.uring, &first);
    if (ret < 0) {
      LOG_ERROR("get_completed: io_uring_wait_cqe failed, ret=%d", ret);
      return IndexError_Runtime;
    }
    int max_batch = std::min((int)n_remaining, 64);
    struct io_uring_cqe *cqes[64];
    unsigned reaped =
        io_uring_peek_batch_cqe(&ctx.uring, cqes, (unsigned)max_batch);
    for (unsigned i = 0; i < reaped; i++) {
      uint32_t idx = (uint32_t)cqes[i]->user_data;
      if (cqes[i]->res < 0) {
        LOG_WARN("get_completed: io_uring read %u failed: res=%d", idx,
                 cqes[i]->res);
      }
      completed_indices.push_back(idx);
    }
    io_uring_cq_advance(&ctx.uring, reaped);
    batch.n_reaped += (uint32_t)reaped;
    return (int)reaped;
  }

  // --- AIO backend (libaio) ---
  std::vector<io_event_t> evts(n_remaining);
  int ret = io_getevents(ctx.aio_ctx, (int64_t)min_req, (int64_t)n_remaining,
                         evts.data(), nullptr);
  if (ret < 0) {
    LOG_ERROR("get_completed: io_getevents failed, ret=%d", ret);
    return IndexError_Runtime;
  }

  for (int i = 0; i < ret; i++) {
    uint32_t idx = (uint32_t)(uintptr_t)evts[i].data;
    if ((int64_t)evts[i].res != (int64_t)batch.cbs[idx].u.c.nbytes) {
      LOG_WARN("get_completed: read %u failed: res=%ld, expected=%ld",
               idx, (long)evts[i].res, (long)batch.cbs[idx].u.c.nbytes);
    }
    completed_indices.push_back(idx);
  }

  batch.n_reaped += (uint32_t)ret;
  return ret;
}
#endif


}  // namespace core
}  // namespace zvec
