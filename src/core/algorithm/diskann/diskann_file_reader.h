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

#define MAX_IO_DEPTH 128

#include <fcntl.h>

#if (defined(__linux) || defined(__linux__))
#include <libaio.h>
#endif

#include <sys/event.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <atomic>
#include <vector>
#include <zvec/core/framework/index_context.h>
#include "diskann_util.h"

namespace zvec {
namespace core {

#if (defined(__linux) || defined(__linux__))
typedef io_context_t IOContext;
#else
typedef uint32_t IOContext;
#endif

int setup_io_ctx(IOContext &ctx);
int destroy_io_ctx(IOContext &ctx);

struct AlignedRead {
  uint64_t offset;
  uint64_t len;
  void *buf;

  AlignedRead() : offset(0), len(0), buf(nullptr) {}

  AlignedRead(uint64_t offset, uint64_t len, void *buf)
      : offset(offset), len(len), buf(buf) {
    ailego_assert(static_cast<size_t>(offset) % 512 == 0);
    ailego_assert(static_cast<size_t>(len) % 512 == 0);
    ailego_assert(reinterpret_cast<size_t>(buf) % 512 == 0);
  }
};

class AlignedFileReader {
 protected:
  std::map<std::thread::id, IOContext> ctx_map;
  std::mutex ctx_mut;

 public:
  virtual IOContext &get_ctx() = 0;

  virtual ~AlignedFileReader() {};

  virtual void register_thread() = 0;
  virtual void deregister_thread() = 0;
  virtual void deregister_all_threads() = 0;

  virtual void open(const std::string &fname) = 0;
  virtual void close() = 0;

  virtual int read(std::vector<AlignedRead> &read_reqs, IOContext &ctx,
                   bool async = false) = 0;
};

class LinuxAlignedFileReader : public AlignedFileReader {
 private:
  uint64_t file_sz;
  int file_desc;

  IOContext bad_ctx = (IOContext)-1;

 public:
  LinuxAlignedFileReader();
  LinuxAlignedFileReader(int file_desc);
  ~LinuxAlignedFileReader();

 public:
  IOContext &get_ctx();

  void register_thread();
  void deregister_thread();
  void deregister_all_threads();
  void open(const std::string &fname);
  void close();

  int read(std::vector<AlignedRead> &read_reqs, IOContext &ctx,
           bool async = false);
};

class KQueueAlignedFileReader {
 public:
  explicit KQueueAlignedFileReader(int fd = -1) : file_desc(fd) {}

  ~KQueueAlignedFileReader() {
    close();
  }

  void open(const std::string &fname) {
    int flags = O_RDONLY;
#ifdef __APPLE__
    // macOS 通常不需要也不完全支持 O_DIRECT 到 block device 的严格语义，
    // 但为了性能最大化，我们尝试使用 O_NOFATCACHING (如果可能) 或仅依赖 OS
    // 缓存策略。 在 macOS 上，通常直接使用 O_RDONLY 配合大页面对齐即可。
#else
#ifdef __linux__
    flags |= O_DIRECT | O_LARGEFILE;
#endif
#endif
    this->file_desc = ::open(fname.c_str(), flags);
    if (this->file_desc == -1) {
      std::perror("Failed to open file");
      return;
    }
    std::cout << "Opened file: " << fname << std::endl;
  }

  void close() {
    if (this->file_desc != -1 && this->file_desc != 0) {
      ::close(this->file_desc);
      this->file_desc = -1;
    }
  }

  /**
   * 模拟 aio_read 的批量执行
   * 注意：kqueue 本质上是边缘触发或水平触发的通知机制。
   * 对于顺序读，我们通常一次性提交所有请求，然后 kqueue 会通知哪些完成了。
   * 为了简化且高效，这里实现一个“收集所有 pending 请求 -> 注册 kq ->
   * 轮询直到全部完成”的逻辑。
   */
  int read(std::vector<AlignedRead> &read_reqs, IOContext &ctx,
           bool async = false) {
    if (this->file_desc == -1) return -1;
    if (read_reqs.empty()) return 0;

    // 1. 为每个请求分配一个 kqueue 事件槽位
    // 在真实的高性能场景中，通常不会为每个小请求开一个 kq
    // entry，而是将多个小请求聚合成一个大请求，或者使用线程池。
    //  这里为了演示 kqueue 的用法，我们假设每个 AlignedRead
    //  对应一次系统调用准备。

    // 初始化 kqueue
    int kq = kqueue();
    if (kq == -1) {
      std::perror("kqueue failed");
      return -1;
    }

    // 我们将所有的 read 请求视为需要被触发的 IO 操作。
    // 但是，kqueue 监听的是 FD 的可读性。
    // 问题：普通的 pread 是阻塞的。要实现类似 aio 的效果，我们需要：
    // 1. 标记所有文件描述符为 non-blocking (O_NONBLOCK)。
    // 2. 发起预读 (pread) 可能会立即返回 EAGAIN 或 EWOULDBLOCK。
    // 3. 将 FD 注册到 kqueue 监听可读事件。
    // 4. 当 kqueue 返回时，再次尝试 pread。

    // 优化策略：由于 kqueue 不能像 io_uring 那样原生支持批量预取，
    // 这里的实现模式是：
    // 1. 将所有需要的数据拷贝到用户态缓冲区。
    // 2. 为了模拟 aio 的“提交即忘”，实际上在单线程模型下，
    //    最高效的方式是利用 kqueue 等待文件描述符可读，然后批量执行 read。

    // 简单且有效的模式：
    // 我们不逐条发送 pread 给内核挂起，而是利用 kqueue 确保数据准备好。
    // 但由于是同一个 fd，我们主要解决的是“如何知道数据何时可用”。
    // 实际上，对于顺序
    // IO，操作系统通常会立即满足本地磁盘请求，除非涉及网络或慢速介质。
    // 真正的价值在于处理大量并发连接或混合读写。

    // 下面实现一个简化的 kqueue 驱动的事件循环来服务这些请求

    // struct kevent change_list[64];
    // struct kevent event_list[64];
    int nchanges = 0;

    // 将当前 fd 注册到 kqueue
    struct kevent ke;
    EV_SET(&ke, this->file_desc, EVFILT_READ, EV_ADD | EV_CLEAR, 0, 0, nullptr);
    const struct kevent *change_arr[] = {&ke};

    // 设置文件为非阻塞，以便 kqueue 工作正常
    int status = fcntl(this->file_desc, F_GETFL, 0);
    if (status != -1) {
      status = fcntl(this->file_desc, F_SETFL, status | O_NONBLOCK);
    }

    // 开始主循环处理请求
    // 为了模拟 aio_submit + aio_wait，我们遍历所有请求
    for (size_t i = 0; i < read_reqs.size(); ++i) {
      AlignedRead &req = read_reqs[i];

      // 尝试直接读取。如果是本地 SSD，这通常是非阻塞成功的，除非数据不在内存。
      // 如果失败 (EAGAIN)，则进入 kqueue 等待。

      // 为了演示 kqueue 的核心逻辑，我们构建一个通用的 wait_and_read 辅助逻辑
      while (true) {
        ssize_t bytes_read =
            pread(this->file_desc, req.buf, req.len, req.offset);

        if (bytes_read > 0) {
          // 成功读取
          break;
        } else if (bytes_read == -1) {
          if (errno == EINTR) continue;
          if (errno == EAGAIN || errno == EWOULDBLOCK) {
            // 数据未就绪，需要等待
            if (kq >= 0) {
              // 确保已注册 (简化起见，每次循环都假设已注册，实际应缓存)
              // 这里做一个简单的 poll 模拟
              struct kevent events[1];
              struct timespec ts;  //= {0};
              // 清空之前的过滤器重新添加比较麻烦，实际生产环境需维护 state
              // machine 这里为了代码简洁，仅在第一次失败时注册
              if (nchanges == 0) {
                if (kevent(kq, change_arr[0], 1, events, 1, &ts) <= 0) {
                  // Error handling
                }
                nchanges = 1;
              }

              // 等待事件
              int n_ev = kevent(kq, change_arr[0], 1, events, 1, &ts);
              if (n_ev > 0) {
                // 事件触发，重试 read
                continue;  // 回到 while 头部重试 pread
              }
            }
          }
        }
        break;
      }
    }

    if (kq >= 0) {
      ::close(kq);
    }

    return 0;
  }

 private:
  int file_desc;
};

}  // namespace core
}  // namespace zvec
