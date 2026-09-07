# collection_test 段错误（SEGFAULT）——根因分析与修复报告

**日期：** 2026-09-04
**问题：** Windows 平台（MSVC 2022 / Ninja）上 `62 - collection_test (SEGFAULT)`
**失败用例：** `CollectionTest.Feature_Optimize_Concurrent_ReadWrite_NonBlocking`（`tests/db/collection_test.cc:2935`）
**状态：** 已修复 —— 过滤后用例连续 45 次通过，整个测试套件全绿。

---

## 1. 摘要

本次崩溃是 **segment 刷盘（flush）路径与并发读 / 索引打开路径之间，对共享索引状态的数据竞争（data race）**，
并非缓冲区溢出导致的堆损坏。涉及两处未同步的结构：

1. `FlatStreamerEntity::segments_` —— 一个由 `get_segment()` 惰性填充的 `std::vector`。
   并发追加会触发缓冲区重分配，而此时另一线程正从中拷贝 `Segment::Pointer`（shared_ptr），
   拿到的是悬空 / 无效指针。
2. `SegmentImpl` 在 `flush()` / `init_memory_components()` 中执行内存态→持久态组件交换时，
   会修改 indexer 映射表（`vector_indexers_`、`memory_vector_indexers_`）和持久 block 偏移，
   而读路径（`Fetch`、`fetch`、`get_combined_vector_indexer`）在遍历这些结构时未持锁。

修复方案：为 `segments_` 引入专用互斥锁并改用地址稳定的 `std::deque`；同时在所有受影响路径上
加 `seg_col_mtx_`（读路径持共享锁，flush/init 交换持独占锁）。

## 2. 现象与复现

- 在原始（未修复）代码上直接运行过滤后的用例，**20 次中崩溃 5 次（25%）**。
  在 `ctest` 下表现为报告中的间歇性 SEGFAULT。
- 崩溃发生在**用例启动后约 37 ms**，即*初始单线程 20,000 条文档写入阶段*——
  早于并发 optimizer/writer/reader 阶段开始。
- 测试配置（`CollectionOptions{false, true, 64 * 1024 * 1024}`）设定了 64 MB 内存缓冲上限。
  前向存储超过该上限后 `ready_for_dump_block()`（segment.cc:2901）触发，
  写入循环因此反复进入
  `flush() → finish_memory_components() → init_memory_components() →
  create_vector_indexer() → VectorColumnIndexer::Open → core_interface::Index::open
  → FlatStreamer<32>::open` 的循环，与此同时其他线程（buffer pool 淘汰、后续插入）
  也在访问同样的状态。

## 3. 证据链

### 3.1 符号化的 minidump

通过自制的调试循环启动器（`tools/dump_launcher.cc`）在首次命中（first-chance）访问违例时
写入全内存 minidump，再由 `tools/dump_analyze.cc` 离线符号化：

```
EXCEPTION code=0xc0000005  AV type=WRITE target=0x1
faulting instruction: memcpy+0xe9  (mov [rax], r8; RAX=1)
R8 = 0x6f725072656e6e49  ("InnerPro" —— 度量名 "InnerProduct" 的前 8 字节)

stack:
  std::string::assign
  zvec::core::IndexMeta::operator=
  zvec::core::FlatStreamer<32>::open+0x327        (flat_streamer.cc:150, *entity_->mutable_meta() = meta_)
  zvec::core_interface::Index::open
  zvec::VectorColumnIndexer::CreateProximaIndex / Open
  zvec::SegmentImpl::create_vector_indexer        (init_memory_components)
  ... CollectionTest TestBody → CreateCollectionWithDoc（初始写入阶段）
```

一个刚由 `make_unique` 创建的 entity，其 meta 赋值却通过野指针（RAX = 1）写入，
说明对象/指针对在创建与使用之间已失效 —— 这是竞争的典型特征，而非溢出拷贝。

### 3.2 排除经典堆损坏

- **Full PageHeap**（IFEO `PageHeapFlags=3`，可在越界 / 释放后写入的精确位置触发异常）：
  **5/5 全部干净通过**。不存在分配边界违规。
- **附加调试器**：被调试的运行 0 次崩溃 —— 竞争窗口对时序敏感，调试扰动会掩盖它。
  这也解释了为何交互式排查时崩溃"消失"。

### 3.3 一次 ODR 事故反向确认了根因

应用修复后，一次增量重编在**10/10 次运行**中以*完全相同*的签名崩溃。排查发现
**本机上 ninja 的头文件依赖跟踪已失效**（中文 locale 的 MSVC `/showIncludes` 输出无法被解析——
touch `flat_streamer_entity.h` 后没有任何目标变脏）。该二进制混合了旧布局的
`flat_streamer.cc.obj` 与新布局的 `flat_streamer_entity.cc.obj`：一种 ODR/布局不匹配，
恰好产生同样的野指针写入签名。而对完全相同的源码做一次干净重编后 45/45 全部通过。

这次事故精确演示了崩溃机制：只要 entity 的内存状态（指针/布局）与读取线程的预期不一致，
`FlatStreamer::open` 中的 meta 拷贝就会写到垃圾地址。在原始构建中，这种不一致来自
未同步的 `segments_` 重分配与未持锁的 flush/init 交换，而非陈旧对象。

## 4. 修复内容

### 4.1 `src/core/algorithm/flat/flat_streamer_entity.{h,cc}`

- `segments_` 从 `std::vector` 改为 `std::deque` —— 写线程追加 segment 时元素地址保持稳定，
  并发搜索取走的 `shared_ptr` 拷贝始终有效。
- 新增 `mutable std::mutex segments_mutex_` 串行化所有 `segments_` 访问：
  `get_segment()`（惰性填充）、`alloc_segment()`、`alloc_block()`（已重构为调用
  `alloc_segment()` 前先释放锁，因为后者自身会加锁）、`add_to_block()` 与 `clone()`。
- `update_head_block()` / `get_head_block()` 改为通过持锁的 `get_segment(0)` 获取
  segment 0，不再裸访问 `segments_[0]`。

### 4.2 `src/db/index/segment/segment.cc`

- `flush()`：在 `finish_memory_components()` 外围加 `unique_lock(seg_col_mtx_)` ——
  内存态→持久态交换以独占方式执行；读者要么看到旧内存 block，要么看到完全迁移后的持久 block，
  绝不会看到迁移到一半的中间态。
- `init_memory_components()`：加 `unique_lock(seg_col_mtx_)` —— 内存组件与 indexer 映射表的
  重建与读者互斥。
- `get_combined_vector_indexer()` / `get_quant_combined_vector_indexer()`：加 `shared_lock`
  —— indexer 映射表在列锁保护下遍历。
- `Fetch()`（向量字段循环）与单行 `fetch()`：block 偏移 / indexer 映射表遍历加 `shared_lock`；
  多 block 回退路径被移到锁释放*之后*，因为它会递归加锁 —— 嵌套共享加锁可能与等待独占锁的
  flush 之间形成死锁。

不改变任何公开 API、磁盘格式或单线程行为。

## 5. 验证

| 构建 | 运行 | 结果 |
|---|---|---|
| 原始（修复前），过滤后用例 | 20 次 | **崩溃 5 次（25%）** |
| 修复后（干净重编），过滤后用例 | 45 次 | **45 次通过，0 崩溃** |
| 修复后，完整 `collection_test` 套件 | 91 个用例 | **89 通过，2 跳过¹，0 失败** |
| 修复后，`flat_streamer_test` | 19 个用例 | **18 通过，1 跳过¹，0 失败** |

¹ 跳过项均为此前已存在的平台性跳过（`Feature_DropIndex_Scalar_FailureKeepsPersistedOldSchema`、
`Feature_DropFtsIndex_FailureKeepsPersistedOldSchema`、`FlatStreamerTest.TestMaxIndexSize`），
与本次修改无关。

所有运行串行执行，每次运行之间删除 `test_collection/`（该测试要求路径不存在）。
构建环境：`build.rel`（Release + PDB，MSVC 2022 BuildTools，Ninja），修复后做了完整干净重编。

## 6. 变更文件

```
src/core/algorithm/flat/flat_streamer_entity.h   | 17 +-
src/core/algorithm/flat/flat_streamer_entity.cc  | 46 +--
src/db/index/segment/segment.cc                  | 262 ++++++------
```

排查过程中编写的诊断工具（未纳入版本管理，保留备查）：
- `tools/dump_launcher.cc` —— 以 `DEBUG_ONLY_THIS_PROCESS` 启动子进程，在首次命中 AV 时
  写全内存 minidump（gtest 的 SEH 会吞掉测试体内的首次命中 AV，等二次命中永远不会触发）。
- `tools/dump_analyze.cc` —— 离线 minidump 符号化工具（模块列表、异常记录、寄存器、
  出错指令、RSP 栈扫描），基于 dbghelp。

## 7. 环境注意事项

1. **本工作区内 ninja 的头文件依赖跟踪已失效**（中文 locale MSVC：`/showIncludes` 输出
   无法解析；touch 头文件不会有任何目标变脏）。修改任何头文件后，请对受影响目标做干净
   重编 —— 增量构建可能悄悄产出混合布局的二进制，并以确定性方式崩溃。建议后续彻底修复
   （例如改用 `/deps` 生成 .d 文件，或为编译器设置英文 locale）。
2. 诊断期间临时使用的注册表键（PageHeap IFEO 键、`collection_test.exe` 的 WER
   LocalDumps）均已删除。
3. 诊断期间用于给测试二进制生成符号的 `tests/db/CMakeLists.txt` PDB 编译选项已还原——
   不属于修复的一部分。
