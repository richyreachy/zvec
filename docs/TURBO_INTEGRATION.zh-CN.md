# zvec 导入独立 Turbo

## 分层与改动

zvec 通过 `turbo::turbo` 导入独立计算库。`src/turbo` 只保留与 IndexMeta、IndexHolder、IndexFactory、训练及模型持久化相关的适配代码。距离、转换、PQ/FastScan 和旋转内核由独立库提供，原 `src/turbo/distance`、`src/turbo/conversion` 的 194 个文件已移除。

| **部分** | **当前职责** |
|:---:|:---:|
| 独立 Turbo | SIMD 内核、CPU/OS 探测、ISA 编译选项和内核选择 |
| `src/turbo/turbo.cc` | 将 `zvec::turbo` 的接口调用转发给独立库，检查枚举值兼容性 |
| zvec Quantizer / Preprocessor | 训练、查询编码、IndexHolder 访问、序列化、工厂注册 |
| zvec core | Flat/HNSW/IVF 等索引构建与查询 |

公开头文件仍为 `<zvec/turbo/turbo.h>`，现有 `zvec::turbo` 命名空间和 DistanceFunc / BatchDistanceFunc 的 `std::function` 语义保留，量化器捕获上下文的 lambda 继续可用。PQ 原始 batch 函数指针的参数改成 `const void *const *`，与外部库的只读指针数组接口一致；普通 `const void **` 调用参数仍可传入，自定义原始回调声明需同步增加外层 const。所有消费者应重新构建。

模型 magic、版本号、量化类型编号和 zvec 的序列化代码保持原有定义；这不等于已承诺所有历史模型的二进制兼容，升级仍需跑现有模型样本回归。FastScan 打包助手通过公开 `<turbo/fast_scan.h>` 引入，不再维护内核私有头文件副本。

## 导入方式

先正常初始化 zvec 自身依赖：`git submodule update --init --recursive`。三种方式在 CMake 配置阶段选择，建议为不同方式使用不同构建目录。

### 本地源码（联调推荐）

```sh
cmake -S . -B build-turbo -DCMAKE_BUILD_TYPE=Release \
  -DZVEC_TURBO_PROVIDER=SOURCE \
  -DZVEC_TURBO_SOURCE_DIR="$HOME/Documents/workspace/zvec/turbo_distribute/turbo_dist/turbo"
cmake --build build-turbo --parallel 8
```

通过 `add_subdirectory` 构建独立 Turbo。独立项目的测试和示例不进入 zvec 构建；安装规则启用，因此源码导入时安装 zvec 也会安装 Turbo 的静态库、公开头文件、CMake 配置和许可证。

### 固定 Git 提交（默认）

```sh
cmake -S . -B build-fetch -DCMAKE_BUILD_TYPE=Release
cmake --build build-fetch --parallel 8
```

`FETCH` 使用 FetchContent 从 `https://github.com/richyreachy/zvec-turbo.git` 获取固定提交 `63e844e7d75e8221eb54a21906ab295d1e5981cf`。不跟随 main；升级时显式调整 `ZVEC_TURBO_GIT_TAG` 并完成回归。首次配置需要网络，离线构建使用 SOURCE 或 PACKAGE。

### 已安装的 Turbo 包

```sh
cmake -S /path/to/turbo -B /path/to/turbo-build \
  -DCMAKE_BUILD_TYPE=Release -DTURBO_BUILD_SHARED=OFF \
  -DCMAKE_INSTALL_PREFIX=/path/to/turbo-sdk
cmake --build /path/to/turbo-build --parallel 8
cmake --install /path/to/turbo-build
cmake -S . -B build-package -DCMAKE_BUILD_TYPE=Release \
  -DZVEC_TURBO_PROVIDER=PACKAGE \
  -DCMAKE_PREFIX_PATH=/path/to/turbo-sdk
```

使用 `find_package(Turbo 0.1.0 EXACT CONFIG REQUIRED)` 导入，避免在布局适配尚未稳定时无意升级。必须是与 zvec 架构、工具链和 Windows CRT 一致的 PIC 静态库。PACKAGE 模式不重新安装外部包；分发静态 zvec SDK 时需同时提供原 Turbo SDK，手动链接静态归档时还需链接 `libturbo`。CMake 构建树内由 target 传递依赖。

当前集成要求静态 Turbo；动态 Turbo 会在配置时明确报错。静态内核经适配层链接进入 zvec 的共享产物，避免新增运行时动态库路径管理。Windows 和其他平台共享产物仍需各自原生 CI 验证。

## OPQ 的 Eigen 依赖

OPQ 训练仍属于 zvec。优先 `find_package(Eigen3 3.3 CONFIG)`；没有独立 Eigen 时使用 zvec 原有 RaBitQ 子模块中的 Eigen 头文件。也可指定 `-DZVEC_TURBO_EIGEN_INCLUDE_DIR=/path/to/eigen-root`，该目录下应有 `Eigen/Dense`。独立 Turbo 本身不依赖 Eigen，也不链接 RaBitQ。

## 测试与升级门禁

`turbo_external_adapter_test` 验证适配层返回的实际函数指针与独立库一致，单条/批量/preprocess 成组转发，捕获式 lambda 仍可用，不支持的类型或 ISA 返回空。原测试改为显式请求公开 backend，不再包含内核私有头文件；不可用后端跳过，不将编译成功当作执行成功。

```sh
cmake --build build-turbo --parallel 8 --target \
  turbo_external_adapter_test turbo_fp32_quantizer_test turbo_fp16_quantizer_test \
  turbo_int8_quantizer_test turbo_int4_quantizer_test \
  turbo_pq_int4_quantizer_test turbo_pq_int8_quantizer_test turbo_pq_fast_quantizer_test \
  turbo_fht_rotator_test turbo_opq_rotator_test turbo_pq_opq_test turbo_neon_distance_test \
  flat_turbo_index_test hnsw_turbo_index_test
ctest --test-dir build-turbo --output-on-failure -R 'turbo_'
```

独立库的支持范围目前为 64 位 x86 和 AArch64；32 位构建、SVE/RVV 和跨工具链稳定 ABI 不属于本次集成范围。独立库负责检查已编译实现与 CPU/OS 能力，显式指定不可用 ISA 不再静默回退。

## 本地验证记录（2026-10-13）

环境为 macOS ARM64、AppleClang 15。SOURCE 和 PACKAGE 两种方式均完成 18 个 CTest 测试程序：12 个 Turbo/适配测试程序、Flat/HNSW 两个索引测试程序，以及 Euclidean、Record INT8、Uniform UINT4/UINT8 四个 metric 测试程序。不可用 ISA 的内部用例会跳过；这些通过记录不代表 AVX2/AVX512 真机已执行。

FETCH 方式已从 GitHub 拉取上述固定提交，在隔离的 CMake 消费工程中编译 zvec 适配源码并完成距离调用测试。`zvec_core_shared` 构建通过，依赖检查未引入 Turbo 动态库，符号检查未发现未解析的 Turbo 引用或原 zvec SIMD 内核定义。

共享库外部消费者已链接成功，但本机启动被 macOS AMFI/XProtect 终止（退出信号 9）；系统日志包含 CT signature 与 rpath 校验错误，codesign 静态验证通过。此项记录为运行验证未完成，未修改系统安全设置。Linux/Windows、Python wheel、完整数据库/C API 构建和历史模型样本兼容仍需 CI/发布环境验证。

已对照独立 Turbo 的 `UPSTREAM.json` 检查本 zvec 基线中的原始内核文件，未发现抽取之后的源码差异。

本目录的第三方子模块最初未初始化；本地复用了同版本依赖，并通过 `ZVEC_TURBO_EIGEN_INCLUDE_DIR` 指定本机现有 Eigen 头文件完成 OPQ 测试。源码、包和拉取三种方式不依赖本机路径作为默认配置。
