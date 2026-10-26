# 阶段78格式化归档副本：隔离重放计划

本文件是只读审阅后的计划。**本次未格式化、复制、编译或运行任何重放，也未修改生产源码或已冻结测试脚本。**唯一新增文件为本计划。根目录记为 `R=/home/xingrui/bench/quantized_graph/research_20261025`；计划中的归档目录为 `A=R/replay78_formatted`。应等待当前等价性检查及计时结束，再由主任务按需要执行。

## 1. 依赖清单

以下是审阅时已经出现的全部阶段78脚本及辅助代码；`screen78.py` 是审阅期间新增的性能流程，归档时应再次枚举、确认它已冻结。

| 文件 | 直接输入/依赖 | 输出/作用 |
|---|---|---|
| `independent78_test.py` | Python标准库、NumPy；`lib/independent78_test.so`；根目录下 `independent78_alloc_test`；`results/independent77_preflight.json` 的 metadata；三个 helper 源文件供摘要哈希 | 生成40节点合成模型及畸形变体；SHA/JSON/加载/验证/复制/并发/生命周期检查；生成 `results/independent78_import_checks_<输出目录basename>.json` |
| `independent78_test.cc` | 同目录 `independent78_import.h`；C++17标准库与Linux/POSIX | ctypes C接口共享库；不依赖Symphony/PQ/SQ8库 |
| `independent78_alloc_test.cc` | 同目录 `independent78_import.h`；C++17标准库与Linux/POSIX | 独立分配计数程序；被Python以6个参数调用 |
| `independent78_import.h` | C++17标准库、`mmap`/文件描述符、`std::from_chars`；无额外第三方库 | helper依赖；冻结原件SHA256为 `de8d451fd0319c6046b971d2ba2be81c51b3973bfee8b9b6aa3fadb4b7647d4e` |
| `guards78.py` | 硬编码活动源码根 `S`；三个生产头；`symphony_qg_test.cc`；`results/build_commands.json`；GCC C++17；RaBitQ/sparsehash/gtest include树 | 19项预处理守卫、3项真实测试TU模板语法检查；不链接、不执行测试 |
| `validate_stage78_default.py` | 硬编码活动根 `S` 的hnsw测试目录；`variants/<variant>/{commands.json,metadata.json,build_source_sha256.json,libcore_knn_hnsw.so}`；metadata指向的冻结源；`results/build_commands.json`；公共链接库和第三方include | 编译/链接/执行28项既有回归；`variants/<variant>/validation78/` 和 `results/<variant>_validation78.json` |
| `import78_integration.py` | NumPy；硬编码 `SRC`；`import78_integration.cc`、`coupled73_codec.h`；`independent77_preflight.json`；四个静态库；RaBitQ include | 编译测试、生成真实旋转合成fixture、运行真实图集成；独占命名的数据目录/二进制/日志/结果/命令清单 |
| `import78_integration.cc` | 生产QuantizedGraph/Symphony/SQ8头；同目录 `coupled73_codec.h`；真实Symphony旋转/FastScan | 40×768合成图、36full+3short+1empty、实际码/分数审计与生命周期；非1M、非训练结果、非QPS |
| `run78_disabled.py` | `explore90.py`；两个冻结variant `independent78_off_v1` / `independent78_native_v1`；`bucket90_w16/metadata.json`；baseline IDs；benchmark配置/二进制/索引/Cohere数据 | 1M索引、1000查询Top10关闭导入等价性；虽然 `duration=-1` 不计时，仍是完整检索负载；**不纳入最小重放** |
| `screen78.py` | `explore90.py`；已通过的disabled/integration结果；`independent78_native_v1`；完整模型、raw/rotated向量及header；benchmark/索引/Cohere | 完整模型加载及6点QPS/recall筛选，`duration=1`；**只归档，不作为格式化验证运行** |

`import78_integration.py` 直接链接：`libcore_framework.a`、`libzvec_turbo.a`、`librabitqlib_runtime.a`、`libzvec_ailego.a`。它不链接某个HNSW variant共享库；实际生产图模板来自指定冻结源码，宏固定为 `RABITQ_SUPPORTED=1,QG_ROTATION_768=1,QG_INDEPENDENT_CODE_IMPORT=1`，无AUGMENT。

28项回归的完整链接命令来自 `results/build_commands.json`，包括上述公共库以及 `libcore_metric.so`、`libcore_quantizer.so`、`libcore_knn_hnsw.so`（替换成对应variant）、`libcore_knn_flat.so`、`libcore_utility.so`、`libgtest.a`、`libgtest_main.a`、`librt.a`、`-ldl`。必须复用此命令，保留原有whole-archive顺序。

第三方include根固定在 `/home/xingrui/workspace/zvec/symphony_qg/zvec/thirdparty/`：RaBitQ-Library 0.1、sparsehash 2.0.4、googletest 1.10.0。Python解释器使用 `/home/xingrui/bench/symphony_qg/.venv/bin/python`，必须记录实际Python/NumPy、编译器和formatter版本；本次在当前PATH及该venv中未找到clang-format/Black/Ruff，因此计划不假设这些命令已存在。

## 2. 路径绑定与覆盖风险

| runner | 原脚本行为 | 归档副本必须处理的事项 |
|---|---|---|
| `independent78_test.py` | `R=__file__.resolve().parent.parent`；参数是 `R/<name>`；目录 `exist_ok=False`，但最终JSON用 `write_text` 可覆盖 | 从真实 `A/scripts` 文件执行，使用新basename；预先断言对应JSON不存在；helper `.so` 和allocation程序必须位于A，不能覆盖原件 |
| `guards78.py` | R随脚本位置；S仍硬编码活动仓库；按精确的 `R/objects/symphony_qg_test.o` 在build_commands中查找；`--label`独占创建结果 | 副本S改为冻结overlay；build_commands副本必须把原R对象路径同步重定位到A，否则 `next(...)` 找不到原命令；不能仅复制原JSON就运行 |
| `validate_stage78_default.py` | R随位置，S仍活动根；variant路径位于R；从S复制测试；从metadata找冻结src；`validation78`目录独占；最终JSON独占 | A/variants下每个variant必须是真目录，禁止把整个variant目录symlink到原件；否则会尝试在原variant下创建validation78。只复用冻结库和源，不调用compile_variant |
| `import78_integration.py` | R随位置，SRC仍活动根；fixture独占；log为append；结果和command manifest用write_text；binary可覆盖 | 副本SRC指冻结overlay；运行前同时检查fixture、binary、log、result、commands五个目标均不存在；不可只依赖fixture目录的独占检查 |

不要把 `A/lib`、`A/results`、`A/logs`、`A/variants` 整个目录链接到原R：这些目录含重放输出。特别是 `A/lib/independent78_test.so` 要重新编译，若lib目录链接回原R会改写原测试库。允许在A的真实目录中为**只读输入单文件**建立链接，或直接复制并验证哈希。

所有运行名只用简单basename，例如 `replay78_helper_v1`、`replay78_guards_v1`、`replay78_integration_v1`；不要传绝对路径、`..` 或含斜杠名称。helper结果名仅用输出目录basename，这会使不同路径同basename的结果冲突。

## 3. 最小隔离准备步骤（尚未执行）

1. 以 `mkdir(..., exist_ok=False)` 新建A；不存在才继续。将原scripts、证据JSON/log/XML、variant metadata/source manifest和相关库的SHA256写入A的输入清单。原文件路径和摘要保持原样，另外保存归档相对路径。对正在新增的screen脚本，应等主任务冻结后再记摘要。
2. 将上表脚本、`coupled73_codec.h`、`explore90.py`复制到A/scripts。保留一份未格式化输入快照（例如A/original_scripts），随后只格式化A/scripts。不要复制 `__pycache__`，以 `PYTHONDONTWRITEBYTECODE=1` 运行。原脚本应在格式化前后保持SHA256不变。
3. 采用冻结源：`F=R/variants/stage78_import_default/build_source`。逐文件校验其1117项 `build_source_sha256.json`；默认variant同样校验其独立manifest。创建 `A/frozen_inputs/src` 指向F/src，并创建 `A/frozen_inputs/tests/core/algorithm/hnsw` 指向 `R/variants/stage78_import_default/validation78/test_source`。这个测试源码副本已由原验证保存，入口 `symphony_qg_test.cc` SHA256为 `55a79f8fe75864e16c154fb60844d4679e3223192b5c07ebd1474f8477aadcd1`。两个variant保存的测试目录均应核验一致。
4. **仅在格式化副本**将guards/validate的 `S`、integration的 `SRC` 常量改为 `Path(A/'frozen_inputs')` 的绝对路径。保留这3处路径修改的独立patch；不要对原脚本做修改。先做纯格式化AST对比，再做这3处受控路径重定位，使语义等价与环境重定位可分别审查。
5. A/results是真目录：复制 `independent77_preflight.json`；复制并重定位 `build_commands.json` 中每条argv的原R路径到A、活动仓库根到A/frozen_inputs。重定位**所有参数**，包括 `-I...`、`-Wl,-rpath,...`、`--dependency-file=...`，也记录cwd为A；使用Python JSON读写，不用shell字符串求值。保留原manifest与重定位映射。其余RaBitQ/sparsehash/gtest系统include路径维持原值并记录依赖摘要。
6. A/lib是真目录：复制或逐文件只读链接现有公共库；`independent78_test.so`预留为A内的新普通文件。A/variants/stage78_default及stage78_import_default是真目录；复制各自 `metadata.json`、`commands.json`、`build_source_sha256.json`，只读复用各自冻结 `libcore_knn_hnsw.so`。metadata中的 `immutable_build_source` 可继续指向原冻结树，因为这是只读输入且已校验。不要复制原 `validation78` 到新variant；它必须由新runner独占创建。
7. 为A建立logs/results等输出目录；环境清除 `QG_*`、`GTEST_*` 及影响测试选择的filter/sharding，设置 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1`。不要启用 `PYTHONOPTIMIZE` 或 `python -O`，否则Python assert检查会被删除。variants和第三方依赖只读，编译输出全部落到A。

建议目录结构：

```text
replay78_formatted/
  original_scripts/         原脚本字节快照
  scripts/                  格式化副本；3处路径重定位有独立patch
  frozen_inputs/src -> 原冻结生产src
  frozen_inputs/tests/core/algorithm/hnsw -> 原冻结测试副本
  lib/                      真目录；原公共库只读输入、新helper共享库普通文件
  variants/stage78_default/ 真目录；独立validation78输出
  variants/stage78_import_default/
  results/                  真目录；输入manifest副本与新的结果
  logs/                     真目录；新的日志
  original_evidence/        原JSON/log/XML字节副本
  input_manifest.json       原件/副本/版本/路径映射
```

## 4. 最小执行顺序（未来主任务决定执行）

在A内，先编译两个helper程序到新位置：

```text
c++ -O3 -std=c++17 -fPIC -shared scripts/independent78_test.cc -o lib/independent78_test.so
c++ -O3 -std=c++17 scripts/independent78_alloc_test.cc -o independent78_alloc_test
```

然后使用上述venv解释器，依次运行：

```text
python scripts/independent78_test.py replay78_helper_v1
python scripts/guards78.py --label replay78_guards_v1
python scripts/validate_stage78_default.py --variant stage78_default
python scripts/validate_stage78_default.py --variant stage78_import_default
python scripts/import78_integration.py replay78_integration_v1
```

每条命令的stdout/stderr由外层runner写入A/logs的独占新文件（`open('x')`）；记录精确argv、cwd、环境、退出码。guards不加 `--preprocess-only`，否则缺3项实际模板检查。28项回归执行前记录variant共享库SHA256，执行后再查一次，确认没有隐式重建。

**不调用** `compile_variant.py`、`run78_disabled.py` 或 `screen78.py`。后两者可归档，但完整检索会干扰主任务当前实验，不是格式化等价所必需。

## 5. 如何验证与原证据一致

### 字节一致的输入

原件SHA256在准备前、准备后、重放后均一致；只读链接解析到的目标也必须校验。验证归档的原证据副本与原JSON/log/XML逐字节一致。不要把格式化文件或新编译二进制要求为原哈希：源码布局、绝对路径、调试/链接build-id均可改变这些摘要。

生产头应继续匹配原integration和guard证据：

- `quantized_graph.h`：`128ce391809918faf2ffd765bf54b987eb1c86fe7661f0b9eaf7adf698d8e6ef`
- `independent_code_import.h`：`de8d451fd0319c6046b971d2ba2be81c51b3973bfee8b9b6aa3fadb4b7647d4e`
- `symphony_qg_quantizer.h`：`61df87d6480b0ce00a202e056180327067402ce77cdb8b3ae22a7d51ca60dcef`

### 代码格式化等价

Python纯格式化前后比较 `ast.dump(ast.parse(text), include_attributes=False)`；三处S/SRC重定位单独核验为唯一AST差异。C++/header使用固定版本formatter，关闭include排序，保留逐文件diff供审阅，禁止宏或字面量改写；编译和实际结果用于补充行为验证。不要用简单去空白后的字符串比较，它会错误处理字符串字面量及预处理续行。

### 结果语义一致

新建一个单独的比较JSON，明确列出允许不同的路径、耗时、源码/二进制哈希和编译器诊断行号；不要覆盖或“更新”原结果。

| 重放对象 | 原证据 | 应一致的内容 |
|---|---|---|
| helper | `results/independent78_import_checks_loader78_unit_final.json` | 105次SHA流式、21次digest后继续更新、同locale下48次数字解析、1280并发检查、40向量/36full/4short、4000分配测试且0分配；62项拒绝的名称/错误类别/行号；状态安全与fd稳定标记。路径性错误消息可按映射归一化；locale变化必须明确报告，不能静默忽略 |
| guards | `results/independent78_compile_guards.json` | 19个预处理和3个模板case的名称、kind、expected_success及退出成败；三个冻结生产头及测试入口摘要匹配；诊断全文只作记录，不要求路径/行号相同 |
| default/import-default回归 | `results/stage78_default_validation78.json`、`results/stage78_import_default_validation78.json`及各自原XML | 每组28 tests、0 failures/errors/disabled/skipped，宏集合一致；XML中suite/test name、status、result及failure内容一致；时间、时间戳和路径可不同 |
| 真实图集成 | `results/integration78_actual_v2.json` | passed、17个检查名、40/768/36/3/1形状、8线程、1152 flips、每轮39非空块覆盖及标量一致性。原计数为26741 FastScan/815958 lane，应在相同环境和同一逻辑下对照；若改变不能仅以passed跳过解释 |
| 真实图fixture | `integration78_actual_v2/`及结果中的fixture_sha256 | 相同NumPy/编译器环境下graph/raw/rotation/native/imported码/signs/header/model均逐字节一致；这些数据不嵌入路径，格式化与重定位不应改变其摘要 |

helper固定seed产生的fixture也可与 `loader78_unit_final/` 逐文件比对（先确认原路径存在）；畸形模型文件内容不含生成路径，摘要应一致。原始日志中并发错误行的先后顺序可能不同，因此比较拒绝集合及结构化结果，保留两个日志，不强求日志全文相等。

这套重放只能证明归档副本保留阶段78的小规模正确性/回归证据；不替代完整1M加载等价性或QPS/recall结论。
