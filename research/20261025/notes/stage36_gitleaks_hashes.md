# 阶段36归档 SHA256 的精确 gitleaks 误报处理

现有 hook 固定使用 gitleaks v8.21.4：`git --pre-commit --redact --staged --verbose`，且 `pass_filenames=false`。仓库此前没有 `.gitleaksignore` 或 `.gitleaks.toml`。本次不修改 hook、默认规则或已经封存的证据。

默认 `generic-api-key` 规则会匹配含 access/token/api 的 JSON 键名及其后的高熵字符串，因此两份归档输入清单各有 25 个源码 SHA256 被报告为凭据。新增根目录 `.gitleaksignore` 仅列出这 50 个完整 `文件:规则:行号` 指纹，不加入路径通配符、规则禁用或内容正则白名单。

## 机器验证

验证结果（本地实验文件：`../results/stage36_gitleaks_hash_verification.json`）逐项记录了以下核对：

1. 归档文件的工作区字节、Git index 字节和阶段36 seal 的 SHA256 一致。
2. gitleaks 报告中的完整 Secret 等于告警行的 JSON 值，且该值属于 `frozen_source_sha256`，是完整的 64 位小写十六进制。
3. 每个值等于 `variants/trace68_native/build_source` 下相应实际源码文件重新计算的 SHA256；50 项对应两个清单中的同 25 份源码。
4. 基线副本扫描检出恰好 50 项；应用精确指纹后，完整暂存区扫描没有告警，现有 gitleaks pre-commit hook 通过。
5. 一个在其他位置生成的公开非凭据测试字符串仍被 `generic-api-key` 检出，确认规则继续启用。
6. 两份证据、seal 和 `.pre-commit-config.yaml` 的 SHA256 在验证前后保持不变。

Gitleaks 会额外读取扫描源目录内的 `.gitleaksignore`，即使指定了其他 ignore 路径。为让验证可重复，基线扫描在独立目录使用与 Git index 逐字节相同的两份副本，不临时移除或改写仓库 ignore。独占证据目录为 `/home/xingrui/bench/quantized_graph/research_20261025/gitleaks36_verified_v3`，保存命令、日志及原始扫描报告；此前两个编排失败目录保留，未改变原始证据。

## 精确指纹的边界和重验

指纹绑定文件、规则和行号，**并不绑定该行的值**。这些证据已经 seal 固定；若今后修改相应文件，应重新执行验证并审阅每个忽略项，不能把旧行号豁免套用到新内容。验证器要求该行仍等于实际冻结源文件 SHA256，内容改变会失败。

```bash
python3 research/20261025/scripts/verify36_gitleaks_hashes.py \
  --repo /home/xingrui/workspace/quantized_graph/symphony_research_20261025 \
  --bench-root /home/xingrui/bench/quantized_graph/research_20261025 \
  --gitleaks /home/xingrui/.cache/pre-commit/repop_izgajs/golangenv-default/bin/gitleaks \
  --out /home/xingrui/bench/quantized_graph/research_20261025/gitleaks36_recheck_v1
```

输出目录必须不存在。`--write-ignore` 仅用于首次创建已核验的精确指纹文件；已有文件必须与重新核验的 50 项完全相同。
