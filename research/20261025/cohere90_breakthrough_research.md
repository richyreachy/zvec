# Cohere Top-10：持续突破研究进展

状态：进行中，尚未达到重大突破验收条件。已有历史最优仍为 `bucket90_w16`。

操作性门槛：同硬件和线程数、Recall@10≥90%，QPS 相对冻结最优稳定提升至少25%；重复计时、正确性验证和额外查询验证均通过。原1,000条查询已参与长期调参，不能作为独立盲测。

## 当前可用结论

冻结方案 `bucket90_w16` 的上一轮5次重复中位数：Recall@10=90.32%，单线程30,857.64 QPS，8线程230,844.74 QPS。其精确距离计算发生在展开节点上；并非完全不做refine。详见[确认报告](cohere90_round3_report.md)。

本轮探索点通常仅1–2轮，最快的单次达标点不能与历史中位数直接解释为稳定提升。FP16减少精排次数、低秩误差修正、64边图及量化感知剪边均尚未达到25%目标。

已汇总 824 条计时记录（每个算法/参数/线程/轮次计一条）；完整数字见下表和JSON。

## 已探索的方向

1. screen36：结果容器、结果门控、访问表和节点预取。
2. screen37：FP16/SQ8 中心区间上界驱动停止，按下界延后精确核算。
3. screen38：两级符号残差量化，按残差内积重新校准以保持理想端点距离。
4. screen39：联合调整导航边数与投影编码维度，探索相近编码字节数下的召回收益。
5. screen40：每次预选1/2/4/8个节点并预取，探索放松展开顺序能否隐藏随机访存延迟。
6. screen41：扩展 SQ8 区间停止的负 margin 扫描，避免只测到远高于90%的工作点。
7. screen42：对齐编码区、去掉 cosine 精确核不读取的尾随范数字段，并组合批末预取。
8. screen43：当前快速图上提前停止导航，比较普通终端精排与缓存中心批量预取精排。
9. screen44：按量化导航的理想距离重建剪枝条件，比较只看估计距离与真实/估计双重判定。
10. screen45：用近似中心距离驱动启发式导航停止，仍以区间下界保证已展开集合的终端精排；比较4候选预取。
11. screen46：以未投影的64边图作消融，比较1/2级残差编码以及完整代码区预取；单独报告内存代价。

所有新方向均为独立编译变体；原仓库不改动。screen37 的初次编译触发了旧实验不允许自适应组合的编译保护，已改为要求显式启用区间上界；保留失败日志，未混入性能数据。

编译可复现性：后续变体先复制并核对完整源码，再从冻结目录编译，防止排队编译与后续研究编辑相互干扰。历史库保留原样。96边测试最初被驱动的32/64限制拦截；使用独立96边驱动续跑，原失败日志保留。

## 已生成的达标测量摘要

下表是各组目前最快已测达标点；未完成组会变化，探索点通常只有1–2轮。不能据此宣布突破。

| 实验 | 线程 | 参数标签 | Recall@10 | QPS | RSS GiB | 轮数 | 该阶段完成 |
|---|---:|---:|---:|---:|---:|---:|---|
| screen36_cohere_batchprefetch3072 | 1 | 800 | 90.32% | 31616 | 9.53 | 2 | 是 |
| screen36_cohere_batchprefetch3072 | 8 | 800 | 90.32% | 234848 | 9.53 | 2 | 是 |
| screen36_cohere_batchprefetch640 | 1 | 800 | 90.32% | 28042 | 9.53 | 2 | 是 |
| screen36_cohere_batchprefetch640 | 8 | 800 | 90.32% | 225181 | 9.53 | 2 | 是 |
| screen36_cohere_batchprefetch6656 | 1 | 800 | 90.32% | 28375 | 9.53 | 2 | 是 |
| screen36_cohere_batchprefetch6656 | 8 | 800 | 90.32% | 225452 | 9.53 | 2 | 是 |
| screen36_cohere_bucket90_w16 | 1 | 800 | 90.32% | 29815 | 9.53 | 2 | 是 |
| screen36_cohere_bucket90_w16 | 8 | 800 | 90.32% | 231208 | 9.53 | 2 | 是 |
| screen36_cohere_control_post | 1 | 800 | 90.32% | 29265 | 9.53 | 2 | 是 |
| screen36_cohere_control_post | 8 | 800 | 90.32% | 226407 | 9.53 | 2 | 是 |
| screen36_cohere_prefetch3072 | 1 | 800 | 90.32% | 29830 | 9.53 | 2 | 是 |
| screen36_cohere_prefetch3072 | 8 | 800 | 90.32% | 223917 | 9.53 | 2 | 是 |
| screen36_cohere_result10 | 1 | 800 | 90.32% | 29774 | 9.53 | 2 | 是 |
| screen36_cohere_result10 | 8 | 800 | 90.32% | 231466 | 9.53 | 2 | 是 |
| screen36_cohere_result10_gate | 1 | 800 | 90.32% | 30197 | 9.52 | 2 | 是 |
| screen36_cohere_result10_gate | 8 | 800 | 90.32% | 230318 | 9.53 | 2 | 是 |
| screen36_cohere_result_gate | 1 | 800 | 90.32% | 30149 | 9.53 | 2 | 是 |
| screen36_cohere_result_gate | 8 | 800 | 90.32% | 229072 | 9.53 | 2 | 是 |
| screen36_cohere_visit64 | 1 | 800 | 90.32% | 29685 | 9.52 | 2 | 是 |
| screen36_cohere_visit64 | 8 | 800 | 90.32% | 231278 | 9.53 | 2 | 是 |
| screen37_cohere_control_post | 1 | 800 | 90.32% | 31484 | 9.53 | 2 | 是 |
| screen37_cohere_control_post | 8 | 800 | 90.32% | 233297 | 9.53 | 2 | 是 |
| screen37_cohere_control_pre | 1 | 800 | 90.32% | 29004 | 9.53 | 2 | 是 |
| screen37_cohere_control_pre | 8 | 800 | 90.32% | 226152 | 9.53 | 2 | 是 |
| screen37_cohere_half_adaptive | 1 | 1000 | 90.41% | 28415 | 8.05 | 1 | 是 |
| screen37_cohere_half_adaptive | 8 | 1000 | 90.41% | 221984 | 8.05 | 1 | 是 |
| screen37_cohere_half_adaptive_anchor | 1 | 1000 | 90.36% | 28509 | 8.05 | 1 | 是 |
| screen37_cohere_half_adaptive_anchor | 8 | 1000 | 90.36% | 224566 | 8.05 | 1 | 是 |
| screen37_cohere_sq8_adaptive | 1 | -1000 | 94.80% | 13750 | 7.31 | 1 | 是 |
| screen37_cohere_sq8_adaptive | 8 | 1000 | 94.80% | 108367 | 7.32 | 1 | 是 |
| screen37_cohere_sq8_adaptive_anchor | 1 | 2000 | 94.69% | 13964 | 7.32 | 1 | 是 |
| screen37_cohere_sq8_adaptive_anchor | 8 | 1000 | 94.69% | 107451 | 7.32 | 1 | 是 |
| screen38_cohere_control_post | 1 | 800 | 90.32% | 29120 | 9.53 | 2 | 是 |
| screen38_cohere_control_post | 8 | 800 | 90.32% | 228020 | 9.53 | 2 | 是 |
| screen38_cohere_control_pre | 1 | 800 | 90.32% | 28621 | 9.53 | 2 | 是 |
| screen38_cohere_control_pre | 8 | 800 | 90.32% | 227927 | 9.53 | 2 | 是 |
| screen38_cohere_two_stage_w16 | 1 | 3000 | 90.41% | 24362 | 12.64 | 1 | 是 |
| screen38_cohere_two_stage_w16 | 8 | 3000 | 90.41% | 188980 | 12.64 | 1 | 是 |
| screen38_cohere_two_stage_w32 | 1 | 3000 | 91.60% | 23248 | 12.63 | 1 | 是 |
| screen38_cohere_two_stage_w32 | 8 | 3000 | 91.60% | 180144 | 12.63 | 1 | 是 |
| screen38_cohere_two_stage_w8 | 1 | 4000 | 91.17% | 23499 | 12.64 | 1 | 是 |
| screen38_cohere_two_stage_w8 | 8 | 4000 | 91.17% | 177321 | 12.64 | 1 | 是 |
| screen39_cohere_control_post | 1 | 800 | 90.32% | 29759 | 9.52 | 2 | 是 |
| screen39_cohere_control_post | 8 | 800 | 90.32% | 227401 | 9.53 | 2 | 是 |
| screen39_cohere_control_pre | 1 | 800 | 90.32% | 29618 | 9.53 | 2 | 是 |
| screen39_cohere_control_pre | 8 | 800 | 90.32% | 227472 | 9.53 | 2 | 是 |
| screen39_cohere_project256_deg96_retry | 1 | 4000 | 90.22% | 12129 | 10.24 | 1 | 是 |
| screen39_cohere_project256_deg96_retry | 8 | 4000 | 90.22% | 95280 | 10.24 | 1 | 是 |
| screen39_cohere_project384_deg64 | 1 | 0 | 90.54% | 20114 | 9.88 | 1 | 是 |
| screen39_cohere_project384_deg64 | 8 | 0 | 90.54% | 163136 | 9.89 | 1 | 是 |
| screen39_cohere_project512_deg32 | 1 | 4000 | 90.99% | 25692 | 8.58 | 1 | 是 |
| screen39_cohere_project512_deg32 | 8 | 4000 | 90.99% | 195445 | 8.58 | 1 | 是 |
| screen39_cohere_project512_deg64 | 1 | 0 | 91.65% | 21604 | 10.83 | 1 | 是 |
| screen39_cohere_project512_deg64 | 8 | 0 | 91.65% | 175059 | 10.84 | 1 | 是 |
| screen40_cohere_control_post | 1 | 800 | 90.32% | 28893 | 9.52 | 2 | 是 |
| screen40_cohere_control_post | 8 | 800 | 90.32% | 227296 | 9.53 | 2 | 是 |
| screen40_cohere_control_pre | 1 | 800 | 90.32% | 29170 | 9.53 | 2 | 是 |
| screen40_cohere_control_pre | 8 | 800 | 90.32% | 228192 | 9.53 | 2 | 是 |
| screen40_cohere_expand_batch1 | 1 | 2000 | 90.32% | 31644 | 9.54 | 1 | 是 |
| screen40_cohere_expand_batch1 | 8 | 2000 | 90.32% | 240880 | 9.54 | 1 | 是 |
| screen40_cohere_expand_batch2 | 1 | 2000 | 90.13% | 28308 | 9.53 | 1 | 是 |
| screen40_cohere_expand_batch2 | 8 | 2000 | 90.13% | 224947 | 9.53 | 1 | 是 |
| screen40_cohere_expand_batch4 | 1 | 2000 | 90.57% | 24539 | 9.53 | 1 | 是 |
| screen40_cohere_expand_batch4 | 8 | 2000 | 90.57% | 194306 | 9.53 | 1 | 是 |
| screen40_cohere_expand_batch4_noprefetch | 1 | 2000 | 90.57% | 25222 | 9.53 | 1 | 是 |
| screen40_cohere_expand_batch4_noprefetch | 8 | 2000 | 90.57% | 197408 | 9.53 | 1 | 是 |
| screen41_cohere_sq8_adaptive | 1 | -10000 | 91.28% | 20047 | 7.34 | 1 | 是 |
| screen41_cohere_sq8_adaptive | 8 | -10000 | 91.28% | 157981 | 7.34 | 1 | 是 |
| screen41_cohere_sq8_adaptive_anchor | 1 | -10000 | 91.35% | 20031 | 7.34 | 1 | 是 |
| screen41_cohere_sq8_adaptive_anchor | 8 | -10000 | 91.35% | 157050 | 7.34 | 1 | 是 |
| screen42_cohere_aligned_center | 1 | 800 | 90.32% | 29347 | 9.53 | 2 | 是 |
| screen42_cohere_aligned_center | 8 | 800 | 90.32% | 230782 | 9.53 | 2 | 是 |
| screen42_cohere_bucket90_w16 | 1 | 800 | 90.32% | 29173 | 9.53 | 2 | 是 |
| screen42_cohere_bucket90_w16 | 8 | 800 | 90.32% | 228553 | 9.53 | 2 | 是 |
| screen42_cohere_control_post | 1 | 800 | 90.32% | 30190 | 9.53 | 2 | 是 |
| screen42_cohere_control_post | 8 | 800 | 90.32% | 233412 | 9.53 | 2 | 是 |
| screen42_cohere_omit_norm | 1 | 800 | 90.32% | 29605 | 9.47 | 2 | 是 |
| screen42_cohere_omit_norm | 8 | 800 | 90.32% | 231082 | 9.47 | 2 | 是 |
| screen42_cohere_omit_norm_batchprefetch | 1 | 800 | 90.32% | 29417 | 9.47 | 2 | 是 |
| screen42_cohere_omit_norm_batchprefetch | 8 | 800 | 90.32% | 234086 | 9.47 | 2 | 是 |
| screen43_cohere_control_post | 1 | 800 | 90.32% | 29653 | 9.53 | 2 | 是 |
| screen43_cohere_control_post | 8 | 800 | 90.32% | 228135 | 9.53 | 2 | 是 |
| screen43_cohere_control_pre | 1 | 800 | 90.32% | 28754 | 9.53 | 2 | 是 |
| screen43_cohere_control_pre | 8 | 800 | 90.32% | 226277 | 9.53 | 2 | 是 |
| screen44_cohere_control_post | 1 | 800 | 90.32% | 28828 | 9.53 | 2 | 是 |
| screen44_cohere_control_post | 8 | 800 | 90.32% | 225699 | 9.53 | 2 | 是 |
| screen44_cohere_control_pre | 1 | 800 | 90.32% | 29961 | 9.53 | 2 | 是 |
| screen44_cohere_control_pre | 8 | 800 | 90.32% | 228911 | 9.53 | 2 | 是 |
| screen44_cohere_quantized_prune1 | 1 | 3000 | 91.35% | 26148 | 9.54 | 1 | 是 |
| screen44_cohere_quantized_prune1 | 8 | 3000 | 91.35% | 204169 | 9.54 | 1 | 是 |
| screen44_cohere_quantized_prune2 | 1 | 2000 | 90.31% | 28690 | 9.54 | 1 | 是 |
| screen44_cohere_quantized_prune2 | 8 | 2000 | 90.31% | 227666 | 9.54 | 1 | 是 |
| screen44_cohere_quantized_prune2_all | 1 | 3000 | 91.31% | 26754 | 9.55 | 1 | 是 |
| screen44_cohere_quantized_prune2_all | 8 | 3000 | 91.31% | 209560 | 9.55 | 1 | 是 |
| screen45_cohere_control_post | 1 | 800 | 90.32% | 28618 | 9.53 | 2 | 是 |
| screen45_cohere_control_post | 8 | 800 | 90.32% | 225266 | 9.53 | 2 | 是 |
| screen45_cohere_control_pre | 1 | 800 | 90.32% | 30127 | 9.53 | 2 | 是 |
| screen45_cohere_control_pre | 8 | 800 | 90.32% | 228816 | 9.53 | 2 | 是 |
| screen45_cohere_half_estimated_batch | 1 | 2000 | 90.29% | 29793 | 8.05 | 1 | 是 |
| screen45_cohere_half_estimated_batch | 8 | 2000 | 90.29% | 234505 | 8.05 | 1 | 是 |
| screen45_cohere_half_estimated_stop | 1 | 2000 | 90.29% | 29135 | 8.05 | 1 | 是 |
| screen45_cohere_half_estimated_stop | 8 | 2000 | 90.29% | 227705 | 8.05 | 1 | 是 |
| screen45_cohere_sq8_estimated_batch | 1 | 3000 | 91.43% | 25486 | 7.33 | 1 | 是 |
| screen45_cohere_sq8_estimated_batch | 8 | 3000 | 91.43% | 195925 | 7.33 | 1 | 是 |
| screen45_cohere_sq8_estimated_stop | 1 | 3000 | 91.43% | 23757 | 7.33 | 1 | 是 |
| screen45_cohere_sq8_estimated_stop | 8 | 3000 | 91.43% | 185325 | 7.33 | 1 | 是 |
| screen46_cohere_control_post | 1 | 800 | 90.32% | 28646 | 9.52 | 2 | 是 |
| screen46_cohere_control_post | 8 | 800 | 90.32% | 227298 | 9.53 | 2 | 是 |
| screen46_cohere_control_pre | 1 | 800 | 90.32% | 29761 | 9.53 | 2 | 是 |
| screen46_cohere_control_pre | 8 | 800 | 90.32% | 226700 | 9.53 | 2 | 是 |
| screen46_cohere_full64 | 1 | 0 | 91.88% | 22714 | 12.75 | 1 | 是 |
| screen46_cohere_full64 | 8 | 0 | 91.88% | 181485 | 12.75 | 1 | 是 |
| screen46_cohere_prefetch32 | 1 | 2000 | 90.32% | 31698 | 9.54 | 1 | 是 |
| screen46_cohere_prefetch32 | 8 | 2000 | 90.32% | 231051 | 9.54 | 1 | 是 |
| screen46_cohere_prefetch64 | 1 | 0 | 91.88% | 23909 | 12.75 | 1 | 是 |
| screen46_cohere_prefetch64 | 8 | 0 | 91.88% | 187885 | 12.75 | 1 | 是 |
| screen46_cohere_two_stage64 | 1 | 0 | 90.66% | 19662 | 18.95 | 1 | 是 |
| screen46_cohere_two_stage64 | 8 | 0 | 90.66% | 158882 | 18.95 | 1 | 是 |
| screen48_cohere_control_post | 1 | 800 | 90.32% | 28981 | 9.53 | 2 | 是 |
| screen48_cohere_control_post | 8 | 800 | 90.32% | 227295 | 9.53 | 2 | 是 |
| screen48_cohere_control_pre | 1 | 800 | 90.32% | 29091 | 9.53 | 2 | 是 |
| screen48_cohere_control_pre | 8 | 800 | 90.32% | 226407 | 9.53 | 2 | 是 |
| screen48_cohere_lowrank48_16 | 1 | 3000 | 91.28% | 24696 | 11.44 | 1 | 是 |
| screen48_cohere_lowrank48_16 | 8 | 3000 | 91.28% | 185996 | 11.44 | 1 | 是 |
| screen48_cohere_lowrank48_4 | 1 | 2000 | 90.33% | 28288 | 10.01 | 1 | 是 |
| screen48_cohere_lowrank48_4 | 8 | 2000 | 90.33% | 219169 | 10.01 | 1 | 是 |
| screen48_cohere_lowrank48_8 | 1 | 2000 | 90.02% | 29056 | 10.49 | 1 | 是 |
| screen48_cohere_lowrank48_8 | 8 | 2000 | 90.02% | 216144 | 10.49 | 1 | 是 |
| screen49_cohere_control_post | 1 | 800 | 90.32% | 28411 | 9.53 | 2 | 是 |
| screen49_cohere_control_post | 8 | 800 | 90.32% | 228384 | 9.53 | 2 | 是 |
| screen49_cohere_control_pre | 1 | 800 | 90.32% | 31363 | 9.53 | 2 | 是 |
| screen49_cohere_control_pre | 8 | 800 | 90.32% | 233380 | 9.53 | 2 | 是 |
| screen49_cohere_cover49_aug2_reference | 1 | 2000 | 90.29% | 28775 | 9.54 | 1 | 是 |
| screen49_cohere_cover49_aug2_reference | 8 | 2000 | 90.29% | 228345 | 9.54 | 1 | 是 |
| screen52_cohere_control_post | 1 | 800 | 90.32% | 28802 | 9.53 | 2 | 是 |
| screen52_cohere_control_post | 8 | 800 | 90.32% | 226529 | 9.53 | 2 | 是 |
| screen52_cohere_control_pre | 1 | 800 | 90.32% | 28895 | 9.52 | 2 | 是 |
| screen52_cohere_control_pre | 8 | 800 | 90.32% | 226469 | 9.53 | 2 | 是 |
| screen53_cohere_control_post | 1 | 800 | 90.32% | 28519 | 9.53 | 2 | 是 |
| screen53_cohere_control_post | 8 | 800 | 90.32% | 227504 | 9.53 | 2 | 是 |
| screen53_cohere_control_pre | 1 | 800 | 90.32% | 29885 | 9.52 | 2 | 是 |
| screen53_cohere_control_pre | 8 | 800 | 90.32% | 227704 | 9.53 | 2 | 是 |
| screen53_cohere_rot53_estimate | 1 | 3000 | 91.59% | 27331 | 7.33 | 1 | 是 |
| screen53_cohere_rot53_estimate | 8 | 3000 | 91.59% | 214188 | 7.33 | 1 | 是 |
| screen53_cohere_rot53_upper_v2 | 1 | -2000 | 91.99% | 25912 | 7.33 | 1 | 是 |
| screen53_cohere_rot53_upper_v2 | 8 | -2000 | 91.99% | 200588 | 7.33 | 1 | 是 |
| screen54_cohere_control_post | 1 | 800 | 90.32% | 32365 | 9.53 | 2 | 是 |
| screen54_cohere_control_post | 8 | 800 | 90.32% | 237182 | 9.53 | 2 | 是 |
| screen54_cohere_control_pre | 1 | 800 | 90.32% | 28616 | 9.53 | 2 | 是 |
| screen54_cohere_control_pre | 8 | 800 | 90.32% | 226512 | 9.53 | 2 | 是 |
| screen54_cohere_mean54_cv | 1 | 25000 | 90.87% | 31247 | 7.33 | 1 | 是 |
| screen54_cohere_mean54_cv | 8 | 25000 | 90.87% | 233846 | 7.33 | 1 | 是 |
| screen54_cohere_mean54_plain | 1 | 30000 | 91.35% | 28899 | 7.33 | 1 | 是 |
| screen54_cohere_mean54_plain | 8 | 25000 | 90.81% | 227610 | 7.33 | 1 | 是 |
| screen54_cohere_mean54_tight | 1 | 25000 | 90.81% | 31838 | 7.33 | 1 | 是 |
| screen54_cohere_mean54_tight | 8 | 25000 | 90.81% | 238360 | 7.33 | 1 | 是 |

未出现达标点的已测组（未完成组仍可能变化）：

- screen40_cohere_expand_batch8：最高已测 Recall@10 85.19%。
- screen43_cohere_tail_bucket_batch：最高已测 Recall@10 83.88%。
- screen43_cohere_tail_bucket_fast：最高已测 Recall@10 83.88%。
- screen49_cohere_cover49_quantized：最高已测 Recall@10 86.33%。
- screen49_cohere_cover49_true：最高已测 Recall@10 86.61%。
- screen52_cohere_progress52_quantized：最高已测 Recall@10 86.26%。
- screen52_cohere_progress52_saturated：最高已测 Recall@10 88.08%。
- screen52_cohere_progress52_true：最高已测 Recall@10 86.30%。

标签在 STOP_GRID 模式下编码 margin/patience，实际 ef=800；各实验 `config.json` 保存完整命令和环境变量。

[逐行测量](results/breakthrough_screen_rows.json)、[摘要JSON](results/breakthrough_screen_summary.json)、[上一轮完整报告](cohere90_round3_report.md)。

额外查询：已下载 [Big ANN 官方数据定义](https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/benchmark/datasets.py) 指向的 Wikipedia-Cohere 5,000 条查询，只做来源和格式核对。已按cosine归一化核对：与原1,000条查询无近重复（阈值0.999），最大相似度0.927826。原始范数差异本身不影响cosine，也不能证明模型不兼容；来源仍不足以确认同模型同分布，不能直接称为同分布盲测。详情见 [来源记录](holdout_sources/provenance.json)。未观察它们的召回结果或据此调参。

额外数据库内查询的[冻结协议](results/additional_query_protocol.json)已保存：均匀随机选1,000个数据库ID，排除自身，穷举生成GT；候选配置冻结前不运行其图搜索结果，不据此调参。准备任务已在计时任务之间完成，穷举扫描用时约10秒；尚未运行额外查询上的图搜索。

已完成验证：screen36 共20,000次逐查询 ID＋精确距离比较零差异；两级量化240项端点/查表误差检查通过；FP16 cosine上下界21,000项检查通过。默认28项单测通过；本轮累计59,000次逐查询ID及距离比较零差异，30,763行B=1展开路径与基线完全一致。FP16/SQ8实际数据上下界审计及与全量精排的结果比较通过；冻结历史库哈希未变。FP16平均展开31.286次、精排11.288次；SQ8平均展开38.927次、精排31.640次。

已完成diagnostic47：用精确邻居距离测量逻辑搜索工作量，仅作诊断、不报告可部署QPS；screen48已测试数据库PCA子空间上的残差误差修正，4/8/16维均未胜过基线，详见[数学说明与相关工作](notes/lowrank_residual_correction.md)。

量化感知剪枝的[数学推导与边界](notes/quantized_pruning.md)。所有局部性质均不能替代全局召回实测。

最新自动核对：[证据JSON](results/breakthrough_correctness.json)，逐查询比较 59,000 次，ID/分数差异 0 次。

## 精确邻居距离诊断

该实验读取每条被扫描邻居的真实向量，成本不代表可部署算法，不报告其QPS。下表在各自已测参数中选择达到90%召回且平均展开次数最低的点。

| 配置 | margin | Recall@10 | 平均展开 | 命中64次上限比例 |
|---|---:|---:|---:|---:|
| diag47_onebit16 | 0.02 | 90.32% | 30.763 | 1.9% |
| diag47_oracle16 | 0.04 | 90.83% | 27.268 | 0.5% |
| diag47_oracle2 | 0.06 | 92.37% | 29.849 | 2.7% |
| diag47_oracle8 | 0.06 | 92.68% | 30.769 | 3.4% |

已完成结构实验screen49：固定32边预算，用局部真实/量化进展半径构成次模覆盖目标；含相同两跳候选的多样性剪枝对照。[推导、适用边界和相关工作](notes/coverage_graph49.md)。

验证 `lowrank48_test`：`low-rank identity/span/LUT checks=3840 worst_identity=1.14312e-05 worst_lut=0.0409895 passed`。

验证 `quantized_prune_geometry_test`：`native ideal score checks=6144 max_gap=3.40216e-06 local-radius checks=29967 passed`。

验证 `cover49_test`：`cover greedy-vs-exhaustive trials=1000 min_ratio=0.925047 local-ball checks=11908 passed`。

## 阶段51–53与提交记录

图可达性诊断排除了新增不可达目标足以解释覆盖图召回下降的假设：两种覆盖图各只有1/10,000次GT出现不可达，却在相同停止参数下损失约15个百分点召回。详见[拓扑诊断](notes/topology51.md)。

距离下降比例选边的三种变体最高召回为86.30%、86.26%、88.08%，均未达90%。8,000项几何检查通过不代表全局检索性能成立。详见[数学推导和负结果](notes/progress52.md)。

旋转SQ8将1,000个数据库样本的重建误差中位数从0.02013降到0.00701，但已测估计停止达标点为Recall@10=91.59%、单线程27,330.80 QPS，未超过冻结基线。8,000项区间与1,000项锚点检查通过；实际主查询按界精排与全部展开节点精排的10,000条ID/分数完全一致。平均展开33.292个节点，按界精排18.178个。详见[实现与边界](notes/rotated_center53.md)。

上述数字属于各自参数下的探索点，不能视为精确90%工作点或稳定收益。研究目标仍未完成。公共均值中心化SQ8的后续实测见阶段54结论。

已按阶段保存11个研究提交，见[提交索引](COMMITS.md)。后续每个完成验证且有实质进展的阶段继续单独提交。

## 阶段54：均值中心化与一维误差修正

三个变体全部完成计时、实际区间audit与全量精排对照。较紧区间版本的最快单线程达标点为31,838.28 QPS、Recall@10=90.81%，8线程238,359.61 QPS。冻结基线本轮前后单线程中位数28,616.41与32,364.92 QPS，漂移显著，不能宣称稳定提升。

均值方向修正将平均精排次数降至11.797（平均展开32.347），但已测单线程达标点31,247.38 QPS未达到突破门槛。3,000条查询的计时版、audit版和全量精排版ID/分数一致；默认28项回归通过。详见[数学依据与完整实测](notes/mean_center54.md)。

下一步验证共享均值下的整数内积，降低每个展开节点的浮点解码成本；目前只有[推导与实验设计](notes/integer_center55_plan.md)，没有实现或QPS结论。额外查询仍未用于调参。
