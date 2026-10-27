from __future__ import annotations

import csv
import hashlib
import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

logging.basicConfig(level=logging.INFO)
R = Path("/home/xingrui/bench/quantized_graph/research_20261025")
SRC = Path("/home/xingrui/workspace/quantized_graph/symphony_research_20261025")
assert "DIAGNOSTIC58 COMPLETE" in (R / "logs/diagnostic58.log").read_text()
summary = json.loads((R / "results/anchor58_summary.json").read_text())
assert len(summary["rows"]) == 17
manifest = {}
query_count = id_count = 0
for count, keep in [(0, 0), (4096, 0), (4096, 1), (65536, 0), (65536, 1)]:
    name = f"anchor58_count{count}_keep{keep}"
    folder = R / "experiments" / name
    recall = list(csv.DictReader((folder / "search_k10_r0_recall.csv").open()))
    ids = list(csv.reader((folder / "search_k10_r0_ids.csv").open()))
    n = 1000 if count == 0 else 4000
    assert len(recall) == n
    assert len(ids) == n * 10
    assert not list(csv.DictReader((folder / "search_k10_r0.csv").open()))
    for label in sorted({r["ef"] for r in recall}):
        queries = [r for r in recall if r["ef"] == label]
        assert len(queries) == 1000
        assert {int(r["query_id"]) for r in queries} == set(range(1000))
        assert all(int(r["returned"]) == 10 for r in queries)
    query_count += n
    id_count += len(ids)
    for p in [
        *folder.glob("*.csv"),
        folder / "config.json",
        R / f"results/{name}_stats.csv",
    ]:
        manifest[str(p.relative_to(R))] = hashlib.sha256(p.read_bytes()).hexdigest()
rows = []
for r in summary["rows"]:
    row = dict(r)
    row["route_exact_calls"] = 5 if r["anchors"] else 0
    row["mean_total_exact_calls"] = r["mean_refined"] + row["route_exact_calls"]
    row["full_dimension_route_coordinates_scored"] = r["anchors"] * 768
    rows.append(row)
(R / "results/anchor58_accounting.json").write_text(
    json.dumps(
        {
            "diagnostic_only": True,
            "qps_measured": False,
            "router_cost_excluded_from_graph_refined_counter": True,
            "exact_call_accounting": "Graph terminal FP32 refinements + original entry and 4 shortlist distances; includes duplicate evaluations if any; full float router scan is additional work.",
            "query_evaluations": query_count,
            "id_score_rows": id_count,
            "rows": rows,
        },
        indent=2,
    )
    + "\n"
)
(R / "results/anchor58_artifact_sha256.json").write_text(
    json.dumps(manifest, indent=2) + "\n"
)
source_checks = {}
for variant in ["anchor58_integer", "stage58_default"]:
    hashes = json.loads(
        (R / f"variants/{variant}/source_files_sha256.json").read_text()
    )
    for path, expected in hashes.items():
        assert hashlib.sha256((SRC / path).read_bytes()).hexdigest() == expected, (
            variant,
            path,
        )
    source_checks[variant] = len(hashes)
root = ET.parse(R / "results/stage58_default_tests.xml").getroot()
assert int(root.attrib["tests"]) == 28
assert int(root.attrib["failures"]) == 0
assert int(root.attrib["errors"]) == 0
for name, count in [("router58_legacy_test", 9000), ("router58_large_test", 600)]:
    s = (R / f"logs/{name}.log").read_text()
    assert str(count) in s
    assert "passed" in s
checks = {
    "query_evaluations": query_count,
    "id_score_rows": id_count,
    "no_timing_rows": True,
    "router_legacy_checks": 9000,
    "large_router_checks": 600,
    "default_tests": 28,
    "source_manifest_files_verified": source_checks,
    "disabled_route_equivalence": json.loads(
        (R / "results/anchor58_disabled_equivalence.json").read_text()
    ),
    "additional_queries_used": False,
    "breakthrough_verified": False,
}
(R / "results/anchor58_validation.json").write_text(json.dumps(checks, indent=2) + "\n")
lines = [
    "# 大规模全维入口路由\uff1a工作量诊断\uff08阶段58\uff09",
    "",
    "结论\uff1a更好的入口明显减少了已测达标点的展开次数\uff0c值得研究低成本路由。但当前实现要扫描4,096或65,536个768维锚点\uff0c并额外计算5次精确距离\uff1b没有测量或证明QPS提升。所有路由开关默认关闭。",
    "",
    "## 配置与控制",
    "",
    "Cohere 1M\uff0ccosine\uff0cTop-10\uff0c原1,000条主查询\uff0c32边图、cap64、ef800\uff1b基于已验证的int55_original。锚点仅由图内部ID固定抽样\uff0c与查询/GT无关\uff1b全维旋转评分后选择4个候选\uff0c计算候选和原入口的精确距离\uff0c保留最佳入口。锚点采样去重改为集合\uff0c避免大样本平方级去重。",
    "",
    "两种桶尺度控制分别使用新入口的整数中心估计距离\uff0c或保留原入口的整数中心估计距离。后者复用已编码查询计算原入口估计\uff0c不能以FP32真值悄悄替换原基线的尺度。margin只扫描0/0.01/0.02/0.04\uff0c禁止把最小已测点称为连续参数最优。",
    "",
    "## 全部诊断结果",
    "",
    "表中的图内精排不含路由成本\uff1b总精确调用数将原入口及4候选一并计入\uff0c还不包含全维锚点评分本身。duration=-1\uff0c所有计时CSV仅含表头。",
    "",
    "| 锚点 | 保留原尺度 | margin | Recall@10 | 平均展开 | 图内精排 | 总精确调用 | 达到cap比例 |",
    "|---:|---|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    lines.append(
        f"| {r['anchors']:,} | {('是' if r['keep_original_bucket_scale'] else '否')} | {r['margin']:.2f} | {100 * r['recall']:.2f}% | {r['mean_expanded']:.3f} | {r['mean_refined']:.3f} | {r['mean_total_exact_calls']:.3f} | {100 * r['cap_fraction']:.1f}% |"
    )
lines += [
    "",
    "相同margin=0.02下\uff0c4,096锚点保留原尺度时Recall@10=90.37%\uff0c平均展开26.121\uff0c对照为90.36%/30.778\uff1b展开减少15.13%\uff0c总精确调用却由13.545增加到18.592。65,536锚点使用新尺度时为90.72%/21.361\uff0c展开减少30.60%\uff0c总精确调用18.368。保留原尺度时为91.24%/22.110\uff0c仍有约28.16%的展开减少\uff0c但召回也更高。",
    "",
    "这些点都达到90%门槛\uff0c但不是严格同召回对比\uff0c也不构成展开下界。全维FP32评分有舍入误差\uff0c选近入口不保证图搜索路径或全局召回单调改善。不能把展开减少比例换算成QPS增幅。",
    "",
    "## 正确性与可审查性",
    "",
    "旧路由接口9,000项检查、新增大规模/原距离输出/禁用路由600项检查、默认28项回归均通过。路由关闭时1,000查询的10,000条ID/精确分数与int55_original完全一致。共17,000次查询评估、170,000条ID/分数记录已检查完整性\uff0c17组统计记录完整。两个构建清单逐文件核对与本阶段源码一致。",
    "",
    "[完整成本核算](../results/anchor58_accounting.json)、[验证摘要](../results/anchor58_validation.json)、[原始文件SHA-256](../results/anchor58_artifact_sha256.json)。原始逐查询数据保留在bench目录的experiments/anchor58_*和results/anchor58_*_stats.csv\uff1b不将诊断扫描混入计时汇总。主查询已经用于调参\uff0c额外查询仍未运行图检索。",
    "",
    "## 下一阶段",
    "",
    "研究数据库样本训练的分层整数入口路由\uff0c复用已编码查询\uff0c并批量计算候选中心的整数内积\uff1b首先核对图内ID映射与训练数据来源\uff0c再测试单位路由成本、召回和完整在线QPS。当前未训练、未实测此候选\uff0c层次聚类或整数路由本身不作为新颖性声明。详见[后续设计](hierarchical_router59_plan.md)。",
]
(R / "notes/anchor58.md").write_text("\n".join(lines) + "\n")
logging.info("%s %s", "ANCHOR58 FINALIZATION COMPLETE", checks)
