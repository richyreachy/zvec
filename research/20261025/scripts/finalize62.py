from __future__ import annotations

import csv
import json
import logging
import statistics
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
assert "AFTER_SCREEN62 COMPLETE" in (R / "logs/after_screen62.log").read_text()
actual = json.loads((R / "results/directional62_actual_validation.json").read_text())
selected = json.loads((R / "results/directional62_selected_primary.json").read_text())
guards = json.loads((R / "results/directional62_guard_checks.json").read_text())
assert len(guards) == 8
rows = []
evaluations = 0
id_rows = 0
controls = []
for prefix in ["screen62", "recover62"]:
    for folder in sorted((R / "experiments").glob(prefix + "_cohere_*")):
        local = []
        for file in sorted(folder.glob("search_k10_r*.csv")):
            if file.stem.endswith("_ids"):
                with file.open() as f:
                    id_rows += sum(1 for _ in f)
            elif file.stem.endswith("_recall"):
                with file.open() as f:
                    for row in csv.DictReader(f):
                        assert int(row["returned"]) == 10
                        evaluations += 1
            else:
                with file.open() as f:
                    local.extend(
                        {**row, "experiment": folder.name} for row in csv.DictReader(f)
                    )
        rows.extend(local)
        if "dir62_" not in folder.name:
            controls.append(
                {
                    "experiment": folder.name,
                    "recall": float(local[0]["recall"]),
                    "qps_single_median": statistics.median(
                        float(x["qps"]) for x in local if x["threads"] == "1"
                    ),
                    "qps_eight_median": statistics.median(
                        float(x["qps"]) for x in local if x["threads"] == "8"
                    ),
                }
            )
assert len(rows) == 156
assert evaluations == 78000
assert id_rows == 780000
for mode in ["identity", "trained", "mse"]:
    log = (R / f"logs/directional62_{mode}_test.log").read_text()
    assert "factor_checks=1024 LUT_checks=8192 endpoint_checks=2028" in log
    assert "unchanged_L2_blocks=1024" in log
    assert "passed" in log
summary = {
    "selected": selected,
    "controls": controls,
    "timing_rows": len(rows),
    "screen_query_evaluations": evaluations,
    "screen_id_score_rows": id_rows,
    "unit_modes": 3,
    "factor_checks": 3072,
    "lut_checks": 24576,
    "endpoint_checks": 6084,
    "unchanged_l2_blocks": 3072,
    "beta1_bit_identical_blocks": 1024,
    "compile_guard_cases": 8,
    "actual_validation": actual,
    "breakthrough_verified": False,
    "conclusion": "Trained shrink loses QPS after recovering recall; per-edge MSE shrink fails90% within tested64-expansion budget even with loose stopping margins. Do not combine this candidate with projected router.",
}
(R / "results/directional62_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
lines = [
    "# \u65b9\u5411\u56e0\u5b50\u6821\u51c6\uff1a\u6062\u590d\u53ec\u56de\u540e\u4ecd\u65e0\u6027\u80fd\u6536\u76ca\uff08\u9636\u6bb562\uff09",
    "",
    "\u7ed3\u8bba\uff1a\u56fa\u5b9a\u8bad\u7ec3\u7cfb\u6570\u7684\u5c40\u90e8\u95f4\u9694MSE\u4e0b\u964d5.42%\uff0c\u6ca1\u6709\u8f6c\u5316\u4e3aCohere Top-10\u3001Recall\u226590%\u7684QPS\u6539\u5584\u3002\u9010\u8fb9\u91cd\u5efaMSE\u7cfb\u6570\u7684\u6548\u679c\u66f4\u5dee\u3002\u505c\u6b62\u6b64\u5355\u7cfb\u6570\u65b9\u5411\uff0c\u4e0d\u4e0e\u6295\u5f71\u8def\u7531\u7ec4\u5408\u3002\u5f53\u524d\u6700\u4f73\u641c\u7d22\u5019\u9009\u4ecd\u662f\u9636\u6bb560\u7684proj60_384\uff0c\u5c1a\u672a\u8fbe\u5230\u91cd\u5927\u7a81\u7834\u95e8\u69db\u3002",
    "",
    "## \u6570\u5b66\u4e0e\u5b9e\u73b0",
    "",
    "[\u9884\u5148\u8bbe\u8ba1](directional_shrink62_plan.md)\u53ca[\u505c\u6b62\u6761\u4ef6\u8026\u5408\u5206\u6790](directional62_mechanism.md)\u7ed9\u51fa\u516c\u5f0f\u3002\u4fdd\u6301\u5f84\u5411\u5e38\u6570\uff0c\u53ea\u7f29\u653e\u65b9\u5411\u9879\uff1a`d\u03b2=d(q,c)-c\xb7r-\u03b2(q-c)\xb7h`\u3002\u6784\u5efa\u65f6\u6539\u5199\u5df2\u6709\u4e24\u4e2a\u6d6e\u70b9\u56e0\u5b50\uff0c\u4e0d\u589e\u52a0\u7801\u957f\u6216\u67e5\u8be2\u7b97\u672f\uff1b\u65b0\u589e\u5168\u7ef4\u70b9\u79ef\u4f1a\u589e\u52a0\u9884\u6784\u5efa\u5de5\u4f5c\uff0c\u672a\u5c06\u6784\u5efa\u6210\u672c\u7b97\u4f5c\u514d\u8d39\u3002",
    "",
    "\u56fa\u5b9a\u8bad\u7ec3\u03b2=0.9359792556459329\uff0c\u7531\u56fe\u5185\u8bad\u7ec3\u4e2d\u5fc3\u62df\u5408\uff0c\u4e0d\u8bfb\u53d6\u4e3b\u67e5\u8be2\u6216GT\u3002\u9010\u8fb9\u5bf9\u7167\u4f7f\u7528`clip(||r||\u2081\xb2/(d||r||\xb2),0.5,1)`\uff0c\u8fd9\u662f\u91cd\u5efaMSE\u76ee\u6807\uff0c\u96f6\u6b8b\u5dee\u91c7\u7528\u03b2=1\u3002\u6240\u6709\u5f00\u5173\u9ed8\u8ba4\u5173\u95ed\uff0c\u62d2\u7edd\u4e0e\u6539\u53d8\u4f30\u8ba1\u516c\u5f0f\u7684\u5176\u4ed6\u539f\u578b\u6df7\u7528\u3002",
    "",
    "## \u5b8c\u6574\u641c\u7d22\u7ed3\u679c",
    "",
    "\u540c\u4e00Cohere 1M\u3001768\u7ef4\u3001Top-10\u300132\u8fb9\u56fe\u3001cap64\u3001ef800\u3001\u56fa\u5b9aCPU\u4eb2\u548c\u6027\uff1b\u4e24\u4e2a\u7ebf\u7a0b\u914d\u7f6e\u5404\u4e24\u8f6e3\u79d2\u8ba1\u65f6\u3002\u539f\u59cb1000\u67e5\u8be2\u5df2\u7528\u4e8e\u5f00\u53d1\uff0c\u4e0d\u662f\u76f2\u6d4b\u3002\u521d\u7b5bmargin\u4e0d\u8db3\uff0c\u968f\u540e\u660e\u786e\u6269\u5c55\u505c\u6b62margin\uff1b\u8bad\u7ec3\u7cfb\u6570\u672a\u6539\u53d8\u3002\u4e0d\u5b58\u572890%\u8fbe\u6807\u70b9\u7684\u53d8\u4f53\u53ea\u62a5\u544a\u6700\u9ad8\u53ec\u56de\uff0c\u4f4e\u53ec\u56de\u9ad8QPS\u4e0d\u4f5c\u4e3a\u76ee\u6807\u6536\u76ca\u3002",
    "",
    "| \u53d8\u4f53 | \u9009\u5b9amargin | Recall@10 | \u5355\u7ebf\u7a0bQPS\u4e2d\u4f4d\u6570 | 8\u7ebf\u7a0bQPS\u4e2d\u4f4d\u6570 | \u8fbe\u6807 |",
    "|---|---:|---:|---:|---:|---|",
]
for variant, row in selected.items():
    q1 = f"{row['qps_single_median']:,.2f}" if row["passing"] else "\u2014"
    q8 = f"{row['qps_eight_median']:,.2f}" if row["passing"] else "\u2014"
    passing_label = (
        "\u662f"
        if row["passing"]
        else "\u5426\uff0c\u5217\u6700\u9ad8\u5df2\u6d4b\u53ec\u56de"
    )
    lines.append(
        f"| {variant} | {row['label'] / 1000000:g} | {row['recall']:.2%} | {q1} | {q8} | {passing_label} |"
    )
lines += [
    "",
    "\u540c\u671f\u5bf9\u7167\uff1a",
    "",
    "| \u5b9e\u9a8c | Recall@10 | \u5355\u7ebf\u7a0bQPS\u4e2d\u4f4d\u6570 | 8\u7ebf\u7a0bQPS\u4e2d\u4f4d\u6570 |",
    "|---|---:|---:|---:|",
]
for row in controls:
    lines.append(
        f"| {row['experiment']} | {row['recall']:.2%} | {row['qps_single_median']:,.2f} | {row['qps_eight_median']:,.2f} |"
    )
lines += [
    "",
    "\u521d\u7b5b\u56fa\u5b9a\u8bad\u7ec3\u7cfb\u6570\u5728\u539fmargin0.02\u65f6\uff0cFP32\u4e2d\u5fc3\u53ec\u56de\u4ec579.04%\uff1b\u653e\u5bbd\u52300.07\u624d\u8fbe90.42%\u3002\u56e0\u4e3a`d\u03b2-d1=(1-\u03b2)(q-c)\xb7h`\uff0c\u4e0e\u8fb9\u65b9\u5411\u4e00\u81f4\u7684\u67e5\u8be2\u88ab\u62ac\u9ad8\u4f30\u8ba1\u8ddd\u79bb\uff0c\u539f\u505c\u6b62\u6761\u4ef6\u53d8\u5f97\u6fc0\u8fdb\u3002\u5c40\u90e8\u5019\u9009\u5dee\u7684MSE\u4e0e\u76f8\u5bf9\u4e8e\u4e2d\u5fc3\u7684\u7edd\u5bf9\u5206\u6570\u9700\u8981\u5206\u522b\u9a8c\u8bc1\u3002\u9010\u8fb9\u7cfb\u6570\u6539\u53d8\u66f4\u5927\uff0c\u5728\u672c\u6b21\u56fa\u5b9acap64\u4e0b\uff0c\u5373\u4f7fmargin\u8fbe\u52300.8\u4ecd\u672a\u8fbe\u523090%\u3002\u8fd9\u4e0d\u662f\u58f0\u79f0\u4efb\u4f55\u66f4\u5927\u641c\u7d22\u9884\u7b97\u90fd\u65e0\u6cd5\u6062\u590d\u53ec\u56de\u3002",
    "",
    "## \u9a8c\u8bc1\u4e0e\u8bc1\u636e",
    "",
    "\u4e09\u6a21\u5f0f\u51713,072\u7ec4\u56e0\u5b50\u300124,576\u6b21\u5b9e\u9645FastScan LUT\u8bef\u5dee\u30016,084\u4e2a\u7aef\u70b9\u68c0\u67e5\u901a\u8fc7\uff1b\u6700\u5927\u7406\u60f3\u516c\u5f0f\u8bef\u5dee\u5c0f\u4e8e1e-6\u3002\u03b2=1\u76841,024\u7801\u5757\u9010\u5b57\u8282\u4e00\u81f4\uff0c3,072\u7ec4L2\u5757\u4fdd\u6301\u4e0d\u53d8\uff1b8\u79cd\u975e\u6cd5\u7ec4\u5408/\u8303\u56f4\u7f16\u8bd1\u4fdd\u62a4\u901a\u8fc7\u3002\u9ed8\u8ba428\u9879\u56de\u5f52\u901a\u8fc7\u3002",
    "",
    "\u56db\u4e2a\u5019\u9009\u9009\u5b9a\u70b9\u7684\u4e24\u8f6e\u8f93\u51fa\u4e00\u81f4\uff0c\u56db\u4e2aaudit\u7248\u672c\u4e0e\u8ba1\u65f6\u8f93\u51fa\u4e00\u81f4\uff0c\u4e24\u4e2a\u6574\u6570\u7248\u672c\u4e0e\u5168\u91cf\u7cbe\u6392\u4e00\u81f4\uff1b\u53e6\u6709\u03b2=1\u76841,000\u67e5\u8be2\u4e0e\u6574\u6570\u5bf9\u7167\u9010ID/\u5206\u6570\u4e00\u81f4\u3002\u5408\u8ba17,000\u6b21\u8bca\u65ad\u67e5\u8be2\uff0c\u53e6\u6709\u4e24\u8f6e\u95f44,000\u6b21\u6bd4\u8f83\u3002audit/full\u4e00\u81f4\u6027\u4ec5\u8bc1\u660e\u5df2\u5c55\u5f00\u96c6\u5408\u4e0a\u7684\u7cbe\u6392\u6b63\u786e\uff0c\u4e0d\u8bc1\u660e\u5bfc\u822a\u7684\u5168\u5c40\u53ec\u56de\u3002",
    "",
    "\u5168\u90e8156\u6761\u8ba1\u65f6\u8bb0\u5f55\u300178,000\u6b21screen/recover\u67e5\u8be2\u8bc4\u4f30\u3001780,000\u6761ID/\u5206\u6570\u5b8c\u6574\uff1b12\u4e2a\u6784\u5efa\u6e90\u7801\u6e05\u5355\u4e0e\u5f53\u524d\u6e90\u7801\u4e00\u81f4\u3002\u6ca1\u6709\u4f7f\u7528\u989d\u5916\u67e5\u8be2\u8c03\u53c2\u6216\u9a8c\u6536\uff0c\u4e5f\u6ca1\u6709\u91cd\u5927\u7a81\u7834\u3002",
    "",
    "[\u7ed3\u679c\u6c47\u603b](../results/directional62_summary.json)\u3001[\u9009\u70b9](../results/directional62_selected_primary.json)\u3001[\u5b9e\u9645\u5ba1\u8ba1](../results/directional62_actual_validation.json)\u3001[\u521d\u7b5b\u914d\u7f6e](../results/directional62_screen_plan.json)\u3001[\u6062\u590d\u53ec\u56de\u7f51\u683c](../results/directional62_recovery_plan.json)\u3002\u539f\u59cb\u5b9e\u9a8c\u76ee\u5f55\u4e3abench/experiments/screen62_*\u4e0erecover62_*\u3002",
    "",
    "\u4e0b\u4e00\u6b65\u5e94\u540c\u65f6\u5173\u6ce8\u4f30\u8ba1\u7684\u5747\u503c\u504f\u5dee\u548c\u505c\u6b62\u6761\u4ef6\uff0c\u6216\u8f6c\u5411\u5b8c\u6574\u56fe\u7ec6\u5316\u5bf9\u7167\uff1b[\u7814\u7a76\u7ea6\u675f\u4e0e\u5df2\u6709\u5de5\u4f5c](research63_options.md)\u3002\u57fa\u7840\u6700\u5c0f\u4e8c\u4e58\u3001\u6536\u7f29\u4e0e\u73b0\u6709\u56fe\u7ec6\u5316\u5747\u4e0d\u80fd\u4f5c\u4e3a\u65b0\u9896\u6027\u4e3b\u5f20\u3002",
]
(R / "notes/directional62.md").write_text("\n".join(lines) + "\n")
logging.info("%s %s %s %s", "FINALIZE62 COMPLETE", len(rows), evaluations, id_rows)
