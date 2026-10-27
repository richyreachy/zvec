from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
M = R / "models70"
assert "POST70 COMPLETE" in (R / "logs/post70.log").read_text()
names = ["baseline", "70", "71", "72"]
errors = np.stack([np.load(M / f"codec_errors_{name}.npy") for name in names])
assert errors.shape == (4, 2048, 4)
base = errors[0]
loo = np.empty_like(base)
selected = np.empty_like(base, dtype=np.uint8)
for heldout in range(4):
    training = np.delete(errors, heldout, axis=2).sum(axis=2)
    chosen = np.argmin(training, axis=0)
    loo[:, heldout] = errors[chosen, np.arange(2048), heldout]
    selected[:, heldout] = chosen
oracle_center = errors.sum(axis=2).min(axis=0).sum() / base.size
oracle_query = errors.min(axis=0).mean()
change = base.mean(axis=1) - loo.mean(axis=1)
b = base.mean(axis=1)
rng = np.random.default_rng(7071)
boot = []
for _i in range(4000):
    sample = rng.integers(0, 2048, 2048)
    boot.append(float(change[sample].mean() / b[sample].mean()))
result = {
    "models": names,
    "baseline_error": float(base.mean()),
    "oracle_per_center_error": float(oracle_center),
    "oracle_per_query_error": float(oracle_query),
    "leave_one_local_query_out_error": float(loo.mean()),
    "leave_one_out_relative_error_gain": float(change.mean() / b.mean()),
    "paired_center_bootstrap95": np.quantile(boot, [0.025, 0.975]).tolist(),
    "frame_selection_counts": np.bincount(selected.ravel(), minlength=4).tolist(),
    "tie_break_baseline_first": True,
    "leave_one_out_selection_uses_three_other_proxy_labels": True,
    "posthoc_reuse_of_validation_centers_not_blind": True,
    "local_queries_not_independent": True,
    "oracles_not_deployable": True,
    "no_retrieval_qps_or_recall_claim": True,
}
(R / "results/rank70_adaptive_probe.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
logging.info("%s", json.dumps(result, indent=2))
logging.info("%s", "ADAPTIVE70 PROBE COMPLETE")
