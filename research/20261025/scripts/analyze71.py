from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
M = R / "models71"
assert "RUN71 COMPLETE" in (R / "logs/run71.log").read_text()
config = json.loads((M / "config.json").read_text())
names = config["models"]
truth = np.memmap(M / "truth.f64", mode="r", dtype="<f8", shape=(2048, 32, 32))
minimum = truth.min(axis=2)
sel = np.fromfile(M / "selection_positions.u32", dtype="<u4")
val = np.fromfile(M / "validation_positions.u32", dtype="<u4")
errors = []
for name in names:
    score = np.fromfile(M / f"codec_scores_{name}.f32", dtype="<f4").reshape(
        2048, 32, 32
    )
    assert np.isfinite(score).all()
    chosen = np.argmin(score, axis=2)
    err = (
        np.take_along_axis(truth, chosen[:, :, None], axis=2)[:, :, 0] > minimum + 1e-07
    )
    errors.append(err)
    np.save(M / f"codec_errors_{name}.npy", err)
errors = np.stack(errors)
selection = errors[:, :, sel].sum(axis=2)
chosen = np.argmin(selection, axis=0)
chosen.astype("<u4").tofile(M / "selected_frames.u32")
base = errors[0][:, val]
selected = errors[chosen, np.arange(2048)][:, val]
np.save(M / "selected_validation_errors.npy", selected)
change = base.mean(axis=1) - selected.mean(axis=1)
b = base.mean(axis=1)
rng = np.random.default_rng(7171)
boot = []
for _ in range(4000):
    sample = rng.integers(0, 2048, 2048)
    boot.append(float(change[sample].mean() / b[sample].mean()))
gain = float(change.mean() / b.mean())
ci = np.quantile(boot, [0.025, 0.975]).tolist()
perturbed = errors.copy()
perturbed[:, :, val] = ~perturbed[:, :, val]
assert np.array_equal(chosen, np.argmin(perturbed[:, :, sel].sum(axis=2), axis=0))
assert np.all(np.argmin(np.zeros((4, 2048)), axis=0) == 0)
result = {
    "config": config,
    "selection_validation_label_isolation_checked": True,
    "baseline_first_ties_checked": True,
    "validation_queries": int(base.size),
    "baseline_validation_errors": int(base.sum()),
    "selected_validation_errors": int(selected.sum()),
    "baseline_validation_error_rate": float(base.mean()),
    "selected_validation_error_rate": float(selected.mean()),
    "relative_error_reduction": gain,
    "paired_center_bootstrap95": ci,
    "quality_gate_passed": gain >= 0.1 and ci[0] > 0,
    "selected_frame_counts": np.bincount(chosen, minlength=4).tolist(),
    "centers_better_equal_worse": [
        int(np.sum(change > 0)),
        int(np.sum(change == 0)),
        int(np.sum(change < 0)),
    ],
    "baseline_selection_error_rate": float(errors[0][:, sel].mean()),
    "selected_selection_error_rate": float(selection.min(axis=0).sum() / (2048 * 16)),
    "fixed_model_validation_error_rates": {
        name: float(errors[i][:, val].mean()) for i, name in enumerate(names)
    },
    "validation_per_center_oracle_error_rate_not_deployable": float(
        errors[:, :, val].sum(axis=2).min(axis=0).sum() / (2048 * 16)
    ),
    "selection_ties": int(
        np.sum(np.sum(selection == selection.min(axis=0), axis=0) > 1)
    ),
    "truth_ties_selection": int(
        np.sum(np.sum(truth[:, sel] <= minimum[:, sel, None] + 1e-07, axis=2) > 1)
    ),
    "truth_ties_validation": int(
        np.sum(np.sum(truth[:, val] <= minimum[:, val, None] + 1e-07, axis=2) > 1)
    ),
    "not_iid_blind_external_query_validation": True,
    "actual_codec_fp32_rotations_used": True,
    "full_search_integration": False,
    "new_retrieval_timing_rows": 0,
    "cumulative_retrieval_timing_rows": 1310,
    "no_qps_or_recall_claim": True,
}
(R / "results/local71_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
logging.info("%s", json.dumps(result, indent=2))
logging.info("%s", "ANALYZE71 COMPLETE")
