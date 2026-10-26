from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
for mode, per_query in [("database", 8), ("primary", 32)]:
    root = R / ("models68" if mode == "database" else "models68/primary")
    pairs = np.load(root / "pairs_v2.npy")
    out = root / "geometry"
    out.mkdir(exist_ok=False)
    rng = np.random.default_rng(6801)
    selected = []
    unique, starts, counts = np.unique(
        pairs["query"], return_index=True, return_counts=True
    )
    for _query, start, count in zip(unique, starts, counts, strict=True):
        selected.extend(
            (
                start + rng.choice(count, min(per_query, int(count)), replace=False)
            ).tolist()
        )
    sample = pairs[np.asarray(selected)]
    np.save(out / "sample.npy", sample)
    ids = np.column_stack(
        (
            sample["target"],
            sample["source1"],
            sample["source2"],
            sample["query_id"] if mode == "database" else sample["query"],
        )
    ).astype("<u4")
    ids.tofile(out / "ids.u32")
    config = {
        "mode": mode,
        "max_pairs_per_query": per_query,
        "seed": 6801,
        "sampled_pairs": len(sample),
        "queries": len(unique),
        "selection_uses_only_query_and_pair_position_not_outcomes": True,
        "lambda_frozen_from_stage67": json.loads(
            (R / "results/multianchor67_calibrated.json").read_text()
        )["lambda_fitted_only_on_training_targets"],
        "no_fit_on_current_queries": True,
        "primary_not_blind": mode == "primary",
        "self_center_pairs_reported_separately": True,
        "all_first_two_pending_arrivals": True,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    logging.info("%s %s", mode, len(sample))
