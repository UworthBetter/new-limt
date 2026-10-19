"""Benchmark-arm forensics: training row counts, boosting/epoch counts,
validation r1 IC for all six Qlib-default benchmark models.

Checks the review's red flag: XGBoost 10 s / DoubleEnsemble 75 s are only
plausible if the training set was complete and the boosting counts match the
published defaults' behaviour. CPU only; reads runs/scores/*.npz + meta.

Usage: bench_forensics.py --cache DIR
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

LIMT_SRC = Path(r"D:\research\limt-repro\src")
SRC = Path(__file__).resolve().parent
for path in (str(LIMT_SRC), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from v2_w4_data import Data  # noqa: E402
from v2_w6_run import DEFAULT  # noqa: E402

ROOT = Path(r"D:\research\new_LiMT")
LABELS = ROOT / "data" / "csi300_w1w2" / "data" / "v2" / "new_v2_labels_csi300.npz"
TAGS = ("XGBoost_seed1", "DoubleEnsemble_seed1", "LSTM_seed1", "GRU_seed1", "ALSTM_seed1", "Transformer_seed1")


def daily_r1_ic(data, days, scores, z_r1, mask_r1):
    values = []
    for i, t in enumerate(days):
        keep = mask_r1[t] & np.isfinite(z_r1[t]) & np.isfinite(scores[i])
        if keep.sum() < 3:
            continue
        x, y = scores[i][keep], z_r1[t][keep]
        if x.std() > 0 and y.std() > 0:
            values.append(float(np.corrcoef(x, y)[0, 1]))
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    args = parser.parse_args()
    data = Data(ROOT / "data" / "csi300_w1w2", args.cache, 21, 0, DEFAULT["evaluation_end"])
    with np.load(LABELS, allow_pickle=False) as labels:
        z_r1, mask_r1 = labels["z_r1"], labels["label_mask_r1"]
    dates = data.dates
    spans = {
        "train": ("2008-01-01", "2018-12-31"),
        "valid": ("2019-01-01", "2020-12-31"),
    }
    counts = {}
    for name, (lo, hi) in spans.items():
        i = int(np.searchsorted(dates, np.datetime64(lo), side="left"))
        j = int(np.searchsorted(dates, np.datetime64(hi), side="right"))
        counts[name] = int((mask_r1[i:j] & np.isfinite(z_r1[i:j])).sum())
    valid_days = data.days("valid")
    out = {"train_row_counts": counts,
           "reference": "ridge r1 fit samples from the 2026-10-17 CPU-baseline run: 768,338 (train)"}
    for tag in TAGS:
        meta = json.loads((ROOT / "runs" / "scores" / f"{tag}.meta.json").read_text(encoding="utf-8"))
        with np.load(ROOT / "runs" / "scores" / f"{tag}.npz", allow_pickle=False) as payload:
            scores_valid = payload["scores_valid"]
        ic = daily_r1_ic(data, valid_days, scores_valid, z_r1, mask_r1)
        out[tag] = {
            "fit_seconds": meta.get("fit_seconds"),
            "evals_result_summary": meta.get("evals_result_summary"),
            "mean_valid_r1_ic": float(np.mean(ic)) if ic else None,
            "ic_days": len(ic),
        }
        print(tag, json.dumps(out[tag]), flush=True)
    dest = ROOT / "protocol" / "checks" / "benchmark_forensics.json"
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("TRAIN_ROWS", counts, flush=True)
    print(f"WROTE {dest}", flush=True)


if __name__ == "__main__":
    main()
