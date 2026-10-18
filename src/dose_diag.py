"""Dose diagnostics (step 5): validation 5-day IC + soft vs top-k on 2021-2022.

(a) 5-day IC on the validation window (2019-2020) for every scored arm,
    reusing v2_w5_results.daily_ic (per-day Pearson corr of the score with
    the z-scored h5 label over the window's loss mask).
(b) Soft layer vs top-k execution on the rule-selection window (2021-2022):
    soft = continuous-weight proportional-fee replay (portfolio_events,
    rule="soft", tau=0.5, 5/15 bp, no min fee / lot); top-k = the executed
    share-ledger default rule (topk 50, drop 5, daily) from the grid artefact.
    Both reported as annualized after-cost excess vs H00300.

Exploratory arms are labelled as such. CPU only; consumes score artefacts.
Usage: dose_diag.py --cache DIR [--tags tag ...]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
LIMT_SRC = Path(r"D:\research\limt-repro\src")
for path in (str(SRC), str(LIMT_SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from cpu_baselines import BENCH, ROOT, share_replay  # noqa: E402
from v2_w4_data import Data  # noqa: E402
from v2_w4_training import portfolio_events  # noqa: E402
from v2_w5_results import benchmark, daily_ic, load_execution_feed  # noqa: E402
from v2_w6_run import DEFAULT  # noqa: E402

FEED = Path(r"D:\research\limt-repro\data\v2_formal_inputs_run3\new_execution_feed.npz")
WINDOW_END_SELECT = "2022-12-31"
TAU = 0.5


def tag_parts(tag):
    kind, _, seed = tag.rpartition("_seed")
    return kind, int(seed)


def soft_annualized_excess(data, days, scores, bench):
    rows = portfolio_events(data, days, scores, TAU, DEFAULT["buy_rate"], DEFAULT["sell_rate"], "soft")
    values = []
    for row in rows:
        day = row["date"]
        if day in bench["values"]:
            values.append(row["net_return"] - bench["values"][day])
    return float(np.mean(values) * 252), len(values), rows[0]["date"], rows[-1]["date"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--tags", nargs="*", default=None)
    args = parser.parse_args()
    scores_dir = ROOT / "runs" / "scores"
    tags = args.tags or sorted(p.stem for p in scores_dir.glob("*.npz"))
    data = Data(ROOT / "data" / "csi300_w1w2", args.cache, 21, 0, DEFAULT["evaluation_end"])
    valid_days = data.days("valid")
    select_days = data.days("rule_select")
    assert str(data.dates[int(valid_days[-1]) + 2]) <= "2020-12-31"
    assert str(data.dates[int(select_days[-1]) + 2]) <= WINDOW_END_SELECT
    print(f"ASSERT windows: valid terminal mark {data.dates[int(valid_days[-1]) + 2]} <= 2020-12-31; "
          f"rule_select terminal mark {data.dates[int(select_days[-1]) + 2]} <= {WINDOW_END_SELECT}", flush=True)
    feed = load_execution_feed(data, FEED)
    bench = benchmark(BENCH)
    expected_valid = [str(d) for d in data.dates[valid_days]]
    expected_select = [str(d) for d in data.dates[select_days]]
    started = time.perf_counter()
    out = {}
    for tag in tags:
        path = scores_dir / f"{tag}.npz"
        if not path.is_file():
            print(f"SKIP {tag}", flush=True)
            continue
        kind, seed = tag_parts(tag)
        with np.load(path, allow_pickle=False) as payload:
            if [str(d) for d in payload["dates_valid"]] != expected_valid:
                raise SystemExit(f"{tag}: valid dates mismatch")
            if [str(d) for d in payload["dates_rule_select"]] != expected_select:
                raise SystemExit(f"{tag}: rule_select dates mismatch")
            valid_scores = payload["scores_valid"]
            select_scores = payload["scores_rule_select"]
        ic = daily_ic(data, valid_days, valid_scores, "valid")
        soft_excess, n, first, last = soft_annualized_excess(data, select_days, select_scores, bench)
        rows = share_replay(data, select_days, select_scores, feed, 5, 1)
        dates = [row["date"] for row in rows]
        topk_excess = float(np.mean([row["net_simple_return"] - bench["values"][d] for row, d in zip(rows, dates)]) * 252)
        topk_drag = float(np.mean([row["fee_fraction"] for row in rows]) * 252)
        out[tag] = dict(
            arm=kind, seed=seed, exploratory="dose003" in kind,
            five_day_ic_valid=dict(mean_ic=ic["mean_ic"], valid_days=ic["valid_days"], samples=ic["samples"]),
            soft_layer_2021_2022=dict(annualized_excess_pp=soft_excess * 100, n_days=n, first_date=first, last_date=last),
            topk_default_2021_2022=dict(annualized_excess_pp=topk_excess * 100, cost_drag_annualized_pp=topk_drag * 100),
            soft_minus_topk_pp=(soft_excess - topk_excess) * 100,
        )
        print(json.dumps({tag: out[tag]}, default=str), flush=True)
    result = dict(window_ic="valid 2019-2020 (h5 label)", window_layer="rule_select 2021-2022",
                  tau=TAU, soft_convention="continuous weights, proportional 5/15 bp, no min fee/lot",
                  topk_convention="share ledger, topk50 drop5 daily, full frictions",
                  seconds=time.perf_counter() - started, arms=out)
    dest = ROOT / "protocol" / "checks" / "dose_diag_2026-10-18.json"
    dest.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"WROTE {dest} arms={sorted(out)}", flush=True)


if __name__ == "__main__":
    main()
