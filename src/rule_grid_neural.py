"""Neural-arm 8-rule grid on the 2021-2022 rule-selection window (CPU).

Consumes the score artefacts written by score_checkpoints.py, replays all
eight Section 5 rules (topk 50; drop in {2,5,10,20} x every {1,5}) on the
share ledger, writes readable daily.csv files under runs/csi300/<arm>/h5/
<rule>/seed<N>/ plus a per-tag grid json, and computes the arm-level rule
selection averaged over the seeds currently on disk (choose_arm from
cpu_baselines: seed-averaged annualized excess, 0.5pp tie -> lower turnover).

Integrity: the runner reads no price dated after 2022-12-31.  Signal days
stop at split-last-2 (v2_w6_policy.signal_days), so the last execution fill
(signal+1) and the terminal accounting mark (signal+2) are both <= 2022-12-31;
asserted and logged before any replay.

Dose arms (G0dose003/C0dose003, seed 1) are labelled exploratory everywhere.

Usage: rule_grid_neural.py --cache DIR [--tags tag ...]
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

from cpu_baselines import (  # noqa: E402
    BENCH, COLUMNS, EVAL_START, ROOT, RULES, annual_excess, choose_arm, choose_rule, share_replay, write_csv,
    _with_benchmark,
)
from v2_w4_data import Data  # noqa: E402
from v2_w5_results import benchmark, load_execution_feed  # noqa: E402
from v2_w6_run import DEFAULT  # noqa: E402

FEED = Path(r"D:\research\limt-repro\data\v2_formal_inputs_run3\new_execution_feed.npz")
WINDOW_END = "2022-12-31"
ARMS = {  # tag prefix -> (output backbone name, seeds expected eventually)
    "H": "H", "H-FT": "H-FT", "G0": "G0", "C0": "C0",
    "G0dose003": "G0dose003", "C0dose003": "C0dose003",
}


def tag_parts(tag):
    kind, _, seed = tag.rpartition("_seed")
    return kind, int(seed)


NEURAL_KINDS = ("H", "H-FT", "G0", "C0")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--tags", nargs="*", default=None)
    parser.add_argument("--label", default="h5",
                        help="output label axis (h5 for the neural arms, r1 for the Qlib-default benchmark arms)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    scores_dir = ROOT / "runs" / "scores"
    available = sorted(p.stem for p in scores_dir.glob("*.npz"))
    tags = args.tags or available
    data = Data(ROOT / "data" / "csi300_w1w2", args.cache, 21, 0, DEFAULT["evaluation_end"])
    days = data.days("rule_select")
    exec_last, terminal_last = int(days[-1]) + 1, int(days[-1]) + 2
    assert str(data.dates[terminal_last]) <= WINDOW_END, (str(data.dates[terminal_last]), WINDOW_END)
    print(
        f"ASSERT no-price-after-window: n_signal_days={len(days)} first={data.dates[days[0]]} "
        f"last_signal={data.dates[int(days[-1])]} last_execution_fill={data.dates[exec_last]} "
        f"terminal_mark={data.dates[terminal_last]} all <= {WINDOW_END}",
        flush=True,
    )
    feed = load_execution_feed(data, FEED)
    bench = benchmark(BENCH)
    expected_dates = [str(d) for d in data.dates[days]]
    started = time.perf_counter()
    grids = {}
    for tag in tags:
        path = scores_dir / f"{tag}.npz"
        if not path.is_file():
            print(f"SKIP {tag}: no score artefact", flush=True)
            continue
        kind, seed = tag_parts(tag)
        backbone = ARMS.get(kind, kind)  # benchmark tags use the model name as backbone
        with np.load(path, allow_pickle=False) as payload:
            dates = [str(d) for d in payload["dates_rule_select"]]
            scores = payload["scores_rule_select"]
        if dates != expected_dates:
            raise SystemExit(f"{tag}: rule_select dates do not match the panel")
        stats = []
        for name, drop, every in RULES:
            rows = share_replay(data, days, scores, feed, drop, every)
            readable = _with_benchmark(rows, bench)
            if any(row["date"] >= EVAL_START for row in readable):
                raise SystemExit(f"{tag} {name}: selection replay produced a 2023+ accounting date")
            out_dir = ROOT / "runs" / "csi300" / backbone / args.label / name / f"seed{seed}"
            target = out_dir / "daily.csv"
            if target.exists():
                # Idempotent re-run (e.g. full 3-seed selection after seeds 2-3 land):
                # the recomputed selection-window rows must match the earlier file.
                existing = target.read_text(encoding="utf-8").splitlines()
                fresh = [",".join(str(row[c]) for c in COLUMNS) for row in readable]
                if [existing[0]] + fresh != existing:
                    raise SystemExit(f"{tag} {name}: recomputed rows differ from existing {target}")
            else:
                write_csv(target, readable)
            excess, turnover = annual_excess(readable, bench)
            stats.append(dict(rule=name, annual_excess=excess, mean_turnover=turnover))
            print(f"rule {backbone} {args.label} seed{seed} {name} selected_on=2021-2022 "
                  f"excess={excess:.4f} turnover={turnover:.4f}", flush=True)
        (ROOT / "runs" / "csi300" / backbone / args.label / f"grid_seed{seed}.json").write_text(
            json.dumps(stats, indent=1), encoding="utf-8")
        if "dose003" in backbone:
            (ROOT / "runs" / "csi300" / backbone / "EXPLORATORY_README.txt").write_text(
                "Exploratory dose arm (deviation log 2026-10-18). All results post-hoc/exploratory; "
                "not in the Holm family; never merged with the primary contrast.\n", encoding="utf-8")
        grids[tag] = stats
    selection = []
    seen = set()
    for tag in sorted(grids):
        kind, _ = tag_parts(tag)
        if kind in seen:
            continue
        seen.add(kind)
        backbone = ARMS.get(kind, kind)
        arm_tags = [t for t in grids if tag_parts(t)[0] == kind]
        seeds = sorted(tag_parts(t)[1] for t in arm_tags)
        seed_grids = [grids[t] for t in sorted(arm_tags)]
        chosen = choose_arm(seed_grids) if len(seed_grids) > 1 else choose_rule(
            [(row["rule"], row["annual_excess"], row["mean_turnover"]) for row in seed_grids[0]])
        provisional = kind in NEURAL_KINDS and len(seeds) < 3
        selection.append(dict(arm=kind, backbone=backbone, label=args.label, seeds_used=seeds,
                              chosen_rule=chosen, exploratory="dose003" in kind,
                              descriptive=kind not in ARMS))
        print(f"SELECT {kind} label={args.label} seeds={seeds} rule={chosen}"
              f"{' PROVISIONAL (seeds 2-3 pending)' if provisional else ''}"
              f"{' DESCRIPTIVE benchmark arm' if kind not in ARMS else ''}",
              flush=True)
    record = dict(
        window="rule_select 2021-2022", label=args.label, topk=50,
        selection_rule="seed-averaged annualized excess, then 0.5pp turnover tie-break",
        integrity_assertion=f"last price date touched {data.dates[terminal_last]} <= {WINDOW_END}",
        grids={tag: grids[tag] for tag in sorted(grids)}, selection=selection,
        seconds=time.perf_counter() - started,
    )
    out = args.out or (ROOT / "runs" / ("csi300_bench_selection.json" if args.label != "h5" else "csi300_neural_selection.json"))
    out.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"WROTE {out} tags={sorted(grids)} seconds={record['seconds']:.1f}", flush=True)


if __name__ == "__main__":
    main()
