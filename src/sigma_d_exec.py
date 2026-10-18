"""Executed-book sigma_d on the validation window (2019-2020), default rule.

Prelim (Gate A, protocol 4.2): ridge_h5 vs H seed 1.
Final  (Gate B input):         C0 vs G0 seed 1 at the seed-1-selected tau=0.5.
Both arms replay the same default execution rule (topk 50, drop 5, daily) on
the share ledger, and the paired daily after-cost difference C0 - G0 (or
ridge - H) gives sigma_d = SD * sqrt(252), MDE80 = 2.487 * sigma_d / sqrt(3.5).

Diagnostics reported alongside the final pair: daily holdings overlap, rank
correlation of the two score vectors over the candidate set, and the
executed-layer cost-drag difference (SC6 preview).

No price dated after the window end is read: the last signal day stops at
last-2, so the terminal accounting mark (signal+2) and every execution fill
(signal+1) stay inside the window.  Asserted in the log.

Usage:
  sigma_d_exec.py --mode prelim|final --cache DIR [--out FILE]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parent
LIMT_SRC = Path(r"D:\research\limt-repro\src")
for path in (str(SRC), str(LIMT_SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from cpu_baselines import DEFAULT as CB_DEFAULT  # noqa: E402
from cpu_baselines import fit_ridge, predict_ridge, share_replay  # noqa: E402
from v2_w4_data import Data  # noqa: E402
from v2_w4_training import make_model, predict, topk_target  # noqa: E402
from v2_w3_ledger import Account, execute_close  # noqa: E402
from v2_w6_policy import listing_eligible  # noqa: E402
from v2_w6_run import DEFAULT, validate_config  # noqa: E402
from v2_w5_results import load_execution_feed  # noqa: E402

ROOT = Path(r"D:\research\new_LiMT")
W0W2 = ROOT / "data" / "csi300_w1w2"
FEED = Path(r"D:\research\limt-repro\data\v2_formal_inputs_run3\new_execution_feed.npz")
RUNS = ROOT / "runs"
WINDOW_END = "2020-12-31"  # validation split end; nothing later may be read
CHECKPOINTS = {
    "H": RUNS / "csi300_H_seed1_20261017" / "H" / "seed1" / "checkpoint.pt",
    "G0": RUNS / "csi300_g0c0_seed1_tau_20261017" / "G0" / "tau0.50" / "seed1" / "checkpoint.pt",
    "C0": RUNS / "csi300_g0c0_seed1_tau_20261017" / "C0" / "tau0.50" / "seed1" / "checkpoint.pt",
}
FIELDS = ("date", "net_simple_return", "fee_fraction", "turnover", "net_log_growth")


def sha256_file(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def replay_with_holdings(data, days, scores, feed, drop=5, every=1):
    """share_replay with the post-trade holdings set recorded per row.

    The numeric rows must be identical to cpu_baselines.share_replay; the
    caller asserts the diagonal before using any number from here.
    """
    prices, states, multiplier, dividend = [
        feed[k] for k in ("raw_close", "state", "share_multiplier", "cash_dividend_per_old_share")
    ]
    listed = listing_eligible(data.dates, data.symbols)
    account = Account.all_cash(len(data.symbols), CB_DEFAULT["capital"])
    selected, rows = set(), []
    for i in range(len(days) + 1):
        terminal = i == len(days)
        t = int(days[-1]) + 2 if terminal else int(days[i]) + 1
        original = account.value()
        action_cash = float(np.dot(account.shares, dividend[t]))
        account = Account(
            account.cash + action_cash,
            account.shares * multiplier[t],
            account.marks / multiplier[t],
            account.date,
        )
        if terminal or i % every != 0:
            target = None
        else:
            candidate = data.candidate(int(days[i])).copy()
            candidate &= listed[int(days[i])]
            target, selected = topk_target(scores[i], candidate, set(np.where(account.shares > 0)[0]), 50, drop)
        quotes = np.where(listed[t], prices[t], np.nan)
        effective = np.where(listed[t], states[t], 1)
        account, event = execute_close(
            account,
            quotes,
            effective,
            target,
            str(data.dates[t - 1]),
            str(data.dates[t]),
            buy=CB_DEFAULT["buy_rate"],
            sell=CB_DEFAULT["sell_rate"],
            min_fee=CB_DEFAULT["min_fee"],
            lot=int(CB_DEFAULT["lot"]),
        )
        pre = event["wealth_before_trade"]
        rows.append(
            dict(
                date=event["execution_date"],
                signal_date=event["signal_date"],
                gross_simple_return=event["gross"] - 1,
                fee_fraction=event["fees"] / pre,
                net_simple_return=event["wealth"] / original - 1,
                net_log_growth=event["log_growth"],
                turnover=event["turnover"],
                fee_value=event["fees"],
                holdings=(frozenset() if terminal else frozenset(selected)),
            )
        )
    return rows


def rows_match(left, right):
    if len(left) != len(right):
        return f"row count {len(left)} != {len(right)}"
    for a, b in zip(left, right):
        for field in FIELDS:
            if a[field] != b[field]:
                return f"{a['date']} {field}: {a[field]!r} != {b[field]!r}"
    return None


def neural_scores(name, data, days, device):
    path = CHECKPOINTS[name]
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    config = validate_config(dict(DEFAULT, **ckpt["configuration"]))
    model = make_model(data, config).to(device)
    model.load_state_dict(ckpt["state_dict"])
    scores = predict(model, data, days, config, device)[..., 0]
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return scores, sha256_file(path)


def rank_correlation(data, days, left, right):
    from scipy.stats import spearmanr

    listed = listing_eligible(data.dates, data.symbols)
    values = []
    for i, t in enumerate(days):
        candidate = data.candidate(int(t)) & listed[int(t)]
        if candidate.sum() < 3:
            continue
        a = np.asarray(left[i], np.float64)[candidate]
        b = np.asarray(right[i], np.float64)[candidate]
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            continue
        values.append(float(spearmanr(a, b).statistic))
    return values


def summarize(tag, rows_a, rows_b, overlap, rankcorr):
    diff = np.array([a["net_simple_return"] - b["net_simple_return"] for a, b in zip(rows_a, rows_b)], float)
    sigma_daily = float(diff.std(ddof=1))
    sigma_ann = sigma_daily * float(np.sqrt(252.0))
    mde80 = 2.487 * sigma_ann / float(np.sqrt(3.5))
    drag_a = float(np.mean([r["fee_fraction"] for r in rows_a]) * 252)
    drag_b = float(np.mean([r["fee_fraction"] for r in rows_b]) * 252)
    overlap_pct = np.array([len(a & b) / max(len(a), len(b), 1) for a, b in overlap if a or b], float)
    rank = np.asarray(rankcorr, float)
    return dict(
        tag=tag, n_days=len(diff), sigma_d_daily=sigma_daily, sigma_d_annualized=sigma_ann,
        mde80_pp=mde80 * 100, mean_daily_diff=float(diff.mean()),
        annualized_diff_pp=float(diff.mean() * 252 * 100),
        cost_drag_annualized_pp=dict(a=drag_a * 100, b=drag_b * 100, a_minus_b=(drag_a - drag_b) * 100),
        holdings_overlap=dict(n=len(overlap_pct), mean=float(overlap_pct.mean()) if len(overlap_pct) else None,
                              median=float(np.median(overlap_pct)) if len(overlap_pct) else None),
        rank_correlation=dict(n=len(rank), mean=float(rank.mean()) if len(rank) else None,
                              median=float(np.median(rank)) if len(rank) else None),
        dates=(rows_a[0]["date"], rows_a[-1]["date"]),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prelim", "final"), required=True)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = Data(W0W2, args.cache, 21, 0, DEFAULT["evaluation_end"])
    valid_days = data.days("valid")
    # Integrity assertion: signal days stop at last-2, so the terminal mark
    # (signal+2) is the last price date touched; it must stay inside the window.
    last_touched = int(valid_days[-1]) + 2
    assert str(data.dates[last_touched]) <= WINDOW_END, (str(data.dates[last_touched]), WINDOW_END)
    print(f"ASSERT no-price-after-window: last_signal={data.dates[int(valid_days[-1])]} "
          f"last_price_touched={data.dates[last_touched]} <= {WINDOW_END}", flush=True)
    feed = load_execution_feed(data, FEED)

    if args.mode == "prelim":
        train_days = data.days("train", training=True)
        coef, count = fit_ridge(data, train_days, data.targets[:, :, 0], data.loss_masks["train"])
        scores_a = predict_ridge(data, valid_days, coef)
        scores_b, ckpt_sha = neural_scores("H", data, valid_days, device)
        tag, meta = "prelim_ridge_h5_vs_H_seed1_default", dict(ridge_samples=count, h_checkpoint_sha256=ckpt_sha)
    else:
        scores_a, sha_a = neural_scores("C0", data, valid_days, device)
        scores_b, sha_b = neural_scores("G0", data, valid_days, device)
        tag, meta = "final_C0_vs_G0_seed1_tau0.50_default", dict(c0_checkpoint_sha256=sha_a, g0_checkpoint_sha256=sha_b)

    rows_a_plain = share_replay(data, valid_days, scores_a, feed, 5, 1)
    rows_b_plain = share_replay(data, valid_days, scores_b, feed, 5, 1)
    rows_a = replay_with_holdings(data, valid_days, scores_a, feed, 5, 1)
    rows_b = replay_with_holdings(data, valid_days, scores_b, feed, 5, 1)
    for name, plain, held in (("A", rows_a_plain, rows_a), ("B", rows_b_plain, rows_b)):
        mismatch = rows_match(plain, held)
        if mismatch:
            raise SystemExit(f"replay diagonal mismatch for {name}: {mismatch}")
    print("DIAGONAL replay_with_holdings == share_replay: PASS (A and B)", flush=True)
    if [r["date"] for r in rows_a] != [r["date"] for r in rows_b]:
        raise SystemExit("accounting dates differ between arms")

    overlap = [(a["holdings"], b["holdings"]) for a, b in zip(rows_a, rows_b)]
    rankcorr = rank_correlation(data, valid_days, scores_a, scores_b)
    record = summarize(tag, rows_a, rows_b, overlap, rankcorr)
    record.update(mode=args.mode, meta=meta,
                  rule="topk50 drop5 every1d (default)", window="valid 2019-2020",
                  sigma_d_definition="SD(daily paired net return diff, ddof=1) * sqrt(252)")
    out = args.out or (ROOT / "protocol" / "checks" / f"sigma_d_{args.mode}_2026-10-18.json")
    out.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k != "meta"}, indent=1), flush=True)
    print(f"WROTE {out}", flush=True)


if __name__ == "__main__":
    main()
