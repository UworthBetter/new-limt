"""Portable verbatim selection/replay functions from the frozen CPU baseline.
Only imports and path-independent constants differ; no training entry point.
"""
import numpy as np
from v2_w3_ledger import Account, execute_close
from v2_w4_training import topk_target
from v2_w6_policy import listing_eligible
from v2_w6_run import DEFAULT
EVAL_START = "2023-01-01"

RULES = (
    ("drop2-every1d", 2, 1),
    ("drop5-every1d", 5, 1),
    ("drop10-every1d", 10, 1),
    ("drop20-every1d", 20, 1),
    ("drop2-every5d", 2, 5),
    ("drop5-every5d", 5, 5),
    ("drop10-every5d", 10, 5),
    ("drop20-every5d", 20, 5),
)


COLUMNS = (
    "date",
    "signal_date",
    "gross_simple_return",
    "fee_fraction",
    "net_simple_return",
    "net_log_growth",
    "turnover",
    "fee_value",
    "benchmark_return",
)


def choose_rule(stats):
    """stats: list of (rule, annual_excess, mean_turnover). 0.5pp tie -> lower turnover."""
    best = max(row[1] for row in stats)
    pool = [row for row in stats if (best - row[1]) * 100 <= 0.5]
    order = {name: i for i, (name, _, _) in enumerate(RULES)}
    pool.sort(key=lambda row: (row[2], order[row[0]]))
    return pool[0][0]


def choose_arm(seed_grids):
    """One rule per arm. Average the selection metric and turnover across seeds first."""
    if not seed_grids:
        raise ValueError("no seed grids")
    names = [row["rule"] for row in seed_grids[0]]
    averaged = []
    for name in names:
        excess, turnover = [], []
        for grid in seed_grids:
            match = next(row for row in grid if row["rule"] == name)
            excess.append(match["annual_excess"])
            turnover.append(match["mean_turnover"])
        averaged.append((name, float(np.mean(excess)), float(np.mean(turnover))))
    return choose_rule(averaged)


def _with_benchmark(rows, bench):
    out = []
    for row in rows:
        item = dict(row)
        item["benchmark_return"] = "" if row["date"] not in bench["values"] else f"{bench['values'][row['date']]:.10g}"
        out.append(item)
    return out


def share_replay(data, days, scores, feed, drop, every):
    prices, states, multiplier, dividend = [
        feed[k] for k in ("raw_close", "state", "share_multiplier", "cash_dividend_per_old_share")
    ]
    listed = listing_eligible(data.dates, data.symbols)
    account = Account.all_cash(len(data.symbols), DEFAULT["capital"])
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
            buy=DEFAULT["buy_rate"],
            sell=DEFAULT["sell_rate"],
            min_fee=DEFAULT["min_fee"],
            lot=int(DEFAULT["lot"]),
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
                benchmark_return="",
            )
        )
    return rows


def annual_excess(rows, bench):
    dates = [row["date"] for row in rows if row["date"] < EVAL_START]
    if not dates:
        raise ValueError("no selection-window rows")
    missing = [day for day in dates if day not in bench["values"]]
    if missing:
        raise ValueError(f"benchmark missing {missing[0]}")
    net = np.array([row["net_simple_return"] for row in rows if row["date"] < EVAL_START], float)
    turnover = np.array([row["turnover"] for row in rows if row["date"] < EVAL_START], float)
    base = np.array([bench["values"][day] for day in dates], float)
    return float((net - base).mean() * 252), float(turnover.mean())
