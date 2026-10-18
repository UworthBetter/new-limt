"""Canonical W5 artifacts, independent IC, benchmark checks and visible cell status."""
import csv
import json
import math
from pathlib import Path
import re
import tempfile
import os
import numpy as np
from v2_w3_run import digest


def json_write(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def canonical(run, model, seed, rule, window):
    if any(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*", x) is None for x in [run, model, rule, window]):
        raise ValueError("identifiers may only contain letters, digits and hyphens")
    return f"v2_{run}__{model}__s{int(seed)}__{rule}__{window}"


def benchmark(path, expected="H00300"):
    path = Path(path)
    meta = json.loads(Path(str(path) + ".meta.json").read_text(encoding="utf-8"))
    if meta.get("benchmark_id") != expected or meta.get("return_kind") != "total_return" or meta.get("sha256") != digest(path):
        raise ValueError("benchmark identity/kind/hash mismatch")
    if not meta.get("source"):
        raise ValueError("benchmark source must be recorded")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    dates = np.array([row["date"] for row in rows], dtype="datetime64[D]")
    values = np.array([float(row["return"]) for row in rows])
    if len(dates) == 0 or np.isnat(dates).any() or not (np.diff(dates) > np.timedelta64(0, "D")).all() or not np.isfinite(values).all() or (values <= -1).any():
        raise ValueError("invalid benchmark rows")
    return dict(identity=expected, values={str(d): float(v) for d, v in zip(dates, values)}, provenance=meta)


def metrics(rows, bench=None):
    dates = np.array([x["date"] for x in rows], dtype="datetime64[D]")
    if not len(rows) or np.isnat(dates).any() or not (np.diff(dates) > np.timedelta64(0, "D")).all():
        raise ValueError("empty, duplicate or unordered result dates")
    required = ["net_return", "log_growth", "wealth", "fee_fraction", "fee_value", "turnover", "target_penalty", "max_weight"]
    values = np.array([[row[key] for key in required] for row in rows], float)
    if not np.isfinite(values).all() or (values[:, 0] <= -1).any() or (values[:, 2] <= 0).any():
        raise ValueError("invalid result numbers")
    net = values[:, 0]; wealth = values[:, 2]; logs = values[:, 1]
    if not np.allclose(np.cumprod(1 + net), wealth, atol=1e-9, rtol=1e-6) or not np.allclose(np.log1p(net), logs, atol=1e-8):
        raise ValueError("return/wealth/log identity mismatch")
    peak = np.maximum.accumulate(np.r_[1., wealth])
    result = dict(events=len(rows), start=str(dates[0]), end=str(dates[-1]),
        cagr=float(np.expm1(logs.sum() * 252 / len(rows))),
        account_max_drawdown=float((1 - np.r_[1., wealth] / peak).max()),
        net_mean_log=float(logs.mean()), mean_turnover=float(values[:, 5].mean()),
        mean_fee_fraction=float(values[:, 3].mean()), fees_initial_capital_units=float(values[:, 4].sum()),
        mean_target_penalty=float(values[:, 6].mean()), max_weight=float(values[:, 7].max()),
        stale_held_events=int(sum(row.get("stale_held", 0) > 0 for row in rows)),
        annualisation="252 accounting events/year", benchmark_status="NOT_REQUESTED_SMOKE" if bench is None else "PASS",
        arithmetic_annual_excess=None, excess_ir=None, cumulative_excess_max_drawdown=None)
    if bench is not None:
        missing = [str(d) for d in dates if str(d) not in bench["values"]]
        if missing:
            raise ValueError(f"benchmark missing {len(missing)} report dates; first {missing[0]}")
        b = np.array([bench["values"][str(d)] for d in dates]); excess = net - b
        deviation = excess.std(ddof=1) if len(excess) > 1 else 0
        cumulative = np.r_[0., np.cumsum(excess)]
        result.update(arithmetic_annual_excess=float(excess.mean() * 252),
            excess_ir=float(excess.mean() / deviation * np.sqrt(252)) if deviation > 0 else None,
            cumulative_excess_max_drawdown=float((np.maximum.accumulate(cumulative) - cumulative).max()),
            benchmark_id=bench["identity"])
    return result


def daily_ic(data, days, scores, split):
    records = []
    for i, t in enumerate(days):
        mask = data.label_mask[t] if split == "full-test" else data.loss_masks[split][t]
        if split == "full-test":
            from v2_w4_data import SPLITS
            if t + 6 >= len(data.dates) or data.dates[t + 6] > np.datetime64(getattr(data,'splits',SPLITS)["eval"][1]):
                mask = np.zeros_like(mask)
        if hasattr(data,'listing_eligible'):
            mask=mask & data.listing_eligible[t]
        mask = mask & np.isfinite(scores[i]) & np.isfinite(data.targets[t, :, 0])
        x, y = scores[i, mask], data.targets[t, mask, 0]
        ic = None
        if len(x) > 1 and x.std() > 0 and y.std() > 0:
            ic = float(np.corrcoef(x, y)[0, 1])
        records.append(dict(signal_date=str(data.dates[t]), samples=int(mask.sum()), ic=ic))
    valid = [x["ic"] for x in records if x["ic"] is not None]
    return dict(days=len(records), valid_days=len(valid), samples=sum(x["samples"] for x in records),
                mean_ic=float(np.mean(valid)) if valid else None, records=records)


def write_daily(path, rows, metadata, window, bench=None):
    path = Path(path)
    result = metrics(rows, bench)
    if result["start"] < window[0] or result["end"] > window[1]:
        raise ValueError("daily result outside declared window")
    required_meta = {"run_id", "model_id", "seed", "rule_id", "data_version", "config_hash", "checkpoint_hash",
        "objective_family", "alpha_aux", "gamma_to", "beta_risk", "fee_convention", "state_approximation", "execution_layer"}
    if required_meta - metadata.keys():
        raise ValueError(f"missing provenance fields: {required_meta - metadata.keys()}")
    if path.exists() or Path(str(path) + ".meta.json").exists():
        raise FileExistsError(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".daily-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        with open(temporary, encoding="utf-8", newline="") as stream:
            restored = list(csv.DictReader(stream))
        if len(restored) != len(rows) or [r["date"] for r in restored] != [r["date"] for r in rows]:
            raise ValueError("CSV readback mismatch")
        os.replace(temporary, path)
        json_write(str(path) + ".meta.json", dict(metadata, sha256=digest(path), rows=len(rows), metrics=result))
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()
    return result


def summary(out, cells, ic_records):
    """Keep every planned cell AND independent IC, even when some cells failed."""
    out = Path(out)
    for cell in cells:
        if cell["status"] == "success":
            path = out / cell["path"]
            try:
                meta = json.loads(Path(str(path) + ".meta.json").read_text(encoding="utf-8"))
                if digest(path) != meta["sha256"]:
                    raise ValueError("changed daily file")
            except Exception as error:
                cell.update(status="failed", error=str(error))
        elif cell["status"] in ("pending", "running"):
            cell.update(status="missing", error="producer did not finish")
    complete = bool(cells) and all(x["status"] == "success" for x in cells)
    json_write(out / "cell_status.json", cells)
    json_write(out / "ic_summary.json", ic_records)
    rows = []
    for cell in cells:
        row = {k: v for k, v in cell.items() if k != "metrics"}
        row.update(cell.get("metrics", {}) if cell["status"] == "success" else {})
        rows.append(row)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (out / "summary.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader(); writer.writerows(rows)
    lines = ["# W4–W6 运行汇总", "", f"整体状态：{'COMPLETE' if complete else 'PARTIAL/FAILED'}。", "",
             "| 模型 | seed | 窗口 | 规则 | 状态 | CAGR | 平均换手 | 错误 |", "|---|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in ["model_id", "seed", "window", "rule_id", "status", "cagr", "mean_turnover", "error"]) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return complete


def load_execution_feed(data, feed_path):
    if isinstance(feed_path, dict):
        arrays = feed_path
    else:
        with np.load(feed_path, allow_pickle=False) as feed:
            dates = feed["dates"].astype("datetime64[D]")
            syms = feed["symbols"].astype(str)
            if not np.array_equal(dates, data.dates) or not np.array_equal(syms, data.symbols):
                raise ValueError("execution feed requires exact calendar/symbol alignment")
            keys=["raw_close", "state", "share_multiplier", "cash_dividend_per_old_share"]
            keys += [k for k in ["is_st","market_state_known","limit_up","limit_down","no_limit"] if k in feed]
            arrays = {k: feed[k] for k in keys}
    shape = (len(data.dates), len(data.symbols))
    prices, states, multiplier, dividend = [arrays[k] for k in ["raw_close","state","share_multiplier","cash_dividend_per_old_share"]]
    if (any(x.shape != shape for x in arrays.values()) or not np.isin(states, [0, 1, 2, 3, 4]).all()
            or not np.isfinite(multiplier).all() or (multiplier <= 0).any()
            or not np.isfinite(dividend).all() or (dividend < 0).any()
            or np.isinf(prices).any() or ((np.isfinite(prices)) & (prices <= 0)).any()):
        raise ValueError("invalid execution/corporate-action arrays")
    market_keys={"is_st","market_state_known","limit_up","limit_down","no_limit"}
    present=market_keys & arrays.keys()
    if present and present != market_keys:
        raise ValueError('incomplete historical market state fields')
    if present:
        st,known,up,down,no_limit=[arrays[k] for k in ['is_st','market_state_known','limit_up','limit_down','no_limit']]
        if (not np.isin(st,[-1,0,1]).all() or not np.isin(known,[0,1]).all() or not np.isin(no_limit,[0,1]).all()
                or not np.array_equal(known.astype(bool),states!=4)):
            raise ValueError('invalid historical ST/known state flags')
        tradable=np.isin(states,[0,2,3])
        restricted=tradable & ~no_limit.astype(bool)
        if ((tradable & ((st<0) | ~np.isfinite(prices)))
                | (restricted & (~np.isfinite(up) | ~np.isfinite(down) | (down<=0) | (up<=down)
                   | (prices>up+1e-5) | (prices<down-1e-5)))
                | ((states==0) & restricted & (np.isclose(prices,up,atol=1e-5,rtol=0) | np.isclose(prices,down,atol=1e-5,rtol=0)))).any():
            raise ValueError('inconsistent tradable market state, price or historical bands')
        for code,band in [(2,up),(3,down)]:
            hit=states==code
            if (hit & (no_limit.astype(bool) | ~np.isfinite(band) | ~np.isclose(prices,band,atol=1e-5,rtol=0))).any():
                raise ValueError('limit state does not match its raw executable price/band')
    return arrays


def actual_execution(data, days, scores, feed_path, config):
    """Share execution for pre-aligned raw quotes + explicit corporate-action feed.

    Required NPZ fields: dates, symbols, raw_close, state, share_multiplier,
    cash_dividend_per_old_share. No corporate actions are inferred from adjusted
    prices. Dividends are applied to old shares before share multiplication.
    """
    from v2_w3_ledger import Account, execute_close
    import torch
    from v2_w4_training import topk_target
    from v2_w6_policy import listing_eligible
    arrays = load_execution_feed(data, feed_path)
    if config.get('mode')=='formal' and 'is_st' not in arrays:
        raise ValueError('formal execution requires historical suspension/limit/ST fields')
    st_policy=config.get('st_policy','limits_only')
    if st_policy not in ('limits_only','exclude_signal'):
        raise ValueError('invalid ST policy')
    if st_policy=='exclude_signal' and 'is_st' not in arrays:
        raise ValueError('signal-day ST exclusion requires historical is_st')
    prices, states, multiplier, dividend = [arrays[k] for k in ["raw_close", "state", "share_multiplier", "cash_dividend_per_old_share"]]
    listed=listing_eligible(data.dates,data.symbols)
    account = Account.all_cash(len(data.symbols), config["capital"])
    selected, rows = set(), []
    for i in range(len(days) + 1):
        terminal = i == len(days)
        t = int(days[-1]) + 2 if terminal else int(days[i]) + 1
        original_value = account.value()
        action_cash = float(np.dot(account.shares, dividend[t]))
        account = Account(account.cash + action_cash,
                          account.shares * multiplier[t], account.marks / multiplier[t], account.date)
        if terminal:
            target = None
        else:
            candidate=data.candidate(days[i]).copy()
            candidate &= listed[days[i]]
            if st_policy=='exclude_signal':
                candidate &= arrays['is_st'][days[i]]==0
            target, selected = topk_target(scores[i], candidate,
                set(np.where(account.shares > 0)[0]), config["topk"], config["drop"])
        # Delisted holdings remain frozen at the last mark. Never synthesize a
        # sale, successor shares, or liquidation cash from an eligibility change.
        quotes=np.where(listed[t],prices[t],np.nan)
        effective_states=np.where(listed[t],states[t],1)
        valuation = np.where(np.isfinite(quotes) & (quotes > 0), quotes, account.marks)
        stock_values = account.shares * np.where(np.isfinite(valuation), valuation, 0.)
        before_wealth = account.cash + stock_values.sum()
        penalty = 0. if target is None else float(np.abs(target[1:] - stock_values / before_wealth).sum())
        account, event = execute_close(account, quotes, effective_states, target, str(data.dates[t - 1]), str(data.dates[t]),
            buy=config["buy_rate"], sell=config["sell_rate"], min_fee=config["min_fee"], lot=config["lot"])
        # Account.value immediately after dividend is NOT the prior day's NAV.
        net = account.value() / original_value - 1
        w = account.shares * np.nan_to_num(account.marks) / account.value()
        rows.append(dict(signal_date=event["signal_date"], date=event["execution_date"],
            event="terminal_mark" if terminal else "rebalance", net_return=net, log_growth=math.log1p(net),
            wealth=account.value() / config["capital"], fee_fraction=event["fees"] / event["wealth_before_trade"],
            fee_value=event["fees"] / config["capital"], turnover=event["turnover"], target_penalty=penalty,
            max_weight=float(w.max()), effective_holdings=float(1 / np.square(w).sum()) if np.square(w).sum() else 0.,
            stale_held=event["stale_held"], cash=event["cash"] / config["capital"],
            corporate_action_cash=action_cash / config["capital"],
            st_policy=st_policy,st_assets_at_execution=int((arrays['is_st'][t]==1).sum()) if 'is_st' in arrays else 0,
            blocked_buy_assets=event['blocked_buy_assets'],blocked_sell_assets=event['blocked_sell_assets'],
            suspended_assets=int((states[t]==1).sum()),limit_up_assets=int((states[t]==2).sum()),
            limit_down_assets=int((states[t]==3).sum()),unknown_state_assets=int((states[t]==4).sum()),
            frozen_delisted_held=int(((~listed[t]) & (account.shares>0)).sum()),
            frozen_delisted_value=float((account.shares[~listed[t]]*np.nan_to_num(account.marks[~listed[t]])).sum()/account.value()),
            executed_orders=json.dumps(event["orders"], ensure_ascii=False)))
    return rows
