"""CPU baselines on the clean panel: ridge, LightGBM regression, LambdaRank.

Both labels, all eight Section 5 rules, full-friction share ledger.
2021-2022 rows stay readable. Any accounting date on or after 2023-01-01
is written under runs/_sealed and is not summarized.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

LIMT_SRC = Path(r"D:\research\limt-repro\src")
sys.path.insert(0, str(LIMT_SRC))

from v2_w3_ledger import Account, execute_close  # noqa: E402
from v2_w4_data import Data  # noqa: E402
from v2_w4_training import topk_target  # noqa: E402
from v2_w5_results import benchmark, load_execution_feed  # noqa: E402
from v2_w6_policy import listing_eligible  # noqa: E402
from v2_w6_run import DEFAULT  # noqa: E402

ROOT = Path(r"D:\research\new_LiMT")
FEED = Path(r"D:\research\limt-repro\data\v2_formal_inputs_run3\new_execution_feed.npz")
BENCH = Path(r"D:\research\limt-repro\data\v2_formal_inputs_run3\new_h00300_returns.csv")
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
LGB_PARAMS = dict(
    colsample_bytree=0.8879,
    learning_rate=0.2,
    subsample=0.8789,
    lambda_l1=205.6999,
    lambda_l2=580.9768,
    max_depth=8,
    num_leaves=210,
    num_threads=8,
    verbosity=-1,
    deterministic=True,
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


class GateLocked(RuntimeError):
    pass


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


def _check_choice():
    chosen = choose_rule(
        [
            ("drop20-every1d", 0.10, 0.50),
            ("drop2-every5d", 0.096, 0.10),
            ("drop5-every1d", 0.10, 0.40),
        ]
    )
    if chosen != "drop2-every5d":
        raise AssertionError(chosen)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _with_benchmark(rows, bench):
    out = []
    for row in rows:
        item = dict(row)
        item["benchmark_return"] = "" if row["date"] not in bench["values"] else f"{bench['values'][row['date']]:.10g}"
        out.append(item)
    return out


def store_rows(market, backbone, label, rule, seed, selection_rows, eval_rows, bench):
    """Selection rows before 2023 stay readable. Evaluation rows are sealed and not returned."""
    open_rows, tail = [], []
    for row in _with_benchmark(selection_rows, bench):
        (tail if row["date"] >= EVAL_START else open_rows).append(row)
    if not open_rows:
        raise ValueError("selection replay produced no pre-2023 rows")
    write_csv(ROOT / "runs" / market / backbone / label / rule / f"seed{seed}" / "daily.csv", open_rows)
    if tail:
        write_csv(ROOT / "runs" / "_sealed" / market / backbone / label / rule / f"seed{seed}" / "selection_tail.csv", tail)
    sealed = _with_benchmark(eval_rows, bench)
    if any(row["date"] < EVAL_START for row in sealed):
        raise GateLocked("evaluation replay produced a pre-2023 accounting date")
    write_csv(ROOT / "runs" / "_sealed" / market / backbone / label / rule / f"seed{seed}" / "daily.csv", sealed)
    return open_rows


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


def samples(data, days, values, mask):
    xs, ys, groups = [], [], []
    for t in days:
        keep = mask[t]
        if not np.any(keep):
            continue
        x = np.asarray(data.features[t][keep], np.float64)
        y = np.asarray(values[t][keep], np.float64)
        ok = np.isfinite(x).all(axis=1) & np.isfinite(y)
        if not ok.any():
            continue
        xs.append(x[ok])
        ys.append(y[ok])
        groups.append(int(ok.sum()))
    if not xs:
        raise ValueError("empty fit")
    return np.vstack(xs), np.concatenate(ys), np.asarray(groups, np.int32)


def r1_days(data, labels, start, end):
    dates = data.dates
    exit_idx = labels["exit_idx_r1"].astype(int)
    inside = (dates >= np.datetime64(start)) & (dates <= np.datetime64(end))
    exit_date = dates[np.clip(exit_idx, 0, len(dates) - 1)]
    ok = inside & (exit_idx < len(dates)) & (exit_date <= np.datetime64(end))
    ok &= np.arange(len(dates)) >= data.lookback - 1
    return np.where(ok)[0]


def predict_ridge(data, days, coef):
    out = np.zeros((len(days), len(data.symbols)), np.float64)
    for i, t in enumerate(days):
        x = np.asarray(data.features[t], np.float64)
        score = np.c_[np.ones(len(x)), x] @ coef
        eligible = data.candidate(t)
        if not np.isfinite(score[eligible]).all():
            raise ValueError(f"nonfinite ridge score on {data.dates[t]}")
        out[i] = score
    return out


def predict_lgb(data, days, booster, iteration):
    out = np.zeros((len(days), len(data.symbols)), np.float64)
    for i, t in enumerate(days):
        x = np.asarray(data.features[t], np.float32)
        score = np.full(len(x), np.nan)
        ok = np.isfinite(x).all(axis=1)
        if ok.any():
            score[ok] = booster.predict(x[ok], num_iteration=iteration)
        eligible = data.candidate(t)
        if not np.isfinite(score[eligible]).all():
            raise ValueError(f"nonfinite model score on {data.dates[t]}")
        out[i] = score
    return out


def fit_ridge(data, days, values, mask):
    width = len(data.feature_names) + 1
    gram, rhs, count = np.zeros((width, width)), np.zeros(width), 0
    for t in days:
        keep = mask[t]
        if not np.any(keep):
            continue
        x = np.asarray(data.features[t][keep], np.float64)
        y = np.asarray(values[t][keep], np.float64)
        ok = np.isfinite(x).all(axis=1) & np.isfinite(y)
        if not ok.any():
            continue
        x, y = x[ok], y[ok]
        x = np.c_[np.ones(len(x)), x]
        gram += x.T @ x
        rhs += x.T @ y
        count += len(y)
    if count == 0:
        raise ValueError("empty ridge fit")
    penalty = np.eye(width) * DEFAULT["ridge_alpha"]
    penalty[0, 0] = 0
    return np.linalg.solve(gram + penalty, rhs), count


def ranks(y):
    order = np.argsort(y, kind="mergesort")
    label = np.empty(len(y), np.int32)
    label[order] = np.arange(len(y), dtype=np.int32)
    return label


def fit_lgb(x, y, groups, x_valid, y_valid, groups_valid, seed, rank):
    import lightgbm as lgb

    params = dict(LGB_PARAMS, seed=seed, bagging_seed=seed, feature_fraction_seed=seed, data_random_seed=seed)
    train_y, valid_y = y, y_valid
    if rank:
        params.update(objective="lambdarank", metric="ndcg", label_gain=list(range(int(max(groups.max(), groups_valid.max())))))
        parts, cursor = [], 0
        for size in groups:
            parts.append(ranks(y[cursor : cursor + size]))
            cursor += size
        train_y = np.concatenate(parts)
        parts, cursor = [], 0
        for size in groups_valid:
            parts.append(ranks(y_valid[cursor : cursor + size]))
            cursor += size
        valid_y = np.concatenate(parts)
    else:
        params.update(objective="regression", metric="l2")
    train = lgb.Dataset(x, train_y, group=groups if rank else None)
    valid = lgb.Dataset(x_valid, valid_y, group=groups_valid if rank else None, reference=train)
    booster = lgb.train(
        params,
        train,
        num_boost_round=1000,
        valid_sets=[valid],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    return booster, int(booster.best_iteration or booster.current_iteration())


def run_cell(data, feed, bench, market, backbone, label, seed, scores, predict_days):
    stats = []
    for name, drop, every in RULES:
        rows = share_replay(data, predict_days["rule_select"], scores["rule_select"], feed, drop, every)
        sealed = share_replay(data, predict_days["eval"], scores["eval"], feed, drop, every)
        readable = store_rows(market, backbone, label, name, seed, rows, sealed, bench)
        excess, turnover = annual_excess(readable, bench)
        stats.append((name, excess, turnover))
        print(f"rule {backbone} {label} seed{seed} {name} selected_on=2021-2022", flush=True)
        del sealed
    chosen = choose_rule(stats)
    return chosen, [{"rule": name, "annual_excess": excess, "mean_turnover": turnover} for name, excess, turnover in stats]


def main():
    _check_choice()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--market", default="csi300")
    args = parser.parse_args()
    if args.market != "csi300":
        raise SystemExit("CSI500 has no execution feed aligned to its panel; refusing a continuous-weight substitute")
    root = Path(args.root)
    cache = ROOT / "runs" / "csi300_cpu" / "cache"
    if cache.exists():
        raise SystemExit(f"cache already exists: {cache}")
    started = time.perf_counter()
    data = Data(root, cache, 21, 0, "2026-06-30")
    feed = load_execution_feed(data, FEED)
    bench = benchmark(BENCH)
    with np.load(root / "data" / "v2" / "new_v2_labels_csi300.npz", allow_pickle=False) as labels:
        z_r1 = labels["z_r1"]
        mask_r1 = labels["label_mask_r1"]
        r1_train = r1_days(data, labels, "2008-01-01", "2018-12-31")
        r1_valid = r1_days(data, labels, "2019-01-01", "2020-12-31")
    label_spec = {
        "h5": (
            data.targets[:, :, 0],
            data.loss_masks["train"],
            data.days("train", training=True),
            data.days("valid"),
            data.loss_masks["valid"],
        ),
        "r1": (z_r1, mask_r1, r1_train, r1_valid, mask_r1),
    }
    predict_days = {"rule_select": data.days("rule_select"), "eval": data.days("eval")}
    import lightgbm  # noqa: F401

    out_root = ROOT / "runs" / "csi300_cpu" / "models"
    out_root.mkdir(parents=True, exist_ok=True)
    summary = []
    for label, (values, mask, train_days, valid_days, valid_mask) in label_spec.items():
        coef, count = fit_ridge(data, train_days, values, mask)
        print(f"ridge {label} samples={count} fit_seconds={time.perf_counter() - started:.1f}", flush=True)
        scores = {
            "rule_select": predict_ridge(data, predict_days["rule_select"], coef),
            "eval": predict_ridge(data, predict_days["eval"], coef),
        }
        chosen, grid = run_cell(data, feed, bench, "csi300", "ridge", label, 1, scores, predict_days)
        summary.append(dict(backbone="ridge", label=label, seed=1, samples=count, grid=grid, chosen_rule=chosen))
        (out_root / f"ridge_{label}_grid.json").write_text(json.dumps(grid), encoding="utf-8")

        x, y, groups = samples(data, train_days, values, mask)
        xv, yv, gv = samples(data, valid_days, values, valid_mask)
        for rank, backbone in ((False, "lgbm-reg"), (True, "lgbm-rank")):
            for seed in (1, 2, 3):
                t0 = time.perf_counter()
                booster, iteration = fit_lgb(x, y, groups, xv, yv, gv, seed, rank)
                print(f"{backbone} {label} seed{seed} iteration={iteration} seconds={time.perf_counter() - t0:.1f}", flush=True)
                model_dir = out_root / backbone / label / f"seed{seed}"
                model_dir.mkdir(parents=True, exist_ok=True)
                booster.save_model(str(model_dir / "model.txt"))
                scores = {
                    "rule_select": predict_lgb(data, predict_days["rule_select"], booster, iteration),
                    "eval": predict_lgb(data, predict_days["eval"], booster, iteration),
                }
                chosen, grid = run_cell(data, feed, bench, "csi300", backbone, label, seed, scores, predict_days)
                summary.append(dict(backbone=backbone, label=label, seed=seed, best_iteration=iteration, grid=grid))
                (model_dir / "grid.json").write_text(json.dumps(grid), encoding="utf-8")
    arms = []
    for label in label_spec:
        ridge_grid = next(row["grid"] for row in summary if row["backbone"] == "ridge" and row["label"] == label)
        arms.append(dict(backbone="ridge", label=label, seeds=[1], chosen_rule=choose_arm([ridge_grid])))
        for backbone in ("lgbm-reg", "lgbm-rank"):
            grids = [row["grid"] for row in summary if row["backbone"] == backbone and row["label"] == label]
            arms.append(dict(backbone=backbone, label=label, seeds=[1, 2, 3], chosen_rule=choose_arm(grids)))
    record = dict(market="csi300", seconds=time.perf_counter() - started, arms=arms,
                  cells=[{k: v for k, v in row.items() if k != "grid"} for row in summary],
                  evaluation_summarized=False, selection="seed-averaged annualized excess, then 0.5pp turnover tie-break")
    (ROOT / "runs" / "csi300_cpu" / "selection.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
