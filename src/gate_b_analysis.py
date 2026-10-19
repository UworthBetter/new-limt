"""FROZEN analysis code for the Gate B unsealing (CSI300).

Commit this file BEFORE any 2023+ output is opened; its sha256 goes into
gate_b.json (analysis_sha256). Scope: the neural primary arms (C0/G0, seeds
1-3, tau 0.5, selected rule = default rule drop5-every1d). It computes:

  - the primary estimand Delta: seed-averaged annualized paired difference of
    after-cost excess returns (C0 - G0), stationary-bootstrap one-sided 5%
    lower bound and 90% two-sided CI (TOST margin 2.0 pp);
  - Newey-West (lag 5) robustness interval on the same paired series;
  - SC1 soft-layer Delta (bootstrap); SC2 Delta in annualized IR (paired
    bootstrap, shared resample indices); SC5 cross-layer tracking correlation
    (Spearman between soft-layer and executed-layer daily excess, per arm);
    SC6 Delta_cost (bootstrap on the paired fee-fraction series); SC7
    Delta_gross (bootstrap) plus the 5-day IC pair;
  - Holm correction over the secondary family present in this pass
    [SC1, SC2, SC5, SC6, SC7]; SC3 coincides with the primary (not
    double-counted); SC4 enters the family when CSI500 lands (Gate C);
  - the mechanical Section 7 S1-condition table (CSI500 condition marked
    pending; the scenario call itself belongs to Gate C);
  - the stamp-duty sensitivity: books replayed with sell fee 10 bp from
    2023-08-28 (HKEX CT/137/23), reported alongside, never decisive.

Inference constants (frozen): stationary bootstrap mean block length 10,
10,000 resamples, rng seed 20261019, one-sided 5%; NW lag 5; TOST 2.0 pp.

--self-test exercises the statistics layer on synthetic data with known
answers. --run requires protocol/gate_b.json whose analysis_sha256 matches
this file, asserts the replay diagonal (fee-variant == share_replay at equal
rates) on the validation window, and only then touches evaluation data.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

LIMT_SRC = Path(r"D:\research\limt-repro\src")
SRC = Path(__file__).resolve().parent
for path in (str(LIMT_SRC), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from cpu_baselines import DEFAULT as CB_DEFAULT  # noqa: E402
from cpu_baselines import benchmark as load_bench  # noqa: E402
from cpu_baselines import share_replay  # noqa: E402
from v2_w4_data import Data  # noqa: E402
from v2_w4_training import make_model, predict, portfolio_events, topk_target  # noqa: E402
from v2_w3_ledger import Account, execute_close  # noqa: E402
from v2_w6_policy import listing_eligible  # noqa: E402
from v2_w6_run import DEFAULT, validate_config  # noqa: E402
from v2_w5_results import load_execution_feed  # noqa: E402

ROOT = Path(r"D:\research\new_LiMT")
W0W2 = ROOT / "data" / "csi300_w1w2"
FEED = Path(r"D:\research\limt-repro\data\v2_formal_inputs_run3\new_execution_feed.npz")
BENCH_CSV = Path(r"D:\research\limt-repro\data\v2_formal_inputs_run3\new_h00300_returns.csv")
RUNS = ROOT / "runs"

TAU = 0.5
BLOCK_MEAN = 10.0
N_BOOT = 10000
RNG_SEED = 20261019
NW_LAG = 5
TOST_MARGIN_PP = 2.0
RULE = (5, 1)  # drop5-every1d: the selected rule of BOTH primary arms (= default)
STAMP_CUTOFF = "2023-08-28"
SEEDS = (1, 2, 3)
HOLM_KEYS = ("SC1", "SC2", "SC5", "SC6", "SC7")  # SC3 == primary; SC4 pending CSI500


def checkpoint(group, seed):
    base = "csi300_g0c0_seed1_tau_20261017" if seed == 1 else f"csi300_g0c0_seed{seed}_tau050_20261018"
    return RUNS / base / group / "tau0.50" / f"seed{seed}" / "checkpoint.pt"


def sha256_file(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


# ---------------------------------------------------------------- statistics

def bootstrap_indices(n, block_mean, n_boot, seed):
    """Stationary-bootstrap index replicates (Politis-Romano)."""
    rng = np.random.default_rng(seed)
    p = 1.0 / block_mean
    out = np.empty((n_boot, n), dtype=np.int64)
    for b in range(n_boot):
        i = int(rng.integers(n))
        row = out[b]
        for t in range(n):
            row[t] = i
            i = int(rng.integers(n)) if rng.random() < p else (i + 1) % n
    return out


def infer(series, n_boot=N_BOOT, seed=RNG_SEED, block_mean=BLOCK_MEAN):
    x = np.asarray(series, float)
    idx = bootstrap_indices(len(x), block_mean, n_boot, seed)
    means = x[idx].mean(axis=1)
    return dict(mean=float(x.mean()),
                lower_5=float(np.percentile(means, 5)),
                ci90=[float(np.percentile(means, 5)), float(np.percentile(means, 95))],
                p_one_sided_pos=float((means <= 0).mean()))


def paired_apply(a, b, fn, n_boot=N_BOOT, seed=RNG_SEED, block_mean=BLOCK_MEAN):
    """Bootstrap a function of two same-length series with SHARED resample indices."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx = bootstrap_indices(len(a), block_mean, n_boot, seed)
    values = np.array([fn(a[row], b[row]) for row in idx])
    return values


def newey_west(series, lag=NW_LAG):
    x = np.asarray(series, float)
    n = len(x)
    m = x.mean()
    e = x - m
    var = float((e @ e) / n)
    for l in range(1, lag + 1):
        var += 2.0 * (1.0 - l / (lag + 1.0)) * float((e[l:] @ e[:-l]) / n)
    se_daily = float(np.sqrt(max(var, 0.0) / n))
    t = float(m / se_daily) if se_daily > 0 else float(np.inf)
    return dict(annualized_mean_pp=float(m * 252 * 100), se_annualized_pp=float(se_daily * np.sqrt(252) * 100),
                t_stat=t, ci95_pp=[float((m - 1.96 * se_daily) * 252 * 100),
                                   float((m + 1.96 * se_daily) * 252 * 100)])


def tost(ci90, margin_pp=TOST_MARGIN_PP):
    return dict(margin_pp=margin_pp, ci90=ci90,
                equivalent=bool(ci90[0] > -margin_pp and ci90[1] < margin_pp))


def conclusion(lower5_pp, ci90_pp, tost_result):
    """Four-way reading of Delta (per the external review's decision table).

    superior: one-sided 5% lower bound > 0 (TOST may simultaneously hold; the
    efficacy direction takes precedence and both flags are reported).
    inferior: the 90% CI lies entirely below 0.
    equivalent: TOST passes and the result is not superior.
    uncertain: everything else.
    """
    superior = bool(lower5_pp > 0)
    inferior = bool(ci90_pp[1] < 0)
    equivalent = bool(tost_result["equivalent"] and not superior)
    verdict = "superior" if superior else ("inferior" if inferior else ("equivalent" if equivalent else "uncertain"))
    return dict(verdict=verdict, flags=dict(superior=superior, inferior=inferior,
                                            tost_equivalent=tost_result["equivalent"]),
                precedence="efficacy direction outranks equivalence; co-occurrence reported via flags")


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def s_mapping(delta_pp, lower5_pp, per_seed_positive, ic_ratio, csi500_delta_pp=None):
    return {
        "S1_1_delta_ge_2pp": bool(delta_pp >= 2.0),
        "S1_1_lower_bound_pos": bool(lower5_pp > 0),
        "S1_2_3of3_seeds_positive": bool(sum(per_seed_positive) >= 3),
        "S1_3_ic_within_5pct": bool(ic_ratio >= 0.95),
        "S1_4_csi500": (None if csi500_delta_pp is None else bool(csi500_delta_pp >= 0)),
        "_note": "tier P1: 'demonstrated' wording requires the lower bound > 0; scenario call at Gate C",
    }


def ir(x):
    return float(x.mean() / (x.std() + 1e-12) * np.sqrt(252))


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


# --------------------------------------------------------------- book replay

def share_replay_fee(data, days, scores, feed, drop, every, sell_before, sell_after, cutoff, buy=0.0005):
    """cpu_baselines.share_replay with a date-dependent sell fee.

    With sell_before == sell_after this must reproduce share_replay row for
    row; the runner asserts that diagonal on the validation window before any
    evaluation data is touched.
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
        sell = sell_before if str(data.dates[t]) < cutoff else sell_after
        original = account.value()
        action_cash = float(np.dot(account.shares, dividend[t]))
        account = Account(account.cash + action_cash, account.shares * multiplier[t],
                          account.marks / multiplier[t], account.date)
        if terminal or i % every != 0:
            target = None
        else:
            candidate = data.candidate(int(days[i])).copy()
            candidate &= listed[int(days[i])]
            target, selected = topk_target(scores[i], candidate, set(np.where(account.shares > 0)[0]), 50, drop)
        quotes = np.where(listed[t], prices[t], np.nan)
        effective = np.where(listed[t], states[t], 1)
        account, event = execute_close(
            account, quotes, effective, target, str(data.dates[t - 1]), str(data.dates[t]),
            buy=buy, sell=sell, min_fee=CB_DEFAULT["min_fee"], lot=int(CB_DEFAULT["lot"]))
        rows.append(dict(
            date=event["execution_date"], signal_date=event["signal_date"],
            net=event["wealth"] / original - 1,
            fee_fraction=event["fees"] / event["wealth_before_trade"],
            turnover=event["turnover"]))
    return rows


def replay_diagonal(data, days, scores, feed):
    plain = share_replay(data, days, scores, feed, RULE[0], RULE[1])
    mine = share_replay_fee(data, days, scores, feed, RULE[0], RULE[1],
                            CB_DEFAULT["sell_rate"], CB_DEFAULT["sell_rate"], "9999-01-01")
    for a, b in zip(plain, mine):
        for key in ("date", "net_simple_return", "fee_fraction", "turnover"):
            left, right = a[key], b["net" if key == "net_simple_return" else key]
            if left != right:
                raise SystemExit(f"replay diagonal mismatch at {a['date']} {key}: {left} != {right}")
    return "PASS"


# ------------------------------------------------------------------- runner

def neural_scores(group, seed, data, days, device):
    import torch
    ckpt = torch.load(checkpoint(group, seed), map_location="cpu", weights_only=True)
    config = validate_config(dict(DEFAULT, **ckpt["configuration"]))
    model = make_model(data, config).to(device)
    model.load_state_dict(ckpt["state_dict"])
    scores = predict(model, data, days, config, device)[..., 0]
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return scores


def rows_to_series(rows, bench_values):
    net = np.array([r["net"] for r in rows], float)
    fee = np.array([r["fee_fraction"] for r in rows], float)
    dates = [r["date"] for r in rows]
    excess = net - np.array([bench_values[d] for d in dates], float)
    return net, fee, excess


def eval_ic(data, days, scores, split="eval"):
    """Native-label IC over the split's LOSS mask.

    The loss mask is essential: it is narrowed in Data.__init__ so that a
    label's exit index stays inside the split -- without it the h5 exit of the
    last four eval signal days would read prices past 2026-06-30.
    """
    listed = listing_eligible(data.dates, data.symbols)
    values = []
    for i, t in enumerate(days):
        keep = data.loss_masks[split][t] & data.candidate(t) & listed[t] \
            & np.isfinite(scores[i]) & np.isfinite(data.targets[t, :, 0])
        if keep.sum() > 3:
            x, y = scores[i][keep], data.targets[t, keep, 0]
            if x.std() > 0 and y.std() > 0:
                values.append(float(np.corrcoef(x, y)[0, 1]))
    return float(np.mean(values)) if values else None


def sc5_pvalue(paired_series, n_boot=N_BOOT, seed=RNG_SEED, block_mean=BLOCK_MEAN):
    """Two-sided stationary-bootstrap p for the soft-vs-executed Spearman
    correlation, computed on the seed-averaged paired daily series. The C0 arm
    is the pre-specified member of the Holm family; both arms are reported."""
    x, y = np.asarray(paired_series[0], float), np.asarray(paired_series[1], float)
    observed = spearman(x, y)
    idx = bootstrap_indices(len(x), block_mean, n_boot, seed)
    cors = np.array([spearman(x[row], y[row]) for row in idx])
    p = 2.0 * min(float((cors <= 0).mean()), float((cors >= 0).mean()))
    return dict(observed=observed, p_two_sided=min(p, 1.0))


def run(args):
    gate_b = json.loads((ROOT / "protocol" / "gate_b.json").read_text(encoding="utf-8"))
    own = sha256_file(Path(__file__))
    if gate_b.get("analysis_sha256") != own:
        raise SystemExit("analysis_sha256 in gate_b.json does not match this file; refusing to run")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = Data(W0W2, args.cache, 21, 0, DEFAULT["evaluation_end"])
    feed = load_execution_feed(data, FEED)
    bench_values = load_bench(BENCH_CSV)["values"]

    # Pre-unseal guard: the fee-variant replay must equal the production replay.
    valid_days = data.days("valid")
    probe = neural_scores("G0", 1, data, valid_days, device)
    print("REPLAY DIAGONAL", replay_diagonal(data, valid_days, probe, feed), flush=True)

    eval_days = data.days("eval")
    per_seed = {}
    for seed in SEEDS:
        c0 = neural_scores("C0", seed, data, eval_days, device)
        g0 = neural_scores("G0", seed, data, eval_days, device)
        c_rows = share_replay_fee(data, eval_days, c0, feed, RULE[0], RULE[1],
                                  CB_DEFAULT["sell_rate"], CB_DEFAULT["sell_rate"], "9999-01-01")
        g_rows = share_replay_fee(data, eval_days, g0, feed, RULE[0], RULE[1],
                                  CB_DEFAULT["sell_rate"], CB_DEFAULT["sell_rate"], "9999-01-01")
        c_net, c_fee, c_ex = rows_to_series(c_rows, bench_values)
        g_net, g_fee, g_ex = rows_to_series(g_rows, bench_values)
        c_soft = portfolio_events(data, eval_days, c0, TAU, CB_DEFAULT["buy_rate"], CB_DEFAULT["sell_rate"], "soft")
        g_soft = portfolio_events(data, eval_days, g0, TAU, CB_DEFAULT["buy_rate"], CB_DEFAULT["sell_rate"], "soft")
        c_soft_ex = np.array([r["net_return"] - bench_values.get(r["date"], np.nan) for r in c_soft])
        g_soft_ex = np.array([r["net_return"] - bench_values.get(r["date"], np.nan) for r in g_soft])
        keep = np.isfinite(c_soft_ex) & np.isfinite(g_soft_ex)
        s_rows = share_replay_fee(data, eval_days, c0, feed, RULE[0], RULE[1], CB_DEFAULT["sell_rate"], 0.0010, STAMP_CUTOFF)
        t_rows = share_replay_fee(data, eval_days, g0, feed, RULE[0], RULE[1], CB_DEFAULT["sell_rate"], 0.0010, STAMP_CUTOFF)
        per_seed[seed] = dict(
            delta=c_ex - g_ex,
            gross=(c_net + c_fee) - (g_net + g_fee),
            drag=c_fee - g_fee,
            soft_delta=(c_soft_ex - g_soft_ex)[keep],
            pair_soft_c=c_soft_ex[keep], pair_exec_c=c_ex[keep],
            pair_soft_g=g_soft_ex[keep], pair_exec_g=g_ex[keep],
            ir_delta=ir(c_soft_ex[keep]) - ir(g_soft_ex[keep]),
            layer_corr_c0=spearman(c_soft_ex[keep], c_ex[keep]),
            layer_corr_g0=spearman(g_soft_ex[keep], g_ex[keep]),
            stamp=rows_to_series(s_rows, bench_values)[2] - rows_to_series(t_rows, bench_values)[2],
            ic_c0=eval_ic(data, eval_days, c0), ic_g0=eval_ic(data, eval_days, g0),
        )
        print(f"seed {seed}: delta_pp={per_seed[seed]['delta'].mean() * 252 * 100:.3f} "
              f"ic_c0={per_seed[seed]['ic_c0']:.4f} ic_g0={per_seed[seed]['ic_g0']:.4f}", flush=True)

    avg = lambda key: np.mean([per_seed[s][key] for s in SEEDS], axis=0)
    primary_series, gross_series, drag_series, soft_series, stamp_series = (
        avg("delta"), avg("gross"), avg("drag"), avg("soft_delta"), avg("stamp"))
    primary = infer(primary_series)
    primary_out = dict(estimate_pp=primary["mean"] * 252 * 100,
                       lower5_pp=primary["lower_5"] * 252 * 100,
                       ci90_pp=[v * 252 * 100 for v in primary["ci90"]],
                       p_one_sided=primary["p_one_sided_pos"])
    # SC2 has no daily-series bootstrap (IR is a ratio of moments); it is reported
    # with per-seed values and excluded from the Holm pass, which covers SC1/SC5/SC6/SC7.
    sc1 = infer(soft_series)
    sc6 = infer(drag_series)
    sc7g = infer(gross_series)
    ic_c0 = float(np.mean([per_seed[s]["ic_c0"] for s in SEEDS]))
    ic_g0 = float(np.mean([per_seed[s]["ic_g0"] for s in SEEDS]))
    # SC5 Holm member (pre-specified): the C0 arm's cross-layer correlation,
    # bootstrap p on the seed-averaged paired daily series; both arms reported.
    sc5_test = sc5_pvalue((np.mean([per_seed[s]["pair_soft_c"] for s in SEEDS], axis=0),
                           np.mean([per_seed[s]["pair_exec_c"] for s in SEEDS], axis=0)))
    pvals = [sc1["p_one_sided_pos"], np.nan,
             sc5_test["p_two_sided"],
             sc6["p_one_sided_pos"], sc7g["p_one_sided_pos"]]
    result = dict(primary=primary_out, tost=tost(primary_out["ci90_pp"]),
                  conclusion=conclusion(primary_out["lower5_pp"], primary_out["ci90_pp"],
                                        tost(primary_out["ci90_pp"])),
                  newey_west=newey_west(primary_series),
                  per_seed={str(s): dict(delta_pp=float(per_seed[s]["delta"].mean() * 252 * 100),
                                         ir_delta=per_seed[s]["ir_delta"],
                                         layer_corr_c0=per_seed[s]["layer_corr_c0"],
                                         layer_corr_g0=per_seed[s]["layer_corr_g0"],
                                         ic_c0=per_seed[s]["ic_c0"], ic_g0=per_seed[s]["ic_g0"]) for s in SEEDS},
                  stamp_duty=dict(delta_pp=float(stamp_series.mean() * 252 * 100),
                                  schedule=f"sell 10bp from {STAMP_CUTOFF}; never decisive"),
                  secondary=dict(
                      SC1_soft_delta_pp=sc1["mean"] * 252 * 100,
                      SC1_ci90_pp=[v * 252 * 100 for v in sc1["ci90"]],
                      SC2_ir_delta_per_seed=[per_seed[s]["ir_delta"] for s in SEEDS],
                      SC2_ir_delta_mean=float(np.mean([per_seed[s]["ir_delta"] for s in SEEDS])),
                      SC5_layer_corr=dict(c0=[per_seed[s]["layer_corr_c0"] for s in SEEDS],
                                          g0=[per_seed[s]["layer_corr_g0"] for s in SEEDS],
                                          holm_member="C0 arm, bootstrap p below"),
                      SC5_p=sc5_test,
                      SC6_delta_cost_pp=sc6["mean"] * 252 * 100,
                      SC6_ci90_pp=[v * 252 * 100 for v in sc6["ci90"]],
                      SC7_delta_gross_pp=sc7g["mean"] * 252 * 100,
                      SC7_ic=dict(c0=ic_c0, g0=ic_g0, ratio=ic_c0 / ic_g0)),
                  s1_conditions=s_mapping(primary_out["estimate_pp"], primary_out["lower5_pp"],
                                          [per_seed[s]["delta"].mean() > 0 for s in SEEDS], ic_c0 / ic_g0),
                  notes=["SC3 coincides with the primary (both arms selected the default rule)",
                         "SC4 (CSI500) enters the Holm family at Gate C",
                         "benchmark/CPU decomposition books are produced by the reporting layer, not this frozen script"])
    # Holm over SC1, SC5, SC6, SC7 (SC2 lacks a daily-series bootstrap; reported with per-seed values)
    holm_members = {k: p for k, p in zip(HOLM_KEYS, pvals) if not np.isnan(p)}
    adjusted = holm(list(holm_members.values()))
    result["holm"] = {k: dict(p_raw=p, p_holm=a) for (k, p), a in zip(holm_members.items(), adjusted)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(primary_out, indent=1), flush=True)
    print(f"WROTE {args.out}", flush=True)


def self_test():
    rng = np.random.default_rng(7)
    r = infer(np.full(300, 0.001), n_boot=500)
    assert abs(r["mean"] - 0.001) < 1e-9 and abs(r["lower_5"] - 0.001) < 1e-9, r
    # under block-mean-10 dependence the effective sample is ~n/10 blocks, so the
    # positive alternative must be strong enough (t ~ 3+) to clear the 5% bar
    assert infer(rng.normal(0.006, 0.01, 300), n_boot=1000)["p_one_sided_pos"] < 0.05
    assert infer(rng.normal(-0.006, 0.01, 300), n_boot=1000)["p_one_sided_pos"] > 0.5
    nw = newey_west(rng.normal(0.002, 0.01, 300))
    assert nw["ci95_pp"][0] < nw["annualized_mean_pp"] < nw["ci95_pp"][1]
    assert tost([-1.0, 1.0])["equivalent"] and not tost([-3.0, 1.0])["equivalent"]
    adj = holm([0.04, 0.01, 0.03])
    # standard Holm: 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> monotone 0.06
    assert adj[1] <= adj[2] <= adj[0] and abs(adj[1] - 0.03) < 1e-12 and abs(adj[2] - 0.06) < 1e-9 and abs(adj[0] - 0.06) < 1e-9, adj
    m = s_mapping(2.5, 0.4, [True, True, True], 0.97)
    assert m["S1_1_delta_ge_2pp"] and m["S1_1_lower_bound_pos"] and m["S1_2_3of3_seeds_positive"] and m["S1_3_ic_within_5pct"]
    m2 = s_mapping(2.5, -0.4, [True, True, False], 0.90)
    assert not m2["S1_1_lower_bound_pos"] and not m2["S1_2_3of3_seeds_positive"] and not m2["S1_3_ic_within_5pct"]
    # conclusion table: precedence and the four outcomes
    t_ok = {"equivalent": True}
    assert conclusion(0.4, [-0.5, 1.0], t_ok)["verdict"] == "superior"
    assert conclusion(0.4, [-0.5, 1.0], t_ok)["flags"]["tost_equivalent"]  # co-occurrence reported
    assert conclusion(-0.4, [-1.5, -0.2], t_ok)["verdict"] == "inferior"
    assert conclusion(-0.4, [-1.0, 0.5], t_ok)["verdict"] == "equivalent"
    assert conclusion(-0.4, [-1.5, 3.0], {"equivalent": False})["verdict"] == "uncertain"
    # SC5 bootstrap p: identical series -> corr 1 everywhere -> p 0; noise -> large p
    x = rng.normal(0, 0.01, 300)
    assert sc5_pvalue((x, x), n_boot=200)["p_two_sided"] < 1e-6
    y1, y2 = rng.normal(0, 0.01, 300), rng.normal(0, 0.01, 300)
    assert sc5_pvalue((y1, y2), n_boot=200)["p_two_sided"] > 0.1
    # paired bootstrap: identical series -> zero IR difference in every replicate
    x = rng.normal(0.001, 0.01, 200)
    vals = paired_apply(x, x, lambda a, b: ir(a) - ir(b), n_boot=200)
    assert np.allclose(vals, 0.0, atol=1e-9)
    print("SELF-TEST PASS: bootstrap/CI/p, NW, TOST, Holm, S-mapping, paired-IR bootstrap")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--cache", type=Path, required=False)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "gate_b_unsealed_csi300.json")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.run:
        parser.error("nothing to do: choose --self-test or --run")
    if args.cache is None:
        parser.error("--run requires a fresh --cache directory")
    run(args)


if __name__ == "__main__":
    main()
