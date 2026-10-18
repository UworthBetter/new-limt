"""Clean H / H-FT / G0 / C0 / C-T training with W3 accounting and TBPTT."""
import copy
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import numpy as np
import torch
from model import LiMT
from v2_w3_ledger import Account, continuous_step, execute_close, signal_target
from v2_w6_policy import listing_eligible

GROUPS = ("H", "H-FT", "G0", "C0", "C-T")
TASK_WEIGHTS = (.65, .30, .05)


def tensor(array, device="cpu", dtype=None):
    """Fast NumPy bridge with a list fallback for older torch/NumPy ABI pairs."""
    array = np.asarray(array)
    try:
        value = torch.from_numpy(np.ascontiguousarray(array))
    except (RuntimeError, TypeError):
        value = torch.tensor(array.tolist(), dtype=torch.bool if array.dtype == bool else dtype)
    return value.to(device=device, dtype=dtype)


def numpy(value):
    value = value.detach().cpu()
    try:
        return value.numpy()
    except RuntimeError:
        return np.asarray(value.tolist())


def batches(days, size):
    if size < 2:
        raise ValueError("batch_days must be >=2")
    out = [days[i:i + size] for i in range(0, len(days), size)]
    if len(out) > 1 and len(out[-1]) == 1:
        tail = out.pop()
        out[-1] = np.r_[out[-1], tail]
    if not out or min(map(len, out)) < 2:
        raise ValueError("too few signal days")
    return out


def auxiliary_loss(outputs, targets, mask):
    if mask.dtype != torch.bool:
        raise ValueError("loss mask must be boolean")
    count = mask.sum(-1)
    active = count > 0
    if not active.any():
        raise ValueError("no valid supervised samples in batch")
    if not torch.isfinite(targets[mask]).all():
        raise ValueError("nonfinite supervised target")
    result = outputs[0].new_zeros(())
    for head, coefficient in enumerate(TASK_WEIGHTS):
        error = torch.where(mask, (outputs[head] - torch.where(mask, targets[..., head], 0)) ** 2, 0)
        result = result + coefficient * (error.sum(-1)[active] / count[active]).mean()
    return result


def group_loss(logs, auxiliary, penalty, group, alpha_aux, gamma_to):
    if group not in GROUPS or alpha_aux < 0 or gamma_to < 0:
        raise ValueError("invalid group or loss coefficients")
    if group in ("H", "H-FT"):
        return auxiliary
    return -torch.stack(logs).mean() + alpha_aux * auxiliary + (gamma_to if group == "C-T" else 0.) * torch.stack(penalty).mean()


def forward(model, data, days, device, amp=False):
    x = tensor(data.windows(days), device, torch.float32)
    visible = tensor(data.feature_mask[days], device, torch.bool)
    good = visible.any(-1)
    output = [x.new_zeros(visible.shape) for _ in range(3)]
    if good.any():
        context = torch.autocast(device_type="cuda", dtype=torch.float16) if amp else nullcontext()
        with context:
            predicted = model(x[good], ~visible[good])
        for j in range(3):
            output[j][good] = predicted[j].float()
    if not all(torch.isfinite(p).all() for p in output):
        raise FloatingPointError("nonfinite network output")
    return tuple(output)


@torch.no_grad()
def predict(model, data, days, config, device):
    model.eval()
    output = []
    for day_batch in batches(days, config["batch_days"]):
        heads = forward(model, data, day_batch, device, config["amp"] and device == "cuda")
        output.append(np.stack([numpy(x) for x in heads], axis=-1))
    return np.concatenate(output)


def portfolio_events(data, days, scores, tau, buy=.0005, sell=.0015, rule="soft", topk=50, drop=5):
    """Fixed-parameter replay; terminal mark always included, no liquidation.

    Topk is a continuous-weight reference. It does not claim lot/minimum-fee fills.
    Rank decisions use only each signal's scores and candidate set.
    """
    w = torch.zeros(len(data.symbols) + 1, dtype=torch.float64); w[0] = 1
    prior_entry, wealth, rows, selected = None, 1., [], set()
    for i, t in enumerate(days):
        eligible = data.candidate(t)
        score = np.asarray(scores[i], np.float64)
        if not np.isfinite(score[eligible]).all():
            raise ValueError("nonfinite candidate score")
        if rule == "soft":
            target = signal_target(tensor(score, dtype=torch.float64), tensor(eligible, dtype=torch.bool), tau)
        else:
            target, selected = topk_target(score, eligible, selected, topk, drop)
            target = None if target is None else tensor(target, dtype=torch.float64)
        relatives = np.ones(len(w)) if prior_entry is None else data.relatives(t + 1, prior_entry)
        step = continuous_step(w, tensor(relatives, dtype=torch.float64), target, buy, sell)
        pre_wealth = wealth * float(step["gross"])
        wealth = pre_wealth * float(step["mu"])
        weights = numpy(step["weights"])
        rows.append(dict(signal_date=str(data.dates[t]), date=str(data.dates[t + 1]), event="rebalance",
            net_return=float(step["gross"] * step["mu"] - 1), log_growth=float(step["log_growth"]),
            wealth=wealth, fee_fraction=float(step["fee"]), fee_value=pre_wealth * float(step["fee"]),
            turnover=float(step["turnover"]), target_penalty=float(step["target_penalty"]),
            max_weight=float(weights[1:].max()), effective_holdings=float(1 / np.square(weights[1:]).sum()) if np.square(weights[1:]).sum() else 0.,
            stale_held=int((data.stale[t + 1] & (weights[1:] > 0)).sum())))
        w, prior_entry = step["weights"], t + 1
    t = int(days[-1])
    step = continuous_step(w, tensor(data.relatives(t + 2, t + 1), dtype=torch.float64), None, buy, sell)
    wealth *= float(step["gross"])
    rows.append(dict(signal_date=str(data.dates[t]), date=str(data.dates[t + 2]), event="terminal_mark",
        net_return=float(step["gross"] - 1), log_growth=float(step["log_growth"]), wealth=wealth,
        fee_fraction=0., fee_value=0., turnover=0., target_penalty=0.,
        max_weight=float(w[1:].max()), effective_holdings=rows[-1]["effective_holdings"],
        stale_held=int((data.stale[t + 2] & (numpy(w[1:]) > 0)).sum())))
    return rows


def topk_target(scores, eligible, selected, topk=50, drop=5):
    if topk < 1 or drop < 0 or drop > topk:
        raise ValueError("invalid topk/drop")
    ranked = sorted(np.where(eligible)[0].tolist(), key=lambda i: (-scores[i], i))
    if not ranked:
        return None, selected.copy()
    current = selected.intersection(ranked)
    additions = [i for i in ranked if i not in current][:max(0, topk - len(current)) + drop]
    pool = current.union(additions)
    rank = sorted(pool, key=lambda i: (-scores[i], i))
    # At most drop replacements of still-eligible existing names; forced exits separate.
    worst = sorted(current, key=lambda i: (scores[i], -i))
    new_best = [i for i in rank if i not in current]
    for old, new in zip(worst[:drop], new_best[:drop]):
        if len(current) >= topk and scores[new] > scores[old]:
            current.remove(old); current.add(new)
    for i in ranked:
        if len(current) >= topk:
            break
        current.add(i)
    w = np.zeros(len(scores) + 1)
    for i in current:
        w[i + 1] = 1 / len(current)
    return w, current


def resolved_arm_config(group, config=None):
    """Resolved schedule for one arm. G0 and C0 differ only in training_fee."""
    if group not in GROUPS:
        raise ValueError(f"unknown arm {group}")
    config = dict(config or {})
    fine = group != "H"
    resolved = dict(
        epochs=int(config.get("epochs_h", 50) if not fine else config.get("epochs_finetune", 20)),
        patience=int(config.get("patience", 8)),
        lr=float(config.get("lr_h", 1e-4) if not fine else config.get("lr_finetune", 1e-5)),
        optimizer="Adam",
        seeds=tuple(config.get("seeds", (1,))),
        alpha_aux=float(config.get("alpha_aux", 0.3)),
        tau_grid=(0.5, 0.75, 1.0),
        monitoring_fee=dict(buy=float(config.get("buy_rate", 0.0005)), sell=float(config.get("sell_rate", 0.0015)),
                            min_fee=float(config.get("min_fee", 5.0)), lot=int(config.get("lot", 100))),
    )
    if group == "G0":
        resolved["training_fee"] = dict(buy=0.0, sell=0.0, min_fee=0.0, lot=int(config.get("lot", 100)))
    elif group in ("C0", "C-T"):
        resolved["training_fee"] = dict(resolved["monitoring_fee"])
    else:
        resolved["training_fee"] = None
    return resolved


def select_checkpoint_epoch(real_fee_logs, zero_fee_logs=None):
    """Epoch index of the real-fee monitoring mean. The zero-fee series is not eligible."""
    real = np.asarray(real_fee_logs, float)
    if real.ndim != 1 or len(real) == 0 or not np.isfinite(real).all():
        raise ValueError("real-fee monitoring log is empty or nonfinite")
    if zero_fee_logs is not None:
        zero = np.asarray(zero_fee_logs, float)
        if zero.shape != real.shape or not np.isfinite(zero).all():
            raise ValueError("zero-fee monitoring log does not match the real-fee log")
    return int(np.argmax(real))


def _monitoring_feed(data, execution_feed):
    if execution_feed is None:
        close = np.asarray(data.close, np.float64)
        shape = close.shape
        return dict(raw_close=close, state=np.zeros(shape, np.int8),
                    share_multiplier=np.ones(shape), cash_dividend_per_old_share=np.zeros(shape))
    if isinstance(execution_feed, dict):
        return execution_feed
    from v2_w5_results import load_execution_feed
    return load_execution_feed(data, execution_feed)


def monitoring_mean_log(data, days, scores, tau, execution_feed, buy, sell, min_fee, lot, capital):
    """Soft targets on the share ledger. Mean log growth is the fine-tune selection metric.

    buy/sell/min_fee/lot are the monitoring schedule. This ledger is not differentiated.
    """
    feed = _monitoring_feed(data, execution_feed)
    prices, states = feed["raw_close"], feed["state"]
    multiplier, dividend = feed["share_multiplier"], feed["cash_dividend_per_old_share"]
    listed = listing_eligible(data.dates, data.symbols)
    account = Account.all_cash(len(data.symbols), float(capital))
    logs = []
    lot = int(lot)
    for i in range(len(days) + 1):
        terminal = i == len(days)
        t = int(days[-1]) + 2 if terminal else int(days[i]) + 1
        action_cash = float(np.dot(account.shares, dividend[t]))
        account = Account(account.cash + action_cash, account.shares * multiplier[t],
                          account.marks / multiplier[t], account.date)
        if terminal:
            target = None
        else:
            candidate = data.candidate(int(days[i])).copy() & listed[int(days[i])]
            soft = signal_target(tensor(np.asarray(scores[i], np.float64), dtype=torch.float64),
                                 tensor(candidate, dtype=torch.bool), tau)
            target = None if soft is None else numpy(soft)
            if target is not None:
                target = np.asarray(target, np.float64)
                target = target / target.sum()
        quotes = np.where(listed[t], prices[t], np.nan)
        effective = np.where(listed[t], states[t], 1)
        account, event = execute_close(account, quotes, effective, target, str(data.dates[t - 1]), str(data.dates[t]),
                                       buy=float(buy), sell=float(sell), min_fee=float(min_fee), lot=lot)
        logs.append(event["log_growth"])
    if not logs:
        raise ValueError("empty monitoring window")
    return float(np.mean(logs))


def make_model(data, config):
    return LiMT(len(data.feature_names), config["dimension"], 4, 2, 4)


def train(data, config, group, seed, out, device, identity, initial=None, execution_feed=None):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = make_model(data, config).to(device)
    if group != "H":
        if initial is None:
            raise ValueError("fine-tuning requires clean H checkpoint")
        checkpoint = torch.load(initial, map_location="cpu", weights_only=True)
        if checkpoint["data_version"] != identity or checkpoint["seed"] != seed or checkpoint["group"] != "H":
            raise ValueError("incompatible/legacy initialization checkpoint")
        model.load_state_dict(checkpoint["state_dict"])
    train_days = data.days("train", training=True, limit=config["train_days"])
    valid_days = data.days("valid", limit=config["valid_days"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr_h"] if group == "H" else config["lr_finetune"])
    enabled = config["amp"] and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=enabled) if hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler(enabled=enabled)
    epochs = config["epochs_h"] if group == "H" else config["epochs_finetune"]
    best, best_state, bad, updates, history, best_index = -math.inf, None, 0, 0, [], None
    for epoch in range(epochs):
        model.train(); w = torch.zeros(len(data.symbols) + 1, device=device, dtype=torch.float32); w[0] = 1
        previous_entry, epoch_rows = None, []
        for day_batch in batches(train_days, config["batch_days"]):
            heads = forward(model, data, day_batch, device, config["amp"] and device == "cuda")
            mask = tensor(data.loss_masks["train"][day_batch], device, torch.bool)
            labels = tensor(data.targets[day_batch], device, torch.float32)
            need_aux = group in ("H", "H-FT") or config["alpha_aux"] != 0
            auxiliary = auxiliary_loss(heads, labels, mask) if need_aux else heads[0].new_zeros(())
            logs, penalties, fees, turns, concentrations = [], [], [], [], []
            if group not in ("H", "H-FT"):
                cb, cs = (0., 0.) if group == "G0" else (config["buy_rate"], config["sell_rate"])
                for j, t in enumerate(day_batch):
                    y = np.ones(len(w)) if previous_entry is None else data.relatives(t + 1, previous_entry)
                    target = signal_target(heads[0][j], tensor(data.candidate(t), device, torch.bool), config["tau"])
                    row = continuous_step(w, tensor(y, device, torch.float32), target, cb, cs)
                    logs.append(row["log_growth"]); penalties.append(row["target_penalty"])
                    fees.append(row["fee"]); turns.append(row["turnover"]); concentrations.append(row["weights"][1:].max())
                    w, previous_entry = row["weights"], t + 1
                if day_batch[-1] == train_days[-1]:
                    t = int(day_batch[-1]); tail = continuous_step(w, tensor(data.relatives(t + 2, t + 1), device, torch.float32), None, cb, cs)
                    logs.append(tail["log_growth"])
            loss = group_loss(logs, auxiliary, penalties, group, config["alpha_aux"], config["gamma_to"])
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_norm"], error_if_nonfinite=True)
            before_scale = scaler.get_scale(); scaler.step(optimizer); scaler.update()
            skipped = scaler.get_scale() < before_scale
            if skipped:
                raise FloatingPointError("AMP update skipped; run failed explicitly")
            updates += 1
            epoch_rows.append(dict(loss=float(loss.detach()), auxiliary=float(auxiliary.detach()), gradient_norm=float(grad),
                mean_log_growth=float(torch.stack(logs).mean().detach()) if logs else None,
                mean_fee=float(torch.stack(fees).mean().detach()) if fees else None,
                mean_turnover=float(torch.stack(turns).mean().detach()) if turns else None,
                mean_target_penalty=float(torch.stack(penalties).mean().detach()) if penalties else None,
                max_weight=float(torch.stack(concentrations).max().detach()) if concentrations else None,
                batch_signals=len(day_batch), skipped=False))
            w = w.detach()  # TBPTT: values carried; no complete-epoch gradient claim.
        predictions = predict(model, data, valid_days, config, device)
        val_mask = tensor(data.loss_masks["valid"][valid_days], dtype=torch.bool)
        val_aux = float(auxiliary_loss(tuple(tensor(predictions[..., j], dtype=torch.float32) for j in range(3)),
                                      tensor(data.targets[valid_days], dtype=torch.float32), val_mask))
        if group == "H":
            validation_zero_fee_mean_log = None
            validation_real_fee_mean_log = None
            metric = -val_aux
        else:
            if config.get("mode") == "formal" and execution_feed is None:
                raise ValueError("formal fine-tune selection requires the real-fee execution feed")
            fees = resolved_arm_config(group, config)
            validation_zero_fee_mean_log = monitoring_mean_log(
                data, valid_days, predictions[..., 0], config["tau"], execution_feed,
                0.0, 0.0, 0.0, fees["monitoring_fee"]["lot"], config["capital"])
            validation_real_fee_mean_log = monitoring_mean_log(
                data, valid_days, predictions[..., 0], config["tau"], execution_feed,
                fees["monitoring_fee"]["buy"], fees["monitoring_fee"]["sell"],
                fees["monitoring_fee"]["min_fee"], fees["monitoring_fee"]["lot"], config["capital"])
            # Selection is the real-fee mean. The zero-fee value is logged and ignored.
            metric = validation_real_fee_mean_log
        chosen_now = None if group == "H" else select_checkpoint_epoch(
            [*(row["validation_real_fee_mean_log"] for row in history), validation_real_fee_mean_log],
            [*(row["validation_zero_fee_mean_log"] for row in history), validation_zero_fee_mean_log])
        if metric > best:
            best, best_state, bad, best_index = metric, copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}), 0, len(history)
        else:
            bad += 1
        if group != "H" and chosen_now != best_index:
            raise RuntimeError("checkpoint drifted from the real-fee monitoring argmax")
        history.append(dict(epoch=epoch + 1, updates=updates, validation_aux=val_aux,
                            validation_zero_fee_mean_log=validation_zero_fee_mean_log,
                            validation_real_fee_mean_log=validation_real_fee_mean_log,
                            validation_net_mean_log=validation_real_fee_mean_log, selection_value=metric, batches=epoch_rows))
        if group == "H":
            print(f"{group} seed={seed} epoch={epoch + 1} updates={updates} val_aux={val_aux:.6g}", flush=True)
        else:
            print(f"{group} seed={seed} epoch={epoch + 1} updates={updates} val_aux={val_aux:.6g} "
                  f"val_net_log={validation_real_fee_mean_log:.6g} val_zero_fee_log={validation_zero_fee_mean_log:.6g}", flush=True)
        if config["patience"] and bad >= config["patience"]:
            break
    if best_state is None:
        raise RuntimeError("no valid checkpoint")
    checkpoint = dict(state_dict=best_state, data_version=identity, seed=seed, group=group,
        objective_family="supervised-h5" if group in ("H", "H-FT") else "log-growth",
        selection="valid_aux_min" if group == "H" else "valid_common_net_mean_log_max",
        validation_selection=best, updates=updates, configuration=config,
        state_approximation="chronological TBPTT; detach weights at update; all-cash epoch reset")
    path = Path(out) / "checkpoint.pt"
    with path.open("xb") as stream:
        torch.save(checkpoint, stream)
    (Path(out) / "history.json").write_text(json.dumps(history, allow_nan=False, indent=1), encoding="utf-8")
    model.load_state_dict(best_state)
    return model, checkpoint


def ridge_fit(data, config):
    """CPU ridge_h5 reference with one fixed alpha; fit only purged train labels."""
    f = len(data.feature_names)
    gram, rhs, count = np.zeros((f + 1, f + 1)), np.zeros(f + 1), 0
    for t in data.days("train", training=True, limit=config["train_days"]):
        mask = data.loss_masks["train"][t]
        if not mask.any():
            continue
        x = np.asarray(data.features[t, data.indices, :], np.float64)[mask]
        x = np.c_[np.ones(len(x)), x]; target = data.targets[t, mask, 0].astype(float)
        if not np.isfinite(x).all() or not np.isfinite(target).all():
            raise ValueError("nonfinite ridge sample")
        gram += x.T @ x; rhs += x.T @ target; count += len(x)
    if count == 0:
        raise ValueError("empty ridge fit")
    regularizer = np.eye(f + 1) * config["ridge_alpha"]; regularizer[0, 0] = 0
    return np.linalg.solve(gram + regularizer, rhs), count


def ridge_predict(data, days, coefficients):
    output = np.zeros((len(days), len(data.symbols), 3), np.float32)
    for i, t in enumerate(days):
        x = np.asarray(data.features[t, data.indices, :], np.float64)
        output[i, :, 0] = np.c_[np.ones(len(x)), x] @ coefficients
    return output
