"""W3 accounting core: signal-only targets, differentiable proxy and share execution.

Cash is column zero. Rates are fees / executed notional. The discrete executor
expects unadjusted quotes in currency/share, with corporate actions already
reflected in the account; adjusted Qlib prices must not be passed as raw quotes.
"""
from dataclasses import dataclass
import math
import numpy as np
import torch


class AccountingError(ValueError):
    pass


def _rates(buy, sell):
    if not all(math.isfinite(x) and 0 <= x < 1 for x in (buy, sell)):
        raise AccountingError("fee rates must be finite and in [0, 1)")


def _weights(w):
    if w.dtype not in (torch.float32, torch.float64) or w.shape[-1] < 2:
        raise AccountingError("weights need float32/64 and cash plus risky columns")
    tol = 1e-6 if w.dtype == torch.float32 else 1e-10
    if not torch.isfinite(w).all() or (w < 0).any() or ((w.sum(-1) - 1).abs() > tol).any():
        raise AccountingError("weights must be finite, nonnegative and sum to one")


def signal_target(scores, candidate, tau=0.5):
    """One signal day's scores + history-visible candidate mask -> weights or HOLD.

    No execution state, future returns or label masks enter this interface.
    None means preserve existing holdings and mark them at the execution event.
    """
    if scores.ndim != 1 or candidate.shape != scores.shape or candidate.dtype != torch.bool:
        raise AccountingError("scores and boolean candidate must be aligned 1D vectors")
    if not math.isfinite(tau) or tau <= 0:
        raise AccountingError("tau must be positive")
    if not candidate.any():
        return None
    if not torch.isfinite(scores[candidate]).all():
        raise AccountingError("nonfinite eligible score")
    risky = torch.softmax((scores / tau).masked_fill(~candidate, -torch.inf), -1)
    return torch.cat([scores.new_zeros(1), risky])


def drift(w_previous, relatives):
    _weights(w_previous)
    if relatives.shape != w_previous.shape or relatives.dtype != w_previous.dtype:
        raise AccountingError("price relatives and weights must have identical shape/dtype")
    if not torch.isfinite(relatives[..., 0]).all() or not torch.allclose(relatives[..., 0], torch.ones_like(relatives[..., 0])):
        raise AccountingError("cash relative must equal one")
    held = w_previous > 0
    if ((~torch.isfinite(relatives) | (relatives <= 0)) & held).any():
        raise AccountingError("held asset requires a positive finite valuation relative")
    # Missing unheld quotes do not affect valuation; this is not execution permission.
    y = torch.where(held, relatives, torch.ones_like(relatives))
    value = y * w_previous
    growth = value.sum(-1)
    if not torch.isfinite(growth).all() or (growth <= 0).any():
        raise AccountingError("nonpositive or nonfinite wealth")
    return growth, value / growth.unsqueeze(-1)


def fee_factor(target, before, buy=0.0005, sell=0.0015):
    """Exact active-piece fee solution; gradients flow through the closed form."""
    _rates(buy, sell); _weights(target); _weights(before)
    if target.shape != before.shape or target.dtype != before.dtype:
        raise AccountingError("target and before must have identical shape/dtype")
    if buy == sell == 0:
        return target[..., 0] * 0 + 1
    with torch.no_grad():
        w, p = target.double(), before.double()
        lo, hi = torch.zeros_like(w[..., 0]), torch.ones_like(w[..., 0])
        for _ in range(64):
            mid = (lo + hi) / 2
            d = mid.unsqueeze(-1) * w[..., 1:] - p[..., 1:]
            residual = mid + buy * d.clamp(min=0).sum(-1) + sell * (-d).clamp(min=0).sum(-1) - 1
            lo, hi = torch.where(residual < 0, mid, lo), torch.where(residual >= 0, mid, hi)
        d = ((lo + hi) / 2).unsqueeze(-1) * w[..., 1:] - p[..., 1:]
        b, s = (d > 0).to(target.dtype), (d < 0).to(target.dtype)
    numerator = 1 + (before[..., 1:] * (buy * b - sell * s)).sum(-1)
    denominator = 1 + (target[..., 1:] * (buy * b - sell * s)).sum(-1)
    mu = numerator / denominator
    d = mu.unsqueeze(-1) * target[..., 1:] - before[..., 1:]
    residual = mu + buy * d.clamp(min=0).sum(-1) + sell * (-d).clamp(min=0).sum(-1) - 1
    tol = 2e-6 if mu.dtype == torch.float32 else 1e-11
    if not torch.isfinite(mu).all() or (mu <= 0).any() or (mu > 1 + tol).any() or (residual.abs() > tol).any():
        raise AccountingError("fee equation did not converge")
    return mu


def continuous_step(previous, relatives, target=None, buy=0.0005, sell=0.0015):
    """Mark old holdings, then trade at this event. target was fixed on signal day."""
    _rates(buy, sell)
    gross, before = drift(previous, relatives)
    after = before if target is None else target
    mu = fee_factor(after, before, buy, sell)
    delta = mu.unsqueeze(-1) * after[..., 1:] - before[..., 1:]
    buy_value = delta.clamp(min=0).sum(-1)
    sell_value = (-delta).clamp(min=0).sum(-1)
    fee = buy * buy_value + sell * sell_value
    cash_delta = mu * after[..., 0] - before[..., 0]
    return dict(weights=after, before=before, gross=gross, mu=mu,
                log_growth=torch.log(gross) + torch.log(mu), fee=fee,
                buy_value=buy_value, sell_value=sell_value,
                turnover=buy_value + sell_value,
                target_penalty=(after[..., 1:] - before[..., 1:]).abs().sum(-1),
                cash_residual=cash_delta + delta.sum(-1) + fee)


@dataclass
class Account:
    cash: float
    shares: np.ndarray
    marks: np.ndarray
    date: str | None = None

    @classmethod
    def all_cash(cls, n_assets, capital=1_000_000.0):
        if n_assets < 1 or not math.isfinite(capital) or capital <= 0:
            raise AccountingError("invalid initial account")
        return cls(float(capital), np.zeros(n_assets), np.full(n_assets, np.nan))

    def value(self):
        held = self.shares > 0
        if (not math.isfinite(self.cash) or self.cash < -1e-8 or not np.isfinite(self.shares).all()
                or (self.shares < 0).any() or not np.isfinite(self.marks[held]).all()
                or (self.marks[held] <= 0).any()):
            raise AccountingError("invalid cash, shares or held valuation marks")
        return float(self.cash + np.dot(self.shares[held], self.marks[held]))


def execute_close(account, quotes, states, target, signal_date, execution_date,
                  buy=0.0005, sell=0.0015, min_fee=5.0, lot=100):
    """Daily close matching: sells first, cash-limited lot buys, asymmetric limits.

    states: 0 normal, 1 suspended, 2 limit-up (sell allowed), 3 limit-down
    (buy allowed), 4 unknown (both forbidden). Missing quotes forbid fills;
    old marks value existing shares. ST is independent of this direction state.
    Buys use a common cash scale then fixed input-symbol order; no ranking by
    execution-day returns. One event per date enforces daily-model T+1 selling.
    Partial sells use lots; full liquidation permits remaining odd shares.
    """
    _rates(buy, sell)
    if not math.isfinite(min_fee) or min_fee < 0 or not isinstance(lot, int) or lot < 1:
        raise AccountingError("invalid min_fee or lot")
    quotes, states = np.asarray(quotes, float), np.asarray(states)
    n = len(account.shares)
    if quotes.shape != (n,) or states.shape != (n,) or not np.isin(states, [0, 1, 2, 3, 4]).all():
        raise AccountingError("invalid quotes or execution states")
    sd, ed = np.datetime64(signal_date, "D"), np.datetime64(execution_date, "D")
    if np.isnat(sd) or np.isnat(ed) or sd >= ed or (account.date is not None and ed <= np.datetime64(account.date, "D")):
        raise AccountingError("signal must precede execution; event dates must increase")
    old_value = account.value()
    valid = np.isfinite(quotes) & (quotes > 0)
    marks = np.where(valid, quotes, account.marks)
    working = Account(account.cash, account.shares.copy(), marks, str(ed))
    pre_value = working.value()
    if old_value <= 0 or pre_value <= 0:
        raise AccountingError("nonpositive wealth")
    before = np.r_[working.cash, working.shares * np.where(np.isfinite(marks), marks, 0)] / pre_value
    if target is None:
        desired = working.shares.copy()
        target_np = before
    else:
        target_np = np.asarray(target, float)
        if target_np.shape != (n + 1,):
            raise AccountingError("wrong target shape")
        # Avoid the optional NumPy C bridge (older torch wheels + NumPy 2).
        mu = float(fee_factor(torch.tensor(target_np.tolist(), dtype=torch.float64),
                              torch.tensor(before.tolist(), dtype=torch.float64), buy, sell))
        desired = np.divide(mu * pre_value * target_np[1:], quotes,
                            out=working.shares.copy(), where=valid)
    delta = desired - working.shares
    requested_buy = np.floor(np.maximum(delta, 0) / lot) * lot
    orders = []
    total_fees, buys, sells = 0.0, 0.0, 0.0

    def fill(i, qty, side):
        nonlocal total_fees, buys, sells
        if qty <= 0:
            return
        value = qty * quotes[i]
        fee = max(min_fee, value * (buy if side == "buy" else sell))
        if side == "buy":
            working.cash -= value + fee; working.shares[i] += qty; buys += value
        else:
            working.cash += value - fee; working.shares[i] -= qty; sells += value
        total_fees += fee
        orders.append(dict(asset=int(i), side=side, quantity=float(qty), price=float(quotes[i]), notional=float(value), fee=float(fee)))

    for i in range(n):
        if delta[i] >= -1e-10 or not valid[i] or states[i] in (1, 3, 4):
            continue
        qty = working.shares[i] if desired[i] < 1e-10 else min(working.shares[i], math.floor(-delta[i] / lot) * lot)
        if qty > 0 and working.cash + qty * quotes[i] >= max(min_fee, qty * quotes[i] * sell):
            fill(i, qty, "sell")
    allowed = valid & ~np.isin(states, [1, 2, 4])
    requested_buy = np.where(allowed, requested_buy, 0)
    notionals = requested_buy * np.where(valid, quotes, 0)
    need = float((notionals + np.where(requested_buy > 0, np.maximum(min_fee, notionals * buy), 0)).sum())
    scale = min(1.0, working.cash / need) if need > 0 else 1.0
    for i in range(n):
        if requested_buy[i] <= 0:
            continue
        affordable = min(max(0, working.cash - min_fee) / quotes[i], working.cash / (quotes[i] * (1 + buy)))
        qty = math.floor(min(requested_buy[i] * scale, affordable) / lot) * lot
        fill(i, qty, "buy")
    post_value = working.value()
    residual = post_value - (pre_value - total_fees)
    if abs(residual) > max(1e-8, pre_value * 1e-12) or post_value <= 0:
        raise AccountingError("execution self-financing invariant failed")
    actual_weights = np.r_[working.cash, working.shares * np.where(np.isfinite(marks), marks, 0)] / post_value
    return working, dict(signal_date=str(sd), execution_date=str(ed), mark_date=str(ed), cost_date=str(ed),
        wealth_before_mark=old_value, wealth_before_trade=pre_value, wealth=post_value,
        gross=pre_value / old_value, log_growth=math.log(post_value / old_value),
        fees=total_fees, buy_notional=buys, sell_notional=sells,
        turnover=(buys + sells) / pre_value, cash=working.cash, cash_residual=residual,
        stale_held=int(((~valid) & (working.shares > 0)).sum()),
        blocked_buy_assets=int(((delta > 1e-10) & ((~valid) | np.isin(states,[1,2,4]))).sum()),
        blocked_sell_assets=int(((delta < -1e-10) & ((~valid) | np.isin(states,[1,3,4]))).sum()),
        allocation_l1=float(np.abs(actual_weights - target_np).sum()), orders=orders)


def replay_daily(dates, quotes, states, targets, capital=1_000_000.0, **kwargs):
    """targets[i] from signal i trades at i+1; last event marks without liquidation.

    N dates require N-2 targets; final holding receives its last close-to-close
    return at date N-1. Targets are fixed inputs, not generated from future quotes.
    """
    dates = np.asarray(dates, dtype="datetime64[D]")
    quotes, states = np.asarray(quotes), np.asarray(states)
    if (dates.ndim != 1 or len(dates) < 3 or np.isnat(dates).any() or not (np.diff(dates) > np.timedelta64(0, "D")).all()
            or quotes.ndim != 2 or quotes.shape[0] != len(dates) or states.shape != quotes.shape
            or len(targets) != len(dates) - 2):
        raise AccountingError("misaligned daily replay inputs")
    account = Account.all_cash(quotes.shape[1], capital)
    rows = []
    for i in range(1, len(dates)):
        target = targets[i - 1] if i < len(dates) - 1 else None
        account, row = execute_close(account, quotes[i], states[i], target, str(dates[i - 1]), str(dates[i]), **kwargs)
        row["event"] = "rebalance" if i < len(dates) - 1 else "terminal_mark"
        rows.append(row)
    return account, rows
