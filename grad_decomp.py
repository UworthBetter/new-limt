"""Gradient decomposition of the fine-tune objective (DIAGNOSTIC; validation window only).

Splits the per-batch objective into three terms, exactly as v2_w4_training.train builds it:
    gross = -mean(log g_k)          (return part of the ledger)
    fee   = -mean(log mu_k)         (cost part of the ledger; zero for G0 fees)
    aux   = alpha_aux * auxiliary_loss
and reports gradient norms w.r.t. (a) the return-head scores on candidate stocks and
(b) all shared MRE parameters, plus cosines between the terms.

Reads only validation days (2019-2020). Never touches rule_select or eval data.
Does not train or modify anything. Runs on CPU or GPU.

Usage (Windows example):
  python grad_decomp.py --src D:\\research\\limt-repro\\src --root D:\\research\\limt-repro ^
      --cache D:\\tmp\\gd_cache_c0 --checkpoint <path to C0 checkpoint.pt> --out gd_c0.json
Optional: --fee-mult 100   (measure the fee gradient share under 100x fees)
          --tau 0.5        (override the checkpoint tau)
          --max-batches 0  (0 = all validation batches)
Note: --cache must be a NEW directory (Data extracts features.npy into it, several GB).
"""
import argparse
import json
import math
import sys

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fee-mult", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=None)
    p.add_argument("--max-batches", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    sys.path.insert(0, a.src)
    from v2_w4_data import Data
    from v2_w4_training import make_model, forward, batches, auxiliary_loss, tensor
    from v2_w3_ledger import continuous_step, signal_target

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    cfg = ck["configuration"]
    tau = float(a.tau if a.tau is not None else cfg["tau"])
    alpha = float(cfg["alpha_aux"])
    cb, cs = float(cfg["buy_rate"]) * a.fee_mult, float(cfg["sell_rate"]) * a.fee_mult
    if not (cs < 1):
        raise ValueError("fee multiplier too large (sell rate must stay < 1)")

    data = Data(a.root, a.cache, cfg["lookback"], cfg.get("max_symbols", 0),
                cfg.get("evaluation_end", "2026-06-30"))
    dev = a.device
    model = make_model(data, cfg)
    model.load_state_dict(ck["state_dict"])
    model.to(dev).eval()
    shared = [(n, q) for n, q in model.named_parameters() if n.startswith("mre.")]
    shared_params = [q for _, q in shared]

    days = data.days("valid")
    day_batches = batches(days, cfg["batch_days"])
    if a.max_batches:
        day_batches = day_batches[:a.max_batches]

    w = torch.zeros(len(data.symbols) + 1, device=dev, dtype=torch.float32)
    w[0] = 1
    prev = None
    rec = {k: [] for k in ("s_gross", "s_fee", "s_aux", "p_gross", "p_fee", "p_aux",
                           "cos_fee_gross_p", "cos_fee_aux_p", "cos_gross_aux_p",
                           "score_std", "neff", "turnover")}

    def flat(gs):
        parts = [g.reshape(-1) for g in gs if g is not None]
        return torch.cat(parts) if parts else None

    def cos(u, v):
        if u is None or v is None or u.norm() == 0 or v.norm() == 0:
            return float("nan")
        return float(torch.dot(u, v) / (u.norm() * v.norm()))

    for b in day_batches:
        heads = forward(model, data, b, dev, False)
        scores = heads[0]
        mask = tensor(data.loss_masks["valid"][b], dev, torch.bool)
        labels = tensor(data.targets[b], dev, torch.float32)
        aux = alpha * auxiliary_loss(heads, labels, mask) if (alpha and mask.any()) else None
        lg, lm, cand_all = [], [], torch.zeros_like(scores, dtype=torch.bool)
        for j, t in enumerate(b):
            y = np.ones(len(w)) if prev is None else data.relatives(t + 1, prev)
            cand = tensor(data.candidate(t), dev, torch.bool)
            cand_all[j] = cand
            target = signal_target(scores[j], cand, tau)
            row = continuous_step(w, tensor(y, dev, torch.float32), target, cb, cs)
            lg.append(torch.log(row["gross"]))
            lm.append(torch.log(row["mu"]))
            rec["score_std"].append(float(scores[j][cand].std()))
            rec["neff"].append(float(1 / (target[1:] ** 2).sum()))
            rec["turnover"].append(float(row["turnover"]))
            w, prev = row["weights"], t + 1
        terms = dict(gross=-torch.stack(lg).mean(), fee=-torch.stack(lm).mean(), aux=aux)
        sv, pv = {}, {}
        for name, term in terms.items():
            if term is None or not term.requires_grad:
                sv[name], pv[name] = None, None
                continue
            gs = torch.autograd.grad(term, [scores] + shared_params, retain_graph=True, allow_unused=True)
            g_s = gs[0]
            sv[name] = None if g_s is None else g_s[cand_all]
            pv[name] = flat(gs[1:])
        for name in ("gross", "fee", "aux"):
            rec[f"s_{name}"].append(float("nan") if sv[name] is None else float(sv[name].norm()))
            rec[f"p_{name}"].append(float("nan") if pv[name] is None else float(pv[name].norm()))
        rec["cos_fee_gross_p"].append(cos(pv["fee"], pv["gross"]))
        rec["cos_fee_aux_p"].append(cos(pv["fee"], pv["aux"]))
        rec["cos_gross_aux_p"].append(cos(pv["gross"], pv["aux"]))
        w = w.detach()

    med = {k: float(np.nanmedian(v)) if len(v) else None for k, v in rec.items()}
    ratio = lambda x, y: (med[x] / med[y]) if med[x] is not None and med[y] else None
    summary = dict(
        checkpoint=a.checkpoint, group=ck.get("group"), tau=tau, alpha_aux=alpha,
        fee_mult=a.fee_mult, buy=cb, sell=cs, batches=len(day_batches), median=med,
        ratios_scores=dict(fee_over_gross=ratio("s_fee", "s_gross"), fee_over_aux=ratio("s_fee", "s_aux"),
                           gross_over_aux=ratio("s_gross", "s_aux")),
        ratios_shared=dict(fee_over_gross=ratio("p_fee", "p_gross"), fee_over_aux=ratio("p_fee", "p_aux"),
                           gross_over_aux=ratio("p_gross", "p_aux")),
        reading=("fee_over_aux ~0.005-0.01 and score_std ~0.1-0.2 => dose/scale explanation; "
                 "fee gradient exactly 0 or NaN with C0 fees => plumbing bug"))
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=True)
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()