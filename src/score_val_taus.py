"""Score checkpoints on the validation window (2019-2020) at three taus.

Fills N5: per (checkpoint, tau) the soft-book concentration on the validation
window -- N_eff = 1/sum(w^2) from signal_target weights, plus max single-name
weight -- and the monitoring-ledger mean log growth (real fee and zero fee).

Read-only with respect to runs/; writes protocol/checks/n5_neff_validation.json.
Usage: score_val_taus.py [--only name ...]  (default: every checkpoint on disk)
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

LIMT_SRC = Path(r"D:\research\limt-repro\src")
sys.path.insert(0, str(LIMT_SRC))

from v2_w4_data import Data
from v2_w4_training import make_model, predict, monitoring_mean_log
from v2_w6_run import DEFAULT, validate_config
from v2_w3_ledger import signal_target
from v2_w6_policy import listing_eligible
from v2_w5_results import load_execution_feed

W0W2 = Path(r"D:\research\new_LiMT\data\csi300_w1w2")
FEED = Path(r"D:\research\limt-repro\data\v2_formal_inputs_run3\new_execution_feed.npz")
OUT_JSON = Path(r"D:\research\new_LiMT\protocol\checks\n5_neff_validation.json")
CACHE = Path(r"D:\research\new_LiMT\runs\score_val_taus_cache")
TAUS = (0.5, 0.75, 1.0)
RUNS = Path(r"D:\research\new_LiMT\runs")
CHECKPOINTS = {
    "H": RUNS / "csi300_H_seed1_20261017" / "H" / "seed1" / "checkpoint.pt",
    "G0_tau0.50": RUNS / "csi300_g0c0_seed1_tau_20261017" / "G0" / "tau0.50" / "seed1" / "checkpoint.pt",
    "G0_tau0.75": RUNS / "csi300_g0c0_seed1_tau_20261017" / "G0" / "tau0.75" / "seed1" / "checkpoint.pt",
    "G0_tau1.00": RUNS / "csi300_g0c0_seed1_tau_20261017" / "G0" / "tau1.00" / "seed1" / "checkpoint.pt",
    "C0_tau0.50": RUNS / "csi300_g0c0_seed1_tau_20261017" / "C0" / "tau0.50" / "seed1" / "checkpoint.pt",
    "C0_tau0.75": RUNS / "csi300_g0c0_seed1_tau_20261017" / "C0" / "tau0.75" / "seed1" / "checkpoint.pt",
    "C0_tau1.00": RUNS / "csi300_g0c0_seed1_tau_20261017" / "C0" / "tau1.00" / "seed1" / "checkpoint.pt",
    "H-FT_tau0.50": RUNS / "csi300_HFT_seed1_20261018" / "H-FT" / "seed1" / "checkpoint.pt",
    "G0_dose003_tau0.50": RUNS / "csi300_dose003_seed1_20261018" / "G0" / "tau0.50" / "seed1" / "checkpoint.pt",
    "C0_dose003_tau0.50": RUNS / "csi300_dose003_seed1_20261018" / "C0" / "tau0.50" / "seed1" / "checkpoint.pt",
}


def sha256_file(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def concentration(data, days, scores, tau, listed):
    """Per-day soft-book N_eff and max weight, mirroring the monitoring target."""
    neff, maxw, ncand = [], [], []
    for i, t in enumerate(days):
        candidate = data.candidate(int(t)) & listed[int(t)]
        if not candidate.any():
            continue
        w = signal_target(
            torch.tensor(np.asarray(scores[i], np.float64)),
            torch.tensor(candidate, dtype=torch.bool), tau)
        if w is None:
            continue
        risky = numpy_or(w[1:])
        neff.append(float(1.0 / np.square(risky).sum()))
        maxw.append(float(risky.max()))
        ncand.append(int(candidate.sum()))
    return neff, maxw, ncand


def numpy_or(tensor_value):
    return tensor_value.detach().cpu().numpy()


def stats(values):
    a = np.asarray(values, float)
    return dict(
        n=len(a), median=float(np.median(a)), mean=float(a.mean()),
        p10=float(np.percentile(a, 10)), p90=float(np.percentile(a, 90)),
        min=float(a.min()), max=float(a.max()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--cache", type=Path, default=CACHE,
                        help="Data cache dir; Data() refuses an existing dir, so pass a fresh one per invocation")
    args = parser.parse_args()
    wanted = args.only or [k for k, v in CHECKPOINTS.items() if v.is_file()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # All formal checkpoints on this tree are lookback-21 d=256; Data is shared.
    LOOKBACK = 21
    for name in wanted:
        path = CHECKPOINTS[name]
        if not path.is_file():
            continue
        ck = torch.load(path, map_location="cpu", weights_only=True)
        if (ck.get("configuration") or {}).get("lookback") != LOOKBACK:
            raise SystemExit(f"{name}: checkpoint lookback is not {LOOKBACK}")
    data = Data(W0W2, args.cache, LOOKBACK, 0, DEFAULT["evaluation_end"])
    valid_days = data.days("valid")
    listed = listing_eligible(data.dates, data.symbols)
    feed = load_execution_feed(data, FEED)
    existing = json.loads(OUT_JSON.read_text(encoding="utf-8")) if OUT_JSON.is_file() else {}
    for name in wanted:
        path = CHECKPOINTS[name]
        if not path.is_file():
            print(f"SKIP {name}: {path} not on disk", flush=True)
            continue
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        config = validate_config(dict(DEFAULT, **ckpt["configuration"]))
        model = make_model(data, config).to(device)
        model.load_state_dict(ckpt["state_dict"])
        scores = predict(model, data, valid_days, config, device)[..., 0]
        entry = dict(checkpoint=str(path), sha256=sha256_file(path),
                     group=ckpt.get("group"), seed=ckpt.get("seed"),
                     data_version=ckpt.get("data_version"),
                     trained_tau=ckpt["configuration"].get("tau"),
                     trained_alpha_aux=ckpt["configuration"].get("alpha_aux"),
                     n_valid_days=len(valid_days), by_tau={})
        for tau in TAUS:
            neff, maxw, ncand = concentration(data, valid_days, scores, tau, listed)
            net = monitoring_mean_log(data, valid_days, scores, tau, feed,
                                      0.0005, 0.0015, 5.0, 100, 100000000.0)
            gross = monitoring_mean_log(data, valid_days, scores, tau, feed,
                                        0.0, 0.0, 0.0, 100, 100000000.0)
            entry["by_tau"][f"{tau:.2f}"] = dict(
                neff=stats(neff), max_weight=stats([w for w in maxw]),
                n_candidates=stats(ncand),
                monitoring_real_fee_mean_log=net,
                monitoring_zero_fee_mean_log=gross,
                val_cost_drag_bp_day=(gross - net) * 1e4)
            print(
                f"{name} tau={tau}: N_eff med={entry['by_tau'][f'{tau:.2f}']['neff']['median']:.1f} "
                f"maxw med={entry['by_tau'][f'{tau:.2f}']['max_weight']['median'] * 100:.3f}% "
                f"net={net:.8f} gross={gross:.8f}",
                flush=True)
        existing[name] = entry
        OUT_JSON.write_text(json.dumps(existing, indent=1), encoding="utf-8")
    print(f"WROTE {OUT_JSON} entries={sorted(existing)}", flush=True)


if __name__ == "__main__":
    main()
