"""G0/C0 seed 1, tau in {0.5, 0.75, 1.0}. Validation ledger only."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

LIMT_SRC = Path(r"D:\research\limt-repro\src")
sys.path.insert(0, str(LIMT_SRC))

from v2_w4_data import Data
from v2_w4_training import train
from v2_w6_run import DEFAULT, validate_config

TRAINING = LIMT_SRC / "v2_w4_training.py"
EXPECTED_TRAINING_SHA256 = "a960a9951ab52c5455c15f8e0c5d60a783a6beb4beef5b4003ee578f0c21ee43"
OUT = Path(r"D:\research\new_LiMT\runs\csi300_g0c0_seed1_tau_20261017")
W0W2 = Path(r"D:\research\new_LiMT\data\csi300_w1w2")
FORMAL = Path(r"D:\research\limt-repro\configs\w4w6_formal.json")
FEED = Path(r"D:\research\limt-repro\data\v2_formal_inputs_run3\new_execution_feed.npz")
ARTIFACTS = (
    "data/v2/new_v2_labels_csi300.npz",
    "data/v2/new_v2_panel_csi300.npz",
    "data/v2/new_v2_splits_csi300.npz",
)
GROUPS = ("G0", "C0")
TAUS = (0.5, 0.75, 1.0)
TAGS = {0.5: "tau0.50", 0.75: "tau0.75", 1.0: "tau1.00"}


def sha256_file(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def panel_version(root):
    hasher = hashlib.sha256()
    for name in ARTIFACTS:
        hasher.update(name.encode())
        hasher.update(b"\0")
        hasher.update(sha256_file(root / name).encode())
        hasher.update(b"\n")
    return hasher.hexdigest()


def load_config():
    supplied = json.loads(FORMAL.read_text(encoding="utf-8-sig"))
    return validate_config(dict(DEFAULT, **supplied))


def require_clean_h(path, identity):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint.get("configuration") or {}
    reasons = []
    if checkpoint.get("group") != "H" or checkpoint.get("seed") != 1:
        reasons.append("group/seed is not H seed 1")
    if config.get("mode") != "formal" or config.get("dimension") != 256 or config.get("lookback") != 21:
        reasons.append("not the formal d=256 lookback=21 schedule")
    if config.get("max_symbols") not in (0, None) or config.get("evaluation_end") != "2026-06-30":
        reasons.append("not the full-universe 2026-06-30 configuration")
    if checkpoint.get("data_version") != identity:
        reasons.append(f"data_version {checkpoint.get('data_version')} != panel {identity}")
    if reasons:
        raise SystemExit("refusing H init: " + "; ".join(reasons))
    return checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h-checkpoint", required=True, type=Path)
    args = parser.parse_args()
    training_hash = sha256_file(TRAINING)
    if training_hash != EXPECTED_TRAINING_SHA256:
        raise SystemExit(f"training file hash is {training_hash}")
    if not FEED.is_file():
        raise SystemExit(f"missing execution feed: {FEED}")
    if not torch.cuda.is_available():
        raise SystemExit("formal fine-tune requires CUDA")
    identity = panel_version(W0W2)
    require_clean_h(args.h_checkpoint, identity)
    base = load_config()
    if base["seeds"] != [1] or base["epochs_finetune"] != 20 or base["patience"] != 8:
        raise SystemExit("formal config is not seed 1 / 20 epochs / patience 8")
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "data_version.txt").write_text(identity + "\n", encoding="utf-8")
    os.environ["OMP_NUM_THREADS"] = "8"
    torch.set_num_threads(8)
    data = Data(W0W2, OUT / "cache", base["lookback"], 0, base["evaluation_end"])
    for group in GROUPS:
        for tau in TAUS:
            config = validate_config(dict(base, tau=float(tau)))
            location = OUT / group / TAGS[tau] / "seed1"
            location.mkdir(parents=True)
            print(f"START {group} seed=1 tau={tau} data_version={identity} -> {location}", flush=True)
            train(data, config, group, 1, location, "cuda", identity, args.h_checkpoint, FEED)
            history = json.loads((location / "history.json").read_text(encoding="utf-8"))
            chosen = max(history, key=lambda row: row["validation_real_fee_mean_log"])
            print(
                f"DONE {group} seed=1 tau={tau} epochs_run={len(history)} "
                f"selected_epoch={chosen['epoch']} "
                f"val_net_log={chosen['validation_real_fee_mean_log']:.8g} "
                f"monitoring_mean_log={chosen['validation_real_fee_mean_log']:.8g}",
                flush=True,
            )
    print("DONE grid runs=6", flush=True)


if __name__ == "__main__":
    main()
