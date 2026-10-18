"""Seeds 2-3 chain for CSI300: H -> H-FT -> G0(tau 0.5) -> C0(tau 0.5).

Mirrors the seed-1 runners exactly (src/run_h_seed1.py, run_hft_seed1.py,
run_g0_c0_seed1_tau.py restricted to the seed-1-selected tau=0.5):
same formal config dict (identical config hash; only the `seed` argument of
train() differs), same trainer-sha pin, same H-init provenance logging.
Each arm gets its own run directory and its own Data() cache folder.

Registered route: Deviation Log 2026-10-18 option (b) (commit c8c6c27).
Usage: run_seed_chain.py --seed 2|3
"""
import argparse
import hashlib
import json
import os
import shutil
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
ROOT = Path(r"D:\research\new_LiMT")
W0W2 = ROOT / "data" / "csi300_w1w2"
FORMAL = Path(r"D:\research\limt-repro\configs\w4w6_formal.json")
FEED = Path(r"D:\research\limt-repro\data\v2_formal_inputs_run3\new_execution_feed.npz")
ARTIFACTS = (
    "data/v2/new_v2_labels_csi300.npz",
    "data/v2/new_v2_panel_csi300.npz",
    "data/v2/new_v2_splits_csi300.npz",
)
TAU = 0.5  # seed-1-selected, both arms (protocol/tau_grid_report_2026-10-18.md)


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


def fresh_dir(path):
    path.mkdir(parents=True, exist_ok=False)
    return path


def make_data(tag):
    # Data() itself mkdirs the cache with exist_ok=False; pass a fresh path.
    return Data(W0W2, ROOT / "runs" / f"csi300_{tag}_cache", 21, 0, "2026-06-30")


def require_clean_h(path, identity, seed):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint.get("configuration") or {}
    reasons = []
    if checkpoint.get("group") != "H" or checkpoint.get("seed") != seed:
        reasons.append(f"group/seed is not H seed {seed}")
    if config.get("mode") != "formal" or config.get("dimension") != 256 or config.get("lookback") != 21:
        reasons.append("not the formal d=256 lookback=21 schedule")
    if config.get("max_symbols") not in (0, None) or config.get("evaluation_end") != "2026-06-30":
        reasons.append("not the full-universe 2026-06-30 configuration")
    if checkpoint.get("data_version") != identity:
        reasons.append(f"data_version {checkpoint.get('data_version')} != panel {identity}")
    if reasons:
        raise SystemExit("refusing H init: " + "; ".join(reasons))
    return checkpoint


def report(group, seed, history_path):
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if group == "H":
        chosen = min(history, key=lambda row: row["validation_aux"])
        key = "validation_aux"
    else:
        chosen = max(history, key=lambda row: row["validation_real_fee_mean_log"])
        key = "validation_real_fee_mean_log"
    print(
        f"DONE {group} seed={seed} epochs_run={len(history)} selected_epoch={chosen['epoch']} "
        f"{key}={chosen[key]:.8g}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", required=True, type=int, choices=(2, 3))
    args = parser.parse_args()
    seed = args.seed
    training_hash = sha256_file(TRAINING)
    if training_hash != EXPECTED_TRAINING_SHA256:
        raise SystemExit(f"training file hash is {training_hash}")
    if not FEED.is_file():
        raise SystemExit(f"missing execution feed: {FEED}")
    if not torch.cuda.is_available():
        raise SystemExit("formal training requires CUDA")
    identity = panel_version(W0W2)
    supplied = json.loads(FORMAL.read_text(encoding="utf-8-sig"))
    base = validate_config(dict(DEFAULT, **supplied))
    expected = dict(mode="formal", dimension=256, lookback=21, epochs_h=50, patience=8,
                    seeds=[1], max_symbols=0, train_days=0, valid_days=0, device="cuda", amp=False)
    for key, value in expected.items():
        if base[key] != value:
            raise SystemExit(f"formal config {key}={base[key]!r}")
    # The config dict is passed to train() UNCHANGED from the seed-1 formal
    # config (config hash identical); only the seed argument differs.
    config_hash = hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest()
    os.environ["OMP_NUM_THREADS"] = "8"
    torch.set_num_threads(8)
    print(f"CHAIN START seed={seed} data_version={identity} config_hash={config_hash}", flush=True)

    out_h = fresh_dir(ROOT / "runs" / f"csi300_H_seed{seed}_20261018")
    (out_h / "data_version.txt").write_text(identity + "\n", encoding="utf-8")
    location = fresh_dir(out_h / "H" / f"seed{seed}")
    data = make_data(f"H_seed{seed}")
    print(f"START H seed={seed} data_version={identity} -> {location}", flush=True)
    train(data, base, "H", seed, location, "cuda", identity, None, None)
    report("H", seed, location / "history.json")
    h_checkpoint = location / "checkpoint.pt"
    h_sha = sha256_file(h_checkpoint)
    print(f"H CHECKPOINT seed={seed} sha256={h_sha}", flush=True)
    del data

    out_hft = fresh_dir(ROOT / "runs" / f"csi300_HFT_seed{seed}_20261018")
    (out_hft / "data_version.txt").write_text(identity + "\n", encoding="utf-8")
    require_clean_h(h_checkpoint, identity, seed)
    config_hft = validate_config(dict(base, tau=TAU))
    location = fresh_dir(out_hft / "H-FT" / f"seed{seed}")
    data = make_data(f"HFT_seed{seed}")
    print(
        f"START H-FT seed={seed} tau={TAU}(monitoring) data_version={identity} "
        f"init={h_checkpoint} init_sha256={h_sha} -> {location}",
        flush=True,
    )
    train(data, config_hft, "H-FT", seed, location, "cuda", identity, h_checkpoint, FEED)
    report("H-FT", seed, location / "history.json")
    del data

    out_g0c0 = fresh_dir(ROOT / "runs" / f"csi300_g0c0_seed{seed}_tau050_20261018")
    (out_g0c0 / "data_version.txt").write_text(identity + "\n", encoding="utf-8")
    data = make_data(f"g0c0_seed{seed}")
    for group in ("G0", "C0"):
        require_clean_h(h_checkpoint, identity, seed)
        config = validate_config(dict(base, tau=TAU))
        location = fresh_dir(out_g0c0 / group / "tau0.50" / f"seed{seed}")
        print(
            f"START {group} seed={seed} tau={TAU} data_version={identity} "
            f"init={h_checkpoint} init_sha256={h_sha} -> {location}",
            flush=True,
        )
        train(data, config, group, seed, location, "cuda", identity, h_checkpoint, FEED)
        report(group, seed, location / "history.json")
    print(f"CHAIN DONE seed={seed}", flush=True)


if __name__ == "__main__":
    main()
