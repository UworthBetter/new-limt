"""CSI500 H, seeds 1-3, on Kaggle. Formal config, sha-pinned trainer.

Same formal configuration as the CSI300 runs (config hash must equal
393b1993...); only the seed argument differs. The Data() loader reads
csi300-named files, so the caller (kaggle_run.sh) exposes the CSI500
artifacts under those names via symlinks -- the same convention as the
local N6 timing run (experiment_report_2026-10-17.md).

H trains the supervised loss only: no execution feed is needed. CSI500
reuses the CSI300 configuration with no tuning (preregistration 6.2).

Resumable: seeds whose checkpoint already exists are skipped, and the
result zip is refreshed after every seed, so a session that hits the 12h
cap loses nothing.

Usage: run_h_csi500_seed.py --root W0W2 --out-root OUT --seeds 1 2 3
"""
import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from v2_w4_data import Data  # noqa: E402
from v2_w4_training import train  # noqa: E402
from v2_w6_run import DEFAULT, validate_config  # noqa: E402

TRAINING = CODE / "v2_w4_training.py"
EXPECTED_TRAINING_SHA256 = "a960a9951ab52c5455c15f8e0c5d60a783a6beb4beef5b4003ee578f0c21ee43"
FORMAL = CODE / "w4w6_formal.json"
ARTIFACTS = (
    "data/v2/new_v2_labels_csi300.npz",
    "data/v2/new_v2_panel_csi300.npz",
    "data/v2/new_v2_splits_csi300.npz",
)
EXPECTED_CONFIG_HASH = "393b19932ea61cdb542cf970cf23a0e82741a3b59f7a8b76be7f9083ac3230d9"


def sha256_file(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def data_version(root):
    hasher = hashlib.sha256()
    for name in ARTIFACTS:
        hasher.update(name.encode())
        hasher.update(b"\0")
        hasher.update(sha256_file(root / name).encode())
        hasher.update(b"\n")
    return hasher.hexdigest()


def refresh_zip(out_root):
    base = out_root / "runs"
    target = out_root / "csi500_H_kaggle_results"
    if target.with_suffix(".zip").exists():
        target.with_suffix(".zip").unlink()
    shutil.make_archive(str(target), "zip", str(base))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    args = parser.parse_args()
    training_hash = sha256_file(TRAINING)
    if training_hash != EXPECTED_TRAINING_SHA256:
        raise SystemExit(f"training file hash is {training_hash}")
    if not torch.cuda.is_available():
        raise SystemExit("formal H requires CUDA")
    supplied = json.loads(FORMAL.read_text(encoding="utf-8-sig"))
    config = validate_config(dict(DEFAULT, **supplied))
    expected = dict(mode="formal", dimension=256, lookback=21, epochs_h=50, patience=8,
                    seeds=[1], max_symbols=0, train_days=0, valid_days=0, device="cuda", amp=False)
    for key, value in expected.items():
        if config[key] != value:
            raise SystemExit(f"formal config {key}={config[key]!r}")
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if config_hash != EXPECTED_CONFIG_HASH:
        raise SystemExit(f"config hash {config_hash} != {EXPECTED_CONFIG_HASH}")
    identity = data_version(args.root)
    print(
        f"ENV python={platform.python_version()} torch={torch.__version__} "
        f"numpy={np.__version__} gpu={torch.cuda.get_device_name(0)} "
        f"cuda={torch.version.cuda}",
        flush=True,
    )
    print(f"CSI500 DATA_VERSION {identity}", flush=True)
    print(f"CONFIG_HASH {config_hash} (identical to the CSI300 formal runs)", flush=True)
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "data_version.txt").write_text(identity + "\n", encoding="utf-8")
    os_env = {"OMP_NUM_THREADS": "4"}
    import os
    for key, value in os_env.items():
        os.environ.setdefault(key, value)
    torch.set_num_threads(4)
    data = Data(args.root, args.out_root / "cache", config["lookback"], 0, config["evaluation_end"])
    manifest_path = args.out_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else []
    done = {row["seed"]: row for row in manifest}
    for seed in args.seeds:
        location = args.out_root / "runs" / f"csi500_H_seed{seed}_kaggle" / "H" / f"seed{seed}"
        if (location / "checkpoint.pt").is_file():
            print(f"SKIP seed {seed}: checkpoint already present", flush=True)
            continue
        location.mkdir(parents=True, exist_ok=False)
        started = time.perf_counter()
        print(f"START H seed={seed} data_version={identity} -> {location}", flush=True)
        train(data, config, "H", seed, location, "cuda", identity, None, None)
        history = json.loads((location / "history.json").read_text(encoding="utf-8"))
        chosen = min(history, key=lambda row: row["validation_aux"])
        ckpt_sha = sha256_file(location / "checkpoint.pt")
        row = dict(seed=seed, epochs_run=len(history), selected_epoch=chosen["epoch"],
                   selected_val_aux=chosen["validation_aux"], checkpoint_sha256=ckpt_sha,
                   data_version=identity, config_hash=config_hash, seconds=time.perf_counter() - started)
        done[seed] = row
        manifest = [done[s] for s in sorted(done)]
        manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        print(f"DONE H seed={seed} epochs_run={len(history)} selected_epoch={chosen['epoch']} "
              f"selected_val_aux={chosen['validation_aux']:.8g} sha256={ckpt_sha}", flush=True)
        refresh_zip(args.out_root)
    print("ALL REQUESTED SEEDS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
