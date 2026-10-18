"""Formal CSI300 H, seed 1. Validation loss only. No evaluation-window files."""
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
OUT = Path(r"D:\research\new_LiMT\runs\csi300_H_seed1_20261017")
W0W2 = Path(r"D:\research\new_LiMT\data\csi300_w1w2")
FORMAL = Path(r"D:\research\limt-repro\configs\w4w6_formal.json")
ARTIFACTS = (
    "data/v2/new_v2_labels_csi300.npz",
    "data/v2/new_v2_panel_csi300.npz",
    "data/v2/new_v2_splits_csi300.npz",
)


def sha256_file(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def data_version(root):
    """Identity of the three arrays Data() loads. This tree has no W1 manifest."""
    hasher = hashlib.sha256()
    for name in ARTIFACTS:
        hasher.update(name.encode())
        hasher.update(b"\0")
        hasher.update(sha256_file(root / name).encode())
        hasher.update(b"\n")
    return hasher.hexdigest()


def main():
    training_hash = sha256_file(TRAINING)
    if training_hash != EXPECTED_TRAINING_SHA256:
        raise SystemExit(f"training file hash is {training_hash}")
    supplied = json.loads(FORMAL.read_text(encoding="utf-8-sig"))
    config = validate_config(dict(DEFAULT, **supplied))
    expected = dict(mode="formal", dimension=256, lookback=21, epochs_h=50, patience=8,
                    seeds=[1], max_symbols=0, train_days=0, valid_days=0, device="cuda", amp=False)
    for key, value in expected.items():
        if config[key] != value:
            raise SystemExit(f"formal config {key}={config[key]!r}")
    if not torch.cuda.is_available():
        raise SystemExit("formal H requires CUDA")
    identity = data_version(W0W2)
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "data_version.txt").write_text(identity + "\n", encoding="utf-8")
    os.environ["OMP_NUM_THREADS"] = "8"
    torch.set_num_threads(8)
    location = OUT / "H" / "seed1"
    location.mkdir(parents=True)
    data = Data(W0W2, OUT / "cache", config["lookback"], 0, config["evaluation_end"])
    print(f"START H seed=1 data_version={identity} -> {location}", flush=True)
    train(data, config, "H", 1, location, "cuda", identity, None, None)
    checkpoint = torch.load(location / "checkpoint.pt", map_location="cpu", weights_only=True)
    history = json.loads((location / "history.json").read_text(encoding="utf-8"))
    chosen = min(history, key=lambda row: row["validation_aux"])
    print(
        f"CHECKPOINT group={checkpoint['group']} seed={checkpoint['seed']} "
        f"data_version={checkpoint['data_version']} selection={checkpoint['selection']} "
        f"mode={checkpoint['configuration']['mode']} dimension={checkpoint['configuration']['dimension']} "
        f"lookback={checkpoint['configuration']['lookback']} epochs_h={checkpoint['configuration']['epochs_h']} "
        f"patience={checkpoint['configuration']['patience']}",
        flush=True,
    )
    print(
        f"DONE epochs_run={len(history)} selected_epoch={chosen['epoch']} "
        f"selected_val_aux={chosen['validation_aux']:.8g} "
        f"last_epoch={history[-1]['epoch']} last_val_aux={history[-1]['validation_aux']:.8g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
