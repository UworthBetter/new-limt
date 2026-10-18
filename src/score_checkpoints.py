"""Predict-and-save pass for neural checkpoints (valid + rule_select windows).

Writes runs/scores/<tag>.npz with dates/scores/candidate plus checkpoint
sha256 and config echo, so every later replay (8-rule grid, diagnostics,
sealing) is CPU-only and reproducible from a pinned artefact.  One Data()
per invocation; the caller passes a fresh --cache dir.

Usage: score_checkpoints.py --cache DIR [--tags tag ...] [--windows valid,rule_select]
Tags: H_seed1 H-FT_seed1 G0_seed1 C0_seed1 G0dose003_seed1 C0dose003_seed1
      H_seed2 H-FT_seed2 G0_seed2 C0_seed2 H_seed3 H-FT_seed3 G0_seed3 C0_seed3
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
from v2_w4_training import make_model, predict
from v2_w6_run import DEFAULT, validate_config

ROOT = Path(r"D:\research\new_LiMT")
RUNS = ROOT / "runs"
SEED1_H = RUNS / "csi300_H_seed1_20261017"
SEED1_G0C0 = RUNS / "csi300_g0c0_seed1_tau_20261017"
SEED1_HFT = RUNS / "csi300_HFT_seed1_20261018"
SEED1_DOSE = RUNS / "csi300_dose003_seed1_20261018"


def resolve(tag):
    kind, _, seed = tag.rpartition("_seed")
    if kind == "H":
        return RUNS / f"csi300_H_seed{seed}_20261017" / "H" / f"seed{seed}" / "checkpoint.pt"
    if kind == "H-FT":
        return RUNS / f"csi300_HFT_seed{seed}_20261018" / "H-FT" / f"seed{seed}" / "checkpoint.pt"
    if kind in ("G0", "C0"):
        base = SEED1_G0C0 if seed == "1" else RUNS / f"csi300_g0c0_seed{seed}_tau050_20261018"
        return base / kind / "tau0.50" / f"seed{seed}" / "checkpoint.pt"
    if kind in ("G0dose003", "C0dose003"):
        return SEED1_DOSE / kind[:2] / "tau0.50" / "seed1" / "checkpoint.pt"
    raise SystemExit(f"unknown tag {tag}")


def sha256_file(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--tags", nargs="*", default=None)
    parser.add_argument("--windows", default="valid,rule_select")
    args = parser.parse_args()
    wanted = args.tags or [
        "H_seed1", "H-FT_seed1", "G0_seed1", "C0_seed1", "G0dose003_seed1", "C0dose003_seed1",
        "H_seed2", "H-FT_seed2", "G0_seed2", "C0_seed2",
        "H_seed3", "H-FT_seed3", "G0_seed3", "C0_seed3",
    ]
    paths = {tag: resolve(tag) for tag in wanted}
    missing = [tag for tag, path in paths.items() if not path.is_file()]
    for tag in missing:
        print(f"SKIP {tag}: {paths[tag]} not on disk", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = Data(ROOT / "data" / "csi300_w1w2", args.cache, 21, 0, DEFAULT["evaluation_end"])
    windows = {name: data.days(name) for name in args.windows.split(",") if name}
    out_dir = RUNS / "scores"
    out_dir.mkdir(exist_ok=True)
    for tag, path in paths.items():
        if tag in missing:
            continue
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        config = validate_config(dict(DEFAULT, **ckpt["configuration"]))
        expected_group = tag.rpartition("_seed")[0].replace("dose003", "")
        if ckpt.get("group") != expected_group:
            raise SystemExit(f"{tag}: checkpoint group is {ckpt.get('group')}")
        model = make_model(data, config).to(device)
        model.load_state_dict(ckpt["state_dict"])
        payload = {}
        for name, days in windows.items():
            scores = predict(model, data, days, config, device)[..., 0]
            payload[f"dates_{name}"] = data.dates[days].astype(str)
            payload[f"scores_{name}"] = scores.astype(np.float32)
            payload[f"candidate_{name}"] = np.stack([data.candidate(int(t)) for t in days])
        meta = dict(
            tag=tag, checkpoint=str(path), sha256=sha256_file(path),
            group=ckpt.get("group"), seed=ckpt.get("seed"),
            data_version=ckpt.get("data_version"),
            trained_tau=config["tau"], trained_alpha_aux=config["alpha_aux"],
            epochs_cap=(config["epochs_h"], config["epochs_finetune"]),
            selection=ckpt.get("selection"), validation_selection=ckpt.get("validation_selection"),
        )
        out = out_dir / f"{tag}.npz"
        with out.open("wb") as stream:
            np.savez_compressed(stream, **payload, **{k: v for k, v in meta.items() if isinstance(v, (str, int, float))})
        (out_dir / f"{tag}.meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(f"SCORED {tag} group={meta['group']} seed={meta['seed']} sha256={meta['sha256'][:16]}... -> {out}", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    print("SCORING PASS DONE", flush=True)


if __name__ == "__main__":
    main()
