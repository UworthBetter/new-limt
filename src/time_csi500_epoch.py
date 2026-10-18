"""Time one H epoch on the full CSI500 W1/W2 panel.

Data() still opens csi300 filenames. This script hardlinks the CSI500
artifacts under those names and does not copy the panel. The run is
train plus the validation pass inside train(). It does not write an
evaluation-period report.
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

LIMT_SRC = Path(r"D:\research\limt-repro\src")
sys.path.insert(0, str(LIMT_SRC))

from v2_w4_data import Data  # noqa: E402
from v2_w4_training import train  # noqa: E402
from v2_w6_run import DEFAULT  # noqa: E402

SRC = Path(r"D:\research\new_LiMT\data\csi500_w1w2\data\v2")
RUN = Path(r"D:\research\new_LiMT\runs\n6_csi500_1epoch")
ALIAS = {
    "new_v2_panel_csi500.npz": "new_v2_panel_csi300.npz",
    "new_v2_labels_csi500.npz": "new_v2_labels_csi300.npz",
    "new_v2_splits_csi500.npz": "new_v2_splits_csi300.npz",
}


def hardlink_alias(root: Path):
    directory = root / "data" / "v2"
    directory.mkdir(parents=True, exist_ok=True)
    for source_name, alias_name in ALIAS.items():
        source = SRC / source_name
        target = directory / alias_name
        if target.exists():
            continue
        os.link(source, target)


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the N6 epoch timing")
    hardlink_alias(RUN / "loader_root")
    config = dict(
        DEFAULT,
        mode="formal",
        seeds=[1],
        dimension=256,
        lookback=21,
        batch_days=4,
        max_symbols=0,
        train_days=0,
        valid_days=0,
        report_days=0,
        epochs_h=1,
        epochs_finetune=1,
        patience=0,
        amp=False,
        device="cuda",
        evaluation_end="2026-06-30",
    )
    started = time.perf_counter()
    data = Data(RUN / "loader_root", RUN / "cache", config["lookback"], 0, config["evaluation_end"])
    load_seconds = time.perf_counter() - started
    print(
        f"loaded symbols={len(data.symbols)} train_days={len(data.days('train', training=True))} "
        f"valid_days={len(data.days('valid'))} load_seconds={load_seconds:.1f}",
        flush=True,
    )
    out = RUN / "H_s1"
    out.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    epoch_started = time.perf_counter()
    train(data, config, "H", 1, out, "cuda", "csi500-w1w2-n6", None)
    torch.cuda.synchronize()
    epoch_seconds = time.perf_counter() - epoch_started
    history = json.loads((out / "history.json").read_text(encoding="utf-8"))
    record = dict(
        market="csi500",
        group="H",
        epochs=1,
        symbols=int(len(data.symbols)),
        train_days=int(len(data.days("train", training=True))),
        valid_days=int(len(data.days("valid"))),
        updates=history[-1]["updates"],
        batch_days=config["batch_days"],
        dimension=config["dimension"],
        lookback=config["lookback"],
        amp=False,
        device=torch.cuda.get_device_name(0),
        load_seconds=load_seconds,
        epoch_seconds=epoch_seconds,
        epoch_includes_validation_replay=True,
        peak_allocated_mib=torch.cuda.max_memory_allocated() / (1024 ** 2),
        filename_alias="Data() still requests csi300 names; files are hardlinks to the csi500 W1/W2 artifacts",
        evaluation_outputs_written=False,
    )
    text = json.dumps(record, indent=2)
    (RUN / "timing.json").write_text(text + "\n", encoding="utf-8")
    checks = Path(r"D:\research\new_LiMT\protocol\checks")
    checks.mkdir(parents=True, exist_ok=True)
    (checks / "n6_csi500_epoch.json").write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
