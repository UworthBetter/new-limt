#!/usr/bin/env bash
# Kaggle CSI500 H x seeds 1-3. Attach the uploaded dataset, enable GPU (T4 or P100), run all.
# Dataset layout is FLAT (kaggle CLI skips subdirectories): all code files and the
# three CSI500 npz artifacts sit at the dataset root.
set -euo pipefail
SRC="/kaggle/input/csi500-newlimt"
WORK="/kaggle/working"
mkdir -p "$WORK/w0w2/data/v2" "$WORK/runs_kaggle"
# The Data() loader reads csi300 filenames; expose the CSI500 artifacts under those names.
ln -sf "$SRC/new_v2_panel_csi500.npz"  "$WORK/w0w2/data/v2/new_v2_panel_csi300.npz"
ln -sf "$SRC/new_v2_labels_csi500.npz" "$WORK/w0w2/data/v2/new_v2_labels_csi300.npz"
ln -sf "$SRC/new_v2_splits_csi500.npz" "$WORK/w0w2/data/v2/new_v2_splits_csi300.npz"
python "$SRC/run_h_csi500_seed.py" --root "$WORK/w0w2" --out-root "$WORK/runs_kaggle" --seeds 1 2 3
ls -la "$WORK/runs_kaggle"
