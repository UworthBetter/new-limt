#!/usr/bin/env bash
# Overnight queue: wait for the seeds 2-3 chain + scoring, then GPU benchmark
# models, then the full-seed neural grid, the six-model benchmark grid and the
# full dose diagnostics. Launched 2026-10-18 after the DL/Gate A commits.
cd /d/research/new_LiMT
while [ ! -f runs/scores/C0_seed3.npz ]; do sleep 300; done
sleep 120
./venv_bench/Scripts/python.exe src/run_benchmarks.py --cache runs/bench_gpu_cache --models LSTM GRU ALSTM Transformer > logs/bench_gpu_20261019.log 2>&1
py -3.13 src/rule_grid_neural.py --cache runs/grid_full3seed_cache --tags H_seed1 H_seed2 H_seed3 H-FT_seed1 H-FT_seed2 H-FT_seed3 G0_seed1 G0_seed2 G0_seed3 C0_seed1 C0_seed2 C0_seed3 G0dose003_seed1 C0dose003_seed1 > logs/rule_grid_full_20261019.log 2>&1
py -3.13 src/rule_grid_neural.py --label r1 --cache runs/grid_bench_all_cache --tags XGBoost_seed1 DoubleEnsemble_seed1 LSTM_seed1 GRU_seed1 ALSTM_seed1 Transformer_seed1 --out runs/csi300_bench_selection.json > logs/rule_grid_bench_all_20261019.log 2>&1
py -3.13 src/dose_diag.py --cache runs/diag_full_cache --tags H_seed1 H_seed2 H_seed3 H-FT_seed1 H-FT_seed2 H-FT_seed3 G0_seed1 G0_seed2 G0_seed3 C0_seed1 C0_seed2 C0_seed3 G0dose003_seed1 C0dose003_seed1 > logs/dose_diag_full_20261019.log 2>&1
echo QUEUE_ALL_DONE
