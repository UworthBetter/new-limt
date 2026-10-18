# Experiment report: clean-pipeline preparations for cost-aware LiMT

Date: 17 October 2026. Updated the same day after the selection-metric code change and the behavioral A-G0 rerun.  
Governing protocol: `PREREGISTRATION_v0.5.md` (draft). The preregistration text was not edited.  
This note records what was run and what the code now does. It is not the paper, and it is not a Gate A or Gate B freeze.

## Question this round actually answered

The preregistration fixes a clean label calendar and a cost-aware training comparison on CSI300 and CSI500. This round did not estimate the primary contrast, Delta. It did four things:

1. Checked the training code against the locked definitions, then changed the checkpoint rule so it follows Section 6.1.
2. Rebuilt the clean CSI500 and CSI300 panels and passed W2.
3. Timed one full-universe CSI500 epoch.
4. Ran the CSI300 CPU baselines on both labels and all eight execution rules, with 2023–2026 returns sealed.

No 2023–2026 model return was opened. Sealed files were written and left unread.

## Code check, then the selection-metric fix

The first comparison found `C:\Users\admin\Downloads\v2_w4_training.py` and `D:\research\limt-repro\src\v2_w4_training.py` identical: 15,074 bytes, SHA256 `7647b83640eceb4cddfb3dd53cc51786ec753f4d5b51d4613fe531b8be74a97a`. That hash describes the file before the edit below. The Downloads copy was not updated. The working copy is `limt-repro\src\v2_w4_training.py`.

These points already matched the preregistration and were left as they were:

- H loss weights are 0.65 / 0.30 / 0.05 on cross-sectional z-scores, with squared error.
- The optimizer is Adam. Epoch caps in the merged formal config are 50 and 20, with patience 8.
- H-FT trains the supervised loss only.
- G0 and C0 share `train()`. Inside the differentiable ledger, G0’s proportional fees are zero and C0 uses the configured buy and sell rates.
- Capital is 100,000,000, lot size is 100, fees are 5 bp buy and 15 bp sell with a CNY 5 minimum, top-k is 50, and the default drop is 5.
- Delistings are SH600837 on 2025-03-04 and SH601989 on 2025-09-05.
- The h5 label is close(t+6) / close(t+1) − 1. The 1-day label r1 is close(t+2) / close(t+1) − 1.

The checkpoint rule did not match Section 6.1. Fine-tunes were selecting on a continuous-weight replay with proportional fees only. That replay has no CNY 5 minimum and no 100-share lot, and it did not log a zero-fee validation value beside the selection metric. The code was changed. The preregistration was not.

What the training code does now:

- H still selects the checkpoint by validation supervised loss.
- H-FT, G0, C0, and C-T select by validation after-cost mean log growth on a separate share ledger, `monitoring_mean_log`. Targets are the soft portfolio at the arm’s tau. Fills use buy 5 bp, sell 15 bp, minimum CNY 5, and lot 100. The shared checkpoint key is `valid_common_net_mean_log_max`.
- The same function is also called with zero fees. That value is stored as `validation_zero_fee_mean_log`. It is not the selection metric and it does not enter patience.
- A formal fine-tune with no execution feed raises, instead of falling back to adjusted closes. Smoke runs may build a quote grid from `data.close` so the unit tests can execute the share ledger.
- The gradient step is unchanged. G0’s training ledger is still the continuous zero-fee proxy. C0’s training ledger is still the continuous proxy at 5 bp / 15 bp. Minimum fee and lot size are applied on the monitoring ledger, because lot rounding is not differentiated.

`resolved_arm_config` records the two schedules. For G0 and C0 the resolved dicts differ only in `training_fee`.

## A-G0 rerun

`tests/test_a_g0_a_lbl.py` was rewritten so A-G0-1 through A-G0-4 run a short training job. They no longer pass by matching source text. The file was run with the limt-repro virtual environment: 5 passed in 12.7 seconds, including A-LBL-1.

| Assertion | What was checked | Result |
|---|---|---|
| A-G0-1 | G0 training steps all use fees (0, 0). `monitoring_mean_log` is called twice per epoch. `execute_close` sees (0, 0, min fee 0, lot 100) and then (5 bp, 15 bp, min fee 5, lot 100). | Pass |
| A-G0-2 | G0 and C0 checkpoints both use `valid_common_net_mean_log_max`. Every `selection_value` equals `validation_real_fee_mean_log`. The real-fee arguments match across the two arms. | Pass |
| A-G0-3 | Three scripted epochs. Zero-fee logs are 0.02, 0.05, 0.01, so the zero-fee peak is epoch 2. Real-fee logs are 0.01, −0.02, 0.04, so the real-fee peak is epoch 3. The saved checkpoint has `validation_selection` 0.04, not 0.05. | Pass |
| A-G0-4 | Resolved G0 and C0 configs differ only in `training_fee`. Recorded training fees are (0, 0) for G0 and (0.0005, 0.0015) for C0. | Pass |

A-LBL-1, on the fix1 CSI300 label file: SZ002074 on 2018-12-26 has r1 = 2.03236% with numerator date 2018-12-28. SH600549 on 2019-06-26 has r1 = 1.27604% with numerator date 2019-06-28. Both signals enter at t+1, exit h5 at t+6, and match close(t+6) / close(t+1) − 1.

A synthetic end-to-end smoke of `run_pipeline` (`test_full_cpu_pipeline_and_gamma_zero_reuse`) also passed after the edit, in 15.3 seconds. That smoke is not a formal result.

## Decisions locked before further training code

Recorded in `protocol/structural_decisions.md`.

- S5. H-FT is the existing group name. No second flag. Its checkpoint now uses the same real-fee monitoring ledger as G0 and C0. A formal H-FT job has not been launched.
- S6. Daily files use the key market / backbone / label / rule / seed. `Delta_cost = drag(G0*) − drag(C0*)`.
- S7. Accounting dates on or after 2023-01-01 may be written under `runs/_sealed/`. Readers do not open them before `protocol/gate_b.json` exists. That file does not exist.
- S9. A later `Data` loader must take `market`. CSI500 passes on endpoint misalignment zero, a passed purge, and evaluation end 2026-06-30. The two named counterexamples are CSI300 audit points.
- S10. This id was not in v0.5. It names the label axis `{h5, r1}` and the eight rules `drop{2,5,10,20}-every{1,5}d`. Neural arms stay on h5. `default` means `drop5-every1d`.

## Data gates

The usable provider is `D:\research\limt-repro\data_new\cn_data`. The older `qlib_cn_data` calendar ends on 2020-09-25 and was not used.

### CSI500

W1 took 811 seconds. The calendar has 4,707 sessions from 2007-06-01 through 2026-10-13, and 1,802 symbols ever in the index. The truncation probe passed (`schema_version` 2, max absolute feature difference 0). The panel is about 5.4 GB, at `data/csi500_w1w2/data/v2/`.

W2 passed. Endpoint misalignment is 0 for r1, h5, zv, and zs, in total and after standardization. Purge is intact. The evaluation split ends on 2026-06-30, and the last label end in that split is the same date. The label calendar contains 843 signal days in that window. The preregistration text says 842 trading days. W2 does not treat that difference as a failure.

The two named counterexamples match the published r1 percentages and numerator dates. On those signal dates their CSI500 universe flag and label mask are false, so they are not in-index samples that day. The check only requires the price identity, and it passed.

### CSI300

The local fix1 tree had labels but not the feature panel. W1 was rerun into `data/csi300_w1w2/` and finished in 400 seconds, with 882 symbols and a passed truncation probe. W2 then passed, including the same two counterexamples with universe and label mask true, and endpoint misalignment 0. The execution feed `new_execution_feed.npz` has the same 4,707 dates and the same 882 symbols.

## One CSI500 epoch (N6)

One H epoch was timed on the full 1,802-name panel, not on a 32-name smoke. Shape: dimension 256, lookback 21, `batch_days` 4, AMP off, seed 1, one epoch. The loader still asks for `csi300` filenames, so the run used hardlinks to the CSI500 artifacts and did not copy the panel.

| Item | Value |
|---|---|
| Device | NVIDIA GeForce RTX 4050 Laptop GPU |
| Train signal days | 2,671 |
| Valid signal days | 485 |
| Updates | 668 |
| Panel load | 10.5 s |
| Epoch wall time | 510.4 s |
| Peak allocated memory | 2,947 MiB |

The 510.4 seconds include the validation replay inside `train()`. That replay is 2019–2020, and it predates the monitoring-ledger edit, so this timing is not the cost of the new share-ledger selector. No evaluation-period report was written. The preregistration’s 204–245 s/epoch figure is a CSI300 run on a Kaggle T4. These are different GPUs, so 510 seconds is the CSI500 measurement on this machine, not a same-device ratio.

The preregistration’s N6 blank was not edited.

## CPU baselines (CSI300 only)

Script: `src/cpu_baselines.py`. Wall time 1,593 seconds.

Arms: ridge (seed 1), LightGBM regression, and LightGBM LambdaRank (seeds 1, 2, and 3). Labels: h5 and r1. Rules: all eight top-k 50 grids, with n_drop in {2, 5, 10, 20} and rebalance every 1 or 5 signal days. On other days the share account is marked and held. Costs are the formal schedule: 5 bp buy, 15 bp sell, minimum CNY 5, lot 100, capital 100,000,000. The book is the share ledger in `v2_w3_ledger.py`, with the CSI300 execution feed (raw close, explicit corporate actions, historical limit and ST state).

LightGBM uses the Qlib Alpha158 tree parameters (`num_leaves` 210, learning rate 0.2, and the published L1/L2 values), 1,000 rounds, early stopping 50 on the 2019–2020 validation window. Regression fits the z-scored label. LambdaRank uses the within-day rank of that z-score as relevance, with linear label gains. Best iterations differ by seed, so the fits are not interchangeable and three seeds were kept.

| Backbone | Label | Best iteration by seed 1, 2, 3 | Training rows |
|---|---|---|---|
| ridge | h5 | deterministic | 738,742 |
| ridge | r1 | deterministic | 768,338 |
| lgbm-reg | h5 | 11, 6, 10 | same h5 rows as ridge |
| lgbm-rank | h5 | 13, 14, 27 | same h5 rows as ridge |
| lgbm-reg | r1 | 10, 5, 4 | same r1 rows as ridge |
| lgbm-rank | r1 | 4, 4, 2 | same r1 rows as ridge |

Early stopping fired. The round cap was not changed.

Rule choice uses only accounting dates before 2023-01-01. The score is the arithmetic mean of daily excess over H00300 total return, times 252. Section 5 requires one rule per arm: the metric is averaged over seeds before the choice. Ties within 0.5 percentage points go to the lower mean turnover, then to the earlier rule in the fixed grid.

The first pass of the runner chose a rule inside each seed. That disagreed with Section 5. The runner was corrected. The 2021–2022 grids already on disk were averaged. The backtests were not replayed, and the sealed files were not read. `runs/csi300_cpu/selection.json` now stores the arm-level choice.

| Backbone | Label | Seeds | Chosen rule |
|---|---|---|---|
| ridge | h5 | 1 | drop5-every5d |
| ridge | r1 | 1 | drop2-every1d |
| lgbm-reg | h5 | 1, 2, 3 | drop10-every5d |
| lgbm-reg | r1 | 1, 2, 3 | drop5-every1d |
| lgbm-rank | h5 | 1, 2, 3 | drop2-every1d |
| lgbm-rank | r1 | 1, 2, 3 | drop5-every5d |

There are 112 readable daily files under `runs/csi300/`. None of them contains a date on or after 2023-01-01. An example file has 484 rows and ends on 2022-12-30. There are 112 sealed daily files under `runs/_sealed/csi300/`. Their returns were not read. The selection grid used to break ties stays in `runs/csi300_cpu/` and is not copied into `results/`.

## CSI500 baselines

Not run. The only execution feed on disk is the 882-name CSI300 book. A continuous-weight, proportional-fee replay would not be the Section 5 rule, so it was not substituted.

## Not done

- Formal GPU fine-tunes, the tau grid {0.5, 0.75, 1.0}, seeds 2 and 3, and C-T have not been run. A-G0 now passes on the short toy. That does not start those jobs.
- The N6 timing was measured before the monitoring ledger existed, so it does not include two share-ledger replays per epoch.
- sigma_d, MDE80, and the Section 4.3 tier have not been computed.
- Gate B has not been frozen. `protocol/gate_b.json` does not exist.
- Section 16.5 prior-art checks, including the authors’ repository backtest configuration, were not done.
- Blank fields in the preregistration were left blank.
- `Data()` still opens `csi300` filenames. The CSI500 epoch used hardlinks.

## Artefacts

| Path | What it is |
|---|---|
| `protocol/structural_decisions.md` | S5–S10 |
| `protocol/checks/verify_in_code_6.5.md` | Training-code check from before the selection-metric edit |
| `protocol/checks/n6_csi500_epoch.json` | CSI500 epoch timing |
| `tests/test_a_g0_a_lbl.py` | Behavioral A-G0-1..4 and A-LBL-1 |
| `limt-repro/src/v2_w4_training.py` | Working trainer, with the monitoring ledger |
| `data/csi500_w1w2/` | CSI500 labels, panel, W2 acceptance |
| `data/csi300_w1w2/` | Rebuilt CSI300 labels, panel, W2 acceptance |
| `runs/csi300/` | Pre-2023 baseline daily files |
| `runs/_sealed/csi300/` | Evaluation-period daily files, unread |
| `runs/csi300_cpu/selection.json` | One chosen rule per arm |
| `src/cpu_baselines.py` | Baseline runner, seed-averaged selection |
| `src/time_csi500_epoch.py` | N6 timer |
