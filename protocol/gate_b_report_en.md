# Pre-Gate-B Status Report — Cost-Aware Training for LiMT (Clean-Pipeline Study)

Workspace: `D:\research\new_LiMT`. Governing protocol: `PREREGISTRATION_v0.6.md`. All thresholds, grids and decision rules were fixed in advance; every deviation was logged before the deviating experiment ran. No evaluation-period (2023-01 to 2026-06-30) output has been opened. This report summarizes everything completed through Gate B readiness; `gate_b.json` has **not** been written and the 2023+ outputs remain sealed.

---

## 1. Executive summary

- The three-seed CSI300 matrix (H, H-FT, G0, C0 at the frozen tau = 0.5) is complete: 8/8 training runs, zero failures, identical resolved configuration across all runs (config hash `393b1993…`).
- Measured on executed books, sigma_d(final) = 1.19% annualized, MDE80 = 1.58 pp → **Tier P1** (assigned mechanically, Section 4.3).
- After seed-averaging, **both primary arms (G0 and C0) selected the default execution rule** (topk 50, drop 5, daily) outright, with no tie-break. The primary estimand therefore reduces to the default-rule paired difference, and SC3 (C0-default vs G0-tuned) coincides with it.
- Six Qlib-default benchmark arms (LSTM, GRU, ALSTM, Transformer, XGBoost, DoubleEnsemble) are trained on the clean panel with the clean 1-day label and have selected rules; these are descriptive arms outside the Holm family.
- CSI500 H is trained for all three seeds on Kaggle T4 GPUs and verified against three integrity anchors; this is Gate C material and does not block Gate B.
- The only remaining prerequisite for Gate B is the investigator's review of this report and the underlying artefacts.

## 2. Protocol integrity

- The Deviation Log entries (benchmark reframe with six Qlib-default arms; stamp-duty sensitivity with sell fee 10 bp from 2023-08-28, HKEX circular CT/137/23, primary schedule unchanged at buy 5 / sell 15 bp with CNY 5 minimum; TOST equivalence margin ±2.0 pp/yr on the same stationary bootstrap; rule-selection seeds option (b); diagnostics and code disclosures) were committed before any run they cover. Commit chain: baseline `8a7a4c45` → deviation log `c8c6c27` → Gate A fills `841f00be` → record `8820fa5`; subsequent runner and fix commits continue the chain.
- Gate A was completed explicitly **after** the seed-1 runs, with no backdating: all `[FILL]` fields of the preregistration were filled or explicitly marked open (authors' repository backtest configuration check treated as unspecified with sensitivity reported; hard write-up date left as a user decision). The Section 6.5 checklist and the five behavioral assertions (A-G0-1..4, A-LBL-1) were ticked on the trainer in force, sha256 `a960a995…`, with a 14/14 rerun recorded under `protocol/checks/`.
- sigma_d(prelim) = 6.78% (ridge_h5 vs H on executed books — an upper-bound-style proxy, since the pair consists of two different models; recorded as such).
- Integrity guard for every selection-window computation: signal days stop at split-last-minus-2, so the terminal accounting mark lands on 2022-12-30 and the last execution fill on 2022-12-29; both are asserted in every runner log, and no price dated after 2022-12-31 is read.

## 3. Power and mechanical tier

| Quantity | Value |
|---|---|
| sigma_d(final), annualized SD of the daily paired after-cost difference (C0 − G0, seed 1, tau 0.5, default rule, validation window, 486 paired days) | **1.189%** |
| MDE80 = 2.487 × sigma_d / sqrt(3.5) | **1.581 pp** |
| Mechanical tier (Section 4.3) | **P1** (MDE80 ≤ 2.5 pp) |

Diagnostics reported alongside: daily holdings overlap 96.12% (median 96.08%); rank correlation of the two score vectors over the candidate set 0.99975 (mean Spearman); executed-layer cost-drag difference C0 − G0 = **+0.019 pp/yr, i.e. C0 pays slightly more drag at the default rule on the validation window — the sign is opposite to the soft layer and is recorded as measured.**

## 4. Three-seed training results (CSI300)

All runs use the identical resolved formal configuration (hash `393b1993…`); only the seed argument of `train()` differs. Trainer file sha pinned and verified in every runner.

| Seed | H | H-FT | G0 | C0 (validation soft-layer monitoring value, mean log growth/day) |
|---|---|---|---|---|
| 1 | 11 epochs, ep3 selected | 0.00108742 | 0.00109438 | 0.00109494 |
| 2 | 11 epochs, ep3 selected | 0.00106132 | 0.00106691 | 0.00106799 |
| 3 | 11 epochs, ep3 selected | 0.00108067 | 0.00108432 | 0.00108599 |

Cross-seed regularities: H early-stops at 11 epochs with the checkpoint at epoch 3 in all three seeds; every fine-tune runs to the 20-epoch cap with the monitoring metric monotonically rising (cap selection, as in the seed-1 grid report). The soft-layer paired difference C0 − G0 is **positive in all three seeds** but of order 0.03–0.04 pp/yr — roughly 1/50 of the S1 threshold.

## 5. Tuned-execution rule selection (2021–2022, seed-averaged, then 0.5 pp lower-turnover tie-break)

| Arm | Selected rule | Seed-averaged annualized excess vs H00300 total return |
|---|---|---|
| **G0** | **drop5-every1d (the default rule)** | 20.69% (runner-up drop2-every1d 18.28%; margin 2.4 pp, no tie) |
| **C0** | **drop5-every1d (the default rule)** | 20.23% (runner-up drop2-every1d 19.25%; margin 0.98 pp, no tie) |
| H | drop20-every5d | 18.33% |
| H-FT | drop2-every1d | 20.03% (0.2 pp from the best rule → tie-break to lower turnover exercised for the first time) |
| G0dose003 (exploratory) | drop5-every1d | — |
| C0dose003 (exploratory) | drop2-every1d | — |

**Consequence for the estimand.** Because both primary arms converge on the default rule, Delta = C0* − G0* equals the default-rule paired difference, and secondary contrast SC3 (C0-default vs G0-tuned) coincides with it. The Gate B unsealing therefore evaluates one clean paired comparison at the executable layer, plus the always-reported default rule (identical here).

## 6. Qlib-default benchmark arms (descriptive; not in the Holm family)

Hyperparameters are pinned verbatim from the official qlib v0.9.6 `examples/benchmarks` Alpha158 configurations (`configs/benchmarks_qlib_defaults_v1.json`, sha256 `da5b7425…`); no tuning. Training uses the clean panel (FilterCol 20-column subset) with the clean 1-day label (z_r1, t+1 entry convention, same definition as the ridge/LightGBM 1-day cells); single run per model, seed 1 (Transformer keeps its configuration seed 0). Selected rules on 2021–2022:

| Model | Selected rule |
|---|---|
| LSTM | drop5-every1d |
| GRU | drop10-every5d |
| ALSTM | drop2-every5d |
| Transformer | drop5-every5d |
| XGBoost | drop2-every5d |
| DoubleEnsemble | drop2-every1d |

Declarations carried with these arms: MASTER is not included because no Qlib-contrib implementation exists (checked at v0.9.6 and main); qlib's CSRankNorm label processor is not re-applied (the clean label is used as defined); Qlib's published benchmark numbers come from different windows and its own data handler, so any comparison to published values must state the pipeline and window difference. Environment disclosure: pyqlib installed from the cp312 wheel with the version tag renamed (no cp313 wheel exists) with the compiled `qlib.data._libs` extensions replaced by unused pure-Python stubs; the qlib expression engine is never invoked by this study's code path.

## 7. Dose-arm diagnostics (validation 5-day IC; soft vs top-k on 2021–2022)

- 5-day IC (validation, h5 label): H 0.0382; H-FT 0.0556; G0 0.0549; C0 0.0548; G0dose003 0.0396; C0dose003 0.0435 (three-seed values in the dose-diagnostics record under `protocol/checks/`).
- Soft layer vs executed top-k (2021–2022, default rule): regular arms soft − topk ≈ −12 to −13 pp/yr; the concentrated dose books collapse at the soft layer (G0dose003 −29.7 pp/yr soft vs +20.3 pp/yr top-k; gap −50 pp). The concentration explosion is fatal before lot-level frictions even enter.
- At the executed default rule on 2021–2022, C0 − G0 (seed 1) = −0.59 pp (selection-window diagnostic, not the estimand).

## 8. CSI500 H (Kaggle; Gate C material)

Trained on Kaggle T4 by splitting across two kernels under the two-session GPU cap; identical runner, formal configuration and trainer sha as the local CSI300 runs.

| Seed | Epochs (early stop) | Selected epoch | validation auxiliary loss | checkpoint sha256 |
|---|---|---|---|---|
| 1 | 12 | 4 | 0.94321 | `06cd06e5…` |
| 2 | 13 | 5 | 0.94280 | `d6ef1297…` |
| 3 | 14 | 6 | 0.94195 | `4d961d0b…` |

Integrity: data_version `bae7115e…` (identical across the three seeds and matching the local pre-computed anchor), config hash `393b1993…` (identical to the CSI300 formal runs), trainer sha self-check passed in-kernel; local recomputation of each checkpoint sha matches the manifest and the run log. Execution environment recorded (T4, torch 2.11+cu128, Python 3.13.15); cross-device bit-level trajectory identity is not claimed, which the preregistration framework permits. CSI500 fine-tunes (G0/C0) await the 1,802-symbol execution-feed build and are not on the Gate B path.

## 9. Remaining steps

1. Investigator review of this report and the underlying artefacts.
2. Write `gate_b.json` once (config hashes; tau = 0.5; Tier P1 with sigma_d and MDE80; selected rule per arm including benchmark arms; seeds used; TOST margin ±2.0 pp; stamp-duty sensitivity flag; list of exploratory/descriptive arms).
3. Evaluation-window scoring for every arm at the selected and default rules; sealed files are written and then read in one unsealing pass.
4. Mechanical S1/S2/S3 mapping at Gate C on the three-seed CSI300 results; CSI500 confirmation afterwards.

## 10. Artefact map

| Artefact | Contents |
|---|---|
| `protocol/gate_b_ready_*.md` | Readiness checklist with full numeric detail (Chinese) |
| `protocol/checks/sigma_d_prelim_*.json`, `sigma_d_final_*.json` | Power inputs and diagnostics |
| `runs/csi300_neural_selection.json` | 14 per-seed grids + seed-averaged selections (h5 arms) |
| `runs/csi300_bench_selection.json` | Six benchmark grids + selections (r1 arms) |
| `runs/csi300/<arm>/h5\|r1/<rule>/seedN/daily.csv` | Readable selection-window daily books, all arms |
| `runs/scores/*.npz` + `.meta.json` | Pinned score artefacts (checkpoint sha per tag) |
| `protocol/checks/dose_diag_*.json` | Full dose diagnostics, 14 tags |
| `runs/csi500_H_kaggle/` | CSI500 H checkpoints, histories, manifests (verified) |
| `configs/benchmarks_qlib_defaults_v1.json` | Pinned Qlib-default benchmark specification |
