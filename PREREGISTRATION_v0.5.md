# Preregistration: Cost-Aware Training for LiMT (Clean-Pipeline Study)

Version: v0.5 DRAFT (2026-10-16, Asia/Shanghai). Fields marked `[FILL]` must be completed before the commit. Items marked `[DECISION]` need an explicit sign-off before Gate A. Items marked `[VERIFY-IN-CODE]` are stated from reports or memory and must be checked against the code before commit.
Author(s): [FILL]
Commit hash / timestamp of this file: [FILL at Gate A]
Pipeline provenance: fix1 labels sha e15ed6f6...; formal pre-run gate run3 (87 pytest items passed; SHA256SUMS consistent for 39 files). Full hashes: [FILL]

Changes in v0.5: locked decisions S2 (tau grid approved), S3 (L0 = No), S4/D2 (1-day H retrain = No), S8 (bridge ladder = No for main matrix). Updated Section 8, 8A, and Section 17 accordingly. G0 real-fee selection requirement confirmed and retained. No thresholds, grids or decision rules changed.

Changes in v0.4: added Section 8A (pre-registered multi-lever decomposition: backbones x {default, tuned rule} x {1-day, h5 label}, with h5 and C0 declared as chosen representatives); G0 selection-metric wording in Section 6.1 made unambiguous and unit assertions added to Section 6.5; Section 8 lines on the 1-day LiMT retrain made conditional on decision D2; added Section 17 (open-items register with code-structure flags); note on exploratory variants added to Section 14. No thresholds, grids or decision rules changed.

Changes in v0.3: added Section 16 (prior-art register and must-cite list from a partial novelty search on 2026-10-16); added optional listwise-ranking baseline as a [DECISION] item in Section 8; limitation 8 updated; framing note added to Section 1. No thresholds, grids or decision rules changed.

Changes from v0.1 to v0.2: scenarios renamed S1-S3 and secondary contrasts renamed SC1-SC5; compute section rewritten from measured timings (epoch-cut advice removed); full formal config added; delisting dates explicit; verify-in-code checklist added; power analysis added with a mechanical rule and tiered S1 wording; tau grid changed from {0.5, 1, 2} to {0.5, 0.75, 1.0}; bridge analysis to the original LiMT paper added.

All thresholds below are fixed in advance. Any change after Gate A must be recorded in the Deviation Log (Section 14) with a date and reason, and the affected result must be labelled "post-hoc".

---

## 0. Integrity rules

1. No evaluation-period (2023-01 to 2026-06-30) output of any new-pipeline model is opened before Gate B (configuration freeze).
2. The 2023-2026 window was inspected during earlier exploration (the h5 idea and the rule grid were chosen after looking at it). It is therefore labelled **exploratory**, never "clean hold-out".
3. The confirmatory evaluation is the post-cutoff holdout (Section 9), evaluated once, with code, config and hashes frozen first.
4. All results from before 2026-10-15 (legacy label-timing error) are INVALID and must not appear as evidence. They may appear only in the RQ1 legacy-vs-clean comparison, clearly marked.
5. Failures must be visible. Missing cells, empty tables and silent job failures are reported, never dropped.

## 1. Research questions

- RQ1: How much do label-calendar misalignment and future-label masking inflate IC and returns? (legacy vs clean)
- RQ2: After realistic costs, which models survive, and how much comes from execution rules alone?
- RQ3 (primary): Does putting realistic costs into the training objective (C0) add anything over a matched control without the cost term (G0), when both use their own tuned execution rule?
- RQ4: Does a soft-portfolio gain transfer to TopkDropout with full frictions? (rank correlation across layers)
- RQ5: Is the result robust across seeds, markets and periods?

Framing note (v0.3): cost-aware training is an established idea (Section 16). RQ3 is therefore posed as a test of that idea in a new regime (cross-sectional, long-only, A-share, executed through TopkDropout), not as a claim of a new method. LiMT is treated as one representative cross-stock transformer backbone, not as the subject of the paper.

## 2. Arms

- H: multi-task LiMT, loss 0.65*h5 + 0.30*zv + 0.05*zs, checkpoint by validation supervised loss. Initialization source for all fine-tuned arms. [VERIFY-IN-CODE: weights, MSE on z-scored targets]
- H-FT: initialized from H, continued with the supervised loss only (control for extra epochs). Seed 1 only unless Section 8 says otherwise.
- G0: initialized from H; fine-tuned with the same soft-portfolio ledger code as C0 on a ZERO-FEE ledger (gross log growth + alpha_aux * h5 auxiliary loss). Matched control: C0 and G0 differ only in the fee schedule inside the ledger. [VERIFY-IN-CODE: this exact definition, same code path, same optimizer, same epochs, same selection metric]
- C0: initialized from H; fine-tuned on the self-financing ledger with real fees (buy 5 bp, sell 15 bp, minimum CNY 5), after-cost mean log growth, alpha_aux = 0.3, gamma_to = 0, beta_risk = 0.
- C-T: C0 plus turnover penalty gamma_to = 0.001 in the formal seed-1 run (candidates must include 0, which is C0). Conditional beyond seed 1 (Section 8).
- Baselines (CPU): ridge_h5 (alpha = 10), LightGBM regression on h5, LightGBM LambdaRank on h5.

## 3. Primary estimand (one only)

Delta = C0* - G0*, the paired difference in annualized after-cost excess return, where * means each arm uses its own tuned execution rule (Section 5).

- Layer: TopkDropout with full frictions (Section 6).
- Benchmark: H00300 total-return index.
- Window: 2023-01 to 2026-06-30 (842 trading days; exploratory label).
- Market: CSI300.
- Computation: seed-averaged daily return difference (C0 minus G0, each averaged over seeds), annualized.
- Inference: stationary block bootstrap, mean block length >= 10 trading days, 10,000 resamples, one-sided 5% lower bound (5th percentile of the bootstrap distribution of Delta).
- Robustness (not decisive): Newey-West HAC with lag >= 5 on the same paired series.

### Secondary contrasts (Holm-corrected as one family)

- SC1. Soft-layer Delta (same estimator).
- SC2. Delta in annualized IR.
- SC3. C0 with default rule (topk 50, drop 5, daily) vs G0 with tuned rule.
- SC4. Delta on CSI500 (confirmatory; Section 9).
- SC5. Rank correlation of arms/models across soft and TopkDropout layers.
- SC6 (mechanism). Delta_cost: the difference in annual cost drag (fees as % of capital, computed from executed trades) between G0* and C0*. Cost drag is computed from holdings and is far less noisy than returns, so this is the best-powered test in the study.
- SC7 (mechanism). Delta_gross: the difference in gross (pre-fee) annualized excess return, plus the 5-day IC difference.
Delta = Delta_gross + Delta_cost (accounting identity). SC6 and SC7 are reported together in every scenario.
Everything else is exploratory and labelled so.

## 4. Power analysis and mechanical rule

### 4.1 Arithmetic (checked)
Evaluation length T = 3.5 years. For annualized SD of the daily difference series sigma_d: SE = sigma_d / sqrt(3.5). One-sided 5% test: MDE at 50% power = 1.645 * SE; at 80% power = (1.645 + 0.842) * SE = 2.487 * SE.

| sigma_d | SE | MDE (50% power) | MDE (80% power) | Power if true Delta = 2.0pp |
|---|---|---|---|---|
| 2% | 1.07pp | 1.76pp | 2.66pp | about 54% |
| 3% | 1.60pp | 2.64pp | 3.99pp | about 35% |
| 4% | 2.14pp | 3.52pp | 5.32pp | about 24% |
| 5% | 2.67pp | 4.40pp | 6.65pp | about 18% |
| 6% | 3.21pp | 5.28pp | 7.98pp | about 14% |

These assume independent daily differences. Autocorrelation and the block bootstrap make the SE larger, never smaller. Seed-averaging shrinks only training noise, not market-sampling noise, so it does not rescue power.

### 4.2 What sigma_d actually is
5% is an assumption. Two fine-tunes of the same H in the same universe may track each other more closely, or less closely if their tuned rules differ in turnover. It must be measured, not assumed.

- Preliminary estimate (Gate A): validation window 2019-2020 (486 days), using ridge_h5 vs H after-cost daily returns under the default rule. Record sigma_d(prelim) = [FILL].
- Final estimate (Gate B, before any 2023+ output is opened): validation-window daily after-cost return difference between seed-1 C0 and seed-1 G0 under the default rule. Record sigma_d(final) = [FILL], MDE80 = 2.487 * sigma_d / sqrt(3.5) = [FILL]. The validation window gives a noisy estimate of sigma_d (roughly +/-15%); the tier below is chosen from the point value and not revisited.

### 4.3 Mechanical rule (applied at Gate B using sigma_d(final); no discretion)
- Tier P1: MDE80 <= 2.5pp. Use S1 as written in Section 7, with lower bound > 0.
- Tier P2: 2.5pp < MDE80 <= 7.0pp. Use S1 as written but with the tiered wording of Section 7: "demonstrated" requires lower bound > 0; "supported" requires the alternative conditions in Section 7. Delta_cost (SC6) becomes a required condition.
- Tier P3: MDE80 > 7.0pp. The window cannot discriminate any plausible effect on net return. S1 "demonstrated" is declared unreachable in advance. The paper is framed by design around the mechanism results (SC6, SC7) and the surrogate gap (SC5, SC1 vs Delta), i.e. an S2-type paper, and no method-effectiveness claim is made.

### 4.4 Options considered
- Pooling CSI300 + CSI500 as primary: NOT adopted. If the two daily difference series have correlation rho, pooling shrinks the SE by sqrt((1 + rho)/2). For rho of 0.5 to 0.8 that is only an 11% to 5% reduction, and it needs CSI500 to pass the W1-W2 gates first. Pooled Delta is reported as exploratory.
- Lengthening the window: NOT adopted. The only way is to reuse 2021-2022 (about 5.5 years in total, SE falls by about 20%), which removes the independent rule-selection window. Post-cutoff holdout days arrive about one per day, so they add no real power in 2026.
- Reframing the primary claim: adopted through the tiered wording in Section 7 and the mechanism tests SC6/SC7.

## 5. Tuned-execution control

Both arms receive the same execution grid and the same selection rule.

- Grid (8 rules, fixed, no additions): n_drop in {2, 5, 10, 20} x rebalance interval in {1, 5 days}; topk = 50.
- Selection window: 2021-2022 (rule-selection window).
- Selection metric: after-cost annualized excess return vs H00300 total return. Ties within 0.5pp go to the lower-turnover rule.
- One rule per arm (not per seed): the selection metric is averaged over the seeds before choosing.
- The default rule (topk 50, drop 5, daily) is always also reported.
- Selection-window grids are not reported as results.

## 6. Formal configuration (frozen at Gate B)

### 6.1 Training
- Network: d = 256, lookback 21 days, batch_days = 4. [VERIFY-IN-CODE: heads, number of experts]
- H: up to 50 epochs, patience 8, lr_h = 1e-4. Loss 0.65*h5 + 0.30*zv + 0.05*zs. [VERIFY-IN-CODE]
- Fine-tunes (H-FT, G0, C0, C-T): up to 20 epochs, patience 8, lr_ft = 1e-5. Checkpoint chosen by validation after-cost mean log growth (G0 included: its TRAINING ledger is zero-fee, but its checkpoint is selected by the real-fee after-cost mean log growth (buy 5 bp, sell 15 bp, minimum CNY 5) computed on a separate monitoring ledger; the zero-fee validation value is logged next to it and is never the selection metric). [VERIFY-IN-CODE: G0 selection metric must be the same real-fee metric as C0, otherwise the arms are not budget-matched]
- Optimizer: Adam. [VERIFY-IN-CODE] clip_norm = 1, AMP off.
- alpha_aux = 0.3, gamma_to = 0.001 (C-T only), beta_risk = 0.
- Periods: train 2008-2018, validate 2019-2020, rule-selection 2021-2022, evaluation 2023-01 to 2026-06-30, plus full test from 2021-01-04.

### 6.2 Tau grid (budget-matched, both G0 and C0) [APPROVED in v0.5]
If scores are roughly standard normal and weights are softmax(score / tau), the effective number of names is about N * exp(-1/tau^2) for N eligible names. For N of about 300: tau 0.5 gives about 5.5 (matches the measured median N_eff of 5.24 for Ai), tau 0.75 gives about 51, tau 1.0 gives about 110, tau 2.0 gives about 270 (almost equal-weight). The v0.1 grid therefore skipped the region (N_eff of about 50) that corresponds to a 50-name TopkDropout book. New grid: tau in {0.5, 0.75, 1.0}.
- Seed 1 formal run (tau = 0.5) is one grid point. The two extra points per arm are 4 extra fine-tunes in total.
- Heavy-tailed learned scores change N_eff; verify empirically on the validation window and record N_eff median and max single-name weight per tau and arm: [FILL].
- Each arm selects its own tau by validation after-cost mean log growth under the same real-fee ledger. Tune on seed 1 only, freeze, then run seeds 2 and 3 unchanged.
- CSI500 reuses the CSI300 configuration. No CSI500 tuning.

### 6.3 Fees and execution
- Buy 5 bp, sell 15 bp, minimum fee CNY 5 per trade, lot size 100 shares, initial capital CNY 100 million (1e8).
- TopkDropout reference: topk = 50, drop = 5 (default rule); rules evaluated: soft, topk-reference, topk-execution.
- 6 models x 3 windows x 3 rules = 54 cells in the seed-1 formal run.
- Pre-tax cash dividends booked on the ex-dividend date; share multiplier = 1 + stock dividend + capitalization issue; discrepant dividend records keep the original-source values (27 amount and 3 ex-date discrepancies vs CNINFO; 21 of 27 inside the membership universe; in the evaluation period 3 for SH600989 and 1 rounding difference for SH600519; all 3 ex-date discrepancies are in the training period).
- Benchmark: H00300 total-return index (RESSET, 2007-06-01 to 2026-09-01, 4683 trading days).

### 6.4 State, halt and delisting policy
- Unknown state (code 4): retained in the grid; trading forbidden; with missing quotes the last valid valuation is carried forward. Full sample has 250 unknown-state cells (204 in the training period, 46 in 2007); within the eligible universe over the evaluation period the count is 0.
- Delisting / exit from the eligible universe, with existing holdings frozen at the last valid valuation (no share-swap simulation):
  - SH600837 (Haitong Securities): from 2025-03-04.
  - SH601989 (CSSC Holdings, referred to in the project files as CSIC): from 2025-09-05.
- Evaluation end 2026-06-30 (single-stock data cutoff); all 842 required benchmark days present.

### 6.5 Verify-in-code checklist (tick before commit)
- [ ] H loss weights 0.65 / 0.30 / 0.05 and MSE on cross-sectionally z-scored targets
- [ ] Optimizer is Adam in all training scripts
- [ ] G0 = gross log growth + alpha_aux * h5 loss on a zero-fee ledger, same code path as C0
- [ ] G0 and C0 use the same selection metric, same epoch cap, same patience, same seeds
- [ ] H-FT definition (supervised loss only, same epoch cap)
- [ ] Epoch caps (50 / 20), patience 8 in the committed config files
- [ ] Capital 1e8, lot 100, fees 5/15 bp, minimum 5 in the committed config files
- [ ] topk 50 and drop 5 in the TopkDropout config
- [ ] Delisting dates in the policy file match this document

Status of this checklist: NOT verified. This document cannot confirm any item against the code. The formal config as reported states that H-FT, G0, C0 and C-T all select by validation after-cost mean log growth, which is consistent with the G0 requirement above, but that is a statement about the config and not a code check. Tick an item only after the corresponding assertion below passes.

Required unit assertions (write before the formal runs; each must fail loudly):
- [ ] A-G0-1: the G0 training ledger is constructed with fees = 0, and a separate monitoring ledger is constructed with the real-fee schedule (5 bp buy, 15 bp sell, minimum CNY 5, lot 100).
- [ ] A-G0-2: the checkpoint-selection metric key for G0 and C0 is the same config string and is computed by the same function on the real-fee monitoring ledger.
- [ ] A-G0-3: on a fixed toy validation panel, the epoch selected for G0 equals the argmax of the real-fee monitoring metric over epochs, not the argmax of the zero-fee metric (use a toy case where the two argmaxes differ).
- [ ] A-G0-4: G0 and C0 share epoch cap, patience, learning rate, optimizer, seeds, alpha_aux and tau grid; the only differing argument is the training-ledger fee schedule (diff of resolved configs shows exactly that one field).
- [ ] A-LBL-1: the 1-day and h5 labels in the committed label files match the definitions in Section 8A, with endpoint misalignment = 0 on the two known counterexamples.

## 7. Go/no-go criteria and scenario mapping

Rules are applied mechanically to the 3-seed average, in order: S1, then S2, otherwise S3. Seed 1 alone decides nothing. Boundaries are hard: a result between 1.0 and 2.0pp is never "almost S1".

### S1: method effective
ALL of the following:
1. Delta_exec >= +2.0pp, and
   - "Demonstrated": one-sided 5% bootstrap lower bound > 0. [required in tier P1]
   - "Supported" (allowed only in tier P2): lower bound > -1.0pp AND SC6 (Delta_cost) lower bound > 0 AND SC7 shows gross-alpha non-inferiority (Delta_gross lower bound > -1.0pp).
2. C0 > G0 in 3 of 3 seeds (each seed compared with its own tuned rule).
3. C0 5-day IC is no more than 5% below G0 5-day IC.
4. CSI500 point estimate of Delta >= 0, with at least 2 of 3 seeds positive.
In tier P3 S1 is unreachable by construction (Section 4.3).
Narrative: method headline. In wording, "demonstrated" and "supported" are never merged; the "supported" tier must say "consistent with a gain; the evaluation window cannot rule out zero".

### S2: surrogate gap or substitutable
S1 not met, and ANY of:
- Soft-layer Delta has a lower bound > 0 but S1 fails at the executable layer.
- Delta_exec >= +2.0pp but the CI is too wide for even the "supported" tier (underpowered).
- Delta_exec passes but C0-default does not beat G0-tuned (SC3): simple execution rules achieve the same effect.
- All S1 conditions pass except the IC guard (degenerate-solution risk).
- Delta_cost (SC6) is positive with lower bound > 0 but Delta_net is not: costs fell but gross alpha fell with them.
Narrative: "Training gains do not transfer to, or are replaceable by, execution engineering." Lead with the surrogate-gap analysis and the audit.

### S3: null
Neither of the above: soft-layer CI includes 0 and Delta_exec < 2.0pp, or Delta is negative.
Narrative: audit and honest-evaluation study. Cost-aware training is reported as a tested negative result with full uncertainty.

### Required reporting in every scenario
Delta with its CI, SC1-SC7, per-seed values, selected rules, measured sigma_d and MDE80, and the soft-vs-TopkDropout rank correlation.

## 8. Experiment matrix

### Essential
- CPU, both markets, all seeds: ridge_h5, LightGBM regression, LightGBM LambdaRank.
- GPU, per (market, seed): H, G0, C0.
- Seed 1 CSI300 formal run: H, H-FT, G0, C0, C-T (already configured), plus 4 extra fine-tunes for the tau grid.
- Evaluation per model: the selected rule and the default rule only.
- Label comparison: h5 vs 1-day labels using ridge/LightGBM only (no clean 1-day LiMT GPU retrain; D2 = No).

### Conditional
- H-FT and C-T for seeds 2-3: only if S1 or S2 is reached on CSI300 seed-averaged results and the sign is in favour of C0.
- Extra analyses: only those whose trigger is listed in Section 11.

### Optional arm
- L0 (listwise ranking): **Not included** (decision S3 = No in v0.5). May appear only as future work.

### Cut (not run)
MASTER, SOFT/NeuralSort, risk term (beta_risk), Sharpe objective, JEF-style overlays, CSI500 tuning, clean 1-day LiMT GPU retrain (D2 = No), full bridge ladder to original LiMT protocol (S8 = No for main matrix). These may appear as future work only.

## 8A. Pre-registered multi-lever decomposition (RQ2)

Purpose: attribute after-cost performance to three levers (label, execution rule, training objective) across backbones, so that the paper's main message does not depend on cost-aware training working. This analysis is descriptive. The only confirmatory estimand remains Section 3.

### Chosen representatives (not fixed)
- Multi-day label = h5: close(t+6)/close(t+1) - 1, cross-sectionally z-scored. 1-day label = the clean r1 label with the same t+1 entry convention. [VERIFY-IN-CODE: both definitions]
- Cost-aware arm = C0.
- h5 and C0 are chosen representatives, not the outcome of a pre-declared search. The choice of h5 was informed by the signal-persistence analysis and by exploration that included 2023-2026 data (Section 0, rule 2), so it is itself exploratory-informed.
- Any other option (other horizons such as h3 or h10, other label normalizations, C-T at other gamma_to, risk or Sharpe objectives, other cost-aware arms) is outside the confirmatory matrix. If it is run, a Deviation Log entry (Section 14) must be made BEFORE it runs, and every result from it is reported as exploratory and post-hoc.

### Decomposition table (per market)
Execution rules: default = topk 50, drop 5, daily; tuned = one rule per (backbone, label) cell, selected on 2021-2022 from the Section 5 grid with the metric averaged over seeds. Cells that cannot be run are reported as N/A, never dropped.

| Backbone | 1-day / default | 1-day / tuned | h5 / default | h5 / tuned | Seeds | Markets |
|---|---|---|---|---|---|---|
| ridge | run | run | run | run | 1 (deterministic) | CSI300, CSI500 |
| LightGBM regression | run | run | run | run | 3 if stochastic, else 1 | CSI300, CSI500 |
| LightGBM LambdaRank | run | run | run | run | 3 if stochastic, else 1 | CSI300, CSI500 |
| H (LiMT-style, multi-task) | N/A (D2 = No) | N/A (D2 = No) | run | run | 3 | per Section 8 matrix |
| G0 | N/A (not run) | N/A (not run) | run | run | 3 | per Section 8 matrix |
| C0 | N/A (not run) | N/A (not run) | run | run | 3 | per Section 8 matrix |

Per cell, report: after-cost annualized excess vs H00300 total return, IR, annual cost drag, gross (pre-fee) excess, turnover, 5-day IC and native-label IC, per-seed values, and the stationary block bootstrap CI of the excess series (Section 3 estimator).

### Decomposition contrasts (descriptive; paired-difference bootstrap CIs, same estimator as Section 3)
- DC1 label lever: h5 minus 1-day, same backbone and rule. (Reported only for ridge and LightGBM, since D2 = No.)
- DC2 execution lever: tuned minus default, same backbone and label.
- DC3 interaction: DC2 at h5 minus DC2 at 1-day. (Reported only for ridge and LightGBM.)
- DC4 training lever: C0 minus G0. This is the primary estimand of Section 3 (tuned rule), also reported at the default rule (SC3 context).
- DC5 ladder: not run on the neural backbone (D2 = No). Paper must state that the label lever was not tested on the neural backbone.
DC1-DC3 and DC5 are not in the Holm family and are described with their intervals only; no significance claim is made from them. DC4 follows Section 7.

### Decision D2
**Decided (v0.5): No.** Clean 1-day H retrain is not included. The label lever is tested only on ridge and the two LightGBM backbones. The paper must state this explicitly.

## 9. Markets and holdout

- CSI500 must pass the same W1-W2 gates (endpoint misalignment = 0, out-of-bound = 0) before any CSI500 result is used. Until then it is not evidence.
- CSI500 is confirmatory only.
- Post-cutoff holdout: data after 2026-06-30. Freeze code, config, hashes and selected rules first. Evaluate once, when at least [FILL, suggest 120] trading days are available. Report whatever results, whatever the sign.

## 10. Compute plan (from measured timings)

- Measured on a Kaggle T4: 204-245 s per model per epoch (the earlier "31 min" was one epoch of all six models together; the v0.1 "67 hours" reading was wrong and is withdrawn).
- Epoch caps stay as configured: H 50, fine-tunes 20, patience 8. No caps or grid reductions are imposed before Gate A.
- Seed-1 CSI300 formal run: H 50 + 4 fine-tunes x 20 = 130 epochs, about 8.1 h worst case (reported; arithmetic at about 224 s/epoch; at the 245 s upper figure the worst case is about 8.9 h). Single-session limit 12 h: fits.
- Remaining work in epochs (worst case, all epochs run): seeds 2-3 CSI300 H+G0+C0 = 2 x 90 = 180; CSI500 3 seeds = 270; tau-grid extras = 80; total 530 epochs = about 30-36 h, so about 38-44 h including seed 1.
- Risk to measure before budgeting CSI500: cross-stock attention scales with S^2, so CSI500 epochs may take up to about 2-3 times longer than CSI300 (estimate, not measured). If so, the CSI500 block rises to about 15-21 h and the total toward about 60 h. Measure CSI500 s/epoch on one short run: [FILL].
- Early stopping with patience 8 should cut the actual figures; record actual epochs used.
- Kaggle weekly GPU quota: [FILL] h/week. Total GPU-hour budget for the study: [FILL].

## 11. Relationship to the original LiMT paper (arXiv:2609.25617)

- What the original paper does: MSE on z-scored targets; costs appear only in the APO long-short backtest (10 bp per trade, 2% participation cap, CNY 1e9 per side). The headline top-50 / drop-20 daily-rebalancing table does not state its cost, benchmark or limit/halt handling in the text. [Checked in the paper text on 2026-10-16.] Action: check the authors' public repository configuration for the backtest settings (open/close cost, minimum cost, benchmark): [FILL].
- Its protocol: Alpha158, lookback 21, 1-day label defined as Close(t+2)/Close(t+1) - 1, train 2008-2014, validation 2015-2016, test 2017-2020-08.
- Rules for our text: we do not claim to have "reproduced" the headline portfolio numbers; we do not describe the original as wrong; we state which settings are unspecified and report sensitivity to them.
- Bridge analysis (descriptive, not a hypothesis test): **Not run in the main matrix** (decision S8 = No in v0.5). May be considered as future work. If later added, move from the original protocol to ours one change at a time, reporting IC and after-cost excess at each step. The legacy rung is reported only after confirming whether the legacy pipeline carries the label misalignment.

## 12. Gates

- Gate A (now): complete the [FILL] and [DECISION] fields, tick Section 6.5, run the preliminary power check, hash and commit this file.
- Gate B (after the tau grid and seed-1 validation results): freeze configuration and selected taus; compute sigma_d(final) and apply Section 4.3. Only then may 2023+ outputs of seed 1 be opened.
- Gate C (after 3 seeds on CSI300): apply Section 7 mechanically. S1 means run CSI500 as confirmation (and conditional arms). S2 means run CSI500 as a gap-analysis robustness check. S3 means run CSI500 as a robustness check for the audit.
- Gate D (after CSI500 and the holdout): decide headline, title, abstract and target journal; run only extensions whose triggers were declared above.

Decisions deferred until Gate D: headline claim, title, venue and tier, extra baselines. Venue quartile must be checked against the official JCR list and the school's recognized list.

## 13. Stopping rule

- Maximum GPU budget: [FILL] Kaggle sessions / [FILL] hours.
- Hard date after which the paper is written with whatever has completed: [FILL].
- No experiment outside this document is started without a Deviation Log entry made before it runs.

## 14. Deviation log

Rule (v0.4): variants of the chosen representatives (h5, C0; see Section 8A) and any other experiment outside the matrix are entered here before they run and are labelled exploratory/post-hoc in every table and in the text.

| Date | Change | Reason | Affected results labelled post-hoc |
|---|---|---|---|
| | | | |

## 15. Known limitations to carry into the paper

1. 2023-2026 is exploratory, not a clean hold-out.
2. Evaluation ends 2026-06-30 because single-stock data stop there.
3. Share-swap mechanics for merged stocks are not simulated (freeze convention); no holdings-exposure analysis yet.
4. 250 unknown-state cells are treated as non-tradable; dividend discrepancies kept at original values.
5. A surrogate gap exists between soft-portfolio training and TopkDropout execution.
6. Starting hyperparameters were chosen by GPT, not tuned, apart from the declared tau grid.
7. Legacy-pipeline (2008-2020) reproduction numbers remain unverified for label misalignment; the CSI300 excess return of 0.26 vs the paper's 0.18 and the "after costs" figures from the unidentified script are unresolved and must be resolved before any legacy number is used.
8. Only a partial novelty search has been done (about seven English-language searches on 2026-10-16; no CNKI, no SSRN full texts). Section 16 lists what was found; absence of a paper there is not evidence that none exists.
9. The evaluation window (3.5 years) has limited power (Section 4); a non-significant Delta is not evidence of no effect.
10. It is not established that the label-calendar bug (membership filter, then per-stock shift) exists in Qlib's own pipeline. The bug is documented for our pipeline only; any generality claim needs a test on an independent pipeline.

## 16. Prior-art register (partial; found 2026-10-16)

Status labels: [seen] = abstract or relevant passage read in search results; [recollection] = not verified in this pass; [preprint] = arXiv only as far as seen. Verify details and publication status against the originals before citing.

### 16.1 Must cite: cost in the training objective
- Jensen, Kelly, Malamud, Pedersen, "Machine Learning and the Implementable Efficient Frontier" [seen]. Net-of-cost evaluation; trading-cost-aware optimization integrated with ML; weights learned with an economic objective. Listed by SFI as forthcoming in the Review of Financial Studies (confirm status). US setting. Closest finance prior art.
- DeePM, arXiv 2601.05975 [seen, preprint]. Gross vs net training ablation: zero-cost training over-trades, net Sharpe 0.56 vs 0.93 with an intermediate cost scaler. Macro futures portfolios.
- Soft-Radial Projection, arXiv 2602.03461 [seen, preprint]. "Training gross" vs "training net" over five seeds; costs in training lower test-time turnover.
- End-to-end parametric portfolio policies for cross-asset futures timing, arXiv 2607.00475 [seen, preprint]. Turnover penalty reduced LSTM trading but did not change the net ranking; too restrictive for the low-turnover transformer. Evidence that a null or small effect is a plausible outcome.
- Machine-learning Bitcoin trading under transaction costs (walk-forward), arXiv 2606.00060 [seen, preprint]. A simple cost-aware execution filter cut turnover by over an order of magnitude and restored positive net results in selected configurations. Supports the tuned-execution control (Section 5).
- Jiang, Xu, Liang 2017 (arXiv 1706.10059); Zhang et al. 2020 (cost-sensitive portfolio selection via DRL); Lim, Zohren, Roberts 2019 (Deep Momentum Networks, arXiv 1904.04912); Zhang, Zohren, Roberts 2020 (arXiv 2005.13665) [recollection; not verified in this pass]. Confirm that each does or does not put costs and holding drift in the training objective before describing them.

### 16.2 Must cite: Top-K alignment and A-share work
- STN-TGAT, arXiv 2607.19385 [seen, preprint]. Head-weighted listwise ranking loss plus auxiliary regression, to align training with Top-K portfolio construction. Competes with the surrogate-gap framing; motivates optional arm L0 (not included in v0.5).
- CEGP, "An efficient loss function and deep learning approach for ranking stock returns in the absence of prior knowledge" [seen]. RL-Loss to bridge predictions and profits on CSI100/300/500; no cost treatment seen.
- Stockformer, arXiv 2401.06139 [seen]. CSI300 Topk-Dropout backtest with per-side cost and stamp duty; multi-task attention.
- MiM-StocR, arXiv 2509.10461 [seen, preprint]. Multi-task stock recommendation with a listwise ranking loss on SSE50/CSI100/CSI300.
- DSPO, arXiv 2405.15833 [seen]. End-to-end sorted portfolio construction; authors include IDEA Research (also a LiMT affiliation). Check author overlap with LiMT.

### 16.3 Leakage and evaluation integrity
- "Information leakage in financial machine learning research," Algorithmic Finance 2019 [seen]. Names off-by-one feature/target misalignment and use of current constituents. The general family our label bug belongs to.
- "Look-Ahead Benchmark Bias in Portfolio Performance Evaluation," arXiv 0810.1922 [seen]. Ex-post constituent lists inflate performance.
- Kapoor and Narayanan, leakage and reproducibility in ML-based science [seen]. Template for a re-examination paper.
- Leakage-controlled stock-prediction benchmark/audit (Zenodo record 21592983) and "When Alpha Disappears" decision-time-leakage benchmark (arXiv 2605.23959, cites Qlib) [seen, preprints]. Crowded neighbourhood: our contribution here must be specific (label calendar after membership filtering, A-share frictions, cost ledger), not generic.
- Not found: any paper describing membership filtering followed by a per-stock shift that makes labels jump across membership gaps. We call it a sub-case of a known leakage family with a quantified effect and a gate suite, not a new discovery.

### 16.4 Original paper and follow-ups
- LiMT, arXiv 2609.25617 (2026-09-22). No citing papers found as of 2026-10-16.

### 16.5 Open prior-art checks before Gate A
- [ ] Verify the five [recollection] items in 16.1 against the originals.
- [ ] Search Chinese-language sources (e.g. CNKI) for cost-aware or turnover-aware training of A-share cross-sectional models.
- [ ] Search SSRN for gross vs net training in cross-sectional equity ML.
- [ ] Check whether Qlib's own label and universe pipeline shows the label-calendar bug (independent-pipeline test).

## 17. Open-items register (as of v0.5)

"Code" = changes how the code must be structured, so it must be settled before coding starts.

### 17.1 Items that affect code structure (settle first)
| ID | Item | Where | What it changes in code | Status |
|---|---|---|---|---|
| S1 | G0 definition and real-fee selection (assertions A-G0-1 to A-G0-4) | 2, 6.1, 6.5 | Ledger API must take separate training-fee and monitoring-fee schedules; checkpoint selector keyed to the monitoring-fee metric | **Confirmed / required** |
| S2 | Approve tau grid {0.5, 0.75, 1.0} | 6.2 | Config sweep per arm; daily logging of N_eff and max single-name weight | **Approved (v0.5)** |
| S3 | Include optional arm L0 (listwise) | 8 | New loss module and arm registry entry; SC8; extra compute | **No (v0.5)** |
| S4 | D2: clean 1-day H retrain | 8A | Label-selectable H training and evaluation; extra runs | **No (v0.5)** |
| S5 | H-FT definition (supervised-only continued training, same caps) [VERIFY-IN-CODE] | 2 | Fine-tune mode flag | Open |
| S6 | Evaluation outputs keyed by (market, backbone, label, rule, seed) with gross/cost split and stored daily return series | 3, 8A | Output schema for SC6/SC7, the decomposition table and the bootstrap | Open |
| S7 | Gate lock: block all 2023+ outputs until the Gate B hash exists; validation-only mode for the sigma_d computation | 0, 4.2 | Date mask in the evaluation harness, enforced in code | Open |
| S8 | Run the bridge ladder to the original LiMT protocol? | 11 | Legacy-protocol switches in configs | **No for main matrix (v0.5)** |
| S9 | CSI500 pipeline passes W1-W2 gates | 9 | Market parameter threaded through every config; CSI500 label rebuild | Open |

### 17.2 Items that do not affect code structure
| ID | Item | Where |
|---|---|---|
| N1 | Author(s) | header |
| N2 | Commit hash and timestamp of this file | header, Gate A |
| N3 | Full pipeline hashes | header |
| N4 | sigma_d (preliminary at Gate A, final at Gate B) and MDE80 | 4.2 |
| N5 | N_eff median and max weight per tau and arm | 6.2 |
| N6 | CSI500 seconds per epoch (measure on one short run) | 10 |
| N7 | Kaggle weekly GPU quota and total GPU-hour budget | 10 |
| N8 | Maximum sessions/hours and hard write-up date | 13 |
| N9 | Minimum post-cutoff holdout days (suggested 120) | 9 |
| N10 | Authors' repository backtest config check (open/close cost, minimum cost, benchmark) | 11 |
| N11 | Whether the legacy pipeline carries the label misalignment (needed for any future ladder) | 11 |
| N12 | Verify-in-code items other than those in 17.1: H loss weights, Adam, epoch caps, capital, lot, fees, topk/drop, delisting dates, network heads and experts | 2, 6.1, 6.5 |
| N13 | Prior-art checks in 16.5 (five [recollection] items, Chinese-language and SSRN searches, optional Qlib independent-pipeline test) | 16.5 |
