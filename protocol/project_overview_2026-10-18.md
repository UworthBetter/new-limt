# Project Overview: Cost-Aware Training for LiMT — A Clean-Pipeline, Preregistered Study

Status date: 18 October 2026. Workspace: `D:\research\new_LiMT`. Governing protocol: `PREREGISTRATION_v0.6.md` (v0.5 of 2026-10-16 retained unchanged). Upstream implementation and data lineage: `D:\research\limt-repro`.

## 1. What this project is

A new research line opened on 2026-10-16. The question is narrow and falsifiable: **after execution rules are tuned fairly on both sides, does putting realistic trading costs into the training objective of a cross-sectional stock-ranking model add anything that execution engineering alone does not?** The backbone is LiMT (a multi-task cross-stock transformer, arXiv:2609.25617), treated as one representative backbone rather than as the subject of the paper.

The study is preregistered and gate-disciplined. All thresholds, grids and decision rules were fixed in advance; deviations must be logged before the deviating experiment runs. Cost-aware training itself is established finance practice (prior-art register in protocol Section 16, closest: Jensen/Kelly/Malamud/Pedersen); the intended contributions are (i) a budget-matched controlled test of it in a new regime — cross-sectional, long-only, A-share frictions, TopkDropout execution — and (ii) an evaluation-integrity audit of a label-calendar defect discovered in our own earlier pipeline.

## 2. Research questions

- **RQ1**: How much do label-calendar misalignment and future-label masking inflate IC and returns? (legacy vs clean pipeline)
- **RQ2**: After realistic costs, which models survive, and how much performance comes from execution rules alone?
- **RQ3 (primary)**: Does a cost-aware training objective (C0) beat a matched zero-fee control (G0), each with its own tuned execution rule?
- **RQ4**: Does a soft-portfolio training gain transfer to TopkDropout execution with full frictions? (the surrogate gap)
- **RQ5**: Robustness across seeds, markets (CSI300 primary, CSI500 confirmatory) and periods.

## 3. Design essentials

**Arms.** H = multi-task LiMT backbone (0.65/0.30/0.05 loss on z-scored h5/zv/zs targets). Fine-tunes initialize from H. **G0** trains on a zero-fee self-financing ledger; **C0** trains on the same ledger code with real fees (buy 5 bp, sell 15 bp, CNY 5 minimum, lot 100). The resolved configurations of G0 and C0 differ in exactly one field (`training_fee`); both select checkpoints on a separate real-fee monitoring ledger — the selection-metric asymmetry was a real code defect, found and fixed on 2026-10-17 with behavioral assertions (A-G0-1..4, A-LBL-1, all passing). **H-FT** is the extra-epochs control (supervised loss only). **C-T** (turnover penalty) is deferred, see Section 6. Baselines: ridge and two LightGBM variants on CPU.

**Primary estimand (one only).** Delta = C0\* − G0\*: the paired difference in annualized after-cost excess return versus the H00300 total-return index over 2023-01 to 2026-06-30, each arm using its own tuned execution rule; stationary block bootstrap (block ≥ 10 days, 10,000 resamples, one-sided 5% lower bound). Scenario mapping S1 (method effective) / S2 (surrogate gap or substitutable by execution engineering) / S3 (honest null) is fixed in advance and applied mechanically to 3-seed averages.

**Fair execution control.** Both arms share the same 8-rule grid (top-k 50; drop ∈ {2,5,10,20} × rebalance ∈ {1,5} days), selected on 2021–2022 with the metric averaged over seeds; the default rule (drop 5, daily) is always also reported.

**Power discipline.** Tiers P1–P3 are assigned mechanically from sigma_d measured on the validation window (MDE80 ≤ 2.5 pp / ≤ 7.0 pp / above); in P3 an S1 claim is unreachable by construction and the paper is framed around mechanism results.

**Windows and integrity.** Train 2008–2018, validation 2019–2020, rule selection 2021–2022, evaluation 2023-01–2026-06-30 — declared *exploratory* because it was inspected during earlier exploration — plus a post-cutoff holdout evaluated once as the confirmatory test. No evaluation-period (2023+) output of any new-pipeline model may be opened before Gate B; sealed files exist and remain unread. All results predating 2026-10-15 are invalid owing to the label-calendar defect and may appear only in the RQ1 legacy-vs-clean comparison.

## 4. Data and pipeline integrity

The discovered defect: the legacy pipeline applied a membership filter and then shifted labels per stock, so labels can bridge membership gaps (a form of feature/target misalignment in a known leakage family; protocol Section 16.3). The clean pipeline rebuilds label calendars end-to-end. Both CSI300 (882 symbols) and CSI500 (1,802 symbols, calendar through 2026-10-13) panels passed the W1/W2 acceptance gates: endpoint misalignment 0 for all label horizons in total and after standardization, purge intact, and two published counterexamples (SZ002074 2018-12-26, SH600549 2019-06-26) verified to the basis-point level. Delisting (SH600837, SH601989) and unknown-state policies are fixed in advance; corporate actions are booked explicitly on the execution ledger.

## 5. What is complete (as of 2026-10-18)

1. Trainer audit and the selection-metric fix, with behavioral assertions passing (2026-10-17).
2. CSI300 CPU baselines: ridge / LightGBM regression / LightGBM LambdaRank × {h5, r1} × all 8 rules × 3 seeds; arm-level rules selected seed-averaged on 2021–2022; all evaluation-period daily files (112) written to `runs/_sealed/` and never read.
3. Seed-1 neural arms: H (11 epochs, early stop, epoch 3 selected) and the **G0/C0 tau grid — six fine-tunes at tau ∈ {0.5, 0.75, 1.0}**, each 20/20 epochs, 13,360 updates, zero skipped batches, gradient norms ≤ 0.503 against a clip of 1.0. The loss identity `−log growth + 0.3·aux` holds to 3.9e-8; G0's training fees are exactly zero, C0's median is 1.90 bp/day.
4. N5 concentration measurement: validation-window soft-book N_eff and max single-name weight for all seven checkpoints at three taus (`protocol/checks/n5_neff_validation.json`), produced by an independent scoring script that reproduces the in-training monitoring values bit-exactly on every diagonal.

## 6. Principal findings so far (all validation-window, soft-layer; no evaluation output opened)

1. **The preregistered dose makes the cost term nearly inert.** The ledger term is ~0.28% of the loss by signed mean (~1.9% by per-batch absolute median); C0 − G0 is **+0.014 pp per year** at tau 0.5 (+0.008 at 0.75, +0.001 at 1.0) — about 1/140 of the S1 threshold. Mechanically: C0's turnover is only 2.46% lower; its validation cost drag is 1.925 vs 1.976 bp/day; gross growth is slightly lower; the net difference is +0.006 bp/day.
2. **C-T's penalty magnitude equals the fee term.** gamma_to × median turnover = 1.90e-4, identical to C0's fee median; at this dose C-T is predictably as inert, and its seed-1 run is deferred (recorded in v0.6 and the protocol notes).
3. **The surrogate gap is a concentration gap.** Measured validation N_eff is 260–295 at all three taus — near-equal-weight soft books — versus the 50-name executed book. The v0.5 Section 6.2 rationale (a unit-variance-score formula predicting N_eff ≈ 5–110) is wrong for this pipeline; v0.6 corrects the rationale, keeps the grid (the three taus still order concentration monotonically and span the achievable range).
4. **Both selections sit at boundaries, not interior optima.** The monitoring metric rises monotonically every epoch (real-fee and zero-fee alike; all six runs select epoch 20 = the cap), falls monotonically in tau (both arms select tau 0.5 = the grid edge), the validation supervised loss worsens monotonically, and per-day max single-name weight rises monotonically (0.71% → 1.19% at tau 0.5). Interpretation (untested): score dispersion widens during fine-tuning and the 2019–2020 window rewards tilt; tau and epochs are two paths to the same degree of freedom.
5. **The fine-tuning gain so far comes from training dynamics, not the cost term.** Either fine-tune raises the soft-layer monitoring metric by ≈ +2.8 pp/yr over un-finetuned H — twenty times the C0−G0 difference — and the first fine-tune epoch already takes the supervised validation loss below H's historical best. The H-FT control (in flight) quantifies exactly this share.

## 7. Decisions recorded before Gate B (records precede runs)

- C-T seed 1 deferred (finding 2).
- H-FT seed 1 launched as the extra-epochs control (a matrix arm, not exploratory), monitoring tau 0.5 to match the arms it controls.
- **Exploratory dose arm**: G0 and C0 at alpha_aux = 0.03 (one tenth of the preregistered 0.3, i.e. a 10× relative dose for the ledger term), tau 0.5, seed 1 — registered in the Deviation Log on 2026-10-18 *before* running; all its results are labelled exploratory/post-hoc, stay outside the Holm family, and never merge with the primary contrast. Purpose: separate "dose too small" from "mechanism absent" — without it the paper could only report a null at an uncalibrated dose.
- Provenance hardening: new runners log the initialization checkpoint's path and SHA256 in the first log line.

## 8. Status and what remains

In flight (2026-10-18, chained): scoring of the seven existing checkpoints (done) → H-FT seed 1 (running, past epoch 11 of 20; interim monitoring 0.001064/day, between H's 0.000978 and G0's 0.001094) → the dose pair → full re-scoring. Measured cost on the local RTX 4050: ~1.5–1.6 h per 20-epoch fine-tune.

Remaining before Gate B: neural-arm 8-rule selection on 2021–2022; sigma_d(final) on the executed default-rule books — measured, not assumed (protocol 4.2), reported together with daily holdings overlap, rank correlation and the executed-layer cost-drag difference, because near-identical scores can still flip names at the top-50 boundary; then the mechanical tier assignment, `gate_b.json`, and only then the one-time unsealing of 2023+ outputs. After Gate B: seeds 2–3, CSI500 confirmation, the post-cutoff holdout, and the Gate C mechanical scenario decision (S1/S2/S3).

## 9. Honest limitations (selected, full list in protocol Section 15)

The 2023–2026 window is exploratory by declaration; the evaluation window (3.5 years) has limited power, so a non-significant Delta is not evidence of no effect; starting hyperparameters (including alpha_aux = 0.3) were chosen by GPT and not tuned — the dose arm addresses this partially and is exploratory; the novelty search is partial (no CNKI/SSRN full texts); the label-calendar defect is documented for our pipeline only, with no claim about Qlib's own pipeline.

## 10. Artefact map

| Path | Contents |
|---|---|
| `PREREGISTRATION_v0.5.md` / `_v0.6.md` | Locked draft and corrected rationale; no rule changes between them |
| `protocol/experiment_report_2026-10-17.md` | Trainer fix, data gates, CPU baselines |
| `protocol/tau_grid_report_2026-10-18.md` | Six-run audit, gap decomposition, N5, decisions (Chinese) |
| `protocol/deviation_log.md` | First entry: the exploratory dose arm, registered before running |
| `protocol/checks/n5_neff_validation.json` | Validation-window N_eff / max-weight / monitoring values, 7 checkpoints × 3 taus |
| `runs/_sealed/csi300/` | 112 evaluation-period daily files, unread until Gate B |
| `src/` | Runners (H, tau grid, H-FT, dose arm) and the validation scoring script |
