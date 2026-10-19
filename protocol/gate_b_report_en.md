# Pre-Gate-B Status Report (Revision 2) — Cost-Aware Training for LiMT (Clean-Pipeline Study)

Workspace: `D:\research\new_LiMT`. Governing protocol: `PREREGISTRATION_v0.6.md`. This revision incorporates the external review round: benchmark forensics, CPU-baseline arms, unseal scope, frozen inference settings, engineering-fix hashes, and the frozen benchmark seed policy. **All timestamps in this report are in America/Los_Angeles time (PDT).** No evaluation-period (2023-01 to 2026-06-30) output has been opened; `gate_b.json` has not been written.

---

## 1. Executive summary

- Three-seed CSI300 matrix complete (8/8 runs, zero failures, config hash `393b1993…` throughout); measured sigma_d(final) = 1.19% → MDE80 = 1.58 pp → **Tier P1**.
- **Both primary arms selected the default execution rule** after seed-averaging, so Delta equals the default-rule paired difference and SC3 coincides with the primary (stated explicitly; SC3 is not double-counted in the Holm family).
- Six Qlib-default benchmark arms trained and forensically verified (full training rows, boosting counts, validation IC — the "implausibly fast" flag is resolved); stochastic benchmark arms are being extended to seeds 2–3 under a Deviation Log entry recorded before the runs; XGBoost and Transformer are frozen as deterministic single runs.
- CPU baseline arms (ridge, LightGBM regression, LightGBM LambdaRank × {h5, r1}) with their selected rules and configuration hash are now part of the readiness list.
- Unseal scope at Gate B: **CSI300 only**.
- CSI500 H is complete for three seeds on Kaggle and verified (Gate C material).

## 2. Hash chain (all before the runs they cover)

| Event | Commit | Time (PDT) |
|---|---|---|
| Baseline snapshot | `8a7a4c45` | 2026-10-18 01:45:27 |
| Deviation Log, five entries (benchmark reframe; stamp-duty sensitivity; TOST ±2.0 pp; rule-selection seeds option (b); diagnostics disclosures) | `c8c6c27` | 2026-10-18 01:45:28 |
| Gate A fills ([FILL] completed or explicitly marked open; Section 6.5 ticked; 14/14 assertions rerun) | `841f00be` | 2026-10-18 01:54:08 |
| Gate A record | `8820fa5` | 2026-10-18 01:54:51 |
| Engineering fix 1 — H seed-2/3 scoring path (date-agnostic glob; see §3) | `080a1c7` | 2026-10-18 12:57:35 |
| Engineering fix 2 — benchmark adapter (Timestamp master index; `n_jobs=0` under Windows spawn) | `48a78d6` | 2026-10-18 13:26:24 |
| Benchmark forensics + Deviation Log entry freezing the benchmark seed policy (before any seed-2/3 run) | `9a578bb` | 2026-10-18 18:50:27 |
| Readiness revision 2 (this document) | `1dc78d7` + this commit | — |

Engineering fixes are "engineering fix, no protocol change" and are logged separately in `protocol/engineering_log.md`.

## 3. Ordering disclosure (H scoring-path fix)

The H seed-2/3 score artefacts were written at 12:56:48 and 12:57:13 PDT; the fix commit `080a1c7` is at 12:57:35 PDT — the code edit was applied to the working tree before the scoring ran, and the commit landed 22–47 s after the artefacts. Evidence chain: the pre-fix run demonstrably SKIPped both tags with the stale path printed in its log; each score artefact's meta records the correct checkpoint path and sha256; every H rule-selection output (the selection json, 13:12:46 PDT) strictly postdates the commit. Recorded as-is, not papered over.

## 4. Power and mechanical tier

| Quantity | Value |
|---|---|
| sigma_d(prelim) — ridge_h5 vs H on executed books (upper-bound-style proxy, recorded as such) | 6.78% |
| **sigma_d(final)** — C0 − G0, seed 1, tau 0.5, default rule, executed books, 486 paired validation days | **1.189%** |
| **MDE80 = 2.487 × sigma_d / sqrt(3.5)** | **1.581 pp** |
| **Mechanical tier** | **P1** |

Diagnostics: holdings overlap 96.12% (median 96.08%); rank correlation 0.99975; executed cost-drag difference C0 − G0 = **+0.019 pp/yr (C0 pays slightly more at the default rule on validation; opposite sign to the soft layer; recorded as measured)**.

## 5. Three-seed training (CSI300)

| Seed | H | H-FT | G0 | C0 (validation soft-layer monitoring value, mean log growth/day) |
|---|---|---|---|---|
| 1 | 11 epochs, ep3 | 0.00108742 | 0.00109438 | 0.00109494 |
| 2 | 11 epochs, ep3 | 0.00106132 | 0.00106691 | 0.00106799 |
| 3 | 11 epochs, ep3 | 0.00108067 | 0.00108432 | 0.00108599 |

H early-stops at 11 epochs with the checkpoint at epoch 3 in all seeds; every fine-tune runs to the 20-epoch cap. Soft-layer C0 − G0 is positive in all three seeds but of order 0.03–0.04 pp/yr.

## 6. Tuned-execution rule selection (2021–2022, seed-averaged, 0.5 pp lower-turnover tie-break)

| Arm | Selected rule | Seed-averaged annualized excess vs H00300 |
|---|---|---|
| **G0** | **drop5-every1d (the default rule)** | 20.69% (runner-up 18.28%; margin 2.4 pp, no tie) |
| **C0** | **drop5-every1d (the default rule)** | 20.23% (runner-up 19.25%; margin 0.98 pp, no tie) |
| H | drop20-every5d | 18.33% |
| H-FT | drop2-every1d | 20.03% (0.2 pp from best → tie-break to lower turnover) |
| G0dose003 (exploratory) | drop5-every1d | — |
| C0dose003 (exploratory) | drop2-every1d | — |

**Estimand statement.** Delta = C0\* − G0\* equals the default-rule paired difference; SC3 coincides with the primary and is excluded from the Holm family to avoid double counting. The t+2 integrity assertion is logged in every runner (terminal accounting mark 2022-12-30; last execution fill 2022-12-29; no price dated after 2022-12-31 is read).

## 7. CPU baseline arms (Section 8 essential arms)

Same selection convention; configuration = `src/cpu_baselines.py`, sha256 `4c585a0f…` (unchanged since the baseline commit): ridge_alpha = 10, deterministic single run; LightGBM arms use the Qlib Alpha158 tree parameters (num_leaves 210, lr 0.2, published L1/L2), seeds 1–3, 1000-round cap with early stopping 50 on validation.

| Arm | Label | Seeds | Selected rule |
|---|---|---|---|
| ridge | h5 | 1 (deterministic) | drop5-every5d |
| ridge | r1 | 1 | drop2-every1d |
| lgbm-reg | h5 | 1–3 | drop10-every5d |
| lgbm-reg | r1 | 1–3 | drop5-every1d |
| lgbm-rank | h5 | 1–3 | drop2-every1d |
| lgbm-rank | r1 | 1–3 | drop5-every5d |

## 8. Qlib-default benchmark arms (descriptive; outside the Holm family)

**Specification wording**: Qlib defaults, **except label normalization: CSRankNorm not re-added** — all other hyperparameters verbatim from the official qlib v0.9.6 `examples/benchmarks` Alpha158 configurations (`configs/benchmarks_qlib_defaults_v1.json`, sha256 `da5b7425…`); clean panel, FilterCol 20-column subset, z_r1 label; train 2008–2018 with early stopping on 2019–2020, confirmed per model: the four neural models early-stop on validation loss (11/11/11/7 epochs, best at epoch 0–1 — recorded as a descriptive fact); XGBoost runs under qlib's own fit defaults (num_boost_round 1000, early_stopping_rounds 50 on validation — the yaml's n_estimators is not consumed by qlib's train path; disclosed); DoubleEnsemble trains its configured 3 sub-models × 28 rounds with no early stopping.

**Forensics (resolving the "implausibly fast" flag)** — `protocol/checks/benchmark_forensics.json` plus the correction below: training rows **768,935** (train) / 145,719 (valid), within 0.08% of the CPU baseline's 768,338 (the baseline additionally requires all 158 features finite; no training-set shrinkage). The initial XGBoost/DoubleEnsemble runs used the 20-column FilterCol subset; the external review caught that **their official yamls specify no FilterCol** (it appears only in the four neural ts configs), so both were re-run on the full 158-column handler under a Deviation Log entry recorded first (DoubleEnsemble's num_models = 3 was verified correct). Corrected values: XGBoost 40 s, validation r1 IC **0.0445**; DoubleEnsemble 574 s, IC **0.0401**; the CPU LightGBM-r1 reference (158 features) is 0.0352 — the corrected GBDTs now rank sensibly above it. The four neural models ran on FilterCol-20 as their yamls specify (early stop 11/11/11/7 epochs, best at epoch 0–1; IC 0.020–0.029 — published defaults, untuned, by design, reported plainly).

**Seed policy (frozen by the Deviation Log entry committed before any run)**: LSTM / GRU / ALSTM / DoubleEnsemble use seeds 1–3 with rules selected on the seed average (Section 8A "3 seeds if stochastic"); seeds 2–3 are running now. XGBoost and Transformer remain deterministic single runs recorded at **seed 0** (xgboost's default seed with the deterministic single-machine hist algorithm; Transformer's official yaml pins seed 0 — replicating it would produce identical copies).

**Environment disclosure**: pyqlib installed from the relabelled cp312 wheel; the `qlib.data._libs` stubs **raise NotImplementedError on any invocation** — there is no silent fallback path; the qlib expression engine is never on this study's code path.

## 9. Dose-arm diagnostics

Validation 5-day IC: H 0.0382; H-FT 0.0556; G0 0.0549; C0 0.0548; G0dose003 0.0396; C0dose003 0.0435. Soft vs top-k on 2021–2022: regular arms soft − topk ≈ −12 to −13 pp/yr; the concentrated dose book G0dose003 loses −29.7 pp/yr at the soft layer while the same model executed as top-k earns +20.3 pp/yr (gap −50 pp). **The soft-book gain observed on validation does not survive outside the selection window — direct evidence for the surrogate-gap section.**

## 10. CSI500 H (Kaggle; Gate C material)

| Seed | Epochs (early stop) | Selected epoch | validation auxiliary loss | checkpoint sha256 |
|---|---|---|---|---|
| 1 | 12 | 4 | 0.94321 | `06cd06e5…` |
| 2 | 13 | 5 | 0.94280 | `d6ef1297…` |
| 3 | 14 | 6 | 0.94195 | `4d961d0b…` |

All three pass the anchors: data_version `bae7115e…` (identical across seeds and to the local pre-computed value), config hash `393b1993…`, trainer sha; local recomputation of every checkpoint sha matches manifest and log. Execution environment recorded (T4, torch 2.11+cu128, Python 3.13.15); cross-device bit-level identity is not claimed. Prerequisites recorded for the future CSI500 execution feed (baostock): board-specific limit bands (main board 10%; ChiNext 20% from 2020-08-24; STAR 20%; ST 5%; no-limit sessions such as listing days), vendor difference from RESSET disclosed, spot-checks against an independent source.

## 11. Unseal scope

**CSI300 only.** All CSI500 evaluation outputs remain sealed until CSI500 G0/C0 exist and their rules are selected on 2021–2022.

## 12. gate_b.json draft (to be written only after final review)

The **frozen analysis code** is committed at `src/gate_b_analysis.py`, sha256 `5742585a…` (commit `4f60ef4`), with a passing synthetic self-test; it computes Delta, the stationary bootstrap (block 10 / 10,000 resamples / rng seed 20261019 / one-sided 5%), TOST (2.0 pp, 90% CI), Newey-West (lag 5), SC1/SC2/SC5/SC6/SC7, Holm over SC1/SC5/SC6/SC7 (SC2 is a ratio reported per-seed; SC4 enters at Gate C), the S1-condition table, and the stamp-duty sensitivity via a date-dependent sell fee whose replay is diagonal-verified against the production replay. The script refuses to run unless gate_b.json records its own sha256.

```json
{
  "unseal_scope": "CSI300 only; CSI500 evaluation outputs stay sealed until CSI500 G0/C0 exist and their rules are selected on 2021-2022",
  "analysis_sha256": "5742585ac41a9a291b0eaefed0086cc337220ad9cdf73aee290f16d3430325b6",
  "config_hashes": {
    "neural_formal": "393b1993...", "trainer": "a960a995...", "data_version_csi300": "3c70f663...",
    "execution_feed_resset": "aff735c3...", "benchmark_h00300": "d6b21b61...",
    "cpu_baselines": "4c585a0f...", "benchmarks_spec": "<sha256 of the corrected benchmarks_qlib_defaults_v1.json>",
    "code_commit": "<HEAD at write time>"
  },
  "fee_schedule": {"buy_bp": 5, "sell_bp": 15, "min_fee_cny": 5, "lot": 100, "capital_cny": 1e8,
                   "stamp_duty_sensitivity": "sell 10bp from 2023-08-28 (HKEX CT/137/23), reported alongside, never decisive"},
  "tau": 0.5,
  "tier": "P1", "sigma_d_final": 0.011894, "mde80_pp": 1.5811,
  "inference": {"bootstrap": "stationary, mean block length 10, 10000 resamples, rng_seed 20261019",
                "primary_bound": "one-sided 5% lower bound of Delta",
                "newey_west_lag": 5,
                "holm_family": ["SC1", "SC2", "SC4", "SC5", "SC6", "SC7"],
                "sc3_note": "SC3 coincides with the primary (both arms selected the default rule); not double-counted",
                "sc4_note": "SC4 enters the family when CSI500 lands at Gate C; the family closes then",
                "tost": {"margin_pp": 2.0, "ci": "90% stationary bootstrap, same settings"}},
  "selected_rules": {"H": "drop20-every5d", "H-FT": "drop2-every1d", "G0": "drop5-every1d", "C0": "drop5-every1d",
                     "G0dose003_exploratory": "drop5-every1d", "C0dose003_exploratory": "drop2-every1d",
                     "ridge_h5": "drop5-every5d", "ridge_r1": "drop2-every1d",
                     "lgbm-reg_h5": "drop10-every5d", "lgbm-reg_r1": "drop5-every1d",
                     "lgbm-rank_h5": "drop2-every1d", "lgbm-rank_r1": "drop5-every5d"},
  "default_rule": "drop5-every1d (topk 50, drop 5, daily) — also the selected rule of both primary arms",
  "benchmark_rules": {"<filled from the 3-seed selection once seeds 2-3 land>": "..."},
  "seeds": {"neural": [1, 2, 3], "cpu_lgbm": [1, 2, 3], "ridge": [1],
            "benchmarks_stochastic": [1, 2, 3],
            "benchmarks_deterministic": {"XGBoost": 0, "Transformer": 0},
            "dose": [1]},
  "deferred_arms": {"C-T": "deferred per v0.6 Section 8 (penalty magnitude equals the fee term already shown near-inert)"},
  "holdout_minimum_days": 120,
  "exploratory_arms": ["G0dose003_seed1", "C0dose003_seed1", "six Qlib-default benchmark arms (descriptive)"],
  "estimand_note": "Delta = default-rule paired difference (both arms selected the default rule)"
}
```

After writing and committing, the sha256 of gate_b.json is recorded in `protocol/gates.md`.

**Gate A open items proposed for closure (user may veto before unsealing):** hard write-up date **2027-01-15** (chosen so the post-cutoff holdout can reach its 120-day minimum, which arrives around end of 2026); GPU budget **no experiments beyond the preregistered matrix and Deviation-Log-registered arms; remaining planned compute bounded at 60 GPU-hours** (CSI500 feed build is CPU/network work; CSI500 fine-tunes are the bulk of the remaining GPU budget).

## 13. Post-unseal execution

Evaluation-window scoring for every arm at the selected and default rules; sealed files written first, then read in one pass; mechanical S1/S2/S3 mapping at Gate C on the three-seed CSI300 results; CSI500 confirmation afterwards.

## 14. Artefact map

| Artefact | Contents |
|---|---|
| `protocol/engineering_log.md` | Engineering fixes with commit hashes ("engineering fix, no protocol change") |
| `protocol/checks/benchmark_forensics.json` | Rows / rounds / validation IC for all six benchmark arms |
| `protocol/checks/sigma_d_prelim_*.json`, `sigma_d_final_*.json` | Power inputs and diagnostics |
| `runs/csi300_neural_selection.json` | 14 per-seed grids + seed-averaged selections (h5 arms) |
| `runs/csi300_cpu/selection.json` | CPU baseline grids + selections (h5/r1 arms) |
| `runs/csi300_bench_selection.json` | Benchmark grids + selections (r1 arms; 3-seed version pending) |
| `runs/csi300/<arm>/h5\|r1/<rule>/seedN/daily.csv` | Readable selection-window daily books |
| `runs/scores/*.npz` + `.meta.json` | Pinned score artefacts (checkpoint sha per tag) |
| `protocol/checks/dose_diag_*.json` | Full dose diagnostics, 14 tags |
| `runs/csi500_H_kaggle/` | CSI500 H checkpoints, histories, manifests (verified) |
| `configs/benchmarks_qlib_defaults_v1.json` | Pinned Qlib-default benchmark specification |
