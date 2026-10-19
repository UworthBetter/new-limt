# Deviation-log draft (pre-Gate-B, revision 2) — English version

> Revised 2026-10-19 01:50 PDT after external review round 4. Status: DRAFT. Not committed. No gate_b.json written. No training run. Chinese master: `new_DL_draft_pre_gate_b.md` (that version prevails if the two differ).
>
> Code baseline:
> - GitHub HEAD: `bc0dc13d848c8c7beeddaf9d4e0bca5c3a11aca4` (2026-10-18 23:03:26 PDT); `git pull` shows no new commits.
> - The laptop (`D:\research\new_LiMT`) is **1 commit ahead and not pushed**: `b7306e0d4b0d37c8456456264ed8c7a2a9eca10a` (2026-10-19 00:53:04 PDT, "frozen analysis v2"). It also has uncommitted log changes and an untracked `protocol/checks/qlib_b3_master_verification_2026-10-19.md`.
>
> Analysis artefact:
> - v1: sha256 `5742585ac41a9a291b0eaefed0086cc337220ad9cdf73aee290f16d3430325b6` (commit `4f60ef436b271ab07b0c5d02fcf2f7c04a89c215`).
> - v2: sha256 `7d90d218e661a242512e542e309ea49e4858c428da6f2ea85018ef22e6db64fa` (commit `b7306e0…`). v2 overwrote v1 at the same path, so v1 now survives only in git.
>
> qlib reference: v0.9.6, `5a84aaf1dc402232646ad2836dea33ff15886552`.
>
> Blinding: this draft contains no metric value from eval/full-test (2023+) or from rule-select. Only code, dates, shapes, masks, meta fields, timestamps and the min/max bounds of pre-2019 training-segment features were inspected.

## DL-1 (B1): XGBoost and Transformer get seeds 1–3; seed 0 kept as a descriptive public-config reproduction

This supersedes "XGBoost and Transformer stay single deterministic runs" (DL commit `9a578bb0ccee6d33be2198b31cad51adfee1e46f`).

Why:
- **XGBoost** uses subsample 0.8789 and colsample 0.8879, which are below 1, so the model is stochastic in xgboost's own `seed`.
- **The seed never reached XGBoost.** qlib `xgboost.py:18-20,58-67` passes kwargs verbatim into `xgb.train`, and neither the yaml nor the spec sets a seed. The global seeds in `run_benchmarks.py:146-148,177` do not affect xgboost. `:143-144` actually forbids seed≠1. Meta shows `effective_seed=0` and `nthread=20`.
- **Synthetic test** (xgboost 3.4.1, the same version as laptop venv_bench):
  - passing no seed gives exactly the same result as seed 0;
  - seeds 1, 2 and 3 give different results;
  - the global numpy seed has no effect.
- **Transformer:** `pytorch_transformer_ts.py:63-65` reseeds from the config seed before the model is built (`:67`). `run_benchmarks.py:185-186` together with spec `seed: 0` forces seed 0 regardless of `--seed`.
- **Disclosure:** `n_estimators: 647` is unused. Training actually runs `num_boost_round=1000` with early stopping 50.

Pre-run gate: XGB params must contain a seed, and predictions must differ across seeds. For the NNs, the initial state_dict hash must be stable within a seed and different across seeds. If either check fails, stop.

## DL-2 (B2(a), confirmed): CSRankNorm restored for LSTM/GRU/ALSTM/Transformer, 3 seeds each

- **What the label becomes:** per-day `(rank(pct) − 0.5) × 3.46`, as in qlib `processor.py:331,360-362`, matching the official NN yamls. It applies to train and valid learning labels only, including the early-stopping loss. Scores are not transformed.
- **CSRankNorm(z_r1) ≡ CSRankNorm(raw r1):** `cs_z` (`v2_calendar_labels.py:47-53`) is, for each day, a positive-slope affine map (std > 0) over the set `label_mask_r1 = universe & isfinite(raw r1)` (`:98`). It is strictly increasing, so the ranks (ties included) are identical. The result also equals qlib's CSRankNorm on raw 1-day returns, as long as the ranking set is the same.
- **Ranking set:** every stock whose label is finite that day, i.e. `label_mask_r1[t]`. It must **not** be `isfinite(z_r1)`, because `cs_z` zero-fills outside the mask (`:48`) so z is finite everywhere. It must also **not** be "the rows kept by flt", which adds window and other filters.
- **XGB/DE are unchanged:** the Alpha158 default learn processors are DropnaLabel + CSZScoreNorm (`handler.py:37-40,106`), which matches z_r1.
- **Old z-score NN runs** become descriptive controls only. After the reruns:
  - write new forensics;
  - re-select rules for LSTM, GRU, ALSTM, Transformer and XGBoost, averaging over 3 seeds;
  - keep the DoubleEnsemble rule as it is.

## DL-3: corrections to the spec JSON and feature-processing disclosure → new `configs/new_benchmarks_qlib_defaults_v2.json`

1. **Correct "class-default CSRankNorm"** (v1 `:9`). The XGB/DE class default is CSZScoreNorm.
2. **Remove the "only deviation" claim** everywhere it appears: v1 `:9`, `run_benchmarks.py:7-8`, `gate_b_report_en.md:84`, `gate_b_ready_2026-10-19.md:64`.
3. **Feature processing (verified).**
   - Official XGB/DE pipeline: raw Alpha158 (`handler.py:105` infer_processors=[]), meaning no normalisation, no clip, NaN kept.
   - Our pipeline, for all arms (`v2_calendar_labels.py` sha `3abd456f…`):
     - `:119-120` raw Alpha158;
     - `:124` RobustZScoreNorm, fitted 2008-01-01..2018-12-31, `clip_outlier=True` (= `np.clip(-3,3)`, qlib `processor.py:299-300`);
     - `:126` fillna(0);
     - `:127` **VWAP0 set to 0**.
   - Panel bounds, training segment only. Laptop panel `a25879f4…`, shape (4707, 882, 158) float32:
     - min −3.0, max +3.0;
     - 98 features reach +3 and 84 reach −3;
     - 0 NaN;
     - VWAP0 is all zeros.
     - The ModelScope fix1 panel `cb774f59…` gives identical results.
   - Consequences to disclose:
     - the clip is irreversible;
     - zero-fill replaces the trees' native missing-value handling;
     - VWAP0 is a constant feature;
     - the normalisation fit window is 2008–2018, whereas the NN yamls use 2008–2014.
   - No rerun is needed.
4. **Build environment.** Alpha158 was computed by real qlib with compiled `_libs`, not stubs.
   - **Laptop panel:** built 2026-10-17 05:48–05:54 PT (meta: workers=4, 400 s).
     - **There is no CSI300 build log and no environment freeze (evidence gap).**
     - The sibling CSI500 log from the same session shows `D:\research\limt-repro\.venv`.
     - That venv today has Python 3.13.9, **pyqlib 0.9.8.dev34**, numpy 2.4.1, pandas 2.3.3, and compiled `rolling/expanding.cp313-win_amd64.pyd`. The `.pyd` is what actually gets imported.
   - **ModelScope fix1 build:** its freeze (`3878b1a0…`) shows Python 3.11.11, qlib built from source at main `be725493…` (2026-09-16), and Cython 0.29.36.
   - **Benchmark venv_bench:** Python 3.13.9, pyqlib 0.9.6 with stub `_libs`. It only reads the panel, so the stubs do not matter.
   - §8 must disclose the two qlib versions. **Alpha158 diff, verified read-only on 2026-10-19:** the expressions are identical.
     - **Method:** sha256 and difflib, comparing laptop `venv_bench` (pyqlib 0.9.6) against limt-repro `.venv` (0.9.8.dev34).
     - **Byte-identical files:** `contrib/data/loader.py` (Alpha158DL) `472f9dc8…`, `data/ops.py` `6f785624…`, `contrib/data/handler.py` `19722325…`, `data/base.py` `cc29103b…`, `_libs/rolling.pyx` `54824a8d…`, `_libs/expanding.pyx` `2aca0fbe…`.
     - **`data/dataset/processor.py` differs in two places only:**
       - the Fillna implementation style (same semantics; our build calls pandas `fillna(0)` directly at `v2_calendar_labels.py:126`);
       - CSRankNorm's `groupby(..., group_keys=False)` (affects index shape only, and the panel build does not use it).
       - RobustZScoreNorm is unchanged.
     - **`__init__.py`** differs only in version sourcing and the NFS mount code.
     - **Why the laptop hashes differ from the v0.9.6 tag source** (`5a84aaf1…`): line endings only. Converting the tag's LF to CRLF reproduces the laptop sha for all 7 files (e.g. loader.py: LF `814b7f7a…` → CRLF `472f9dc8…`).
     - **Conclusion:** Alpha158DL and its ops have identical content in the 0.9.6 tag, venv_bench and 0.9.8.dev34. The §8 text and the "official defaults" claim are therefore unblocked.
5. **Seed policy:** rewrite `seed_policy` to follow DL-1/DL-2.
6. **New fields** to add: `label_processor`, `feature_processing_official/ours`, `spec_supersedes`, `benchmark_note`.

## DL-4: Holm family aligned to frozen v2

- v2 `:71` still reads `HOLM_KEYS=("SC1","SC2","SC5","SC6","SC7")`.
- SC2 is a structural NaN (`:368-370`) and is filtered out (`:401-403`), so effectively m=4.
- **The v2 docstring at `:11-18` is stale.** It still lists SC2 as a paired bootstrap inside the family. The implementation governs.
- gate_b.json should record `holm_family_gate_b=["SC1","SC5","SC6","SC7"]` (m=4, provisional) together with an `sc2_note`.
- The Gate C family (SC1, SC4, SC5, SC6, SC7; m=5) comes from a separate `src/new_gate_c_holm.py`. It must be frozen and hashed before CSI500 is unsealed.

## DL-5 (B6, confirmed option (b)): holdout becomes descriptive

- **Looks:** at most two.
  - An interim look at submission.
  - A final look at ≥120 trading days, reported regardless of outcome.
- **Deadline:** 2027-01-15 stays as a deadline only, decoupled from the holdout.
- **Places to re-label:** PREREG `:28` (rule 3, see DL-11), `:258-262`, `:286`, `:293`, `:380`, `gates.md:34`, `paper/README.md:4`, `plan.md:21`.
- **Integrity assertion — gap in v2.** No `max_date_touched ≤ 2026-06-30` assertion exists (no `max_date` anywhere; `2026-06-30` only appears in the docstring at `:273`). Status by data type:
  - **Labels:** `eval_ic` now uses `loss_masks["eval"]` (`:278`), but nothing asserts or logs the latest label endpoint touched.
  - **Prices:** replay reaches t+2. The last signal day is 2026-06-26, and t+2 is 2026-06-30, which is in bounds by logic but not asserted.
  - **Benchmark:** the full H00300 file is loaded (`:308`) and only sampled at row dates. Nothing is asserted.
  - **Resolution (review round 4):** the guard goes inside analysis v3, so there is one hash (DL-12). It asserts on the dates actually accessed: replay prices, label endpoints through the eval mask, and the benchmark rows sampled. It logs `max_date_touched` and fails hard if any date is after 2026-06-30. No separate script.
- **The eval_ic leak itself is confirmed and fixed in v2.** Evidence comes from the laptop data (labels `7bf5633b…`, splits `0ab69b33…`); the code shas are identical between the box and the laptop.
  - The h5 label endpoint is t+6.
  - There are 841 eval signal days (2023-01-03..2026-06-26).
  - Signal days 06-23..06-26 exit on 07-01..07-06, each with 300 real labels.
  - `loss_mask_eval` is 0 for those 4 days and 300 up to 06-22, so v2 excludes them.
  - v1 also scored unlabelled stocks as label 0, because `cs_z` zero-fills. v2 fixes this too, since loss_mask ⊆ label_mask (`v2_w4_data.py:64`).
  - Affected: SC7 IC and the S1.3 IC guard only.

## DL-6 (B4): TOST bound unchanged (pointing at v2)

- **Bound:** ±2.0 pp/yr (`deviation_log.md:14`, `c8c6c2799f4ea60a9ab6228a3b283490a7b4c72b`). v2 implements it in `tost()` (`:136-138`) and `conclusion()` (`:141-156`).
- **Pre-committed reporting:**
  - the CI half-width and its ratio to the measured fee drag;
  - the fact that the bound was set after the seed-1 validation results.
- **What the CI covers:** it resamples days only, on the seed-averaged series (`:350-351`), so it excludes seed variance. Per-seed deltas are always reported.
- **Method:** bootstrap is primary; Newey–West (lag 5) is a sensitivity check.

## DL-7: TOST verdict table — implemented in v2

- **Precedence:** superior > inferior > equivalent > uncertain.
- **Flags:** inferior and equivalent can both hold (CI inside (−2, 0)), and so can superior and equivalent. Each flag is set independently.
- **Wording:** "uncertain" is never written as "no effect". The required wording is "neither 0 nor an effect ≥2 pp can be excluded".
- **Definitions** (LB = one-sided 5% bound = lower end of the 90% CI):
  - superior: LB > 0;
  - inferior: UB < 0;
  - equivalent: TOST, i.e. −2 < LB and UB < 2.
- **Relation to S1–S3:** they remain mechanical, as in §7. The table only governs wording:
  - superior with Δ < 2 pp is not S1;
  - inferior falls under the S3 "negative" case;
  - equivalent falls under S3 as a "bounded null", unless S2 fires.
- **Check of v2 `conclusion()`:** precedence and flags match, and the string "uncertain" is used. Remaining issues:
  - (1) The output (`:371-373`) is not wired into `s_mapping`/S1–S3, and the "no effect" ban is not enforced in code.
  - (2) The self-tests at `:430-434` use lower5 ≠ CI lower end. This is not representative, though harmless.
  - (3) `lower5 == ci90[0]` holds in real runs, so the code is consistent with the definitions.

## DL-8 (A2): which spec hash governs which run

The governing spec is the one in force when the run started, judged by the log CreationTime (PT). Meta files carry no spec hash; this is a gap that the v2 runner must fill.

| Spec | sha256 | Runs governed |
|---|---|---|
| v1-a | `da5b742579bf8c5a15e049ec03dd59e0ba6f39da59bd6582b1e338b5a883ecff` (`6be7ae58…`, 02:48:37) | • first 20-column XGB/DE runs (02:35/02:40, superseded)<br>• LSTM/GRU/ALSTM/Transformer seed1 (bench_gpu_fix2 13:26:24 → 16:07:55)<br>• **LSTM_seed2** (bench_seed2 started 18:50:57, before 19:15:55) |
| v1-b | `e189f6cbe3453d172fc407b6bf2117be15bcba7c72c698f53f125d92533e5165` (`5a8b9a42…`, 19:15:55) | • XGBoost_seed1 158 columns (19:16:05; effective_seed 0; nthread 20; fit 40 s)<br>• DE seed1 (19:29:18; 158 columns)<br>• GRU/ALSTM seed2 (20:13; 20 columns)<br>• GRU/LSTM/ALSTM seed3 (20 columns)<br>• **DE seed3** (21:55:46 → 22:05:05; n_features=158)<br>• **DE seed2** (22:33:51 → 22:43:13; n_features=158) |
| v2 | pending | the 15 new tags |

- **Selection log:** the 3-seed selection log `rule_grid_bench_final2.log` (22:45:16–22:57:22) post-dates every 158-column tree artefact. Its end time matches the mtime of the selection JSON. The JSON's grid values were not opened.
- **Hyperparameters:** v1-a and v1-b have identical `models.*.kwargs` (diffed).
- **Stale references:** `gate_b_report_en.md:84` and `gate_b_ready_2026-10-19.md:94` still cite da5b7425.

## DL-9 (A3/A6): hash chain and dose-arm erratum

**Hash chain** (full shas as in the Chinese master):

| Commit | Time | Content |
|---|---|---|
| `8a7a4c45493b562712177688c9d52118a51b3fb3` | 01:45:27 | baseline |
| `c8c6c2799f4ea60a9ab6228a3b283490a7b4c72b` | 01:45:28 | first DLs |
| `841f00be…` | 01:54:08 | |
| `8820fa5b…` | 01:54:51 | |
| `080a1c73…` | 12:57:35 | |
| `48a78d6b…` | 13:26:24 | |
| `9a578bb0…` | 18:50:27 | |
| `d3a97b62…` | 18:50:57 | |
| `5a8b9a42…` | 19:15:55 | |
| `4f60ef43…` | 19:44:05 | analysis v1 |
| `9430acff…` | 20:00:13 | |
| `caee9164…` | 20:02:10 | |
| `30ff5389…` | 23:03:10 | |
| `bc0dc13d…` | 23:03:26 | |
| `b7306e0d…` | 2026-10-19 00:53:04 | analysis v2, local only |

**Erratum (translation of the adopted Chinese text):**

> The repository's first commit (`8a7a4c45493b562712177688c9d52118a51b3fb3`, 01:45:27 PDT) and the first DL batch (`c8c6c2799f4ea60a9ab6228a3b283490a7b4c72b`, 01:45:28) both post-date the dose-arm runs, whose results are in that same commit. "Logged before run" (v0.6 §8 line 215; gate_a_record "precedes every run it covers") does not hold for the dose arm and is withdrawn; the dose arm was always a post-hoc exploratory analysis. Every other DL row in c8c6c279 concerns a run or analysis that started after 01:45:28, as verified by the hash chain.

**Row-by-row check of the last sentence: it is FALSE for one row.** Evidence: laptop file CreationTime/LastWriteTime (Pacific) plus git. These timestamps are corroborative only, not tamper-proof.

| Row | Earliest evidence | After 01:45:28? |
|---|---|---|
| ① dose arm | run dir created 10-17 20:26:28 | no (the erratum) |
| ② benchmark reframe | first bench log 02:35:47; code first appears in `6be7ae58` | yes |
| ③ stamp duty | code first appears in `4f60ef4` (19:44:05); not yet run on eval | yes |
| ④ TOST ±2 pp | same as ③ (bound set after the seed-1 results, see DL-6) | yes |
| ⑤ option (b), seeds 2–3 | `seed2_chain` log 02:02:31 | yes |
| ⑥ diagnostics (gradient decomposition, "H-FT explains 94%", dose-arm cost term) | `grad_decomp` log 10-17 23:50 – 10-18 00:13; `hft_dose_score` 10-17 19:13 – 10-18 00:20; HFT run dir 10-17 19:18 | **no** |

**Replacement for the last sentence (review round 4 wording, verbatim):**

> "Rows ②–⑤ concern runs that started after 01:45:28 PDT (② and ⑤, by git history and file timestamps, which are corroborative and not tamper-proof) or analyses not yet run (③ and ④). Row ⑥ discloses diagnostic analyses completed before the commit (2026-10-17 19:13 to 10-18 00:20 PDT). It is a disclosure, not a registration, and those analyses are exploratory."

Row ⑥ never claimed "before run", so nothing is withdrawn; it is only classified.

## DL-10: analysis artefact v2

- **Change:** v1 `5742585a…` → v2 `7d90d218…` (`b7306e0…`, local only).
- **Bug fix:** `eval_ic` now uses `loss_masks["eval"]` (`:268-284`). This removes:
  - (a) the 4 post-cutoff h5 endpoints;
  - (b) unlabelled stocks entering as label 0.
- **Protocol changes made before unsealing:**
  - the SC5 Holm p is now a stationary-bootstrap p (`sc5_pvalue`, `:287-297`), with the C0 arm pre-specified and both arms reported;
  - the `conclusion()` table was added.
- **SC5 p construction:** `p = 2·min(P*(ρ*≤0), P*(ρ*≥0))`. This is a two-sided p obtained by inverting a percentile CI, not a p from a null-centred distribution.
  - The hypothesis is directional, and SC1/SC6/SC7 use one-sided `P*(mean*≤0)`.
  - Suggestion: make SC5 one-sided, `P*(ρ*≤0)`. The current two-sided version is conservative but inconsistent with the rest of the family.
  - **Resolved in review round 4:** one-sided in v3 (DL-12).
- **New finding — the SC6 sign looks reversed:**
  - Code: `:336` defines `drag = c_fee − g_fee`, and SC6 tests mean > 0, i.e. "C0 has higher fees".
  - Prereg: §3 (`:70-72`) gives Δ = Δ_gross + Δ_cost, and S2 (`:189`) reads "Δ_cost positive … costs fell".
  - Since the code has delta = gross − drag, consistency requires Δ_cost = g_fee − c_fee = −drag.
  - So the SC6 estimate, CI and Holm p are all sign-flipped. v1 has the same issue.
  - **Resolved in review round 4:** this is a bug, fixed in v3 (DL-12).
- **Run status:** neither v1 nor v2 has ever been run with `--run`, and no eval-window value has been seen.
- **gate_b.json `analysis_sha256`:** points to **v3** (draft placeholder: "pending v3 sha"). v2 will never be run.

## DL-11: §0 rule 3 rewritten (verbatim)

> Rule 3 (revised by DL-11, 2026-10-19, before any clean-pipeline evaluation output was opened). No window in this study is fully confirmatory. All decisions (S1–S3, the SC tests and the TOST verdict) are made mechanically by the frozen analysis artefact on the 2023-01-01 to 2026-06-30 CSI300 window, and on CSI500 at Gate C. Because this window was exposed before registration (rule 2), these are reported as preregistered analyses on a previously exposed window, not as confirmatory tests. The post-cutoff sample after 2026-06-30, which no version of any model or analysis has touched, is reported descriptively per DL-5: one interim look at submission and one final look at 120 or more trading days, reported regardless of outcome. The paper uses 'preregistered', never 'confirmatory'.

(Note: "DL-xx" in the supplied text is filled in as DL-11.)

## DL-12: analysis artefact v2 → v3

- **Old:** v2, `7d90d218e661a242512e542e309ea49e4858c428da6f2ea85018ef22e6db64fa`, commit `b7306e0d4b0d37c8456456264ed8c7a2a9eca10a`.
- **New:** v3, `src/new_gate_b_analysis_v3.py` (to be written by GPT). sha and commit: **pending v3 sha**.
- **Scope:** this is the single v2→v3 revision. The full spec and acceptance criteria are in `/workspace/uploads/new_GPT_code_spec_gate_b.md` §9.
- **gate_b.json** will point to v3.

1. **SC6 sign fix (bug fix).**
   - **Prereg check (read-only, `PREREGISTRATION_v0.6.md` @ bc0dc13):**
     - `:70` "Delta_cost: the difference in annual cost drag … between G0* and C0*";
     - `:72` "Delta = Delta_gross + Delta_cost (accounting identity)";
     - `:176` S1 "supported" requires "SC6 (Delta_cost) lower bound > 0";
     - `:189` (S2) "Delta_cost (SC6) is positive with lower bound > 0 but Delta_net is not: costs fell but gross alpha fell with them".
     - So the registered direction is cost reduction: Δ_cost = g_fee − c_fee.
   - **Why the sign is wrong:** with net = gross − fee, Δ = Δ_gross + (g_fee − c_fee) = Δ_gross + Δ_cost. The code's `delta = gross − drag` (drag = c_fee − g_fee) is correct; only SC6 is reversed, because it tests "C0 pays more".
   - **v3 changes:**
     - `dcost = g_fee − c_fee`, tested one-sided with P*(mean* ≤ 0);
     - every v2 use of drag is fixed (`:14`, `:336`, `:350-351`, `:360`, `:370`, `:391-392`);
     - `s_mapping` (`:170-178`) does not read Δ_cost, so v3 adds a parallel `s2_conditions` that uses dcost.
   - **Self-tests:**
     - (a) a synthetic case where C0 pays lower fees gives SC6 > 0;
     - (b) Δ = Δ_gross + Δ_cost holds to 1e-12.
   - **Disclosure:**
     - no evaluation data had been seen when the fix was made;
     - v1 had the same bug;
     - in the paper, label the Rev2 validation figure "+0.019 pp/yr" explicitly as c_fee − g_fee, so that Δ_cost = −0.019.
2. **SC5 becomes one-sided, P*(ρ* ≤ 0).**
   - This matches SC1/SC6/SC7 and is decided before any data are seen.
   - Disclose that v2 was two-sided; v3 keeps the two-sided field for disclosure only.
   - **Premise check (read-only):**
     - `:69` "Rank correlation of arms/models across soft and TopkDropout layers." states no direction;
     - `:37` RQ4 "Does a soft-portfolio gain transfer to TopkDropout with full frictions? (rank correlation across layers)" is directional only by implication.
     - Applying the condition "provided its registered hypothesis is directional (positive association)" therefore needs explicit user or reviewer confirmation. Until then, v3 keeps the two-sided test.
     - Also disclose that `:69` is about arms/models, whereas the code computes a daily-series Spearman for the C0 arm.
3. **Date guard inside v3:** it covers replay prices (including the terminal t+2), the h5 endpoints t+6 through `loss_masks["eval"]`, and the H00300 rows sampled. It logs `max_date_touched` and fails hard if any date is after 2026-06-30.
4. **Docstring `:11-18` corrected:**
   - SC2 is outside Holm;
   - the family is [SC1, SC5, SC6, SC7];
   - SC5 is one-sided; SC6 uses dcost;
   - the guard is described;
   - SC4 joins at Gate C.
5. **Self-tests `:430-434` corrected:**
   - inputs are consistent (lower5 == ci90[0]) and the real `tost()` is used;
   - each of the four verdicts is tested;
   - the superior+equivalent and inferior+equivalent co-occurrence cases are tested.
6. **conclusion():** it stays out of `s_mapping`, because DL-7 says it only governs wording. It is emitted next to `s1_conditions`/`s2_conditions` in the output JSON, with a `conclusion_note` stating that the separation is intentional.
- **Run status:** v1, v2 and v3 have never been run with `--run`, and no evaluation-window value has been seen.

## DL-13: compute exemption (to be committed before T3)

> "Reruns mandated by DL-1/DL-2 are exempt from the 60 GPU-h cap on any device, because their length is fixed by published defaults and cannot be truncated. Actual device-hours are recorded. The cap continues to bind all discretionary compute, including the CSI500 fine-tunes."

- **Rationale (reviewer):** the cap exists to stop discretionary experimentation. Ruling instead on CPU vs GPU, after the worst case is known, would look like moving the goalposts.
- **Procedure:** run the NN arms on the GPU and measure LSTM seed 1 first. Record `device_hours` per run.

## DL-14: Data identifier erratum

- **Formal CSI300 inputs** (`D:\research\new_LiMT\data\csi300_w1w2\data\v2\`):

  | File | sha256 |
  |---|---|
  | panel | `a25879f4b94e60b6224bf52ec777d25e4f4511af4be3591711647992b601a62e` |
  | labels | `7bf5633b7ad9d146e5154a120307d8dbf771a9d5a5c9852f9eb7aba0bd1064a5` |
  | splits | `0ab69b33d0c8e76c020ebd100791eaf5535cac0e4c4c075207b03890baec17ad` |

  - **Source of these hashes:** computed read-only on the laptop on 2026-10-19, using the trainer's own `sha256_file` (`src/run_seed_chain.py:43-48`, extracted verbatim via ast) in the limt-repro `.venv`.
  - There is no manifest json in that directory. The values match the earlier Get-FileHash results and the prereg header prefixes (`:6`).
- **"fix1"** in the preregistration (`:6`) refers to these three files.
- **ModelScope rebuild:** `e15ed6f6…` (labels) and `cb774f59…` (panel) are the ModelScope rebuild. It matches the formal inputs on the checks performed (calendar, shapes, mask counts, ±3 clip, the 4 leak days), but it is **not a formal input**.
- **data_version recomputed** with the trainer's own `panel_version` (`src/run_seed_chain.py:51-58`; `ARTIFACTS` at `:35` = labels, panel, splits):
  - result: `3c70f66384b8a954d18540a3b129da2103d13b04629897d5204e6292d97f7379`;
  - this **matches** the anchor 3c70f663… in every checkpoint meta (e.g. `protocol/checks/n5_neff_validation.json:7`);
  - so this item does not block gate_b.json.

## Compute budget (cap treatment per DL-13)

- **Machine (decided; changed on 2026-10-19):** the ModelScope free GPU environment.
  - NVIDIA A10 with 24 GB GPU memory (driver 550.54.15);
  - 8 CPU cores (Intel Xeon Platinum 8369B) and about 28–32 GB RAM;
  - image ubuntu22.04-cuda13.0.3-py312-torch2.13.0 (torch 2.13.0+cu130, CUDA 13.0, Python 3.12.13).
  - The MI300X and the 55 GB AMD option originally chosen could not start because resources were unavailable.
  - About 11 h of free quota remain. If that is not enough, tags not yet started move whole to Kaggle or Colab; a single tag is never split across platforms.
- **Use of the machine:** all 15 new tags run on this machine, using as much of it as possible and in parallel. The parallel plan is in the run plan.
- **Serial worst case** (based on the laptop per-epoch timings; must be re-measured on the A10):

  | Arm | Worst case per run | Runs | Total |
  |---|---|---|---|
  | LSTM/GRU/ALSTM | 200 × ~225 s ≈ 12.5 h | 9 | ≈110 h |
  | Transformer | 100 × ~318 s ≈ 8.8 h | 3 | ≈26.5 h |
  | XGB | negligible | 3 | — |

  The serial total is ≈137 h. Running in parallel shortens wall-clock time, but the 8 vCPUs limit how far.
- **Procedure:** measure LSTM seed 1 alone first.
- **Epoch caps are never cut.**
- **60 GPU-h cap:** DL-1/DL-2 reruns are exempt under DL-13. Device-hours are recorded for every run.

## Platform (decided)

- **Where:** the ModelScope A10. All 15 new tags run there where possible; quota overflow moves whole tags only, under the rule above, and the platform is recorded per tag.
- **Disclosure:**
  - The old z-score controls and the seed-0 reproductions came from a laptop NVIDIA GPU (CUDA 11.8). The new tags run on the A10 with CUDA 13.0, so the results will not be bit-identical.
  - Even on CUDA, a same-seed rerun is not guaranteed bit-identical, because of cuDNN RNN kernels and atomics.
- **Recorded for every run:**
  - Python, pyqlib, xgboost and torch versions, plus CUDA;
  - GPU model (A10);
  - XGB nthread, `torch.get_num_threads()` and the number of concurrent jobs.

## Other items

- **MASTER** is excluded (user-confirmed). The reason is that v0.9.6 has no built-in MASTER implementation or official benchmark config; verified.
- **README** must be updated.
- **gate_b.json placeholders:**
  - `<HEAD at write time>` and truncated hashes become full values;
  - fix the typo `execution_feed_resselt`.
- **Data naming:**
  - the prereg calls `7bf5633b…` the "fix1 labels", but the ModelScope fix1 build is `e15ed6f6…`;
  - the formal inputs are `7bf5633b` / `a25879f4` / `0ab69b33`;
  - resolved by DL-14: data_version recomputed and matching.

## User sign-off record

- **Time:** 2026-10-19 00:41 PDT
- **Signatory:** beelingz
- **Own words:** "我已阅读 DL-3、DL-12～15，同意按此执行。beelingz，2026-10-19" ("I have read DL-3 and DL-12 to DL-15 and agree to proceed accordingly."; default wording chosen by the user in chat at 2026-10-19 04:37 PDT)
- **Items confirmed:**
  1. N8: the 2027-01-15 hard deadline, decoupled from the holdout, and the 60 GPU-h cap (proposal `caee9164a65a6026c7fb7016f925441db556ab87`).
  2. B6 option (b): the holdout is descriptive, with an interim look at submission and a final look at ≥120 trading days, at most two looks.
  3. B2(a).
  4. MASTER is excluded.
- **60 GPU-h:** handled by DL-13 (to be committed). Platform: originally the ModelScope MI300X; because resources were unavailable the user switched on 2026-10-19 to the ModelScope free GPU environment, an NVIDIA A10 24 GB (CUDA 13.0, torch 2.13.0+cu130). T1 passed on the A10 at 2026-10-19 04:31 PDT.
- **Decided (2026-10-19 03:25 PDT, confirmed by beelingz in chat):** the SC5 hypothesis is positive association; v3 uses the one-sided P*(ρ* ≤ 0) (DL-12 item 2 takes effect). The estimand mismatch between line 69 (cross-arm rank correlation) and the implementation (C0 soft vs executed daily Spearman) is disclosed in the v3 docstring; the algorithm is unchanged.

## DL-15 (draft) float32 tie exception in the CSRankNorm ranking basis
- Upstream computes z_r1 from float64 raw returns, then stores raw_r1 and z_r1 separately as float32. The per-day affine map preserves order in float64, but independent float32 rounding can create a tie on one side and not the other, so the spec's acceptance check "z_r1 ranks bit-identical to archived raw_r1 ranks" fails on some slices.
- Resolution: keep ranking by z_r1 as specified (the label the benchmarks actually consume); algorithm and data unchanged.
- Replacement acceptance: (1) every rank mismatch must involve a stock that sits in a float32 tie group on raw_r1 or z_r1, otherwise FAIL; (2) report affected days, stock-days, max rank difference, and the minimum daily Spearman against raw_r1 ranks; (3) no rank difference may exceed the size of its tie group.
- Recorded 2026-10-19 03:28 PDT, before ranking; no 2023+ metrics examined.

- DL-3 approval record: beelingz confirmed DL-3 (Alpha158 defaults and 0.9.6 vs 0.9.8.dev34 differences) in chat at 2026-10-19 04:23 PDT. The v2 spec configs/new_benchmarks_qlib_defaults_v2.json (sha 49010f9b…) line 204 "FROZEN: DL-3 approved by user" is consistent with this.
