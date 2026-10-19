# DL-24 (new) h5 benchmark addition: LSTM / Transformer × {default-h5, tuned-h5} (r14, pre-unseal; signed)

Experiment assistant, 2026-10-19. All times PDT. **Revision r14** supersedes `new_DL24_h5_benchmarks_r13_draft_en.md` (b5a22555…, kept unchanged); it only fills in the verbatim 09:29 card option text. r13 superseded r11 (4ce459c0…) and added only the sign-off record and the frozen spec sha; the body is unchanged. The r11 note follows. **Revision r11** supersedes `new_DL24_h5_benchmarks_r8_draft_en.md` (ec9fe32b…, kept unchanged). The content is unchanged. Stale file references now point to the final versions (spec draft json r8, GPT spec r9, guard spec v3.1d, retrain list r9), and the round-9 cfg1-kwargs assertion and cutoff rules are added. The r8 note follows. **Revision r8**, applying Claude's round-8 rulings (the 07:52 PDT reply and `/workspace/uploads/claude_round8_reply.md`) on top of r7. It supersedes `new_DL24_h5_benchmarks_r7_draft_en.md` (974d73c7…), which is kept unchanged. New file, for Claude's review. Source: Claude's h5 plan (`/workspace/uploads/claude_h5_plan_reply.md`). `[sha]` and `[time]` are placeholders.

**Numbering.** The round-6 draft `new_DL_rows_round6_draft.md` (c45a7bdc…, left unchanged) had used DL-24 for v3 → v3.1. Per Claude's decision at 07:33 PDT, **DL-24 is this entry (h5 benchmarks)** and v3 → v3.1 becomes **DL-25** (see `new_DL_rows_round6b_draft.md`).

---

- **Who and when.** The PI, beelingz, decided on 2026-10-19 at 07:28 PDT. At that point no evaluation output (2023 onward) had been opened, and no 2021–22 rule-selection result had been viewed: the T7 output archive is still packed. The design was proposed by the external review (Claude).
- **Purpose.** Baseline fairness: separate the label effect from the model effect. The existing Qlib-default benchmarks use the r1 label (DL-2, CSRankNorm(r1)), while LiMT uses h5. Switching the label and tuning at the same time would mix the two effects, so two separate arms are added:

| Arm | Label | Hyperparameters | Status |
|---|---|---|---|
| default-r1 | CSRankNorm(r1) | published | already run (LSTM and Transformer tags among the 15) |
| default-h5 | CSRankNorm(h5) | published | new |
| tuned-h5 | CSRankNorm(h5) | the configuration selected from the fixed grid below | new |

  default-h5 − default-r1 gives the label effect, and tuned-h5 − default-h5 gives the tuning effect.

- **Status.** These are descriptive arms **outside the Holm family**. The primary arms (C0/G0), the primary estimand and all inference settings are unchanged. The tuned arms are always named "tuned-h5" and **must never be called "official defaults" or "published defaults"**. default-h5 may be described as "published default hyperparameters (h5 label)".
- **Models.**
  - LSTM: the canonical baseline, and the worst case, with best epoch 0–1 under r1.
  - Transformer: the only NN benchmark that trains past the first epoch.
  - No GBDTs: the CPU LightGBM arms already exist for both h5 and r1 with Qlib parameters.
- **Label.** h5 = C(t+6)/C(t+1) − 1. Source: `new_v2_labels_csi300.npz`, sha `7bf5633b…`, keys `z_h5`, `label_mask_h5`, `exit_idx_h5`, `entry_idx`.
  - Apply CSRankNorm per day over `label_mask_h5`: (rank(pct) − 0.5) × 3.46, mirroring DL-2.
  - Use it for the train and validation labels only, as with r1.
  - **Endpoint purge: on** (round 7, item 1). Drop every train or valid sample whose label endpoint `dates[t+6]` falls after its segment end (train 2018-12-31, valid 2020-12-31). Without the purge, training labels would read validation-period prices and compromise early stopping.
  - Cost, per Claude's estimate: about 5 boundary days out of roughly 2,500 in training, and 4 out of roughly 485 in validation. Neither setting touches sealed data, since the validation endpoint reaches only into early 2021.
  - **Asymmetry disclosed**: the default-r1 arms (v2) were **not** purged. Their endpoint is t+2, which overlaps by 1 day at each boundary.
- **Tuning grid** (8 configurations per model, fixed now):

| Factor | Levels |
|---|---|
| Learning rate | published; published ÷ 10 (LSTM 0.001 / 0.0001; Transformer 0.0001 / 0.00001) |
| Width: LSTM `hidden_size` / Transformer `d_model` | published; 2 × published (64 / 128) |
| Early-stopping criterion | validation loss (published); validation 5-day IC |

  - Everything else stays at the published values, including the epoch caps (LSTM 200, Transformer 100), early-stop patience (LSTM 10, Transformer 5), batch size, layers, dropout, loss and n_jobs.
  - The published values are copied verbatim from spec v2 `configs/new_benchmarks_qlib_defaults_v2.json` (49010f9b…), `models.LSTM` / `models.Transformer`. The Transformer class defaults come from qlib v0.9.6 `pytorch_transformer_ts.py:26-43`.
  - Configurations are numbered cfg1–cfg8 in the spec draft. cfg1 uses all published values, i.e. the default-h5 configuration. **cfg1 seed 1 is a single run that counts in both places**: as default-h5 seed 1 and as a member of the tuning pool. Rerunning the same configuration and seed would only add GPU-nondeterminism pseudo-replication (round 7, item 2).
- **Validation 5-day IC.** For each 2019–2020 validation signal day, take the Pearson correlation between the score and `z_h5`. The mask is: loss_mask_valid, endpoint no later than 2020-12-31, listing-eligible, and both values finite. The metric is the mean over days, the same as limt-repro `v2_w5_results.daily_ic`.
- **Selection.** Seed 1 only. For each model, pick among the 8 configurations by the validation mean daily 5-day IC of the restored best checkpoint, the same window every arm uses for early stopping. **Tie-break, in exact order:**
  1. best = the highest IC of the 8 configurations;
  2. pool = configurations with IC ≥ best − 0.002;
  3. within the pool, fewest non-published factors, which favours the published values;
  4. then the highest IC;
  5. then the lowest cfg index.

  If any configuration fails or has a non-finite IC, stop and report without selecting.

  **Selection bias, disclosed** (round 7, item 3): configurations that early-stop on IC are selected on the same metric and window they stopped on, so the selection is optimistic toward them. This favours the tuned arms, which errs toward fairness to the benchmarks and is conservative for the LiMT claim.

  **If cfg1 is selected**, tuned-h5 is identical to default-h5. That is reported as a finding ("the tuning budget selected the published configuration"), not as two arms, and no extra runs are made. Only the selected configuration names are recorded in logs. Per-configuration ICs go to a file and are not printed.
- **Seeds.** tuned-h5 is the selected configuration at seeds 1–3. Seed 1 **reuses the seed-1 tuning run**, which took part in the selection; this is disclosed here. default-h5 is run at seeds 1–3, where seed 1 is the cfg1 tuning run.
- **Execution rule.** As for every other arm: the same 8-rule grid on 2021–2022, averaged over seeds, with a 0.5 pp tie going to the lower-turnover rule. DE is unchanged.
- **Run count (declared now, round 7).** Per model: 7 tuning-only runs (cfg2–cfg8, seed 1) + 3 default-h5 (seed 1 doubles as cfg1) + 2 for the selected configuration's seeds 2–3 = 12.
  - Total: **24 runs**.
  - The count depends on how many of the two models select cfg1: **24 if neither, 22 if one, 20 if both** (confirmed by Claude, round 8 (c)).
  - The plan's original count of 26 is withdrawn.
- **Compute.** About 5–8 h on the A10 (Claude's estimate, made for 26 runs). The work is discretionary and **counts against the 60 GPU-h cap**; it is not covered by the DL-1/2 rerun exemption. **Accounting (round 7, item 5):** GPU hours are the GPU instance's wall-clock time whenever it is up, including idle and debugging time. The sum of per-tag device time is logged as well, for information only. The assistant's estimate is in the appendix of the Chinese draft.
- **Spec file.**
  - New: `configs/new_benchmarks_h5_v3.json`. Draft: `/workspace/limt_gateb/new_benchmarks_h5_v3_r8_draft.json` (039ee1cb…); final sha `77a2d8db5bac48e7db4a6fef597690add5b084bd26c16498e786954e7dadeae9` (see Sign-off below). **Its status field must not say FROZEN until beelingz personally approves this entry.** **v2 is not edited.**
  - Code spec: `/workspace/uploads/new_GPT_spec_h5_benchmarks_r9.md` (84f25f17…); guard spec: `/workspace/uploads/new_GPT_spec_v3_1d_guard_fix.md` (7eea4ee1…). GPT writes the code as new files and does not edit any v2 code.
- **Place in the unseal sequence.** All of these steps come before gate_b.json is written, because each arm can be unsealed only once.
  1. Commit DL-24 and the v3 spec, with the spec's sha stated in DL-24, and **push immediately**. The push must land before any run, including T1. The earlier ordering issues are covered in DL-16 and DL-26.
  2. Run T1-type checks on the new runner: seeds, h5 label, IC function, RNG isolation.
  3. Train 24 / 22 / 20 runs (zero / one / both models select cfg1). Select configurations on validation and record only the configuration names.
  4. Run the 8-rule grid on 2021–22 and update forensics, using validation-only numbers.
  5. Resume the existing sequence: §8 (add the two new arms), gate_b.json (add them to the arms list), push, unseal.
- **Evaluation-window books.**
  - v3 (203e6a45…) has no hard-coded benchmark list and computes only C0/G0 (`new_gate_b_analysis_v3.py:147-149, :397-412, :488`). The h5 arms are not added to v3.1 and stay out of Holm (`HOLM_KEYS`, `:66`).
  - Per round 7, item 4, the books for every descriptive arm, including the h5 arms, come from a separate frozen script, `src/new_bench_eval_v1.py`.
  - It shares the guarded-data module `src/new_guarded_data_v1.py` with v3.1 and reads its arm list from gate_b.json.
  - It reports descriptive metrics only: no p-values and no Holm.
  - Its sha goes into gate_b.json, and it runs in the same single unseal step as v3.1.
  - Specs: §7 of the r9 GPT spec, and §7–§8 of the v3.1d spec.
- **Sign-off (r13).**
  1. **09:29 PDT, card click.** beelingz clicked an option whose text the assistant wrote: a conditional approval, verbatim: "签：我批准 DL-24 r11，Claude 审核无修改才生效" ("Signed: I approve DL-24 r11, effective only if Claude's review makes no changes"). The condition was later met: Claude's round 10 confirmed DL-24 r11 needs no changes. It is recorded as given, **but is not the operative signature**. This follows the DL-3 precedent, where a card click was later ratified in his own words.
  2. **09:38 PDT, own typed words.** In a conversation with Claude, beelingz typed: "我已审阅 DL-24 r11（4ce459c0）并同意，内容无改动。beelingz，日期10-19" ("I have reviewed DL-24 r11 (4ce459c0) and agree; no changes to the content. beelingz, date 10-19"). **Disclosure:** the sentence follows the gist Claude suggested, but beelingz typed it himself; it was not a card click. This is the operative signature.
- **Frozen spec.** The final `configs/new_benchmarks_h5_v3.json` is `new_benchmarks_h5_v3_frozen_20261019.json`, sha `77a2d8db5bac48e7db4a6fef597690add5b084bd26c16498e786954e7dadeae9`. Relative to 039ee1cb (DRAFT r8, kept unchanged), only two things differ: status is now `"FROZEN: DL-24 approved by beelingz 2026-10-19 09:38 PDT"`, and `r1_meta_sha256` is added (the six r1 meta shas, each matching DL-27's allowed list). Every other field is byte-identical. Checked on the box with GPT's `assert_cfg1`: for 6/6, the cfg1 kwargs equal the r1 meta kwargs, and the status contains no "draft".

## r8 additions (Claude round 8)

- **Checkpoints are mandatory.** For every run, the h5 runner saves the restored best-epoch checkpoint as `new_<tag>.pt` and records its sha256 as `artifact_sha256` in `<tag>.meta.json`. A missing checkpoint or a sha mismatch is a hard fail. Every checkpoint sha goes into gate_b.json under `descriptive_arms[*].score_source`.
- **Checkpoint location.** The r7 cache path was on the instance and was not included in the sealed archive. In r8, the checkpoints and their metas are packed and uploaded to their own subdirectory of the dataset, separate from the score archive.
- **Evaluation-period scoring** happens only in the unseal step, via `new_bench_eval_v1`, with no evaluation-period predictions before unseal; see **DL-28**.
- How existing benchmark arms without saved models are handled (retraining and comparison) is set out in DL-28 and `new_retrain_list_r9_draft.md`.

## r9/r11 additions (Claude round 9)

- **cfg1 equals default-r1** except for the label and the purge. The runner asserts that cfg1's resolved constructor kwargs equal `kwargs` and `effective_model_settings` in the meta of the matching r1 tag (same model and seed). For Transformer it also checks the qlib v0.9.6 class-default table and the source sha (GPT spec r9 §2.2a).
- **Cutoff and guard.** The h5 runner hard-codes `PRE_UNSEAL_CUTOFF = 2022-12-31`. The guard module `new_guarded_data_v1.py` is frozen and committed first, and runners verify it against the committed sha (GPT spec r9 §2.7, v3.1d §7).
- **Checkpoints.** Hash the file, then `torch.load` it and assert tensor-by-tensor equality before recording the sha (GPT spec r9 §2.6).
- **Platform.** The ModelScope A10, the same platform as the 15 tags. The rule grid and forensics run on the ModelScope CPU instance.
