# DL-25 Analysis artifact v3 → v3.1 (draft r13, pre-unseal)

Experiment assistant, 2026-10-19. All times PDT. New file, superseding `new_DL25_analysis_v3_1_r11_draft_en.md` (af62ff5e…, kept unchanged). r13 fills in the shas after GPT's delivery completed at 09:18; the assistant checked them and re-ran the self-tests on the box. The r11 note follows. r11 superseded the DL-25 section of `new_DL_rows_round6b_draft.md` (4115ae1b…). The only change in r11 is the spec reference: **v3.1d** (`/workspace/uploads/new_GPT_spec_v3_1d_guard_fix.md`, sha 7eea4ee10387797d0241e782f7c3df4267e678ef30ee170b84daf81e097b2a41) replaces v3.1b. Its companion is the h5 spec r9, `new_GPT_spec_h5_benchmarks_r9.md` (84f25f17…). DL-12a, the appendix row to DL-12, and this entry cite each other. Numbering follows the 07:33 PDT decision: DL-24 is the h5 benchmarks, and v3 → v3.1 is this entry.

- **Why.** Review rounds 5 and 6 required the guard to cover every price, mark and feature access. Reading the code showed:
  - v3's `portfolio_events` path reads prices only through the already-logged `relatives()` and `stale`, with no overreach;
  - however, `PriceGuardData.__getattr__` forwarded `close`, `marks` and `features` unguarded;
  - the scoring path (`predict`) used unwrapped data;
  - the self-test did not exercise the guard.
- **v3.1 changes** (v3.1d §1–§3, §7, §8):
  1. A whitelist guard object; every data access in `run()` goes through it.
  2. Per-stream max_date_touched for prices, marks, features, labels and bench. It fails hard if any stream is later than the cutoff, and also fails if any stream is never touched (the second condition was added by the assistant).
  3. Positive controls in the self-test: deliberate out-of-range access must raise.
  4. A runtime import-closure log covering sys.modules, file sha256 values, and third-party versions (torch, pyqlib, xgboost, lightgbm and others).
  5. **The guard classes live in a standalone module**, `src/new_guarded_data_v1.py`, shared by v3.1 and `new_bench_eval_v1.py`. It defines two constants: `UNSEAL_CUTOFF = "2026-06-30"` for unseal and `PRE_UNSEAL_CUTOFF = "2022-12-31"` for pre-unseal retrains and h5. Each entry point selects its constant in code, never from the CLI or JSON. Each constant has positive controls.
  6. **The guard module is frozen first.** It is committed before any retrain or h5 run, and pre-unseal runners check it against the committed `configs/new_guarded_data_v1.sha256`. Any later change needs its own DL row.
  7. **Single-shot unseal.** v3.1 runs first, then bench_eval. If either fails: stop, log, and commit the logs; any rerun needs a DL row first. A bench_eval failure never affects v3.1's outputs.
- **Unchanged:** algorithms; constants (TAU, BLOCK_MEAN, N_BOOT, RNG_SEED, NW_LAG, TOST_MARGIN_PP, RULE, STAMP_CUTOFF, SEEDS, HOLM_KEYS); statistical definitions; gate_b.json validation.
- **shas** (checked 09:2x PDT):
  - v3: `203e6a455796f4eca90fa5a9a0038459eec2324279bb13c72d5622339dd8f75d`
  - **v3.1**, `src/new_gate_b_analysis_v3_1.py`: `4135981bbaf7844bc270a26b0a4de7cbe06983342e4ad9a62d51590d273e4b88`
  - **Guard module**, `src/new_guarded_data_v1.py`: `5de86785d2725a48b03ceb4c865926fd3c6516157c23f4780bd86844cbd7efa3`. `configs/new_guarded_data_v1.sha256` contains this value; the file itself hashes to f45bc04b…. The 08:36 snapshot 6ae87320 is superseded.
  - **Descriptive evaluation**, `src/new_bench_eval_v1.py`: `cc5c12faa2f350a22d4ca441e89dd713ab8bc2cff212b800e65b6e832e2967fd`
  - **One-shot unseal wrapper**, `src/new_unseal_once_v1.py`: `b62b44575269dce50b2b6728288ad709e525bedbf2b4c23df8f71e9ee8e6acad`
  - The laptop copy, the delivery zip and GPT's code manifest agree byte for byte.
- **Self-test outputs** (synthetic data only; every positive control fired):
  **Primary evidence is GPT's laptop output** in `D:\research\new_LiMT\protocol\checks\new_r9_delivery_20261019_release\`. That directory's SHA256SUMS is 3e131896…, byte-identical to the copy in the delivery zip; all 55 entries verify OK.
  - `new_pytest.txt` (5182073448644846…): **41 passed, 0 failed**
  - `new_bench_eval_final_pytest.txt` (c13f6f3a2de5edcf…): **14 passed, 0 failed**; source sha under test was cc5c12fa and did not change during the tests
  - `new_guard_self_test.txt` (c97d54fa…): PASS, all 28 positive controls fired
  - `new_analysis_self_test.txt` (3c98ec1c…): PASS, covering both the guard and statistics parts
  - `new_t1_smoke.txt` (3a0999ac…), `new_runner_smoke.txt` (4be3ddbe…), `new_smoke_schema.txt` (6c6edc37…): all PASS
  - `new_acceptance.json`: passed=true, synthetic_only=true; all 7 checks returned 0; protected_unchanged=true; all 14 protected baseline files match
  - The box re-run (table below) was done before the instruction not to re-run arrived, and is corroboration only. On the first attempt, with pandas 3.0.6, `test_original_ridge_fit_roundtrip` FAILED on a 1.1e-16 float difference; with pandas 2.2.3, 41/41 and 14/14 pass.

  | Output | GPT delivery (laptop) | Box re-run (py3.12, torch 2.13.0+cpu, pyqlib 0.9.6, pandas 2.2.3) |
  |---|---|---|
  | Guard `--self-test` | c97d54fab44f205916d9439907ba116bb553bf3976d8eab881f8a5e5d73b6699 | 63a8c5e582599b47e46b37376cc639312fe67d08e8f9922f46fe63a1db2bccbd |
  | v3.1 `--self-test` | 3c98ec1c168595936434129421a3bb31675b11c901526c5bee7ee8ee3a66fc59 | 7fb074b62e6488fa54df53720e4019b9e63381ac7d3b0074b65fc8b2d9e6a0fa |

  Both PASS. The log shas differ because each log records runtime paths and versions; this does not indicate a different result. Box logs: `/workspace/limt_gateb/new_r9check_root/new_LiMT/protocol/checks/new_r9_box_check_20261019_pd223/`.
- **Unseal order** (Claude round 6): v3.1 in place and self-test passed, including positive controls → shas recorded → PI sign-off → continue unseal. In gate_b.json, `analysis_sha256` is the v3.1 sha and `guarded_data_sha256` is the guard module sha.
- This change was made without viewing any data from 2023 onward.
- **Sign-off:** [awaiting beelingz, with time]
