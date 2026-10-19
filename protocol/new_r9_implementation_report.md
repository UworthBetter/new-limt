# r9 / v3.1d implementation delivery

This delivery adds the implementation and synthetic verification requested by
the h5 r9 and guard v3.1d specifications. It does not run an experiment, open
real evaluation observations, or finalize the experiment protocol.

## Delivered code

All implementation files are new. The protected v2 files and
`limt-repro/src` remain unchanged.

- Shared guard, fixed cutoff identities, five access streams, per-call marks
  checks, import closure logging, and committed-source verification:
  [new_guarded_data_v1.py](D:/research/new_LiMT/src/new_guarded_data_v1.py).
- Main analysis with guarded prediction, replay, soft portfolios, and IC:
  [new_gate_b_analysis_v3_1.py](D:/research/new_LiMT/src/new_gate_b_analysis_v3_1.py).
  Its statistical functions, replay functions, constants, and Holm family
  retain the frozen v3 behavior.
- h5 training, configuration selection, T1 verification, rule grid, and
  validation forensics: the five requested `*_h5_v3.py` entry points.
  Supporting modules contain the label/purge contracts and the two qlib fit
  subclasses.
- Descriptive evaluation, saved-model adapters, and sequential one-shot
  opening: [new_bench_eval_v1.py](D:/research/new_LiMT/src/new_bench_eval_v1.py)
  and [new_unseal_once_v1.py](D:/research/new_LiMT/src/new_unseal_once_v1.py).
- CPU retraining, fixed-tolerance comparison, parameterized v2 rule/forensic
  copies, and CPU reselection. CPU selection directly calls
  `cpu_baselines.choose_arm` (line 78), which calls `choose_rule` (line 69).
- The supplied frozen ridge comparator is copied without modification. Its
  SHA256 remains
  `07f4e1214b619468827bd44002b46cdf5c2c6f0f1b837e7b6faea9528349a9ca`.

The handoff h5 JSON is copied byte-for-byte to its requested configuration
path. Its existing `DRAFT` status remains visible. No reviewed parameter,
protocol approval, or original metadata hash is invented.

## Verification evidence

The authoritative delivery directory is
[new_r9_delivery_20261019_release](D:/research/new_LiMT/protocol/checks/new_r9_delivery_20261019_release).
Earlier delivery directories retain intermediate evidence. The final adapter
regression follows the complete suite because the adapter now preserves each
qlib model's declared device and uses the main analysis device policy for LiMT.

Read these files in the authoritative directory:

- `new_acceptance.json`: command outcomes and production prerequisites.
- `new_pytest.txt`: the complete synthetic test suite.
- `new_bench_eval_final_pytest.txt`: the final evaluation-adapter regression suite,
  with its source hash recorded in `new_bench_eval_final_test_record.json`.
- `new_guard_self_test.txt`: every positive control fires, including both
  cutoff constants and the real `portfolio_events` path.
- `new_analysis_self_test.txt`: the guard checks and inherited statistics
  checks pass.
- `new_t1_smoke.txt` and `new_t1_record.json`: model/config seed checks,
  label/purge checks, known-answer IC, RNG isolation, and best-weight reload.
- `new_runner_smoke.txt`: LSTM and Transformer default-h5 command-line runs
  using synthetic observations. Their scores, metas, and readback-verified
  checkpoints are stored in `new_smoke_scores` and `new_smoke_models`.
- `new_protected_baseline.json`: the specified read-only baseline hashes.
- `new_code_manifest.json`: SHA256 for every implementation, configuration,
  and test file in this delivery.
- `new_analysis_v3_to_v3_1.diff`, `new_lstm_fit.diff`, and
  `new_transformer_fit.diff`: requested source comparisons.
- `new_xgb_retrain_path.diff`, `new_de_retrain_path.diff`, and
  `new_ridge_retrain_path.diff`: comparisons against the original entry points.

The complete suite passes 41 tests. The final evaluation-adapter regression
passes 14 tests. All six command/schema checks and the 20/22/24-run
cardinality audit pass. Fourteen protected baseline files match their specified
SHA256 values.

The tests exercise original qlib training with diagnostics enabled and disabled,
including an exact state comparison to the unmodified classes. They also test
checkpoint truncation, tensor changes, mandatory model-output arguments, score
and NaN perturbations, index ordering, ridge/XGB/DE persistence, all four qlib TS
prediction adapters, LiMT prediction, r1 masking, aliases, and failed opening.

The parameterized grid produces byte-identical daily CSVs for the original
six-kind manifest: 18 tags, including the 15 new v2 tags and three DE tags,
across eight rules, totaling 144 books. Its numerical aggregate equals v2.
The parameterized forensic entry point produces byte-identical synthetic JSON
for the 15-tag v2 manifest when runtime provenance is held fixed.

The main analysis numerical functions also match v3 at the AST level.
Additional provenance fields intentionally differ from the original scripts.

## Guard whitelist

The complete whitelist is documented in the guard module. Its streams are:

| Stream | Reads |
| --- | --- |
| `prices` | Execution columns and `stale` |
| `marks` | Both indices passed to `relatives` |
| `features` | Windows, candidates, features, feature masks, and universe masks |
| `labels` | Loss/label masks at signal dates; h5/r1 values at t+6/t+2 endpoints |
| `bench` | Each requested benchmark row |

Calendar/identifier metadata (`dates`, `symbols`, `lookback`, `indices`,
`feature_names`, `splits`, and `days`) contains no observed values. It requires
no access accounting. Direct `close` and `marks` access, unknown attributes,
and implicit conversion of a guarded array fail closed.

Construction exceptions are confined to canonical `Data`/`BlindData` input
loading, alignment checks, and execution-feed construction. After wrapping,
prediction, replay, soft portfolios, and IC use the guarded object.

## Production handoff

The implementation deliberately refuses production until you complete the
protocol steps in the specifications:

1. Finalize the supplied h5 draft. Add `r1_meta_sha256`, keyed by the six
   `LSTM-csrank_seed{1,2,3}.meta.json` and
   `Transformer-csrank_seed{1,2,3}.meta.json` names, using the approved original
   metadata hashes. These original metas are not present in the local handoff.
2. Commit and push the guard and
   [its checksum](D:/research/new_LiMT/configs/new_guarded_data_v1.sha256) first.
   Production checks both working bytes and the committed `HEAD` copies.
   Commit and push the finalized h5 spec and DL entries before real runs.
3. Run production T1 and h5 on the specified ModelScope A10 runtime. Run CPU
   retraining and reselection on the specified CPU instance. This local runtime
   is synthetic verification evidence, not production-platform verification.
4. Supply explicit reselection manifests and the reviewed `gate_b.json` arm
   inventory. Store persistent models under `/mnt/workspace`.
5. At opening, use the one-shot wrapper: analysis first, then descriptive
   evaluation. A failed attempt keeps its exclusive attempt directory, records
   the failure, and commits only that attempt's logs. A rerun needs a DL entry.

No commit, push, retraining on real data, or opening is performed in this coding
delivery. The supplied draft is not represented as an approved frozen spec.

The installed Windows LSTM source has CRLF bytes. Its normalized source equals
the official v0.9.6 source, whose SHA256 is the specified `4496a8bd...`.
Production still requires the exact file SHA from the specification; the
implementation does not normalize bytes or relax that check.

Legacy score NPZ files omit instrument identifiers. The comparator accepts
matching pinned panel/label provenance as their index contract. If that
provenance is absent, as in the legacy DE metadata, it conservatively reports
`MISMATCH` rather than certifying an unverified column order. The original files
remain available for the replacement audit.

## Manifest interfaces

For v2/CPU reselection, supply a JSON object with `kinds`, `label`, and `tags`.
Each tag record contains `tag`, `kind`, `seed`, and `path`. You can additionally
pin each record's `spec_sha256`. The default seed set per kind is `[1, 2, 3]`;
for ridge, declare `seeds: {"ridge-h5": [1]}` (or the corresponding r1 kind).
The grid writes the requested aggregate plus
`new_reselect_<arm>_r9.json` records containing replacement provenance and the
selected rule. Use a separate aggregate output filename.

For descriptive evaluation, `score_source` maps seed strings to model records.
Each record contains `path`, `sha256`, `adapter`, and the required constructor
or feature metadata. Supported adapters are `qlib_ts`, `xgboost`, `lightgbm`,
`doubleensemble`, `ridge`, and `limt`. DE records also enumerate every submodel
in `files`, each with its own path and hash. TS records add `model`, `kwargs`,
`columns`, and `step_len`. Controls reference a hashed `rule_select_result` and
produce the required no-evaluation-result text. An `identical_to_default` note
or explicit alias flag suppresses duplicate scoring.

The opening wrapper plan contains `analysis_args`, `bench_args`,
`analysis_out`, and `inputs` (path/SHA256 records). `analysis_args` must explicitly
identify the same `--out` path as `analysis_out`. The wrapper fixes both script
paths and their execution order.

To reproduce the local delivery evidence, run the following command with the
existing benchmark Python environment and choose a fresh output directory:

```powershell
$env:PYTHONPATH='D:\research\new_LiMT\src;D:\research\limt-repro\src'
& 'D:\research\new_LiMT\venv_bench\Scripts\python.exe' `
  'D:\research\new_LiMT\src\new_r9_delivery_check.py' `
  --out-dir 'D:\research\new_LiMT\protocol\checks\new_r9_fresh_check'
```
