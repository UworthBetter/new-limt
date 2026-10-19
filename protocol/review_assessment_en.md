# Assessment of the External Review (Two Rounds)

Scope: factual accuracy of the reviewer's claims, checked against primary sources (official qlib v0.9.6 yamls, run logs, recomputed arithmetic).

## Verified correct (credit where due)

- Ordering discipline, hash-chain reading, and the MDE arithmetic (1.189% → 1.58 pp → P1) — all recomputed and confirmed.
- The TOST omission in revision 2 was real (it was restored).
- XGBoost and Transformer ran at seed 0, not seed 1 — correct catch; the draft was fixed.
- The H-fix ordering concern was legitimate; the suggested byte-identical re-scoring was performed and passed (16/16 replay files character-identical).
- The seed-freeze warning (adding seeds after unsealing makes rule selection post-hoc) was sound and was implemented as advised.
- **The productive catch of round 2**: XGBoost and DoubleEnsemble were training on the 20-column FilterCol subset while their official yamls specify no FilterCol. Verified true; both arms were re-run on the full 158-column handler (XGBoost validation r1 IC 0.0323 → 0.0445; DoubleEnsemble 0.0245 → 0.0401; the CPU LightGBM-r1 reference is 0.0352).

## Errors found in the review

1. **"DoubleEnsemble may specify 6 sub-models, not 3" — wrong.** The official v0.9.6 yaml pins `num_models: 3` (verified against the raw file). The error sat inside the same hedged paragraph as the correct FilterCol point.
2. **Round-1 premise: the ~158-feature assumption applied to all six arms.** In fact the four neural ts configs specify FilterCol-20; only XGBoost/DoubleEnsemble use the full handler. The round-1 "implausibly fast" flag therefore rested on a wrong feature count for the arms as configured — though chasing it produced the genuine round-2 finding.
3. **"XGBoost uses hundreds of depth-8 trees" — inaccurate for this code path.** Under qlib's own fit defaults (num_boost_round 1000, early_stopping_rounds 50 on validation) it early-stopped at ~106 rounds; the yaml's `n_estimators: 647` is not consumed by qlib's train path.

## Net assessment

High review quality: every unhedged claim survived verification; both errors were in the explicitly hedged section ("treat this as something to check, not a finding") and were cheap to settle against primary sources. The hedging discipline worked as intended — the errors cost minutes, the correct half of the same paragraph changed two arms.
