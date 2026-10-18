# Pre-Gate-B plan (2026-10-18)

Principle: do everything that must be declared before unsealing; unseal once, after the benchmark arms are frozen.

## A. Do now, in order
1. Deviation Log entries (before any further run):
   - DL: paper reframed as a benchmark/re-evaluation; add benchmark arms (list models; fixed Qlib default configs; same clean data; same 8-rule selection on 2021-2022; same metrics). Declare whether the dose arms are evaluated on 2023+ (exploratory label).
   - DL: stamp-duty sensitivity. VERIFIED: sell-side stamp duty cut from 0.1% to 0.05%, effective 28 Aug 2023 (HKEX circular CT/137/23, 27 Aug 2023). Primary unchanged (sell 15 bp throughout). Sensitivity: sell 10 bp from 2023-08-28, all arms, reported alongside, never decisive.
   - DL: TOST equivalence margin for Delta = +/-2.0 pp/yr (equal to the S1 threshold); same stationary bootstrap; equivalent if the 90% CI lies inside (-2, +2).
   - DL: rule-selection seeds. Section 5 averages over seeds, but only seed 1 exists. Option (b), preferred: run seeds 2-3 for H/H-FT/G0/C0 before Gate B, then select averaged. Option (a): select on seed 1 and freeze, logged.
   - DL: record diagnostics (gradient decomposition, H-FT share 94%, dose arm activates the cost term) and disclosures (truncation gap, model differences vs public code, typical price for VWAP, adjusted-price features not covered by the truncation probe).
2. Gate A formal items: fill [FILL] fields, commit and hash the prereg + deviation log. Record actual dates; state that Gate A was completed after the seed-1 runs (no backdating).
3. sigma_d(final) on validation, default rule, executed books C0 vs G0, plus holdings overlap, rank correlation and cost-drag difference -> mechanical tier. Expect a very small sigma_d (near-identical books), so probably tier P1. Accept it mechanically.
4. 8-rule x neural-arm grid on 2021-2022 (H, H-FT, G0, C0; dose arms labelled exploratory). Tie within 0.5 pp -> lower turnover.
5. Dose diagnostics in the same runner: 5-day IC; executable-layer (topk-execution) vs soft on 2021-2022.

## B. Not yet
gate_b.json and the unseal. Wait until benchmark arms have frozen configs and their rule selection is done, so every arm is unsealed once.

## C. After Gate B
Seeds 2-3 (if not done before), CSI500 (W1-W2 gates first), post-cutoff holdout, mechanical Gate C. C-T stays deferred.