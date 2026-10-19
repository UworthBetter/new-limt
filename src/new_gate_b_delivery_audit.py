"""Package code-only acceptance evidence; never executes an unsealing or real fit."""
import argparse
import ast
import difflib
import json
import os
import subprocess
import sys
from pathlib import Path

from new_gate_b_common import Provenance, path_args, preflight, resolve_paths, sha256_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    path_args(parser)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--protected-before', type=Path, required=True)
    parser.add_argument('--seed-report', type=Path, required=True)
    parser.add_argument('--xgb-smoke-meta', type=Path, required=True)
    parser.add_argument('--nn-smoke-meta', type=Path, required=True)
    parser.add_argument('--nn-snapshot-seed-report', type=Path, required=True,
                        help='Earlier T1 report that records the runner hash at NN smoke launch')
    parser.add_argument('--rank-report', type=Path, required=True)
    parser.add_argument('--dl15-entry', type=Path, required=True)
    args = parser.parse_args()
    resolve_paths(args, ('root', 'spec'))
    targets = [args.out_dir / name for name in
               ('new_gate_b_analysis_v2_to_v3.diff', 'new_gate_b_analysis_v2_to_v3.diff.meta.json',
                'new_pytest_results.txt', 'new_pytest_results.txt.meta.json',
                'new_delivery_report.txt', 'new_delivery_report.txt.meta.json',
                'new_code_manifest.json', 'new_nn_smoke_provenance_addendum.json',
                'new_smoke_runner_initial_to_final.diff', 'new_smoke_runner_initial_to_final.diff.meta.json')]
    baseline_meta = args.out_dir / 'new_protected_baseline.meta.json'
    spec_meta = args.out_dir / 'new_spec_evidence.meta.json'
    snapshot_path = args.nn_snapshot_seed_report.parent / 'new_smoke_runner_initial_snapshot.py'
    snapshot_meta = args.out_dir / 'new_initial_snapshot_evidence.meta.json'
    preflight([args.out_dir, baseline_meta, spec_meta, snapshot_meta, *targets])
    provenance = Provenance(__file__, args, synthetic=True)
    before = json.loads(args.protected_before.read_text(encoding='utf-8'))
    after = {name: sha256_file(args.root / name) for name in before}
    if before != after:
        raise SystemExit('protected original files changed')
    seeds = json.loads(args.seed_report.read_text(encoding='utf-8'))
    rank_report = json.loads(args.rank_report.read_text(encoding='utf-8'))
    diagnostic = rank_report['diagnostic']
    if (diagnostic.get('acceptance') != 'DL-15: user-confirmed replacement criteria'
            or diagnostic['max_label_endpoint'] > '2026-06-30'
            or rank_report['meta']['script_sha256'] != sha256_file(
                args.root / 'src' / 'new_label_rank_check_v2.py')):
        raise SystemExit('missing or stale DL-15 acceptance report')
    runner = args.root / 'src' / 'new_run_benchmarks_v2.py'
    if not seeds['passed'] or seeds['runner_sha256'] != sha256_file(runner):
        raise SystemExit('missing or stale passing T1 report')
    smokes = []
    early_seeds = json.loads(args.nn_snapshot_seed_report.read_text(encoding='utf-8'))
    if sha256_file(snapshot_path) != early_seeds['runner_sha256']:
        raise SystemExit('NN initial source snapshot is not hash-matched to its launch-time T1')
    initial_ast = ast.parse(snapshot_path.read_text(encoding='utf-8'))
    final_ast = ast.parse(runner.read_text(encoding='utf-8'))
    for name in ('csrank_labels', 'model_kwargs', 'construct_model', 'training_summary', 'synthetic_data'):
        old = next(n for n in initial_ast.body if isinstance(n, ast.FunctionDef) and n.name == name)
        new = next(n for n in final_ast.body if isinstance(n, ast.FunctionDef) and n.name == name)
        if ast.dump(old) != ast.dump(new):
            raise SystemExit('NN smoke training helper changed: ' + name)
    for path, model in [(args.xgb_smoke_meta, 'XGBoost'), (args.nn_smoke_meta, 'LSTM')]:
        record = json.loads(path.read_text(encoding='utf-8'))
        if not record['smoke']:
            raise SystemExit('only synthetic smoke records can be packaged')
        rounds = record['training'].get('actual_epochs', record['training'].get('actual_boosting_rounds'))
        if rounds != 2 or record['kwargs']['seed'] != record['seed_passed_to_model']:
            raise SystemExit('smoke training length or seed failed')
        if model == 'LSTM' and record['kwargs']['n_jobs'] != 20:
            raise SystemExit('NN smoke changed pinned worker count')
        snapshot = record.get('dependency_sha256', {}).get(runner.name)
        if model == 'LSTM' and snapshot is None:
            snapshot = early_seeds['runner_sha256']
        smokes.append(dict(model=model, meta_path=str(path), meta_sha256=sha256_file(path),
                           rounds=rounds, runner_snapshot_sha256=snapshot,
                           final_runner_sha256=sha256_file(runner)))
    test_environment = dict(os.environ)
    # DL-15 has hard failure criteria; there is no xfail acceptance override.
    test_environment.pop('LIMT_ACCEPT_STORED_FLOAT32_TIES', None)
    if args.labels:
        test_environment['LIMT_TEST_LABELS'] = str(args.labels.resolve())
    if args.limt_src:
        test_environment['LIMT_TEST_SRC'] = str(args.limt_src.resolve())
    test = subprocess.run([sys.executable, '-B', '-m', 'pytest',
                           'tests/new_test_gate_b_family.py', 'tests/new_test_label_rank_dl15.py',
                           'tests/new_test_modelscope_platform.py',
                           '-q', '-ra', '-p', 'no:cacheprovider'],
                          cwd=args.root, capture_output=True, text=True, env=test_environment)
    files = (sorted((args.root / 'src').glob('new_*.py')) + [args.spec, args.dl15_entry]
             + sorted((args.root / 'tests').glob('new_test*.py')))
    def manifest_key(path):
        return path.relative_to(args.root).as_posix() if path.is_relative_to(args.root) else str(path.resolve())
    hashes = {manifest_key(p): sha256_file(p) for p in files}
    frozen = args.root / 'src' / 'gate_b_analysis.py'
    candidate = args.root / 'src' / 'new_gate_b_analysis_v3.py'
    diff = ''.join(difflib.unified_diff(frozen.read_text(encoding='utf-8').splitlines(keepends=True),
                   candidate.read_text(encoding='utf-8').splitlines(keepends=True),
                   fromfile='src/gate_b_analysis.py (frozen v2)', tofile='src/new_gate_b_analysis_v3.py (draft v3)'))
    runner_diff = ''.join(difflib.unified_diff(snapshot_path.read_text(encoding='utf-8').splitlines(keepends=True),
                         runner.read_text(encoding='utf-8').splitlines(keepends=True),
                         fromfile='NN smoke initial runner (hash-matched)', tofile='final runner'))
    summary = (
        'Gate B code delivery — draft, no commit, no real-data training, no unsealing.\n\n'
        'Acceptance complete: ' + str(test.returncode == 0 and rank_report['passed']) + '\n'
        'Stored-raw rank diagnostic: ' + json.dumps(rank_report['diagnostic']) + '\n'
        'Storage-precision exception explicitly accepted: ' + str(rank_report['stored_precision_exception_accepted']) + '\n'
        'Validation: ' + test.stdout.strip() + '\n'
        'T1: XGB same-seed bitwise repeatability/different-seed predictions; all four NN initial hashes passed.\n'
        'Synthetic integration: 18 tags x 8 rules, 15 validation forensics; protected-file hashes unchanged.\n'
        'DL-15 replaces stored-raw exact equality with tie membership and tie-size bounds.\n'
        'All available label rows with r1 endpoints through 2026-06-30 are checked.\n'
        'Numeric precision diagnostics cover signal dates before 2023 only; later checks disclose booleans only.\n'
        'CSRankNorm pct-rank mean is 1.73/n, approaching zero; tests preserve this exact finite-n convention.\n'
        'Smoke: XGBoost seed 2, 2 boosting rounds; LSTM seed 1, 2 epochs, pinned n_jobs=20.\n'
        'The long Windows NN smoke began during development, before metadata hardening. Its initial\n'
        'runner hash is recorded in the earlier T1 report, exact source snapshot, and provenance addendum. Its live\n'
        'script_sha256 field was computed at finish, so use the addendum for the initial source hash.\n'
        'That snapshot differs from the final runner in metadata and production preflight code.\n'
        'Formal fit/seed/label/worker logic is unchanged between those snapshots.\n'
        'SC5: user explicitly confirmed the positive-correlation hypothesis on 2026-10-19; one-sided p used.\n'
        'SC6: g_fee - c_fee; positive means C0 pays less.\n'
        'Preregistration lines 37/69/70/72/176/189 were verified before implementing changes.\n'
        'No Gate B --run was executed. gate_b.json and deviation logs were not written.\n\n'
        'DL-3 is approved and the v2 spec is frozen by the user.\n'
        'Pending external steps: review the updated v3 diff and enter DL-12;\n'
        'rerun T1 on ModelScope A10/Python 3.12/torch 2.13.0/CUDA 13.0 before any production run.\n'
        'Target image: ubuntu22.04-cuda13.0.3-py312-torch2.13.0.\n'
        'This local validation used Windows/Python 3.13/torch CUDA; the target A10 image was not exercised.\n'
        'All 15 new production tags require matching CPU, GPU models, and software versions.\n'
        'Hostname, OS kernel, and memory-size fields remain in meta but do not enter the fingerprint.\n'
        'The runner checks a passing, hash-matched T1 record from the same runtime across restarts.\n'
        'First production fit is LSTM-csrank_seed1, after which the user chooses concurrency.\n\n'
        'Portable commands (supply explicit paths; every output/cache must be new):\n'
        'python src/new_gate_b_analysis_v3.py --self-test\n'
        'python src/new_gate_c_holm.py --self-test\n'
        'python src/new_verify_seeds.py --root ROOT --spec SPEC --threads 2 --out NEW_T1_JSON\n'
        'python src/new_run_benchmarks_v2.py --root ROOT --limt-src LIMT_SRC --w0w2 W0W2 '
        '--labels LABELS --panel PANEL --splits SPLITS --spec SPEC --cache NEW_CACHE '
        '--models LSTM --seed 1 --label-norm csrank --threads 2 --out-dir NEW_SCORES_DIR '
        '--seed-verification NEW_T1_JSON\n'
        'python src/new_rule_grid_bench_v2.py --root ROOT --limt-src LIMT_SRC --w0w2 W0W2 '
        '--spec SPEC --feed FEED --bench BENCH --cache NEW_GRID_CACHE --scores-dir NEW_SCORES_DIR '
        '--de-scores-dir OLD_DE_SCORES_DIR --label r1csrank --out NEW_SELECTION_JSON\n'
        'python src/new_bench_forensics_v2.py --root ROOT --limt-src LIMT_SRC --w0w2 W0W2 '
        '--spec SPEC --cache NEW_FORENSICS_CACHE --scores-dir NEW_SCORES_DIR --out NEW_FORENSICS_JSON\n\n'
        'v3 candidate SHA256: ' + hashes['src/new_gate_b_analysis_v3.py'] + '\n'
        'Gate C candidate SHA256: ' + hashes['src/new_gate_c_holm.py'] + '\n'
        'v2 spec frozen SHA256 (DL-3 approved): ' + hashes[manifest_key(args.spec)] + '\n'
    )
    args.out_dir.mkdir(parents=True, exist_ok=False)
    for name, text in [('new_gate_b_analysis_v2_to_v3.diff', diff),
                       ('new_smoke_runner_initial_to_final.diff', runner_diff),
                       ('new_pytest_results.txt', test.stdout + test.stderr),
                       ('new_delivery_report.txt', summary)]:
        target = args.out_dir / name
        preflight([target])
        with target.open('x', encoding='utf-8') as stream:
            stream.write(text)
        write_json(str(target) + '.meta.json', provenance.finish(dict(artifact_sha256=sha256_file(target))))
    write_json(baseline_meta, provenance.finish(dict(artifact_sha256=sha256_file(args.protected_before))))
    write_json(spec_meta, provenance.finish(dict(artifact_sha256=sha256_file(args.spec), status='DL-3 frozen')))
    write_json(snapshot_meta, provenance.finish(dict(artifact_sha256=sha256_file(snapshot_path),
               source_evidence_sha256=sha256_file(args.nn_snapshot_seed_report))))
    write_json(args.out_dir / 'new_DL15_entry.meta.json', provenance.finish(dict(
        artifact_path=str(args.dl15_entry), artifact_sha256=sha256_file(args.dl15_entry),
        rank_report_sha256=sha256_file(args.rank_report))))
    write_json(args.out_dir / 'new_nn_smoke_provenance_addendum.json', dict(
        initial_runner_sha256=early_seeds['runner_sha256'],
        initial_runner_snapshot_path=str(snapshot_path),
        initial_source_hash_evidence_sha256=sha256_file(args.nn_snapshot_seed_report),
        nn_smoke_meta_sha256=sha256_file(args.nn_smoke_meta),
        final_runner_sha256=sha256_file(runner),
        note='Development smoke: original live script hash was computed at finish. Initial runner hash comes from the pre-launch T1 record; formal fit/seed/label/worker logic was unchanged.',
        meta=provenance.finish()))
    write_json(args.out_dir / 'new_code_manifest.json', dict(
        status='draft code complete; no commit/real-data training/unsealing',
        sha256=hashes, protected_before=before, protected_after=after,
        protected_unchanged=True, seed_report_sha256=sha256_file(args.seed_report),
        smokes=smokes, pytest_exit_code=test.returncode,
        rank_report_sha256=sha256_file(args.rank_report), rank_precision_check=rank_report,
        acceptance_complete=test.returncode == 0 and rank_report['passed'],
        meta=provenance.finish()))
    passed = test.returncode == 0 and rank_report['passed']
    print('DELIVERY AUDIT ' + ('PASS' if passed else 'NEEDS REVIEW: see recorded acceptance failures'))
    print('ANALYSIS_SHA256=' + hashes['src/new_gate_b_analysis_v3.py'])
    print('GATE_C_SHA256=' + hashes['src/new_gate_c_holm.py'])
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
