"""Validation-only benchmark forensics. Never opens rule-select/eval scores."""
import argparse
import json
from pathlib import Path

import numpy as np

from new_gate_b_common import BlindData, Provenance, path_args, preflight, resolve_paths, sha256_file, write_json


def daily_r1_ic(data, days, scores, z_r1, mask_r1):
    values = []
    for i, t in enumerate(days):
        if str(data.dates[t]) > '2020-12-31':
            raise SystemExit('forensics can only access validation scores')
        keep = mask_r1[t] & np.isfinite(z_r1[t]) & np.isfinite(scores[i])
        if keep.sum() < 3:
            continue
        x, y = scores[i][keep], z_r1[t][keep]
        if x.std() > 0 and y.std() > 0:
            values.append(float(np.corrcoef(x, y)[0, 1]))
    return values


from new_benchmark_manifest_v1 import load_manifest
from new_guarded_data_v1 import DateGuard, PRE_UNSEAL_CUTOFF, GuardedData, verify_guard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    path_args(parser)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--scores-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    resolve_paths(args)
    args.out = args.out or args.root / 'protocol' / 'checks' / 'new_benchmark_forensics_v2p.json'
    preflight([args.out, args.cache])
    provenance = Provenance(__file__, args)
    verify_guard(args.root)
    manifest, tags = load_manifest(args.manifest)
    guard = DateGuard(PRE_UNSEAL_CUTOFF)
    data = GuardedData(BlindData(args), guard)
    counts = {}
    for split, lo, hi in [('train', '2008-01-01', '2018-12-31'), ('valid', '2019-01-01', '2020-12-31')]:
        keep = (data.dates >= np.datetime64(lo)) & (data.dates <= np.datetime64(hi))
        counts[split] = sum(int((data.mask_r1[t] & np.isfinite(data.z_r1[t])).sum()) for t in np.flatnonzero(keep))
    days = data.days('valid')
    results = {}
    for kind in manifest['kinds']:
        for seed in sorted(r['seed'] for r in tags.values() if r['kind']==kind):
            tag = next(t for t,r in tags.items() if r['kind']==kind and r['seed']==seed)
            path = Path(tags[tag]['path'])
            meta_path = path.with_suffix('.meta.json')
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            if meta.get('smoke') or meta.get('tag') != tag or meta.get('score_sha256', meta.get('artifact_sha256')) != sha256_file(path):
                raise SystemExit('forensics score provenance mismatch')
            with np.load(path, allow_pickle=False) as payload:
                dates, scores = payload['dates_valid'].astype(str), payload['scores_valid']
            if (dates.tolist() != [str(d) for d in data.dates[days]]
                    or scores.shape != (len(days), len(data.symbols))):
                raise ValueError('validation score dates mismatch')
            ic = daily_r1_ic(data, days, scores, data.z_r1, data.mask_r1)
            results[tag] = dict(fit_seconds=meta['fit_seconds'], training=meta.get('training'),
                                mean_valid_r1_ic=float(np.mean(ic)) if ic else None, ic_days=len(ic),
                                score_sha256=sha256_file(path), meta_sha256=sha256_file(meta_path))
    write_json(args.out, dict(train_row_counts=counts, benchmarks=results, meta=provenance.finish(dict(manifest_sha256=sha256_file(args.manifest), **guard.metadata(('labels',))))))
    print('FORENSICS DONE (validation only)', flush=True)


if __name__ == '__main__':
    main()
