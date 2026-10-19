"""Three-seed rule selection for 15 new benchmark tags and unchanged DE tags."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from new_gate_b_common import (BlindData, Provenance, blind_benchmark, blind_feed,
                               benchmark_platform_fingerprint, path_args, preflight,
                               resolve_paths, sha256_file, write_json)

KINDS = ('LSTM-csrank', 'GRU-csrank', 'ALSTM-csrank', 'Transformer-csrank',
         'XGBoost-seeded', 'DoubleEnsemble')


def tag_parts(tag):
    kind, marker, seed = tag.rpartition('_seed')
    if not marker or kind not in KINDS or seed not in ('1', '2', '3'):
        raise ValueError('invalid benchmark tag')
    return kind, int(seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    path_args(parser)
    parser.add_argument('--cache', required=True, type=Path)
    parser.add_argument('--scores-dir', required=True, type=Path)
    parser.add_argument('--de-scores-dir', type=Path,
                        help='Read-only source of unchanged DoubleEnsemble seed 1-3 scores')
    parser.add_argument('--label', choices=('r1csrank',), default='r1csrank')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path)
    args = parser.parse_args()
    resolve_paths(args, ('root', 'limt_src', 'w0w2', 'labels', 'panel', 'splits', 'spec', 'feed', 'bench'))
    from new_bench_replay import RULES, COLUMNS, annual_excess, choose_arm, share_replay, _with_benchmark
    args.out_dir = args.out_dir or args.root / 'runs' / 'csi300'
    tags = [f'{kind}_seed{s}' for kind in KINDS for s in (1, 2, 3)]
    sources = {tag: ((args.de_scores_dir or args.scores_dir) if tag.startswith('DoubleEnsemble_')
                     else args.scores_dir) / f'{tag}.npz' for tag in tags}
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    targets = [args.out]
    for tag in tags:
        kind, seed = tag_parts(tag)
        base = args.out_dir / kind / args.label
        targets.append(base / f'grid_seed{seed}.json')
        for rule, _, _ in RULES:
            daily = base / rule / f'seed{seed}' / 'daily.csv'
            targets.extend([daily, Path(str(daily) + '.meta.json')])
    preflight(targets + [args.cache])
    provenance = Provenance(__file__, args)
    input_hashes = {tag: sha256_file(path) for tag, path in sources.items()}
    platforms = set()
    for tag, path in sources.items():
        if tag.startswith('DoubleEnsemble_'):
            continue
        meta_path = path.with_suffix('.meta.json')
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if (meta.get('smoke') or meta.get('tag') != tag
                or meta.get('artifact_sha256') != input_hashes[tag]
                or meta.get('spec_sha256') != sha256_file(args.spec)):
            raise SystemExit('score metadata is smoke, misaligned, stale, or belongs to a different spec')
        platforms.add(json.dumps(benchmark_platform_fingerprint(meta), sort_keys=True))
    if len(platforms) != 1:
        raise SystemExit('all 15 new tags must have the same platform fingerprint')
    data = BlindData(args)
    days = data.days('rule_select')
    exec_last, terminal_last = int(days[-1]) + 1, int(days[-1]) + 2
    assert str(data.dates[terminal_last]) <= '2022-12-31'
    print(f'ASSERT no-price-after-window: last_execution_fill={data.dates[exec_last]} '
          f'terminal_mark={data.dates[terminal_last]} all <= 2022-12-31', flush=True)
    feed, bench = blind_feed(data, args.feed), blind_benchmark(args.bench)
    expected = [str(d) for d in data.dates[days]]
    grids = {}
    for tag in tags:
        kind, seed = tag_parts(tag)
        with np.load(sources[tag], allow_pickle=False) as payload:
            dates = payload['dates_rule_select'].astype(str).tolist()
            scores = payload['scores_rule_select']
        if dates != expected or scores.shape != (len(days), len(data.symbols)):
            raise SystemExit('rule_select scores/date alignment mismatch')
        stats = []
        for rule, drop, every in RULES:
            rows = share_replay(data, days, scores, feed, drop, every)
            if any(r['date'] >= '2023-01-01' for r in rows):
                raise SystemExit('selection replay produced a 2023+ accounting date')
            readable = _with_benchmark(rows, bench)
            target = args.out_dir / kind / args.label / rule / f'seed{seed}' / 'daily.csv'
            preflight([target, Path(str(target) + '.meta.json')])
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('x', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=COLUMNS)
                writer.writeheader()
                writer.writerows(readable)
            excess, turnover = annual_excess(readable, bench)
            stats.append(dict(rule=rule, annual_excess=excess, mean_turnover=turnover))
            write_json(str(target) + '.meta.json', provenance.finish(dict(
                artifact_sha256=sha256_file(target), score_sha256=input_hashes[tag], tag=tag)))
        grids[tag] = stats
        write_json(args.out_dir / kind / args.label / f'grid_seed{seed}.json',
                   dict(stats=stats, meta=provenance.finish(dict(score_sha256=input_hashes[tag]))))
    selection = [dict(arm=kind, backbone=kind, label=args.label, seeds_used=[1, 2, 3],
                      chosen_rule=choose_arm([grids[f'{kind}_seed{s}'] for s in (1, 2, 3)]),
                      descriptive=True) for kind in KINDS]
    write_json(args.out, dict(window='rule_select 2021-2022', label=args.label,
               topk=50, seeds_used=[1, 2, 3], grids=grids, selection=selection,
               integrity_assertion=f'terminal {data.dates[terminal_last]} <= 2022-12-31',
               meta=provenance.finish(dict(score_sha256=input_hashes))))
    print('RULE GRID DONE (numeric selection results written to files only)', flush=True)


if __name__ == '__main__':
    main()
