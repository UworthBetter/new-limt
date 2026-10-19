"""Three-seed rule selection for 15 new benchmark tags and unchanged DE tags."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from new_gate_b_common import (BlindData, Provenance, blind_benchmark, blind_feed,
                               benchmark_platform_fingerprint, path_args, preflight,
                               resolve_paths, sha256_file, write_json)

LEGACY_KINDS = ('LSTM-csrank', 'GRU-csrank', 'ALSTM-csrank', 'Transformer-csrank',
         'XGBoost-seeded', 'DoubleEnsemble')


from new_guarded_data_v1 import (DateGuard, PRE_UNSEAL_CUTOFF, GuardedData,
    GuardedBench, guard_execution_feed, verify_guard)
from new_benchmark_manifest_v1 import load_manifest


def tag_parts(tag):
    row = TAGS[tag]
    return row['kind'], row['seed']


def main(chooser=None):
    parser = argparse.ArgumentParser(description=__doc__)
    path_args(parser)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument('--cache', required=True, type=Path)
    parser.add_argument('--scores-dir', required=True, type=Path)
    parser.add_argument('--de-scores-dir', type=Path,
                        help='Read-only source of unchanged DoubleEnsemble seed 1-3 scores')
    parser.add_argument('--label')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path)
    args = parser.parse_args()
    resolve_paths(args, ('root', 'limt_src', 'w0w2', 'labels', 'panel', 'splits', 'spec', 'feed', 'bench'))
    global KINDS, TAGS
    manifest, TAGS = load_manifest(args.manifest)
    KINDS = tuple(manifest['kinds'])
    if args.label is not None and args.label != manifest['label']:
        raise SystemExit('manifest label mismatch')
    args.label = manifest['label']
    verify_guard(args.root)
    guard = DateGuard(PRE_UNSEAL_CUTOFF)
    from new_bench_replay import RULES, COLUMNS, annual_excess, choose_arm, share_replay, _with_benchmark
    choose_arm = chooser or choose_arm
    seeds_by_kind = {kind:sorted(row['seed'] for row in TAGS.values() if row['kind']==kind) for kind in KINDS}
    args.out_dir = args.out_dir or args.root / 'runs' / 'new_reselect_r9'
    tags = list(TAGS)
    sources = {tag: Path(TAGS[tag]['path']) for tag in tags}
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    targets = [args.out] + [args.out.parent / f"new_reselect_{kind}_r9.json" for kind in KINDS]
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
    source_metas = {}
    for tag, path in sources.items():
        meta_path = path.with_suffix('.meta.json')
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if (meta.get('smoke') or meta.get('tag') != tag
                or meta.get('score_sha256', meta.get('artifact_sha256')) != input_hashes[tag]
                or meta.get('spec_sha256') != TAGS[tag].get('spec_sha256', sha256_file(args.spec))):
            raise SystemExit('score metadata is smoke, misaligned, stale, or belongs to a different spec')
        source_metas[tag] = meta
        platforms.add(json.dumps(benchmark_platform_fingerprint(meta), sort_keys=True))
    if len(platforms) != 1:
        raise SystemExit('manifest tags must have the same platform fingerprint')
    raw = BlindData(args)
    data = GuardedData(raw, guard)
    days = data.days('rule_select')
    exec_last, terminal_last = int(days[-1]) + 1, int(days[-1]) + 2
    assert str(data.dates[terminal_last]) <= '2022-12-31'
    print(f'ASSERT no-price-after-window: last_execution_fill={data.dates[exec_last]} '
          f'terminal_mark={data.dates[terminal_last]} all <= 2022-12-31', flush=True)
    feed = guard_execution_feed(blind_feed(raw, args.feed), raw.dates, guard)
    del raw
    bench = blind_benchmark(args.bench)
    bench['values'] = GuardedBench(bench['values'], guard)
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
                artifact_sha256=sha256_file(target), score_sha256=input_hashes[tag], tag=tag, manifest_sha256=sha256_file(args.manifest), **guard.metadata(('prices','features','bench')))))
        grids[tag] = stats
        write_json(args.out_dir / kind / args.label / f'grid_seed{seed}.json',
                   dict(stats=stats, meta=provenance.finish(dict(score_sha256=input_hashes[tag], manifest_sha256=sha256_file(args.manifest), **guard.metadata(('prices','features','bench'))))))
    selection = [dict(arm=kind, backbone=kind, label=args.label, seeds_used=seeds_by_kind[kind],
                      chosen_rule=choose_arm([grids[next(tag for tag,row in TAGS.items() if row['kind']==kind and row['seed']==s)] for s in seeds_by_kind[kind]]),
                      descriptive=True) for kind in KINDS]
    write_json(args.out, dict(window='rule_select 2021-2022', label=args.label,
               topk=50, seeds_used=sorted({r['seed'] for r in TAGS.values()}), grids=grids, selection=selection,
               integrity_assertion=f'terminal {data.dates[terminal_last]} <= 2022-12-31',
               meta=provenance.finish(dict(score_sha256=input_hashes, manifest_sha256=sha256_file(args.manifest), **guard.metadata(('prices','features','bench'))))))
    for row in selection:
        arm = row['arm']
        arm_sources = {tag:dict(new_score_sha256=input_hashes[tag],
            new_model_sha256=source_metas[tag].get('artifact_sha256'),
            old_artifact=source_metas[tag].get('retrain_of'),
            expected_verdict=source_metas[tag].get('expected_verdict', 'unknown'),
            fingerprint_sha256=source_metas[tag].get('fingerprint_sha256'))
            for tag in tags if TAGS[tag]['kind']==arm}
        write_json(args.out.parent / f'new_reselect_{arm}_r9.json', dict(
            arm=arm, selected_rule=row['chosen_rule'], sources=arm_sources,
            platform_fingerprint=json.loads(next(iter(platforms))),
            manifest_sha256=sha256_file(args.manifest),
            dl_entry=f"Replacement {arm}; selected rule {row['chosen_rule']}; old/new artifact SHA and expected_verdict are recorded in sources.",
            **guard.metadata(('prices','features','bench'))))
        print('RESELECTED ' + row['arm'] + ' ' + row['chosen_rule'], flush=True)


if __name__ == '__main__':
    main()
