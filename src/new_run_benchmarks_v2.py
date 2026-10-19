"""Explicit-seed Qlib benchmarks. Production epochs/workers/batches follow spec.

Only synthetic --smoke may reduce training to two epochs/boosting rounds.
DL-3 is approved; production targets the ModelScope A10 CUDA image.
"""
import argparse
import importlib
import inspect
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from new_bench_dataset import BenchDataset, TreeDataset, build_master, score_matrix
from new_gate_b_common import (BlindData, Provenance, configure_runtime, path_args,
                               preflight, resolve_paths, sha256_file, write_json)

TS_MODELS = {'LSTM', 'GRU', 'ALSTM', 'Transformer'}
MODELS = TS_MODELS | {'XGBoost'}
PRODUCTION_IMAGE = 'ubuntu22.04-cuda13.0.3-py312-torch2.13.0'


def csrank_labels(z_r1, mask_r1, day_lo, day_hi):
    """Rank ALL label_mask_r1 members, independent of candidate/filter masks."""
    out = np.full((day_hi - day_lo + 1, z_r1.shape[1]), np.nan, np.float32)
    for i, t in enumerate(range(day_lo, day_hi + 1)):
        keep = mask_r1[t]
        if not np.isfinite(z_r1[t, keep]).all():
            raise ValueError('nonfinite member of label_mask_r1')
        if keep.any():
            out[i, keep] = (pd.Series(z_r1[t, keep]).rank(method='average', pct=True).to_numpy() - .5) * 3.46
    return out


def model_kwargs(entry, seed):
    if seed not in (1, 2, 3):
        raise ValueError('seed must be 1/2/3')
    return dict(entry['kwargs'], seed=seed)


def construct_model(entry, seed, smoke=False):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    kwargs = model_kwargs(entry, seed)
    if smoke and entry['class'] != 'XGBModel':
        kwargs['n_epochs'] = 2
    cls = getattr(importlib.import_module(entry['module']), entry['class'])
    return cls(**kwargs), kwargs


def training_summary(model, history, name):
    """Qlib NN histories contain NEGATIVE loss; XGB histories contain RMSE."""
    summary = {}
    for key, values in history.items():
        loss = np.asarray(values, float) * (-1 if name in TS_MODELS else 1)
        if not len(loss) or not np.isfinite(loss).all():
            raise ValueError('empty or nonfinite training history')
        best = int(np.argmin(loss))
        summary[key] = dict(actual_rounds=len(loss), best_iteration=best, minimum_loss=float(loss[best]))
    if name == 'XGBoost':
        booster = model.model
        summary['actual_boosting_rounds'] = booster.num_boosted_rounds()
        summary['best_iteration'] = int(booster.best_iteration)
        summary['xgb_actual_nthread'] = int(json.loads(booster.save_config())['learner']['generic_param']['nthread'])
    else:
        summary['actual_epochs'] = summary['valid']['actual_rounds']
        summary['best_epoch'] = summary['valid']['best_iteration']
    return summary


def synthetic_data(spec):
    from types import SimpleNamespace
    rng = np.random.default_rng(120)
    # Enough samples for pinned NN batches (including Transformer's 8192).
    dates = np.array(pd.bdate_range('2018-01-01', periods=180).strftime('%Y-%m-%d'))
    symbols = np.array([f'S{i:04d}' for i in range(256)])
    columns = spec['filter_col_20']
    features = rng.normal(size=(len(dates), len(symbols), len(columns))).astype(np.float32)
    z = features[..., 0] + .1 * rng.normal(size=features.shape[:2])
    return SimpleNamespace(dates=dates, symbols=symbols, features=features,
                           feature_names=np.array(columns), z_r1=z,
                           mask_r1=np.ones(z.shape, bool),
                           candidate=lambda t: np.ones(len(symbols), bool))


def verify_t1(args, spec):
    if args.seed_verification is None:
        raise SystemExit('production requires --seed-verification from new_verify_seeds.py')
    record = json.loads(args.seed_verification.read_text(encoding='utf-8'))
    expected = {'XGBoost', 'LSTM', 'GRU', 'ALSTM', 'Transformer'}
    if (not record.get('passed') or set(record.get('checks', {})) != expected
            or not all(v.get('passed') for v in record['checks'].values())
            or record['meta']['spec_sha256'] != sha256_file(args.spec)
            or record.get('runner_sha256') != sha256_file(__file__)
            or record.get('verifier_sha256') != sha256_file(Path(__file__).with_name('new_verify_seeds.py'))):
        raise SystemExit('T1 record failed, incomplete, stale, or belongs to a different spec/runner')
    import platform
    import importlib.metadata
    import torch
    meta = record['meta']
    if (meta['versions']['python'] != platform.python_version()
            or meta['versions']['torch'] != importlib.metadata.version('torch')
            or meta.get('cuda') != torch.version.cuda
            or meta.get('hip') != torch.version.hip):
        raise SystemExit('rerun T1 with the same Python/torch/CUDA/HIP runtime as production')
    for filename, digest in meta.get('dependency_sha256', {}).items():
        if sha256_file(Path(__file__).with_name(filename)) != digest:
            raise SystemExit('T1 code dependencies changed; rerun verification before production')


def verify_production_platform():
    """Validate exposed runtime versions; record the intended image separately."""
    import platform
    import importlib.metadata
    import torch
    gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if (not platform.python_version().startswith('3.12.')
            or str(torch.__version__).split('+', 1)[0] != '2.13.0'
            or not torch.cuda.is_available() or torch.version.hip is not None
            or str(torch.version.cuda).split('.')[:2] != ['13', '0']
            or not gpu_names or not all(re.search(r'\bA10\b', name) for name in gpu_names)
            or importlib.metadata.version('pyqlib') != '0.9.6'):
        raise SystemExit('production requires ModelScope A10 / CUDA 13.0 / Python 3.12 / '
                         'torch 2.13.0 / pyqlib 0.9.6; image: ' + PRODUCTION_IMAGE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    path_args(parser)
    parser.add_argument('--cache', required=True, type=Path)
    parser.add_argument('--models', nargs='+', required=True, choices=sorted(MODELS))
    parser.add_argument('--seed', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--label-norm', choices=('zscore', 'csrank'), required=True)
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--seed-verification', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    resolve_paths(args, ('root', 'spec') if args.smoke else
                  ('root', 'limt_src', 'w0w2', 'labels', 'panel', 'splits', 'spec'))
    args.out_dir = args.out_dir or args.root / 'runs' / 'new_scores_v2'
    forbidden = args.root / 'runs' / 'scores'
    if args.out_dir.resolve().is_relative_to(forbidden.resolve()):
        raise SystemExit('legacy scores directory is forbidden')
    if args.smoke and ('smoke' not in args.out_dir.name.lower()
                       or args.out_dir.resolve().is_relative_to((args.root / 'runs' / 'new_scores_v2').resolve())):
        raise SystemExit('smoke requires a separate out-dir with smoke in its name')
    if len(set(args.models)) != len(args.models):
        raise SystemExit('duplicate models')
    for name in args.models:
        if (name == 'XGBoost') != (args.label_norm == 'zscore'):
            raise SystemExit('XGBoost requires zscore; all NN require csrank; invoke separately')
    tags = {name: f'{name}-{"seeded" if name == "XGBoost" else "csrank"}_seed{args.seed}' for name in args.models}
    targets = [args.out_dir / (tag + suffix) for tag in tags.values() for suffix in ('.npz', '.meta.json')]
    preflight(targets + [args.cache])
    spec = json.loads(args.spec.read_text(encoding='utf-8'))
    if not args.smoke:
        verify_t1(args, spec)
    provenance = Provenance(__file__, args, synthetic=args.smoke)
    import torch
    if not args.smoke:
        verify_production_platform()
    if TS_MODELS.intersection(args.models) and not args.smoke and not torch.cuda.is_available():
        raise SystemExit('production NN benchmarks require a GPU (CUDA or HIP)')
    if args.smoke:
        args.cache.mkdir(parents=True, exist_ok=False)
        data = synthetic_data(spec)
        train_span = (str(data.dates[20]), str(data.dates[89]))
        valid_days, select_days = np.arange(90, 135), np.arange(135, 178)
    else:
        data = BlindData(args)
        train_span = ('2008-01-01', '2018-12-31')
        valid_days, select_days = data.days('valid'), data.days('rule_select')
    assert str(data.dates[int(select_days[-1]) + 2]) <= '2022-12-31'
    print(f'ASSERT rule_select terminal mark {data.dates[int(select_days[-1]) + 2]} <= 2022-12-31', flush=True)
    valid_dates, select_dates = ([str(d) for d in data.dates[days]] for days in (valid_days, select_days))
    valid_span, select_span = (valid_dates[0], valid_dates[-1]), (select_dates[0], select_dates[-1])
    z, mask = data.z_r1, data.mask_r1
    for name in args.models:
        started = time.perf_counter()
        model_started_at = datetime.now(timezone.utc).isoformat()
        entry = spec['models'][name]
        columns = spec['filter_col_20'] if name in TS_MODELS else list(data.feature_names)
        hi = int(valid_days[-1]) + 2
        learning = z.copy()
        if name in TS_MODELS:
            learning[:hi + 1] = csrank_labels(z, mask, 0, hi)
        master_a = build_master(data, learning, 0, hi, columns)
        def flt(span):
            dates = pd.to_datetime(master_a.index.get_level_values('datetime'))
            keep = (mask[:hi + 1] & np.isfinite(z[:hi + 1])).reshape(-1)
            return pd.Series(keep & (dates >= span[0]) & (dates <= span[1]), index=master_a.index)
        segments = dict(train=train_span, valid=valid_span, test=valid_span)
        filters = dict(train=flt(train_span), valid=flt(valid_span), test=None)
        dataset = (BenchDataset(master_a, segments, filters, spec['step_len_ts']) if name in TS_MODELS
                   else TreeDataset(master_a, segments, filters, columns))
        model, kwargs = construct_model(entry, args.seed, args.smoke)
        history = {}
        checkpoint = args.cache / f'new_{tags[name]}.pt'
        if name in TS_MODELS:
            preflight([checkpoint, Path(str(checkpoint) + '.meta.json')])
            model.fit(dataset, evals_result=history, save_path=str(checkpoint))
        else:
            fit_kwargs = dict(evals_result=history)
            if args.smoke:
                fit_kwargs['num_boost_round'] = 2
            model.fit(dataset, **fit_kwargs)
        fit_seconds = time.perf_counter() - started
        summary = training_summary(model, history, name)
        master_b = build_master(data, z, max(0, int(valid_days[0]) - 25), int(select_days[-1]), columns)
        def predict(span):
            ds = (BenchDataset(master_b, {'test': span}, {'test': None}, spec['step_len_ts']) if name in TS_MODELS
                  else TreeDataset(master_b, {'test': span}, {'test': None}, columns))
            return model.predict(ds)
        values = dict(dates_rule_select=np.array(select_dates),
                      scores_rule_select=score_matrix(predict(select_span), select_dates, list(data.symbols)),
                      candidate_rule_select=np.stack([data.candidate(int(t)) for t in select_days]),
                      dates_valid=np.array(valid_dates),
                      scores_valid=score_matrix(predict(valid_span), valid_dates, list(data.symbols)),
                      candidate_valid=np.stack([data.candidate(int(t)) for t in valid_days]))
        meta = provenance.finish(dict(tag=tags[name], model=entry['module'] + '.' + entry['class'],
                    kwargs=kwargs, effective_seed=args.seed, seed_passed_to_model=args.seed,
                    label_norm=args.label_norm, rank_set='label_mask_r1' if name in TS_MODELS else None,
                    feature_set='filter20' if name in TS_MODELS else 'all158', n_features=len(columns),
                    effective_model_settings={key: getattr(model, key, None) for key in
                        ('n_epochs', 'early_stop', 'batch_size', 'n_jobs', 'device', 'd_feat')},
                    fit_defaults={key: parameter.default for key, parameter in inspect.signature(model.fit).parameters.items()
                                  if key in ('num_boost_round', 'early_stopping_rounds')},
                    xgb_n_estimators_note='qlib forwards n_estimators to xgb.train; it is not num_boost_round' if name == 'XGBoost' else None,
                    seed_policy=spec['data_convention']['seed_policy'], fit_seconds=fit_seconds,
                    training=summary, xgb_actual_nthread=summary.get('xgb_actual_nthread'),
                    model_started_at=model_started_at,
                    model_wall_seconds=time.perf_counter() - started,
                    device_hours=(time.perf_counter() - started) / 3600 if name in TS_MODELS and torch.cuda.is_available() else 0,
                    smoke=args.smoke, descriptive=True,
                    expected_container_image=PRODUCTION_IMAGE if not args.smoke else None,
                    seed_verification_sha256=sha256_file(args.seed_verification) if args.seed_verification else None))
        target = args.out_dir / f'{tags[name]}.npz'
        preflight([target, target.with_suffix('.meta.json')])
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            np.savez_compressed(stream, **values)
        meta['artifact_sha256'] = sha256_file(target)
        write_json(target.with_suffix('.meta.json'), meta)
        if name in TS_MODELS:
            write_json(str(checkpoint) + '.meta.json', dict(meta, artifact_sha256=sha256_file(checkpoint)))
        print(f'BENCH DONE {tags[name]} smoke={args.smoke}', flush=True)
        del model, dataset, master_a, master_b
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
