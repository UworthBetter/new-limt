"""Shared v3.1d access guard. No CLI/JSON cutoff override exists.

Metadata whitelist: dates (calendar), symbols (identifiers), lookback (integer),
indices (symbol positions), feature_names (column identifiers), splits (date
boundaries), days() (calendar positions). These contain no observations.
features, feature_mask, universe, raw_universe, listing_eligible: features stream;
stale and execution-feed columns: prices; relatives(): marks; targets/z_h5/
z_r1: labels at t+6/t+2; label/loss masks: labels at signal date.
close and marks cannot be requested directly. Unknown attributes fail closed.
Expected local import additions are the fixed EXPECTED_LOCAL_MODULES below.
Raw-data construction and alignment validation precede wrapping; no computation
on observations may subsequently use the raw object.
"""
import argparse
from collections.abc import Mapping
from contextlib import contextmanager
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np

# Identity, not string equality, is the admission check.
UNSEAL_CUTOFF = ''.join(('2026-', '06-30'))
PRE_UNSEAL_CUTOFF = ''.join(('2022-', '12-31'))
STREAMS = ('prices', 'marks', 'features', 'labels', 'bench')
EXPECTED_LOCAL_MODULES = frozenset((
    'new_guarded_data_v1', 'new_gate_b_common', 'new_bench_replay',
    'new_bench_dataset', 'new_run_benchmarks_v2', 'new_run_benchmarks_h5_v3',
    'new_gate_b_analysis_v3_1', 'new_bench_eval_v1', 'new_h5_common_v3',
    'v2_w3_ledger', 'v2_w4_data', 'v2_w4_training', 'v2_w5_results',
    'v2_w6_policy', 'v2_w6_run', 'model', 'cpu_baselines'))


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def verify_guard(root, gate=None, committed=True):
    root = Path(root).resolve()
    path = root / 'configs/new_guarded_data_v1.sha256'
    digest = sha256_file(__file__)
    if path.read_text(encoding='utf-8').strip() != digest:
        raise SystemExit('guarded module hash mismatch')
    if committed:
        import subprocess
        try:
            recorded = subprocess.check_output(
                ['git', '-C', str(root), 'show', 'HEAD:configs/new_guarded_data_v1.sha256'],
                stderr=subprocess.DEVNULL, text=True).strip()
            source = subprocess.check_output(
                ['git', '-C', str(root), 'show', 'HEAD:src/new_guarded_data_v1.py'],
                stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            raise SystemExit('guard must be committed before production')
        if recorded != digest or hashlib.sha256(source).hexdigest() != digest:
            raise SystemExit('committed guard differs from working module')
    if gate is not None and gate.get('guarded_data_sha256') != digest:
        raise SystemExit('gate guarded_data_sha256 mismatch')
    return digest


class DateGuard:
    def __init__(self, cutoff):
        if cutoff is not UNSEAL_CUTOFF and cutoff is not PRE_UNSEAL_CUTOFF:
            raise SystemExit('cutoff must be a named constant object')
        self.cutoff = cutoff
        self.cutoff_name = ('UNSEAL_CUTOFF' if cutoff is UNSEAL_CUTOFF else 'PRE_UNSEAL_CUTOFF')
        self.maximum = dict.fromkeys(STREAMS)
        self._scopes = []

    def touch(self, category, day):
        if category not in STREAMS:
            raise SystemExit('unknown guard stream')
        value = str(np.datetime64(day, 'D'))
        if value == 'NaT' or value > self.cutoff:
            raise SystemExit(f'date guard: {category} exceeds cutoff or is invalid')
        for record in (self.maximum, *self._scopes):
            old = record[category]
            record[category] = value if old is None else max(old, value)

    def label_endpoint(self, dates, t, mask, horizon=6):
        if np.any(mask):
            if t < 0 or t + horizon >= len(dates):
                raise SystemExit('label endpoint exceeds available calendar')
            self.touch('labels', dates[t + horizon])

    @contextmanager
    def scope(self):
        record = dict.fromkeys(STREAMS)
        self._scopes.append(record)
        try:
            yield record
        finally:
            assert self._scopes[-1] is record
            self._scopes.pop()

    def report(self, expected=STREAMS, emit=True):
        expected = set(expected)
        if not expected <= set(STREAMS):
            raise SystemExit('unknown expected stream')
        for name, value in self.maximum.items():
            if name in expected and value is None:
                raise SystemExit('guard stream never touched: ' + name)
            if name not in expected and value is not None:
                raise SystemExit('unexpected guard stream: ' + name)
        if emit:
            for name, value in self.maximum.items():
                print(f'ASSERT max_date_touched {name}={value}', flush=True)
        return dict(self.maximum)

    def metadata(self, expected, emit=False):
        return dict(guard_cutoff_name=self.cutoff_name, guard_cutoff_value=self.cutoff,
                    guarded_data_sha256=sha256_file(__file__),
                    max_date_touched=self.report(expected, emit))


class PriceGuardArray:
    """Indexed read wrapper; implicit ndarray conversion is intentionally absent."""
    def __init__(self, values, dates, guard, stream='prices', horizon=0):
        self._values, self._dates, self._guard = values, dates, guard
        self._stream, self._horizon = stream, horizon

    @property
    def shape(self):
        return self._values.shape

    @property
    def dtype(self):
        return self._values.dtype

    def __len__(self):
        return len(self._values)

    def __array__(self, *args, **kwargs):
        raise SystemExit('unguarded array conversion')

    def __getitem__(self, index):
        rows = index[0] if isinstance(index, tuple) else index
        if rows is None or rows is Ellipsis:
            rows = slice(None)
        positions = np.atleast_1d(np.arange(len(self._dates))[rows]).reshape(-1)
        if self._horizon and len(positions) and np.any(positions + self._horizon >= len(self._dates)):
            raise SystemExit('label endpoint exceeds available calendar')
        for t in positions:
            self._guard.touch(self._stream, self._dates[int(t) + self._horizon])
        # A copy prevents a caller from acquiring a view into unguarded rows.
        return np.array(self._values[index], copy=True)


class GuardedData:
    _metadata = frozenset(('dates', 'symbols', 'lookback', 'indices', 'feature_names', 'splits'))
    _features = frozenset(('features', 'feature_mask', 'universe', 'raw_universe', 'listing_eligible'))
    _masks = frozenset(('label_mask', 'mask_r1', 'label_mask_r1', 'mask_h5', 'label_mask_h5'))

    def __init__(self, data, guard):
        self._data, self._guard = data, guard

    def __getattr__(self, name):
        if name in self._metadata:
            value = getattr(self._data, name)
            return value.copy() if hasattr(value, 'copy') else value
        if name in self._features or name in self._masks or name in ('stale', 'targets', 'z_h5', 'z_r1'):
            stream = ('features' if name in self._features else 'prices' if name == 'stale' else 'labels')
            horizon = 6 if name in ('targets', 'z_h5') else 2 if name == 'z_r1' else 0
            return PriceGuardArray(getattr(self._data, name), self._data.dates, self._guard, stream, horizon)
        if name == 'loss_masks':
            return {k: PriceGuardArray(v, self._data.dates, self._guard, 'labels')
                    for k, v in self._data.loss_masks.items()}
        raise SystemExit('unguarded attribute ' + name)

    def days(self, split, *args, **kwargs):
        return self._data.days(split, *args, **kwargs)

    def windows(self, days):
        for t in np.asarray(days, int):
            if t < self._data.lookback - 1:
                raise SystemExit('insufficient causal context')
            self._guard.touch('features', self._data.dates[t])
        return self._data.windows(days)

    def candidate(self, t):
        self._guard.touch('features', self._data.dates[t])
        return self._data.candidate(t).copy()

    def relatives(self, mark_day, previous_day):
        for t in (mark_day, previous_day):
            self._guard.touch('marks', self._data.dates[t])
        return self._data.relatives(mark_day, previous_day)


class GuardedBench(Mapping):
    def __init__(self, values, guard):
        self._values, self._guard = dict(values), guard

    def __getitem__(self, key):
        self._guard.touch('bench', key)
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)


def guard_execution_feed(feed, dates, guard):
    return {k: PriceGuardArray(v, dates, guard) if k in (
        'raw_close', 'state', 'share_multiplier', 'cash_dividend_per_old_share') else v
        for k, v in feed.items()}


def load_evaluation_data(args, guard):
    """Construction-only raw IO. Return guarded objects, never raw observations.

    Data performs canonical input alignment checks. Additional r1 labels use the
    same canonical labels file; their values are only exposed with t+2 accounting.
    """
    from v2_w4_data import Data
    from v2_w5_results import load_execution_feed, benchmark
    raw=Data(args.w0w2,args.cache,21,0,UNSEAL_CUTOFF)
    canonical=Path(args.w0w2)/'data/v2/new_v2_labels_csi300.npz'
    if Path(args.labels).resolve()!=canonical.resolve():
        raise SystemExit('evaluation labels must use canonical Data input')
    for name in ('panel','splits'):
        if Path(getattr(args,name)).resolve()!=(canonical.parent/f'new_v2_{name}_csi300.npz').resolve():
            raise SystemExit('evaluation input must use canonical Data path: '+name)
    with np.load(canonical,allow_pickle=False) as z:
        raw.z_r1=z['z_r1'];raw.mask_r1=z['label_mask_r1']
    raw.z_h5=raw.targets[...,0]
    feed=guard_execution_feed(load_execution_feed(raw,args.feed),raw.dates,guard)
    bench=GuardedBench(benchmark(args.bench)['values'],guard)
    return GuardedData(raw,guard),feed,bench


def guarded_portfolio_events(fn, data, days, scores, guard, records, **kwargs):
    days = np.asarray(days, int)
    if not len(days) or np.any(np.diff(days) <= 0) or days[0] < 0 or days[-1] + 2 >= len(data.dates):
        raise SystemExit('invalid soft portfolio days')
    end = str(data.dates[int(days[-1]) + 2])
    if end > guard.cutoff:
        raise SystemExit('soft terminal exceeds cutoff')
    with guard.scope() as record:
        result = fn(data, days, scores, **kwargs)
    if record['marks'] != end:
        raise SystemExit('soft marks maximum differs from terminal')
    records.append(record['marks'])
    return result


def third_party_versions(required=()):
    result = {}
    for dist, module in (('torch', 'torch'), ('pyqlib', 'qlib'), ('xgboost', 'xgboost'),
                         ('lightgbm', 'lightgbm'), ('numpy', 'numpy'), ('pandas', 'pandas'), ('scipy', 'scipy')):
        try:
            installed = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        loaded = sys.modules.get(module)
        result[dist] = dict(distribution=installed, module=getattr(loaded, '__version__', None)) if installed else None
        if dist in required and installed is None:
            raise SystemExit('missing required package ' + dist)
    if result['torch']:
        result['torch']['cuda'] = getattr(getattr(sys.modules.get('torch'), 'version', None), 'cuda', None)
    return result


def import_closure(root, limt_src, start=None, required=()):
    roots = [Path(root).resolve() / 'src', Path(limt_src).resolve()]
    result = {}
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.is_file() and any(path.is_relative_to(p) for p in roots):
            result[name] = dict(path=str(path), sha256=sha256_file(path))
            print(f'IMPORT {name} {result[name]["sha256"]}', flush=True)
    if start is not None:
        if set(result) - set(start) - EXPECTED_LOCAL_MODULES:
            raise SystemExit('unexpected local module imported after start')
        if any(result.get(k) != v for k, v in start.items()):
            raise SystemExit('local import changed during run')
    return result


def self_test(limt_src=None):
    from types import SimpleNamespace
    if limt_src:
        sys.path.insert(0, str(limt_src))
    import torch
    from v2_w4_training import portfolio_events
    from v2_w6_policy import signal_days
    def blocked(name, fn):
        try:
            fn()
        except SystemExit:
            print('ASSERT positive control ' + name + ' fired')
        else:
            raise AssertionError('guard did not fire: ' + name)
    assert UNSEAL_CUTOFF == '2026-06-30' and PRE_UNSEAL_CUTOFF == '2022-12-31'
    blocked('constant identity', lambda: DateGuard(''.join(['2026', '-06-30'])))
    for cutoff, safe, late in ((UNSEAL_CUTOFF, '2026-06-30', '2026-07-01'),
                               (PRE_UNSEAL_CUTOFF, '2022-12-30', '2023-01-03')):
        dates = np.arange(np.datetime64(safe) - 12, np.datetime64(late) + 10)
        a, b = (int(np.where(dates == np.datetime64(d))[0][0]) for d in (safe, late))
        shape = (len(dates), 4)
        raw = SimpleNamespace(dates=dates, symbols=np.array(list('ABCD')), lookback=2,
            feature_mask=np.ones(shape, bool), features=np.ones((*shape, 2)),
            stale=np.zeros(shape, bool), targets=np.ones((*shape, 3)),
            loss_masks={'eval': np.ones(shape, bool)},
            candidate=lambda t: np.ones(4, bool), windows=lambda ds: np.ones((len(ds), 4, 2, 2)),
            relatives=lambda m, p: np.ones(5))
        guard = DateGuard(cutoff)
        data = GuardedData(raw, guard)
        feed = guard_execution_feed({'raw_close': np.ones(shape)}, dates, guard)
        bench = GuardedBench({safe: 0., late: 0.}, guard)
        accesses = [('prices', lambda t: feed['raw_close'][t]),
                    ('prices tuple', lambda t: feed['raw_close'][t, 0]),
                    ('marks', lambda t: data.relatives(t, t-1)),
                    ('windows', lambda t: data.windows([t])),
                    ('candidate', lambda t: data.candidate(t)),
                    ('feature_mask', lambda t: data.feature_mask[t]),
                    ('labels', lambda t: guard.label_endpoint(dates, t-6, [True])),
                    ('loss_masks', lambda t: data.loss_masks['eval'][t]),
                    ('targets', lambda t: data.targets[t-6]),
                    ('bench', lambda t: bench[str(dates[t])])]
        for name, fn in accesses:
            fn(a)
            blocked(name, lambda fn=fn: fn(b))
        for name in ('close', 'marks', 'unknown'):
            blocked(name, lambda name=name: getattr(data, name))
        assert all(v == safe for v in guard.report(emit=False).values())
        # Invoke the real numerical portfolio function, with calendar-generated days.
        if cutoff is UNSEAL_CUTOFF:
            ds = signal_days(dates, 'eval', safe, 2)
            records = []
            guarded_portfolio_events(portfolio_events, data, ds, np.zeros((len(ds), 4)),
                                     guard, records, tau=.5)
            assert records == [safe]
            blocked('real portfolio terminal', lambda: guarded_portfolio_events(
                portfolio_events, data, np.r_[ds, ds[-1]+1], np.zeros((len(ds)+1, 4)),
                guard, [], tau=.5))
    root = Path(__file__).resolve().parents[1]
    import_closure(root, limt_src or root.parent / 'limt-repro/src')
    print('SELF-TEST PASS: price/mark/feature guard + positive controls')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true', required=True)
    parser.add_argument('--limt-src', type=Path, required=True)
    self_test(parser.parse_args().limt_src)
