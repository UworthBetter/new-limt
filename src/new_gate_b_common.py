"""Shared, exclusive-output provenance and pre-2023 benchmark input contract."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def preflight(paths):
    paths = [Path(p) for p in paths]
    if len({p.resolve() for p in paths}) != len(paths):
        raise ValueError('duplicate output paths')
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)


def write_json(path, value):
    preflight([path])
    path = Path(path)
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(payload)


def benchmark_platform_fingerprint(meta):
    """Stable across instance restarts; retain CPU, GPU models, and software."""
    return dict(cpu=meta.get('cpu'),
                gpus=sorted(gpu['name'] for gpu in meta.get('gpus', [])),
                versions=meta.get('versions'), cuda=meta.get('cuda'), hip=meta.get('hip'))


def path_args(parser):
    for name in ('root', 'limt-src', 'w0w2', 'labels', 'panel', 'splits', 'spec', 'feed', 'bench'):
        parser.add_argument('--' + name, type=Path,
                            default=os.environ.get('LIMT_' + name.upper().replace('-', '_')))
    parser.add_argument('--threads', type=int, default=None)
    parser.add_argument('--miopen-user-db-path', type=Path)


def resolve_paths(args, required=('root', 'limt_src', 'w0w2', 'labels', 'panel', 'splits', 'spec')):
    if args.w0w2:
        directory = args.w0w2 / 'data' / 'v2'
        for key, filename in [('labels', 'new_v2_labels_csi300.npz'),
                              ('panel', 'new_v2_panel_csi300.npz'),
                              ('splits', 'new_v2_splits_csi300.npz')]:
            if getattr(args, key) is None:
                setattr(args, key, directory / filename)
    if args.spec is None and args.root:
        args.spec = args.root / 'configs' / 'new_benchmarks_qlib_defaults_v2.json'
    missing = [k for k in required if getattr(args, k, None) is None]
    if missing:
        raise SystemExit('missing path arguments: ' + ', '.join('--' + k.replace('_', '-') for k in missing))
    for key in required:
        value = Path(getattr(args, key)).resolve()
        if not value.exists():
            raise FileNotFoundError(value)
        setattr(args, key, value)
    if args.limt_src:
        sys.path.insert(0, str(args.limt_src.resolve()))
    configure_runtime(args)
    return args


def configure_runtime(args):
    if args.threads is not None:
        if args.threads < 1:
            raise ValueError('--threads must be positive')
        os.environ['OMP_NUM_THREADS'] = str(args.threads)
        import torch
        torch.set_num_threads(args.threads)
    if args.miopen_user_db_path:
        preflight([args.miopen_user_db_path])
        args.miopen_user_db_path.mkdir(parents=True, exist_ok=False)
        os.environ['MIOPEN_USER_DB_PATH'] = str(args.miopen_user_db_path.resolve())


class Provenance:
    def __init__(self, script, args, synthetic=False):
        self.start = datetime.now(timezone.utc).isoformat()
        self.clock = time.perf_counter()
        self.script, self.args, self.synthetic = Path(script), args, synthetic
        self.script_hash = sha256_file(self.script)
        self.hashes = {}
        self.dependencies = {p.name: sha256_file(p) for p in self.script.parent.glob('new_*.py')}
        for key in ('spec', 'labels', 'panel', 'splits', 'feed', 'bench'):
            path = getattr(args, key, None)
            self.hashes[key] = ({'path': str(path), 'sha256': sha256_file(path)}
                                if path else {'status': 'synthetic' if synthetic else 'not_used'})

    def finish(self, extra=None):
        import torch
        try:
            import psutil
            memory = psutil.virtual_memory().total
            cpu = platform.processor()
            physical_cores = psutil.cpu_count(logical=False)
            if sys.platform == 'linux':
                for line in Path('/proc/cpuinfo').read_text().splitlines():
                    if line.startswith('model name'):
                        cpu = line.split(':', 1)[1].strip()
                        break
            elif sys.platform == 'win32':
                import winreg
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                    r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
                    cpu = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
        except ImportError:
            memory, cpu = None, platform.processor()
            physical_cores = None
            if hasattr(os, 'sysconf'):
                memory = os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
        def git(*cmd):
            try:
                return subprocess.check_output(['git', '-C', str(self.args.root), *cmd],
                                               text=True, stderr=subprocess.DEVNULL).strip()
            except (OSError, subprocess.CalledProcessError):
                return None
        status = git('status', '--porcelain', '--untracked-files=all')
        versions = {'python': platform.python_version()}
        for package in ('pyqlib', 'numpy', 'pandas', 'torch', 'xgboost'):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        seconds = time.perf_counter() - self.clock
        gpu = [{'name': torch.cuda.get_device_name(i),
                'memory_bytes': torch.cuda.get_device_properties(i).total_memory}
               for i in range(torch.cuda.device_count())]
        result = dict(git_head=git('rev-parse', 'HEAD'), git_dirty=None if status is None else bool(status),
                      script_sha256=self.script_hash, inputs=self.hashes,
                      dependency_sha256=self.dependencies,
                      spec_sha256=self.hashes['spec'].get('sha256'), versions=versions,
                      cuda=torch.version.cuda, hip=torch.version.hip,
                      hostname=socket.gethostname(), os=platform.platform(), cpu=cpu,
                      cpu_logical_cores=os.cpu_count(), memory_bytes=memory, gpus=gpu,
                      cpu_physical_cores=physical_cores,
                      cpu_affinity_cores=len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                      torch_num_threads=torch.get_num_threads(),
                      omp_num_threads=os.environ.get('OMP_NUM_THREADS'),
                      miopen_user_db_path=os.environ.get('MIOPEN_USER_DB_PATH'),
                      started_at=self.start, ended_at=datetime.now(timezone.utc).isoformat(),
                      argv=sys.argv, wall_seconds=seconds, device_hours=seconds / 3600 if gpu else 0,
                      synthetic=self.synthetic, xgb_actual_nthread=None)
        result.update(extra or {})
        return result


class BlindData:
    """Only expose pre-2023 days; no eval targets or eval scores are loaded.

    NPZ array decompression is not date-indexed IO. Slice arrays immediately,
    and never calculate a post-selection statistic. Each cache must be fresh.
    """
    def __init__(self, args):
        import numpy as np
        from v2_w6_policy import listing_eligible, signal_days
        self._signal_days = signal_days
        self.lookback = 21
        preflight([args.cache])
        args.cache.mkdir(parents=True, exist_ok=False)
        with zipfile.ZipFile(args.panel) as archive, archive.open('feat_mat.npy') as source:
            with (args.cache / 'new_features.npy').open('xb') as dest:
                shutil.copyfileobj(source, dest, 4 * 1024 * 1024)
        all_features = np.load(args.cache / 'new_features.npy', mmap_mode='r', allow_pickle=False)
        with np.load(args.labels, allow_pickle=False) as labels:
            all_dates = labels['dates'].astype('datetime64[D]')
            n = int(np.searchsorted(all_dates, np.datetime64('2023-01-01')))
            self.dates = all_dates[:n]
            self.symbols = labels['symbols'].astype(str)
            self.z_r1 = labels['z_r1'][:n]
            self.mask_r1 = labels['label_mask_r1'][:n]
        self.features = all_features[:n]
        with np.load(args.panel, allow_pickle=False) as panel:
            if (not np.array_equal(panel['dates'].astype('datetime64[D]'), all_dates)
                    or not np.array_equal(panel['symbols'].astype(str), self.symbols)):
                raise ValueError('panel/labels alignment mismatch')
            self.feature_names = panel['feat_cols'].astype(str)
            self.feature_mask = panel['feature_mask'][:n]
            self.universe = panel['universe_mask'][:n] & listing_eligible(self.dates, self.symbols)
        with np.load(args.splits, allow_pickle=False) as splits:
            if (not np.array_equal(splits['dates'].astype('datetime64[D]'), all_dates)
                    or not np.array_equal(splits['symbols'].astype(str), self.symbols)):
                raise ValueError('splits/labels alignment mismatch')
        if len(self.dates) < 21 or not (np.diff(self.dates) > np.timedelta64(0, 'D')).all():
            raise ValueError('invalid pre-2023 calendar')
        expected = (n, len(self.symbols))
        if (self.features.shape != (*expected, len(self.feature_names))
                or self.features.dtype != np.float32
                or self.z_r1.shape != expected or self.mask_r1.shape != expected
                or any(x.shape != expected or x.dtype != bool for x in
                       (self.mask_r1, self.feature_mask, self.universe))):
            raise ValueError('panel shape/mask contract mismatch')

    def candidate(self, t):
        return self.feature_mask[t] & self.universe[t]

    def days(self, split):
        if split not in ('train', 'valid', 'rule_select'):
            raise SystemExit('blind input forbids requested split')
        return self._signal_days(self.dates, split, lookback=self.lookback)


def blind_feed(data, path):
    import numpy as np
    from v2_w5_results import load_execution_feed
    with np.load(path, allow_pickle=False) as feed:
        if (not np.array_equal(feed['dates'][:len(data.dates)].astype('datetime64[D]'), data.dates)
                or not np.array_equal(feed['symbols'].astype(str), data.symbols)):
            raise ValueError('feed alignment mismatch')
        keys = ('raw_close', 'state', 'share_multiplier', 'cash_dividend_per_old_share',
                'is_st', 'market_state_known', 'limit_up', 'limit_down', 'no_limit')
        arrays = {k: feed[k][:len(data.dates)] for k in keys if k in feed}
    return load_execution_feed(data, arrays)


def blind_benchmark(path):
    """Verify bytes, but convert return values only for pre-2023 rows."""
    import math
    meta = json.loads(Path(str(path) + '.meta.json').read_text(encoding='utf-8'))
    if (meta.get('benchmark_id') != 'H00300' or meta.get('return_kind') != 'total_return'
            or meta.get('sha256') != sha256_file(path) or not meta.get('source')):
        raise ValueError('benchmark identity/kind/hash mismatch')
    values = {}
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['date'] >= '2023-01-01':
                continue
            value = float(row['return'])
            if not math.isfinite(value) or value <= -1 or row['date'] in values:
                raise ValueError('invalid benchmark row')
            values[row['date']] = value
    return dict(values=values, provenance=meta)
