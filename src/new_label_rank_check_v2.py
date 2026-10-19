"""DL-15 bounded label precision acceptance, with no 2023+ numeric disclosure."""
import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from new_gate_b_common import Provenance, path_args, preflight, resolve_paths, write_json

CUTOFF = np.datetime64('2026-06-30')
PUBLIC_END = np.datetime64('2023-01-01')


def read_rows(archive, name, total_dates, lo, hi):
    """Decode only [lo, hi) rows, including Fortran-ordered NPZ members."""
    with archive.open(name + '.npy') as stream:
        version = np.lib.format.read_magic(stream)
        reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                  else np.lib.format.read_array_header_2_0)
        shape, fortran, dtype = reader(stream)
        if (len(shape) != 2 or shape[0] != total_dates or dtype.hasobject
                or not 0 <= lo < hi <= total_dates):
            raise ValueError('unexpected bounded label array contract')
        offset = stream.tell()
        if fortran:
            out = np.empty((hi - lo, shape[1]), dtype=dtype)
            for col in range(shape[1]):
                stream.seek(offset + (col * shape[0] + lo) * dtype.itemsize)
                out[:, col] = np.frombuffer(stream.read((hi - lo) * dtype.itemsize), dtype=dtype)
            return out
        stream.seek(offset + lo * shape[1] * dtype.itemsize)
        return np.frombuffer(stream.read((hi - lo) * shape[1] * dtype.itemsize),
                             dtype=dtype).reshape((hi - lo, shape[1]))


def day_check(raw, z):
    """Average ordinal ranks and exact stored-value tie groups on the same mask."""
    if raw.dtype != np.float32 or z.dtype != np.float32:
        raise ValueError('DL-15 requires stored float32 values; do not recast data')
    if not np.isfinite(raw).all() or not np.isfinite(z).all():
        raise ValueError('nonfinite value inside label_mask_r1')
    left = pd.Series(raw).rank(method='average').to_numpy()
    right = pd.Series(z).rank(method='average').to_numpy()
    groups = []
    for values in (raw, z):
        _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        groups.append(counts[inverse])
    size = np.maximum(*groups)
    delta = np.abs(left - right)
    changed = delta != 0
    return dict(left=left, right=right, changed=changed, delta=delta,
                tie_membership_pass=bool(np.all(size[changed] > 1)),
                tie_size_bound_pass=bool(np.all(delta[changed] <= size[changed])))


def diagnose_labels(path):
    with zipfile.ZipFile(path) as archive:
        with archive.open('dates.npy') as stream:
            dates = np.load(stream, allow_pickle=False).astype('datetime64[D]')
        with archive.open('exit_idx_r1.npy') as stream:
            exits = np.load(stream, allow_pickle=False)
        if (dates.ndim != 1 or exits.shape != dates.shape
                or exits.dtype.kind not in 'iu' or np.isnat(dates).any()
                or not np.all(dates[1:] > dates[:-1])):
            raise ValueError('invalid calendar or r1 endpoint mapping')
        lo = 0
        hi = int(np.searchsorted(dates, CUTOFF, side='right'))
        candidate = np.arange(lo, hi)
        valid_exit = (exits[candidate] >= 0) & (exits[candidate] < len(dates))
        eligible = candidate[valid_exit]
        eligible = eligible[dates[exits[eligible]] <= CUTOFF]
        if not eligible.size:
            raise ValueError('no eligible label endpoints')
        # Read contiguous eligible runs only: an ineligible row is never decoded.
        runs = np.split(eligible, np.flatnonzero(np.diff(eligible) != 1) + 1)
        public = dict(affected_days=0, affected_stock_days=0,
                      maximum_rank_difference=0.0, minimum_daily_spearman=None,
                      undefined_spearman_days=0, checked_days=0)
        checks = {key: dict(tie_membership_pass=True, tie_size_bound_pass=True,
                            data_contract_pass=True)
                  for key in ('pre2023', 'from2023')}
        for run in runs:
            a, b = int(run[0]), int(run[-1]) + 1
            raw, z, mask = (read_rows(archive, k, len(dates), a, b)
                            for k in ('raw_r1', 'z_r1', 'label_mask_r1'))
            if raw.shape != z.shape or raw.shape != mask.shape or mask.dtype != bool:
                raise ValueError('label arrays or mask disagree')
            for j, t in enumerate(run):
                is_public = dates[t] < PUBLIC_END
                status = checks['pre2023' if is_public else 'from2023']
                if exits[t] <= t or dates[t] > CUTOFF:
                    raise ValueError('invalid or out-of-bounds actual label endpoint')
                try:
                    result = day_check(raw[j, mask[j]], z[j, mask[j]])
                except ValueError:
                    status['data_contract_pass'] = False
                    continue
                for key in ('tie_membership_pass', 'tie_size_bound_pass'):
                    status[key] &= result[key]
                if not is_public:
                    continue  # No post-2022 counts, correlations, or magnitudes.
                public['checked_days'] += 1
                public['affected_days'] += int(result['changed'].any())
                public['affected_stock_days'] += int(result['changed'].sum())
                public['maximum_rank_difference'] = max(public['maximum_rank_difference'],
                                                         float(result['delta'].max(initial=0)))
                left, right = result['left'], result['right']
                if len(left) < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
                    public['undefined_spearman_days'] += 1
                else:
                    rho = float(np.clip(np.corrcoef(left, right)[0, 1], -1, 1))
                    current = public['minimum_daily_spearman']
                    public['minimum_daily_spearman'] = rho if current is None else min(current, rho)
    passed = all(all(status.values()) for status in checks.values())
    return dict(passed=passed, acceptance='DL-15: user-confirmed replacement criteria',
                date_start=str(dates[eligible[0]]), date_end=str(dates[eligible[-1]]),
                max_label_endpoint=str(dates[exits[eligible]].max()), cutoff=str(CUTOFF),
                rank_difference_units='average ordinal rank (1..n)',
                tie_bound='maximum size of the exact raw_r1/z_r1 tie groups containing that stock',
                public_pre2023=public, pre2023_acceptance=checks['pre2023'],
                from2023_acceptance=dict(**checks['from2023'], numeric_metrics='not disclosed'),
                runner_policy='unchanged: rank z_r1 over the full label_mask_r1 set')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    path_args(parser)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    resolve_paths(args, ('root', 'labels', 'spec'))
    preflight([args.out])
    provenance = Provenance(__file__, args)
    diagnostic = diagnose_labels(args.labels)
    write_json(args.out, dict(passed=diagnostic['passed'], diagnostic=diagnostic,
                             stored_precision_exception_accepted=True,
                             meta=provenance.finish()))
    print('DL-15 LABEL ACCEPTANCE ' + ('PASS' if diagnostic['passed'] else 'FAIL'))
    if not diagnostic['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
