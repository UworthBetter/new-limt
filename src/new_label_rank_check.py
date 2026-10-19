"""Bounded pre-2019 raw-r1/z-r1 rank diagnostic; no fitting or eval values."""
import argparse
import zipfile
from pathlib import Path

import numpy as np

from new_gate_b_common import Provenance, path_args, preflight, resolve_paths, write_json
from new_run_benchmarks_v2 import csrank_labels


def diagnose_train_labels(path, rows=32):
    if not 1 <= rows <= 32:
        raise ValueError('diagnostic is limited to 1..32 training-date rows')
    with zipfile.ZipFile(path) as archive:
        if 'raw_r1.npy' not in archive.namelist():
            return dict(raw_field_present=False, stored_raw_rank_exact=None)
        with archive.open('dates.npy') as stream:
            dates = np.load(stream, allow_pickle=False).astype('datetime64[D]')
        lo = int(np.searchsorted(dates, np.datetime64('2008-01-01')))
        hi = min(lo + rows, int(np.searchsorted(dates, np.datetime64('2019-01-01'))))
        if lo >= hi or hi + 1 >= len(dates) or str(dates[hi + 1]) > '2018-12-31':
            raise SystemExit('diagnostic label endpoint is outside the training window')

        def selected_rows(name):
            # Seek over compressed bytes, but decode/index only the requested
            # training cells. C-order and Fortran-order NPZ members are supported.
            with archive.open(name + '.npy') as stream:
                version = np.lib.format.read_magic(stream)
                reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                          else np.lib.format.read_array_header_2_0)
                shape, fortran, dtype = reader(stream)
                if len(shape) != 2 or shape[0] != len(dates) or dtype.hasobject:
                    raise ValueError('unexpected label array contract')
                offset = stream.tell()
                if fortran:
                    out = np.empty((hi - lo, shape[1]), dtype=dtype)
                    for col in range(shape[1]):
                        stream.seek(offset + (col * shape[0] + lo) * dtype.itemsize)
                        out[:, col] = np.frombuffer(stream.read((hi - lo) * dtype.itemsize), dtype=dtype)
                    return out
                stream.seek(offset + lo * shape[1] * dtype.itemsize)
                raw = stream.read((hi - lo) * shape[1] * dtype.itemsize)
                return np.frombuffer(raw, dtype=dtype).reshape((hi - lo, shape[1]))

        raw, z, mask = (selected_rows(k) for k in ('raw_r1', 'z_r1', 'label_mask_r1'))
    if not mask.any():
        raise ValueError('empty diagnostic mask')
    left, right = csrank_labels(raw, mask, 0, hi - lo - 1), csrank_labels(z, mask, 0, hi - lo - 1)
    differences = mask & (left != right)
    explained = True
    for t, j in zip(*np.where(differences)):
        group = mask[t] & (raw[t] == raw[t, j])
        explained &= bool(group.sum() > 1 and np.unique(z[t, group]).size > 1
                          and np.isclose(left[t, j], right[t, group].mean(), atol=1e-7))
    return dict(raw_field_present=True, date_start=str(dates[lo]), date_end=str(dates[hi - 1]),
                max_label_endpoint=str(dates[hi + 1]), raw_dtype=str(raw.dtype), z_dtype=str(z.dtype),
                cells_checked=int(mask.size), finite_mask_cells=int(mask.sum()),
                mismatch_cells=int(differences.sum()), stored_raw_rank_exact=not bool(differences.any()),
                all_mismatches_explained_by_stored_raw_ties=explained)


def synthetic_proof():
    raw = np.array([[1., 1. + 1e-9, 0., 2.]], np.float64)
    z = (raw - raw.mean(axis=1, keepdims=True)) / raw.std(axis=1, keepdims=True)
    mask = np.ones(raw.shape, bool)
    exact = np.array_equal(csrank_labels(raw, mask, 0, 0), csrank_labels(z, mask, 0, 0), equal_nan=True)
    rounded = np.array_equal(csrank_labels(raw.astype(np.float32), mask, 0, 0),
                             csrank_labels(z.astype(np.float32), mask, 0, 0), equal_nan=True)
    assert exact and not rounded
    return dict(before_storage_rank_exact=exact, independent_float32_storage_can_change_ties=not rounded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    path_args(parser)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--accept-stored-float32-ties', action='store_true',
                        help='Use only after explicit reviewer/user acceptance of this storage-precision exception')
    args = parser.parse_args()
    resolve_paths(args, ('root', 'labels', 'spec'))
    preflight([args.out])
    provenance = Provenance(__file__, args)
    diagnostic = diagnose_train_labels(args.labels)
    exact = diagnostic['stored_raw_rank_exact']
    accepted = bool(args.accept_stored_float32_ties and
                    diagnostic.get('all_mismatches_explained_by_stored_raw_ties'))
    passed = exact is not False or accepted
    write_json(args.out, dict(passed=passed, diagnostic=diagnostic, synthetic=synthetic_proof(),
                             stored_precision_exception_accepted=accepted,
                             runner_policy='unchanged: rank z_r1 over label_mask_r1',
                             meta=provenance.finish()))
    print('LABEL RANK CHECK ' + ('PASS' if passed else 'REVIEW REQUIRED: stored float32 ties differ'))
    if not passed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
