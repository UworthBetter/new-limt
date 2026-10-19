"""Original benchmark dataset adapters; no global paths or side effects."""
import numpy as np
import pandas as pd

def build_master(data, z_r1, day_lo, day_hi, columns):
    """(datetime, instrument)-indexed frame of the given feature columns + LABEL0."""
    cols = [list(data.feature_names).index(c) for c in columns]
    feats = np.asarray(data.features[day_lo:day_hi + 1, :, cols], np.float32)
    labels = z_r1[day_lo:day_hi + 1].astype(np.float32)
    arr = np.concatenate([feats, labels[..., None]], axis=2)
    dates = pd.to_datetime([str(d) for d in data.dates[day_lo:day_hi + 1]])
    index = pd.MultiIndex.from_product([dates, list(data.symbols)], names=["datetime", "instrument"])
    return pd.DataFrame(arr.reshape(-1, arr.shape[2]), index=index,
                        columns=list(columns) + ["LABEL0"])


class BenchDataset:
    """qlib-style ts dataset: prepare(segment) -> TSDataSampler (features + LABEL0 last)."""

    def __init__(self, master, segments, flt, step_len):
        self.master = master
        self.segments = segments
        self.flt = flt
        self.step_len = step_len
        self._samplers = {}

    def prepare(self, segment="test", col_set=None, data_key=None, **kwargs):
        from qlib.data.dataset import TSDataSampler

        if isinstance(segment, (list, tuple)):
            return tuple(self.prepare(s, col_set, data_key) for s in segment)
        start, end = self.segments[segment]
        flt = self.flt.get(segment)
        if segment not in self._samplers:
            self._samplers[segment] = TSDataSampler(
                self.master.copy(), start, end, self.step_len,
                dtype=np.float32, flt_data=None if flt is None else flt.copy())
        return self._samplers[segment]


class TreeDataset:
    """qlib-style tabular dataset (feature/label column grouping), any feature set."""

    def __init__(self, master, segments, flt, columns):
        self.master = master
        self.segments = segments
        self.flt = flt
        self.columns = list(columns)

    def prepare(self, segment="test", col_set=None, data_key=None, **kwargs):
        if isinstance(col_set, str):
            col_set = [col_set]
        table = col_set is None or "label" in col_set
        if isinstance(segment, (list, tuple)):
            return tuple(self._slice(s, table) for s in segment)
        return self._slice(segment, table)

    def _slice(self, segment, table):
        start, end = self.segments[segment]
        flt = self.flt.get(segment)
        sub = self.master.loc[(slice(start, end), slice(None)), :].copy()
        if flt is not None:
            aligned = flt.reindex(sub.index).fillna(False).astype(bool)
            sub = sub.loc[aligned.values]
        header = [("feature", c) for c in self.columns]
        if table:
            sub.columns = pd.MultiIndex.from_tuples(header + [("label", "LABEL0")])
        else:
            sub = sub[self.columns]  # qlib's col_set="feature" drops the group level
        return sub


def score_matrix(preds, dates, symbols):
    date_at = {d: i for i, d in enumerate(dates)}
    sym_at = {s: j for j, s in enumerate(symbols)}
    out = np.full((len(dates), len(symbols)), np.nan, np.float32)
    for (day, sym), value in preds.items():
        key = day if isinstance(day, str) else pd.Timestamp(day).strftime("%Y-%m-%d")
        i, j = date_at.get(key), sym_at.get(sym)
        if i is not None and j is not None:
            out[i, j] = value
    return out
