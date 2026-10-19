"""Qlib-default benchmark arms on the clean CSI300 panel (descriptive; DL 2026-10-18).

Uses pyqlib 0.9.6 model classes with hyperparameters pinned verbatim from
microsoft/qlib tag v0.9.6 examples/benchmarks (configs/benchmarks_qlib_defaults_v1.json).
No tuning. Data: the clean panel every other arm consumes -- 20 FilterCol
Alpha158 columns, z_r1 label (clean 1-day, t+1 entry), train 2008-2018,
valid 2019-2020; scores saved for valid and rule_select (2021-01-04..2022-12-28)
exactly like the neural scoring pass, so the same CPU 8-rule grid applies.

Single run per model, seed 1 (Transformer keeps its config seed 0).
Descriptive arms: not in the Holm family.

NOTE (environment): pyqlib installed from the cp312 wheel with the version tag
renamed (no cp313 wheel exists); the compiled qlib.data._libs extensions are
replaced by unused pure-Python stubs -- see venv_bench/Lib/site-packages/
qlib/data/_libs/*.py. The expression engine is never invoked by this script.

Usage: run_benchmarks.py --cache DIR --models XGBoost DoubleEnsemble [LSTM GRU ALSTM Transformer]
Run inside venv_bench.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

LIMT_SRC = Path(r"D:\research\limt-repro\src")
BENCH_SRC = Path(__file__).resolve().parent
for path in (str(LIMT_SRC), str(BENCH_SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from v2_w4_data import Data  # noqa: E402
from v2_w6_run import DEFAULT  # noqa: E402

ROOT = Path(r"D:\research\new_LiMT")
W0W2 = ROOT / "data" / "csi300_w1w2"
LABELS = W0W2 / "data" / "v2" / "new_v2_labels_csi300.npz"
SPEC = ROOT / "configs" / "benchmarks_qlib_defaults_v1.json"
FILTER20 = ["RESI5", "WVMA5", "RSQR5", "KLEN", "RSQR10", "CORR5", "CORD5", "CORR10", "ROC60", "RESI10",
            "VSTD5", "RSQR60", "CORR60", "WVMA60", "STD5", "RSQR20", "CORD60", "CORD10", "CORR20", "KLOW"]
TS_MODELS = {"LSTM", "GRU", "ALSTM", "Transformer"}


def build_master(data, z_r1, day_lo, day_hi):
    """(datetime, instrument)-indexed frame of 20 features + LABEL0 over [day_lo, day_hi]."""
    cols = [list(data.feature_names).index(c) for c in FILTER20]
    feats = np.asarray(data.features[day_lo:day_hi + 1, :, cols], np.float32)
    labels = z_r1[day_lo:day_hi + 1].astype(np.float32)
    arr = np.concatenate([feats, labels[..., None]], axis=2)
    dates = pd.to_datetime([str(d) for d in data.dates[day_lo:day_hi + 1]])
    index = pd.MultiIndex.from_product([dates, list(data.symbols)], names=["datetime", "instrument"])
    frame = pd.DataFrame(arr.reshape(-1, arr.shape[2]), index=index,
                         columns=FILTER20 + ["LABEL0"])
    return frame


class BenchDataset:
    """qlib-style ts dataset over the clean panel.

    prepare(segment) returns a TSDataSampler (window features + LABEL0 as the
    last column) for the pytorch ts models; a list of segments returns a tuple.
    """

    def __init__(self, master, segments, flt, step_len):
        self.master = master
        self.segments = segments  # name -> (start_date_str, end_date_str)
        self.flt = flt  # name -> pd.Series(bool) indexed like master, or None
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
    """qlib-style tabular dataset for tree models (feature/label column grouping)."""

    def __init__(self, master, segments, flt):
        self.master = master
        self.segments = segments
        self.flt = flt

    def prepare(self, segment="test", col_set=None, data_key=None, **kwargs):
        if isinstance(col_set, str):
            col_set = [col_set]
        if col_set is None or "label" in col_set:
            table = True
        else:
            table = False
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
        if table:
            sub.columns = pd.MultiIndex.from_tuples(
                [("feature", c) for c in FILTER20] + [("label", "LABEL0")])
        else:
            sub = sub[FILTER20]  # qlib's col_set="feature" selection drops the group level
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=1,
                        help="global torch/numpy/random seed and the tag suffix; Transformer keeps its config seed 0")
    args = parser.parse_args()
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    for name in args.models:
        if name not in spec["models"]:
            raise SystemExit(f"unknown benchmark model {name}")
    seed = args.seed
    random.seed(seed); np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    data = Data(W0W2, args.cache, 21, 0, DEFAULT["evaluation_end"])
    with np.load(LABELS, allow_pickle=False) as labels:
        z_r1 = labels["z_r1"]
        mask_r1 = labels["label_mask_r1"]
    valid_days = data.days("valid")
    select_days = data.days("rule_select")
    assert str(data.dates[int(select_days[-1]) + 2]) <= "2022-12-31"
    print(f"ASSERT rule_select terminal mark {data.dates[int(select_days[-1]) + 2]} <= 2022-12-31", flush=True)
    valid_dates = [str(d) for d in data.dates[valid_days]]
    select_dates = [str(d) for d in data.dates[select_days]]
    train_span = ("2008-01-01", "2018-12-31")
    valid_span = (valid_dates[0], valid_dates[-1])
    select_span = (select_dates[0], select_dates[-1])

    # Phase A master: history through the validation end (train + valid fits).
    day_hi_a = int(valid_days[-1]) + 2  # keep the label horizon inside the frame
    master_a = build_master(data, z_r1, 0, day_hi_a)
    dates_all = [str(d) for d in data.dates]

    def flt_series(lo_date, hi_date):
        """Boolean over master_a rows: r1 label defined inside [lo, hi]."""
        lo = int(np.searchsorted(data.dates, np.datetime64(lo_date), side="left"))
        hi = int(np.searchsorted(data.dates, np.datetime64(hi_date), side="right")) - 1
        keep = np.zeros((len(data.dates), len(data.symbols)), bool)
        keep[lo:hi + 1] = mask_r1[lo:hi + 1] & np.isfinite(z_r1[lo:hi + 1])
        out = pd.Series(keep[:day_hi_a + 1].reshape(-1), index=master_a.index)
        return out

    ts_segments = {"train": train_span, "valid": valid_span, "test": valid_span}
    ts_flt = {"train": flt_series(*train_span), "valid": flt_series(*valid_span), "test": None}
    tree_segments = {"train": train_span, "valid": valid_span, "test": valid_span}
    out_dir = ROOT / "runs" / "scores"
    out_dir.mkdir(exist_ok=True)
    for name in args.models:
        entry = spec["models"][name]
        t0 = time.perf_counter()
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        module = __import__(entry["module"], fromlist=[entry["class"]])
        cls = getattr(module, entry["class"])
        kwargs = dict(entry["kwargs"])
        # n_jobs is a DataLoader infra knob (workers), not a model hyperparameter;
        # 0 keeps loading in the main process (Windows spawn cannot pickle the
        # in-memory dataset). All other kwargs stay verbatim from the yaml.
        kwargs["n_jobs"] = 0
        if name == "Transformer":
            kwargs.setdefault("seed", 0)
        model = cls(**kwargs)
        evals_result = {}
        if name in TS_MODELS:
            dataset = BenchDataset(master_a, ts_segments, ts_flt, spec["step_len_ts"])
        else:
            dataset = TreeDataset(master_a, tree_segments,
                                  {k: (ts_flt[k] if k != "test" else None) for k in tree_segments})
        if name in TS_MODELS or name == "XGBoost":
            model.fit(dataset, evals_result=evals_result)
        else:
            model.fit(dataset)
        fit_seconds = time.perf_counter() - t0

        # Phase B: prediction master (validation start minus window history through rule_select end).
        lo_b = max(0, int(valid_days[0]) - 25)
        master_b = build_master(data, z_r1, lo_b, int(select_days[-1]))
        if name in TS_MODELS:
            pred_ds = BenchDataset(master_b, {"test": select_span}, {"test": None}, spec["step_len_ts"])
            preds = model.predict(pred_ds)
            valid_ds = BenchDataset(master_b, {"test": valid_span}, {"test": None}, spec["step_len_ts"])
            valid_preds = model.predict(valid_ds)
        else:
            pred_ds = TreeDataset(master_b, {"test": select_span}, {"test": None})
            preds = model.predict(pred_ds, segment="test")
            valid_ds = TreeDataset(master_b, {"test": valid_span}, {"test": None})
            valid_preds = model.predict(valid_ds, segment="test")
        matrix = score_matrix(preds, select_dates, list(data.symbols))
        matrix_valid = score_matrix(valid_preds, valid_dates, list(data.symbols))
        tag = f"{name}_seed{seed}"
        meta = dict(tag=tag, model=entry["module"] + "." + entry["class"], kwargs=kwargs,
                    seed_policy=spec["data_convention"]["seed_policy"], fit_seconds=fit_seconds,
                    evals_result_summary={k: (len(v), (v.index(max(v)) if v else None), (max(v) if v else None))
                                          for k, v in evals_result.items()} if evals_result else None,
                    label="z_r1 (clean 1-day)", features="FilterCol 20 from clean panel",
                    descriptive=True)
        with (out_dir / f"{tag}.npz").open("wb") as stream:
            np.savez_compressed(
                stream,
                dates_rule_select=np.array(select_dates), scores_rule_select=matrix,
                candidate_rule_select=np.stack([data.candidate(int(t)) for t in select_days]),
                dates_valid=np.array(valid_dates), scores_valid=matrix_valid,
                candidate_valid=np.stack([data.candidate(int(t)) for t in valid_days]),
            )
        (out_dir / f"{tag}.meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
        n_finite = int(np.isfinite(matrix[np.stack([data.candidate(int(t)) for t in select_days])]).sum())
        print(f"BENCH DONE {name} fit_s={fit_seconds:.0f} finite_select_scores={n_finite} -> {tag}.npz", flush=True)
        del model, dataset, pred_ds, valid_ds, master_b
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("BENCH PASS DONE", flush=True)


if __name__ == "__main__":
    main()
