"""Qlib-default benchmark arms on the clean CSI300 panel (descriptive; DL entries 2026-10-18/19).

Hyperparameters are pinned from the official qlib v0.9.6 examples/benchmarks
Alpha158 workflows (configs/benchmarks_qlib_defaults_v1.json). No tuning.
Feature sets follow the yamls per model: the four neural ts models specify
FilterCol (20 columns); XGBoost and DoubleEnsemble specify no FilterCol and
use the full 158-column handler. Label: z_r1 (clean 1-day; CSRankNorm not
re-applied -- the single declared deviation). Train 2008-2018, early stop on
2019-2020; scores saved for valid and rule_select exactly like the neural
scoring pass, so the same CPU 8-rule grid applies.

Seeds (frozen by the 2026-10-19 DL entry): LSTM/GRU/ALSTM/DE seeds 1-3 via
--seed; XGBoost and Transformer are deterministic single runs (xgboost default
seed 0; Transformer config seed 0) and are only run with --seed 1.

Environment note: pyqlib installed from the relabelled cp312 wheel; the
qlib.data._libs stubs raise NotImplementedError if ever invoked.

Usage (inside venv_bench):
  run_benchmarks.py --cache DIR --models XGBoost DoubleEnsemble [--seed 1]
  run_benchmarks.py --cache DIR --seed 2 --models LSTM GRU ALSTM DoubleEnsemble
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
DETERMINISTIC = {"XGBoost", "Transformer"}


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=1,
                        help="global torch/numpy/random seed and tag suffix; deterministic models only run at 1")
    args = parser.parse_args()
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    for name in args.models:
        if name not in spec["models"]:
            raise SystemExit(f"unknown benchmark model {name}")
        if name in DETERMINISTIC and args.seed != 1:
            raise SystemExit(f"{name} is a deterministic single-run model; invoke only with --seed 1")
    seed = args.seed
    random.seed(seed); np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    data = Data(W0W2, args.cache, 21, 0, DEFAULT["evaluation_end"])
    with np.load(LABELS, allow_pickle=False) as labels:
        z_r1, mask_r1 = labels["z_r1"], labels["label_mask_r1"]
    valid_days = data.days("valid")
    select_days = data.days("rule_select")
    assert str(data.dates[int(select_days[-1]) + 2]) <= "2022-12-31"
    print(f"ASSERT rule_select terminal mark {data.dates[int(select_days[-1]) + 2]} <= 2022-12-31", flush=True)
    valid_dates = [str(d) for d in data.dates[valid_days]]
    select_dates = [str(d) for d in data.dates[select_days]]
    candidate_select = np.stack([data.candidate(int(t)) for t in select_days])
    candidate_valid = np.stack([data.candidate(int(t)) for t in valid_days])
    train_span = ("2008-01-01", "2018-12-31")
    valid_span = (valid_dates[0], valid_dates[-1])
    select_span = (select_dates[0], select_dates[-1])

    def flt_series(master, day_hi_a, lo_date, hi_date):
        lo = int(np.searchsorted(data.dates, np.datetime64(lo_date), side="left"))
        hi = int(np.searchsorted(data.dates, np.datetime64(hi_date), side="right")) - 1
        keep = np.zeros((len(data.dates), len(data.symbols)), bool)
        keep[lo:hi + 1] = mask_r1[lo:hi + 1] & np.isfinite(z_r1[lo:hi + 1])
        return pd.Series(keep[:day_hi_a + 1].reshape(-1), index=master.index)

    out_dir = ROOT / "runs" / "scores"
    out_dir.mkdir(exist_ok=True)
    for name in args.models:
        entry = spec["models"][name]
        columns = FILTER20 if name in TS_MODELS else [str(c) for c in data.feature_names]
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

        day_hi_a = int(valid_days[-1]) + 2
        master_a = build_master(data, z_r1, 0, day_hi_a, columns)
        flt = {"train": flt_series(master_a, day_hi_a, *train_span),
               "valid": flt_series(master_a, day_hi_a, *valid_span), "test": None}
        if name in TS_MODELS:
            dataset = BenchDataset(master_a, {"train": train_span, "valid": valid_span, "test": valid_span},
                                   flt, spec["step_len_ts"])
            model.fit(dataset, evals_result=evals_result)
        else:
            dataset = TreeDataset(master_a, {"train": train_span, "valid": valid_span, "test": valid_span},
                                  flt, columns)
            if name == "XGBoost":
                model.fit(dataset, evals_result=evals_result)
            else:
                model.fit(dataset)
        fit_seconds = time.perf_counter() - t0

        lo_b = max(0, int(valid_days[0]) - 25)
        master_b = build_master(data, z_r1, lo_b, int(select_days[-1]), columns)
        if name in TS_MODELS:
            preds = model.predict(BenchDataset(master_b, {"test": select_span}, {"test": None}, spec["step_len_ts"]))
            valid_preds = model.predict(BenchDataset(master_b, {"test": valid_span}, {"test": None}, spec["step_len_ts"]))
        else:
            preds = model.predict(TreeDataset(master_b, {"test": select_span}, {"test": None}, columns), segment="test")
            valid_preds = model.predict(TreeDataset(master_b, {"test": valid_span}, {"test": None}, columns), segment="test")
        matrix = score_matrix(preds, select_dates, list(data.symbols))
        matrix_valid = score_matrix(valid_preds, valid_dates, list(data.symbols))
        tag = f"{name}_seed{seed}"
        effective_seed = 0 if name in DETERMINISTIC else seed
        meta = dict(tag=tag, model=entry["module"] + "." + entry["class"], kwargs=kwargs,
                    feature_set=("filter20" if name in TS_MODELS else "all158"),
                    n_features=len(columns), effective_seed=effective_seed,
                    seed_policy=spec["data_convention"]["seed_policy"],
                    fit_seconds=fit_seconds,
                    evals_result_summary={k: (len(v), (v.index(max(v)) if v else None), (max(v) if v else None))
                                          for k, v in evals_result.items()} if evals_result else None,
                    label="z_r1 (clean 1-day)", descriptive=True)
        with (out_dir / f"{tag}.npz").open("wb") as stream:
            np.savez_compressed(
                stream,
                dates_rule_select=np.array(select_dates), scores_rule_select=matrix,
                candidate_rule_select=candidate_select,
                dates_valid=np.array(valid_dates), scores_valid=matrix_valid,
                candidate_valid=candidate_valid,
            )
        (out_dir / f"{tag}.meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
        n_finite = int(np.isfinite(matrix[candidate_select]).sum())
        print(f"BENCH DONE {name} seed={effective_seed} features={len(columns)} fit_s={fit_seconds:.0f} "
              f"finite_select_scores={n_finite} -> {tag}.npz", flush=True)
        del model, dataset, master_a, master_b
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("BENCH PASS DONE", flush=True)


if __name__ == "__main__":
    main()
