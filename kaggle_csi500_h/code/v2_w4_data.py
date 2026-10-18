"""Verified v2 data contract, disk-backed features, causal windows and split domains."""
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
from v2_w6_policy import splits_for, listing_eligible, signal_days, DEFAULT_EVALUATION_END, LEGACY_EVALUATION_END, DELISTINGS

SPLITS = splits_for()


def past_marks(close):
    """Last observed positive close, strictly forward fill; no future backfill."""
    close = np.asarray(close, np.float64)
    observed = np.isfinite(close) & (close > 0)
    index = np.where(observed, np.arange(len(close))[:, None], -1)
    np.maximum.accumulate(index, axis=0, out=index)
    marks = close[np.maximum(index, 0), np.arange(close.shape[1])[None, :]]
    marks[index < 0] = np.nan
    return marks, ~observed


class Data:
    def __init__(self, root, cache, lookback, max_symbols=0, evaluation_end=DEFAULT_EVALUATION_END):
        self.splits=splits_for(evaluation_end)
        root, cache = Path(root), Path(cache)
        directory = root / "data/v2"
        cache.mkdir(parents=True, exist_ok=False)
        panel = directory / "new_v2_panel_csi300.npz"
        # Extract only the known member, without materialising the multi-GB tensor.
        with zipfile.ZipFile(panel) as archive, archive.open("feat_mat.npy") as source:
            with (cache / "features.npy").open("xb") as destination:
                shutil.copyfileobj(source, destination, length=4 * 1024 * 1024)
        self.features = np.load(cache / "features.npy", mmap_mode="r", allow_pickle=False)
        with np.load(panel, allow_pickle=False) as p, np.load(directory / "new_v2_labels_csi300.npz", allow_pickle=False) as labels:
            self.dates = labels["dates"].astype("datetime64[D]")
            self.symbols = labels["symbols"].astype(str)
            if not np.array_equal(p["dates"].astype("datetime64[D]"), self.dates) or not np.array_equal(p["symbols"], self.symbols):
                raise ValueError("panel/labels dates or symbols differ")
            self.feature_mask, self.universe = p["feature_mask"], p["universe_mask"]
            self.close = labels["close"].astype(np.float64)
            self.targets = np.stack([labels[f"z_{k}"] for k in ("h5", "zv", "zs")], axis=-1)
            self.label_mask = labels["label_mask_h5"] & labels["label_mask_multi"]
            self.feature_names = p["feat_cols"].astype(str)
        with np.load(directory / "new_v2_splits_csi300.npz", allow_pickle=False) as splits:
            if not np.array_equal(splits["dates"].astype("datetime64[D]"), self.dates) or not np.array_equal(splits["symbols"], self.symbols):
                raise ValueError("split alignment mismatch")
            self.loss_masks = {name: splits[f"loss_mask_{name}"] for name in SPLITS}
            source_end=str(splits['evaluation_end'].item()) if 'evaluation_end' in splits else LEGACY_EVALUATION_END
        self.lookback = int(lookback)
        if self.lookback < 1 or np.isnat(self.dates).any() or not (np.diff(self.dates) > np.timedelta64(0, "D")).all():
            raise ValueError("invalid lookback/calendar")
        expected = (len(self.dates), len(self.symbols))
        if self.features.shape != (*expected, len(self.feature_names)) or self.features.dtype != np.float32:
            raise ValueError("feature tensor shape/dtype mismatch")
        if any(x.shape != expected or x.dtype != bool for x in [self.feature_mask, self.universe, self.label_mask, *self.loss_masks.values()]):
            raise ValueError("mask contract violated")
        if self.close.shape != expected or self.targets.shape != (*expected, 3):
            raise ValueError("label shape mismatch")
        # Validate the immutable upstream split, then narrow the run's loss domain.
        # A new run may shorten an old verified split; it cannot invent new labels.
        for name, (_, end) in splits_for(source_end).items():
            mask = self.loss_masks[name]
            if (mask & ~self.label_mask).any():
                raise ValueError("loss split exceeds label availability")
            days = np.where(mask.any(1))[0]
            if len(days) and (days[-1] + 6 >= len(self.dates) or self.dates[days[-1] + 6] > np.datetime64(end)):
                raise ValueError("loss label endpoint crosses split end")
        self.raw_universe=self.universe.copy()
        self.listing_eligible=listing_eligible(self.dates,self.symbols)
        self.universe=self.universe & self.listing_eligible
        index=np.arange(len(self.dates));endpoint=index+6
        for name,(start,end) in self.splits.items():
            keep=(self.dates>=np.datetime64(start)) & (self.dates<=np.datetime64(end)) & (endpoint<len(index))
            keep &= self.dates[np.minimum(endpoint,len(index)-1)]<=np.datetime64(end)
            self.loss_masks[name]=self.loss_masks[name] & keep[:,None] & self.listing_eligible
        self.indices = np.arange(len(self.symbols))
        if max_symbols:
            # Small engineering smoke subset: choose from first TRAIN signal only.
            first = np.where((self.dates >= np.datetime64(SPLITS["train"][0])) & (np.arange(len(self.dates)) >= self.lookback - 1))[0][0]
            chosen = np.where(self.candidate(first))[0][:max_symbols]
            if len(chosen) < 2:
                raise ValueError("too few history-visible smoke symbols")
            self.indices = chosen
            self.symbols = self.symbols[chosen]
            self.feature_mask = self.feature_mask[:, chosen]; self.universe = self.universe[:, chosen]
            self.raw_universe=self.raw_universe[:,chosen];self.listing_eligible=self.listing_eligible[:,chosen]
            self.close = self.close[:, chosen]; self.targets = self.targets[:, chosen]
            self.label_mask = self.label_mask[:, chosen]
            self.loss_masks = {k: v[:, chosen] for k, v in self.loss_masks.items()}
        self.marks, self.stale = past_marks(self.close)

    def candidate(self, t):
        return self.feature_mask[t] & self.universe[t]

    def windows(self, days):
        days = np.asarray(days, int)
        if len(days) == 0 or (days < self.lookback - 1).any():
            raise ValueError("empty days or insufficient causal context")
        array = np.stack([self.features[t - self.lookback + 1:t + 1, self.indices, :].transpose(1, 0, 2) for t in days])
        if not np.isfinite(array).all():
            raise ValueError("nonfinite input features")
        return array.astype(np.float32, copy=False)

    def days(self, split, training=False, limit=0):
        return signal_days(self.dates,split,self.splits['eval'][1],self.lookback,training,limit)

    def relatives(self, mark_day, previous_day):
        with np.errstate(invalid="ignore", divide="ignore"):
            y = self.marks[mark_day] / self.marks[previous_day]
        return np.r_[1., y]


def write_contract(data, path):
    Path(path).write_text(json.dumps(dict(dates=len(data.dates), symbols=data.symbols.tolist(),
        features=len(data.feature_names), lookback=data.lookback,
        splits=data.splits,delisting_rules=DELISTINGS,
        excluded_delisted_member_cells=int((data.raw_universe & ~data.listing_eligible).sum()),
        input_mask="feature_mask only; candidate=feature_mask & source universe_mask & dated listing eligibility",
        valuation="last positive observed adjusted close; no backfill",
        execution_layer="continuous full-fill reference at valuation marks; stale fills are a proxy assumption"), indent=2), encoding="utf-8")
