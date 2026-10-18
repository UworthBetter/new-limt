"""预注册 v0.5 第 6.5 节断言。

失败就是不允许开正式训练。上游代码在 D:\\research\\limt-repro。
"""
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
LIMT = ROOT.parent / "limt-repro"
sys.path.insert(0, str(LIMT / "src"))

import v2_w4_data  # noqa: E402
import v2_w4_training  # noqa: E402
from v2_calendar_labels import calendar_labels  # noqa: E402
from v2_w4_training import TASK_WEIGHTS, group_loss, monitoring_mean_log, train  # noqa: E402
from v2_w6_policy import DELISTINGS, splits_for  # noqa: E402
from v2_w6_run import DEFAULT, validate_config  # noqa: E402


def test_h_loss_weights_and_mse_targets_are_zscores():
    assert TASK_WEIGHTS == (0.65, 0.30, 0.05)
    data_source = inspect.getsource(v2_w4_data)
    assert 'np.stack([labels[f"z_{k}"] for k in ("h5", "zv", "zs")]' in data_source
    loss_source = inspect.getsource(v2_w4_training.auxiliary_loss)
    assert "** 2" in loss_source


def test_optimizer_is_adam_and_caps_come_from_config():
    source = inspect.getsource(train)
    assert "torch.optim.Adam" in source
    assert "optim.SGD" not in source
    assert 'config["epochs_h"] if group == "H" else config["epochs_finetune"]' in source
    assert 'config["lr_h"] if group == "H" else config["lr_finetune"]' in source
    assert 'config["patience"]' in source


def test_hft_trains_supervised_loss_only():
    source = inspect.getsource(group_loss)
    assert 'if group in ("H", "H-FT")' in source
    assert "return auxiliary" in source


def test_delisting_dates_and_periods():
    assert DELISTINGS["SH600837"]["date"] == "2025-03-04"
    assert DELISTINGS["SH601989"]["date"] == "2025-09-05"
    splits = splits_for("2026-06-30")
    assert splits["train"] == ("2008-01-01", "2018-12-31")
    assert splits["valid"] == ("2019-01-01", "2020-12-31")
    assert splits["rule_select"] == ("2021-01-01", "2022-12-31")
    assert splits["eval"] == ("2023-01-01", "2026-06-30")


def test_label_formulas_match_section_8a():
    close = np.array(
        [[10.0], [11.0], [12.0], [13.0], [14.0], [15.0], [16.0]], dtype=float
    )
    volume = np.ones_like(close)
    high = close + 1.0
    low = close - 1.0
    labels = calendar_labels(close, volume, high, low)
    assert labels["h5"][0, 0] == pytest.approx(16.0 / 11.0 - 1.0)
    assert labels["r1"][0, 0] == pytest.approx(12.0 / 11.0 - 1.0)


def test_resolved_formal_config_matches_upstream_merge():
    supplied = json.loads((LIMT / "configs" / "w4w6_formal.json").read_text(encoding="utf-8"))
    resolved = validate_config(dict(DEFAULT, **supplied))
    saved = json.loads(
        (ROOT / "configs" / "draft" / "resolved_upstream_formal.json").read_text(encoding="utf-8")
    )
    assert saved["gate_b_frozen"] is False
    assert saved["config"] == resolved
    assert resolved["epochs_h"] == 50
    assert resolved["epochs_finetune"] == 20
    assert resolved["patience"] == 8
    assert resolved["capital"] == 100_000_000.0
    assert resolved["lot"] == 100
    assert resolved["min_fee"] == 5.0
    assert resolved["buy_rate"] == pytest.approx(0.0005)
    assert resolved["sell_rate"] == pytest.approx(0.0015)
    assert resolved["topk"] == 50
    assert resolved["drop"] == 5
    assert resolved["amp"] is False
    assert resolved["dimension"] == 256
    assert resolved["lookback"] == 21


def test_g0_training_branch_uses_zero_fees():
    source = inspect.getsource(train)
    assert '(0., 0.) if group == "G0" else (config["buy_rate"], config["sell_rate"])' in source


def test_a_g0_selection_uses_separate_real_fee_monitoring_ledger():
    """A-G0-1、A-G0-2、A-G0-3。

    选轮必须同时记下零费率验证值和真实费率验证值，并且用带最小佣金和整手的监控账本。
    上游目前没有这两列，选轮走的是连续权重回放。
    """
    source = inspect.getsource(train)
    history_line = source.split("history.append", 1)[1]
    replay = inspect.getsource(monitoring_mean_log)
    missing = []
    if "validation_zero_fee_mean_log" not in history_line:
        missing.append("history has no validation_zero_fee_mean_log")
    if "validation_real_fee_mean_log" not in history_line:
        missing.append("history has no validation_real_fee_mean_log")
    if "metric = validation_real_fee_mean_log" not in source:
        missing.append("selection metric is not the real-fee monitoring mean")
    if "min_fee" not in replay or "lot" not in replay:
        missing.append("monitoring ledger has no min_fee/lot")
    assert not missing, "A-G0-1/2/3 not met: " + "; ".join(missing)


def test_make_model_heads_and_experts():
    source = inspect.getsource(v2_w4_training.make_model)
    assert "LiMT(len(data.feature_names), config[\"dimension\"], 4, 2, 4)" in source
