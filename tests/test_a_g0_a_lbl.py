"""A-G0-1..4 与 A-LBL-1。

A-G0 跑短训练：训练账本费率、监控账本费率、玩具面板上的选轮、G0/C0 配置差。
A-LBL-1 检查 fix1 CSI300 标签文件上的两个已知反例，不打开任何模型输出。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
LIMT_SRC = ROOT.parent / "limt-repro" / "src"
LABELS = (
    ROOT.parent
    / "limt-repro"
    / "results"
    / "v2"
    / "fix1"
    / "v2_w0w2_out_fix1"
    / "data"
    / "v2"
    / "new_v2_labels_csi300.npz"
)

import sys

sys.path.insert(0, str(LIMT_SRC))
import v2_w4_training as training  # noqa: E402
from v2_w3_ledger import execute_close  # noqa: E402
from v2_w4_training import monitoring_mean_log, select_checkpoint_epoch, train  # noqa: E402
from v2_w6_demo import make_demo  # noqa: E402
from v2_w6_run import DEFAULT  # noqa: E402

# 每个 epoch 先记零费率，再记真实费率。真实费率的 argmax 在第 3 轮，零费率在第 2 轮。
TOY_LOGS = (0.02, 0.01, 0.05, -0.02, 0.01, 0.04)


COUNTEREXAMPLES = (
    ("SZ002074", "2018-12-26", 2.03236, "2018-12-28"),
    ("SH600549", "2019-06-26", 1.27604, "2019-06-28"),
)


def test_a_lbl_1_counterexamples_and_h5_endpoint():
    assert LABELS.is_file(), LABELS
    with np.load(LABELS, allow_pickle=False) as labels:
        dates = pd.to_datetime(labels["dates"])
        symbols = {str(s): i for i, s in enumerate(labels["symbols"].astype(str))}
        close = labels["close"]
        for symbol, day, expected_pct, numerator in COUNTEREXAMPLES:
            assert symbol in symbols, symbol
            t = dates.get_loc(pd.Timestamp(day))
            s = symbols[symbol]
            realised = float(labels["raw_r1"][t, s]) * 100
            exit_r1 = int(labels["exit_idx_r1"][t])
            assert abs(realised - expected_pct) < 1e-3
            assert str(dates[exit_r1].date()) == numerator
            assert int(labels["entry_idx"][t]) == t + 1
            exit_h5 = int(labels["exit_idx_h5"][t])
            assert exit_h5 == t + 6
            entry = float(close[t + 1, s])
            horizon = float(close[t + 6, s])
            assert np.isfinite(entry) and np.isfinite(horizon) and entry != 0
            assert float(labels["raw_h5"][t, s]) == pytest.approx(horizon / entry - 1.0)


def _config(tmp):
    return dict(DEFAULT, dimension=16, max_symbols=8, train_days=4, valid_days=4, report_days=3,
                batch_days=2, threads=2, epochs_h=1, epochs_finetune=3, patience=0)


def _run_group(data, config, group, out, initial, train_fees, monitor_calls, logs):
    cursor = {"n": 0}

    def wrapped_step(previous, relatives, target=None, buy=0.0005, sell=0.0015):
        train_fees.append((float(buy), float(sell)))
        return train_step(previous, relatives, target, buy, sell)

    def scripted_monitor(data, days, scores, tau, execution_feed, buy, sell, min_fee, lot, capital):
        monitor_calls.append((float(buy), float(sell), float(min_fee), int(lot)))
        value = logs[cursor["n"]]
        cursor["n"] += 1
        return value

    training.continuous_step = wrapped_step
    training.monitoring_mean_log = scripted_monitor
    try:
        return train(data, config, group, 1, out, "cpu", "toy", initial)
    finally:
        training.continuous_step = train_step
        training.monitoring_mean_log = monitoring_mean_log


train_step = training.continuous_step


@pytest.fixture(scope="module")
def toy_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ag0")
    config = _config(tmp)
    data = make_demo(tmp / "input", tmp / "cache", config)
    h_dir = tmp / "H"
    h_dir.mkdir()
    train(data, config, "H", 1, h_dir, "cpu", "toy")
    g0_fees, c0_fees, g0_monitor, c0_monitor = [], [], [], []
    g0_dir, c0_dir = tmp / "G0", tmp / "C0"
    g0_dir.mkdir()
    c0_dir.mkdir()
    initial = h_dir / "checkpoint.pt"
    _run_group(data, config, "G0", g0_dir, initial, g0_fees, g0_monitor, TOY_LOGS)
    _run_group(data, config, "C0", c0_dir, initial, c0_fees, c0_monitor, TOY_LOGS)
    return dict(config=config, data=data, g0_fees=g0_fees, c0_fees=c0_fees,
                g0_monitor=g0_monitor, c0_monitor=c0_monitor, g0_dir=g0_dir, c0_dir=c0_dir)


def test_a_g0_1_training_zero_fee_and_separate_monitoring_ledger(toy_run):
    assert toy_run["g0_fees"] and set(toy_run["g0_fees"]) == {(0.0, 0.0)}
    days = toy_run["data"].days("valid")
    scores = np.zeros((len(days), len(toy_run["data"].symbols)))
    seen = []
    real_close = execute_close

    def spy(account, quotes, states, target, signal_date, execution_date, buy=0.0005, sell=0.0015, min_fee=5.0, lot=100):
        seen.append((float(buy), float(sell), float(min_fee), int(lot)))
        return real_close(account, quotes, states, target, signal_date, execution_date, buy, sell, min_fee, lot)

    training.execute_close = spy
    try:
        monitoring_mean_log(toy_run["data"], days, scores, 0.5, None, 0.0, 0.0, 0.0, 100, toy_run["config"]["capital"])
        monitoring_mean_log(toy_run["data"], days, scores, 0.5, None, 0.0005, 0.0015, 5.0, 100, toy_run["config"]["capital"])
    finally:
        training.execute_close = real_close
    assert seen
    assert set(seen[: len(seen) // 2]) == {(0.0, 0.0, 0.0, 100)}
    assert set(seen[len(seen) // 2 :]) == {(0.0005, 0.0015, 5.0, 100)}
    assert toy_run["g0_monitor"][0] == (0.0, 0.0, 0.0, 100)
    assert toy_run["g0_monitor"][1] == (0.0005, 0.0015, 5.0, 100)


def test_a_g0_2_g0_and_c0_share_real_fee_selection_key(toy_run):
    g0 = json.loads((toy_run["g0_dir"] / "history.json").read_text(encoding="utf-8"))
    c0 = json.loads((toy_run["c0_dir"] / "history.json").read_text(encoding="utf-8"))
    g0_ckpt = training.torch.load(toy_run["g0_dir"] / "checkpoint.pt", map_location="cpu", weights_only=False)
    c0_ckpt = training.torch.load(toy_run["c0_dir"] / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert g0_ckpt["selection"] == c0_ckpt["selection"] == "valid_common_net_mean_log_max"
    assert [row["selection_value"] for row in g0] == [row["validation_real_fee_mean_log"] for row in g0]
    assert [row["selection_value"] for row in c0] == [row["validation_real_fee_mean_log"] for row in c0]
    assert [call[1] for call in toy_run["g0_monitor"]] == [call[1] for call in toy_run["c0_monitor"]]
    real_calls = [call for call in toy_run["g0_monitor"] if call[0] != 0.0]
    assert real_calls and set(real_calls) == {(0.0005, 0.0015, 5.0, 100)}


def test_a_g0_3_toy_panel_selects_real_fee_argmax_not_zero_fee(toy_run):
    history = json.loads((toy_run["g0_dir"] / "history.json").read_text(encoding="utf-8"))
    zero = [row["validation_zero_fee_mean_log"] for row in history]
    real = [row["validation_real_fee_mean_log"] for row in history]
    assert zero == [0.02, 0.05, 0.01]
    assert real == [0.01, -0.02, 0.04]
    assert int(np.argmax(zero)) != int(np.argmax(real))
    selected = select_checkpoint_epoch(real, zero)
    checkpoint = training.torch.load(toy_run["g0_dir"] / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert selected == int(np.argmax(real)) == 2
    assert checkpoint["validation_selection"] == pytest.approx(real[selected])
    assert checkpoint["validation_selection"] != pytest.approx(max(zero))


def test_a_g0_4_resolved_configs_differ_only_by_training_fee(toy_run):
    import v2_w4_training as training

    assert hasattr(training, "resolved_arm_config"), (
        "A-G0-4 not met: train() does not emit resolved_arm_config; "
        "G0 and C0 still share one config dict and hide the fee split in a branch"
    )
    g0 = training.resolved_arm_config("G0")
    c0 = training.resolved_arm_config("C0")
    assert set(g0) == set(c0)
    differ = [key for key in g0 if g0[key] != c0[key]]
    assert differ == ["training_fee"]
    assert g0["training_fee"] == {"buy": 0.0, "sell": 0.0, "min_fee": 0.0, "lot": 100}
    assert c0["training_fee"] == {"buy": 0.0005, "sell": 0.0015, "min_fee": 5.0, "lot": 100}
    for key in ("epochs", "patience", "lr", "optimizer", "seeds", "alpha_aux", "tau_grid"):
        assert g0[key] == c0[key]
    assert set(toy_run["g0_fees"]) == {(0.0, 0.0)}
    assert set(toy_run["c0_fees"]) == {(toy_run["config"]["buy_rate"], toy_run["config"]["sell_rate"])}
