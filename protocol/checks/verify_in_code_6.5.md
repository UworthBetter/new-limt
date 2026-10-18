# 第 6.5 节代码核对

日期：2026-10-17。对照对象是 `D:\research\limt-repro`，不是本目录里的副本。
预注册正文里的勾选框没有改。下面的「通过」只表示这一条和当前上游代码一致。

断言：`tests/test_preregistration_assertions.py`
当次输出：`pytest_6.5_2026-10-17.txt`
解释器：`D:\research\limt-repro\.venv\Scripts\python.exe`
结果：8 通过，1 失败。失败的是 A-G0 选轮账本，这是预期的，不要改成跳过。

这些断言多数是对源码和配置合并结果的检查，不是把正式训练集跑了一遍。

## 清单

| 预注册条目 | 结果 | 依据 |
|---|---|---|
| H 损失 0.65 / 0.30 / 0.05，目标是截面 z-score 上的 MSE | 通过 | `v2_w4_training.TASK_WEIGHTS`；`v2_w4_data.py` 取 `z_h5`、`z_zv`、`z_zs`；`auxiliary_loss` 是平方误差 |
| 正式训练优化器是 Adam | 通过 | `v2_w4_training.train` 调用 `torch.optim.Adam`。旧脚本 `train.py`、`train_new.py` 不在这条干净流水线里，这次没再审 |
| G0 与 C0 同一条训练路径，G0 训练费率为 0 | 通过 | 同一个 `train()`。G0 的 `cb, cs` 是 `(0., 0.)`，C0 用配置里的买卖费率。`group_loss` 对两者都是负的平均 log growth 加 `alpha_aux` 乘辅助损失 |
| G0 与 C0 同一选轮指标、同一 epoch 上限、同一 patience | 部分通过 | 两者都走 `portfolio_events`，选轮字符串都是 `valid_common_net_mean_log_max`。epoch、patience、学习率按「是不是 H」分支，G0 和 C0 落在同一边。选轮账本不是预注册写的那本，见下节 |
| H-FT 只继续监督损失，epoch 上限与其他微调相同 | 通过 | `group_loss` 对 H 和 H-FT 直接返回辅助损失。epoch 用 `epochs_finetune`。选轮用验证集 after-cost log growth，与第 6.1 节一致，不是 H 那种监督损失最小 |
| epoch 50 / 20，patience 8 | 通过 | `configs/w4w6_formal.json` 盖上 `v2_w6_run.DEFAULT` 之后。patience 来自 DEFAULT，不在那份 JSON 里 |
| 资金 1e8，整手 100，买 5 bp，卖 15 bp，最低佣金 5 元 | 配置通过 | 合并后的正式配置里有这些数。选轮回放没有用最低佣金和整手 |
| topk 50，drop 5 | 配置通过 | 合并配置里有。预注册的 8 条执行网格不在这份配置里 |
| 退市日 | 通过 | `SH600837` 为 2025-03-04，`SH601989` 为 2025-09-05，见 `v2_w6_policy.DELISTINGS` |
| 头数与专家数 | 已记下 | `make_model` 调用 `LiMT(..., 4, 2, 4)`，即跨股票 4 头、时间 2 头、4 个专家 |

区间与标签公式也核对过。`splits_for("2026-06-30")` 是训练 2008–2018、验证 2019–2020、规则选择 2021–2022、评价 2023-01-01 至 2026-06-30。
`calendar_labels` 里 h5 是 `close(t+6)/close(t+1)-1`，r1 是 `close(t+2)/close(t+1)-1`。玩具数组上的断言通过。

## 必须先响的断言

### A-G0-1 未通过

训练侧：G0 费率是 0，这一点有。
监控侧：没有另一本账。`train` 在验证时调用 `portfolio_events(..., buy_rate, sell_rate)`。
`portfolio_events` 自己写明 Topk 是连续权重参照，不声称整手或最低佣金成交。
最低佣金和 100 股整手在 `v2_w3_ledger` 的份额账本里，评估时由 `v2_w5_results.actual_execution` 使用，不参与选 checkpoint。
零费率验证值也没有记在 `history.json` 旁边。

### A-G0-2 未通过

G0 和 C0 用的是同一个函数、同一对比例费率。
预注册要求的是同一套真实费率监控账本（含最低 5 元和 100 股）。当前选轮没有这本账，所以不能打勾。

### A-G0-3 未做

预注册要求一个玩具验证面板：零费率 argmax 和真实费率 argmax 不同，被选中的 epoch 必须是后者。
上游没有这个测试。现在的代码根本不算零费率验证值，所以也无从证明它会在两者冲突时选对。

### A-G0-4 部分通过

G0 和 C0 共享 epoch 上限、patience、学习率、优化器、种子列表、`alpha_aux` 和当前这个单一 tau。
运行时真正分开的是训练费率那一对数。
代码没有打出「解析后的配置只差一个字段」这种 diff。`gamma_to` 写在共享配置里，`group_loss` 只在 C-T 上使用它。

### A-LBL-1 未完成

公式与第 8A 节一致，玩具断言通过。
「两个已知反例上端点错位为 0」没有做：预注册没有写出这两个反例是哪两只、哪两天。
上游 `limt-repro/results/v2/fix1/.../new_W0W2_acceptance.md` 里整表 `total_misaligned` 为 0。那是旧验收记录，不是本目录复跑，也不能代替那两个点名反例。

## 还不能开正式训练的原因

1. A-G0-1 至 A-G0-3 没有过。
2. 上游正式配置只有 tau=0.5、种子 `[1]`、三条规则名，不是预注册的 tau 网格和 8 条执行规则。差表在 `configs/draft/DIFF.md`。
3. LightGBM 两臂不在 `v2_w6_run` 的模型列表里。
4. sigma_d(prelim) 还没测。Gate B 之前也不打开 2023–2026 的新模型输出。
