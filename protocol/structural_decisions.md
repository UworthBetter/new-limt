# 结构决定（先于继续写代码）

日期：2026-10-17。对象是预注册第 17.1 节里仍开着的 S5、S6、S7、S9，以及本页新编号的 S10。
S1 仍是未通过的实现缺口（见 `checks/verify_in_code_6.5.md`），定义不再改。
S2、S3、S4、S8 维持 v0.5 已有决定。
这些决定只约束以后要写的代码。本页不改 `limt-repro` 里的训练器。

## S5 H-FT：已定，不再加第二套开关

模式就是 `group="H-FT"`。不新增 `finetune_mode` 之类的第二个旗标。

训练损失只有监督项：`group_loss` 对 H 和 H-FT 返回 `auxiliary_loss`。权重 0.65 / 0.30 / 0.05，目标是截面 z-score 上的平方误差。
初始化必须是同一 seed 的干净 H checkpoint。优化器 Adam，学习率 `lr_finetune`（1e-5），上限 `epochs_finetune`（20），patience 8。
选 checkpoint 与 G0、C0 相同，必须是真实费率监控账本上的 after-cost mean log growth。当前的连续权重回放不算数。
A-G0 断言没通过之前，不启动正式 H-FT。种子 2 和 3 仍按第 8 节有条件才跑。

## S6 输出键与日频列：已定

目录键：

```
runs/{market}/{backbone}/{label}/{rule}/seed{n}/daily.csv
```

`market` 是 `csi300` 或 `csi500`。`label` 是 `h5` 或 `r1`。`rule` 见 S10。`seed` 是 `seed1`、`seed2`、`seed3`。

`daily.csv` 每个会计日一行，列序固定：

| 列 | 含义 |
|---|---|
| date | 会计日 |
| signal_date | 产生目标权重的信号日 |
| gross_simple_return | 费前简单收益，`gross - 1`。`gross` 是 `continuous_step` 里的财富相对数，费还没扣 |
| fee_fraction | 这一日费用占费前财富的比例 |
| net_simple_return | 费后简单收益 |
| net_log_growth | 费后对数增长 |
| turnover | 买入名义金额加卖出名义金额，占再平衡前财富 |
| fee_value | 这一日费用的货币金额 |
| benchmark_return | 同一日 H00300 全收益。还没对齐时该格留空，不能填 0 |

身份：`net_simple_return = gross_simple_return - fee_fraction` 只在费用按费前财富计提、且没有最低佣金的连续账本上成立。带最低佣金和整手的份额账本以 `fee_value / 费前财富` 重算 `fee_fraction`，不沿用比例佣金公式硬套。

SC6 的成本拖累是费用占初始资金的比例，按日 `fee_value` 求和再年化。
`Delta_cost = drag(G0*) - drag(C0*)`。C0 更省费用时这项为正。
`Delta_gross` 用费前超额。`Delta = Delta_gross + Delta_cost` 作为会计恒等式报告，不另造第三个收益定义。

规则选择窗 2021–2022 的网格可以落在 `runs/` 里，但不得写入 `results/`。

## S7 门闩：已定，代码还没写

评价期是 2023-01-01 至 2026-06-30。GPU 和 CPU、全部 backbone 都算新流水线输出。

- 写日频文件的函数默认 `eval_open=False`。
- 任一会计日 `>= 2023-01-01` 且 `eval_open` 为假：在创建文件之前抛出 `GateLocked`，不写半截文件。
- `protocol/gate_b.json` 还不存在。它以后必须含 `status=frozen` 和冻结配置的 sha256。读者 `open_daily()` 在该文件缺失或哈希不符时拒绝打开 `date` 落入评价期的文件。
- sigma_d 只读验证窗 2019-01-01 至 2020-12-31。这个读者拒绝评价期路径，也拒绝规则选择窗。
- 「封存」指文件可以在 Gate B 之前生成到 `runs/_sealed/`，但本目录的读取函数打不开。现在 A-G0 未通过，正式训练还不该生成这批文件。

## S9 CSI500：已定通过条件，重跑用现有 W1/W2 脚本

`Data` 和 `verify_fix1` 现在把文件名写死成 `csi300`。以后的签名是：

```text
Data(root, cache, lookback, max_symbols=0, evaluation_end="2026-06-30", market="csi300")
```

文件名是 `new_v2_labels_{market}.npz`、`new_v2_panel_{market}.npz`、`new_v2_splits_{market}.npz`。
`market` 只允许 `csi300` 和 `csi500`。CSI500 不另做调参。

通过条件，与第 9 节一致，不是把 CSI300 的两只股票检查原样当成 CSI500 的失败条件：

- `0_array_contract` 通过
- `2_endpoint_misalignment` 的 `total_misaligned` 和 `standardized_misaligned` 都是 0
- `4_purge` 通过，且 `--evaluation-end 2026-06-30`
- 没有越界标签

`1_SZ002074_2018-12-26` 和 `1_SH600549_2019-06-26` 是 CSI300 标签文件上的两个已知反例，归 A-LBL-1。
这两只如果不在 CSI500 成分跨度里，W2 会因为「反例不在宇宙里」把检查标成 FAIL。那种失败只说明门脚本把 CSI300 反例写死了，不把 CSI500 端点检查判失败，也不要为此去删检查。端点检查本身必须仍是 0。

数据目录用 `D:\research\limt-repro\data_new\cn_data`（日历到 2026-10-13）。
不要用 `data\qlib_cn_data`：那份日历停在 2020-09-25，CSI500 成分停在 2022-03-21。

输出只进 `data/csi500_w1w2/`。不覆盖 fix1 的 CSI300 产物。
本地 fix1 只有 `new_v2_labels_csi300.npz`（117 MB）。正式 panel 不在这台机器上。

N6（CSI500 每 epoch 秒数）要等这份 panel 出来之后，用全股票池、1 个 epoch 来测。32 只股票的 smoke 不能填 N6。

## S10 标签轴和八条执行规则：本页编号

v0.5 的第 17.1 节没有 S10。2026-10-17 补上，因为第 5 节和第 8A 节已经要求这两维，而 `v2_w6_run.DEFAULT["rules"]` 只有 `soft`、`topk-reference`、`topk-execution`，`drop` 固定为 5，没有再平衡间隔，监督头也写死成 h5。
不先定这个接口，CPU 基线会按错误的三层规则名去写。

`label` 是 `h5` 或 `r1`。定义仍是第 8A 节：h5 = `close(t+6)/close(t+1)-1`，r1 = `close(t+2)/close(t+1)-1`，再做截面 z-score。
神经网络臂在 v0.5 里不做 1 日重训（D2 = No），所以 H、H-FT、G0、C0、C-T 的 `label` 只有 `h5`。`r1` 只给 ridge 和两套 LightGBM。

八条规则，topk 固定 50：

| rule id | n_drop | 再平衡间隔（交易日） |
|---|---|---|
| drop2-every1d | 2 | 1 |
| drop5-every1d | 5 | 1 |
| drop10-every1d | 10 | 1 |
| drop20-every1d | 20 | 1 |
| drop2-every5d | 2 | 5 |
| drop5-every5d | 5 | 5 |
| drop10-every5d | 10 | 5 |
| drop20-every5d | 20 | 5 |

`default` 是 `drop5-every1d` 的别名，不是第九条规则。
间隔大于 1 时，非再平衡日目标权重是 HOLD（沿用 `signal_target` 返回 None 的语义），不是每天都换仓。
间隔必须进执行函数的参数。不能靠把日收益抽样成每 5 日一行来假装。

软组合层（tau 网格）不是这八条之一。它仍叫 `soft`，只用于 SC1 和 SC5，不参与第 5 节的规则选择。

## 现在仍不准写的代码

正式训练器、八条规则的执行实现、CPU 基线作业、Gate B 之后才打开的评价读取。
A-G0-1 至 A-G0-3 的断言还是失败的。按这个失败去改选轮，才是下一处允许动的训练代码。
