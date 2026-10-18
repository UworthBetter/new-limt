# 原始运行输出

一个单元一个目录，键固定为：

```
runs/{market}/{backbone}/{label}/{rule}/seed{n}/
```

列定义和 `Delta_cost` 的符号以 `protocol/structural_decisions.md` 的 S6、S10 为准。

允许的取值：

| 键 | 取值 |
|---|---|
| market | `csi300`, `csi500` |
| backbone | `ridge`, `lgbm-reg`, `lgbm-rank`, `H`, `H-FT`, `G0`, `C0`, `C-T` |
| label | `h5`, `r1` |
| rule | `default`, 或 `drop{n}-every{k}d`（n 为 2/5/10/20，k 为 1 或 5） |
| seed | `seed1`, `seed2`, `seed3`；ridge 只有 `seed1` |

每个单元里只放该次运行自己的文件：`config.json`、`history.json`、`daily_returns.csv`、`metrics.json`、日志。
checkpoint 可以留在单元内，不要放到 `results/`。

没有跑过的单元不要预建空目录。
2023-01 至 2026-06-30 的模型输出，在 Gate B 哈希写进 `protocol/gates.md` 之前不要往这里放，也不要打开上游已经生成的那一批。

验证窗（2019–2020）和规则选择窗（2021–2022）可以先落盘，但规则选择窗的网格不能写进结果表。
