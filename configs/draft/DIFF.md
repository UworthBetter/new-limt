# 预注册目标与上游正式配置的差

左边是 `preregistration_target.json`，右边是 `resolved_upstream_formal.json` 里代码现在真会跑的值。
2026-10-17 对照。两边都还没冻结。

| 项 | 预注册 | 上游合并后的正式配置 |
|---|---|---|
| 种子 | seed 1 先跑，tau 定死后 seeds 2 和 3 原样再跑 | 只有 `[1]` |
| tau | {0.5, 0.75, 1.0}，每臂自己在验证集上选 | 单个 `0.5`，没有网格 |
| 执行规则 | 8 条：n_drop {2,5,10,20} × 间隔 {1,5} 日，topk=50 | `soft`、`topk-reference`、`topk-execution`，drop 固定为 5，没有再平衡间隔 |
| G0 选轮 | 训练账本费率 0；另建真实费率监控账本（5 bp / 15 bp / 最低 5 元 / 100 股） | 训练费率是 0。选轮用连续权重回放，费率是比例佣金，没有最低佣金和整手，也不另记零费率验证值 |
| LightGBM | 回归和 LambdaRank，两个市场 | 编排器里没有这两臂 |
| C-T | seed 1 正式跑；seeds 2–3 有条件 | 与 C0 写在同一次 `seeds` 里，`gamma_to=0.001` 只加在 C-T 上 |
| 网络 | d=256，lookback 21，batch_days 4 | 一致。头数 4 / 2，专家数 4，来自 `make_model` |
| 费用、资金、整手、topk、epoch、patience、Adam、AMP 关 | 与第 6 节一致 | 默认值盖上 `w4w6_formal.json` 后一致 |
| 区间与退市日 | 与第 6.1、6.4 节一致 | `splits_for` 与 `DELISTINGS` 一致 |

差在前四行补上之前，这份上游配置不能当成预注册正式运行。
