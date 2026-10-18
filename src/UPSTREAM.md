# 上游代码对照

本目录不复制实现。下面这些文件都在 `D:\research\limt-repro\src`。

| 预注册对象 | 文件 | 函数或常量 |
|---|---|---|
| H / H-FT / G0 / C0 / C-T 训练 | `v2_w4_training.py` | `train`, `group_loss`, `auxiliary_loss` |
| 网络宽度、头数、专家数 | `model.py`, `v2_w4_training.py` | `LiMT.__init__`, `make_model` |
| 软组合目标 | `v2_objective.py` | `objective`, `TASK_LAM` |
| 连续权重账本 | `v2_w3_ledger.py` | `continuous_step` |
| 含最小佣金和整手的成交 | `v2_w3_ledger.py`, `v2_w5_results.py` | 份额账本, `actual_execution` |
| 正式运行入口 | `v2_w6_run.py` | `DEFAULT`, `validate_config`, `run_pipeline` |
| 标签定义 | `v2_calendar_labels.py` | `calendar_labels`, `cs_z` |
| 退市日 | `v2_w6_policy.py` | `DELISTINGS` |
| 监督目标从哪一列来 | `v2_w4_data.py` | `z_h5`, `z_zv`, `z_zs` |

正式配置草稿的合并规则：`validate_config(dict(DEFAULT, **w4w6_formal.json))`。
未知键会被拒绝，所以预注册里的 tau 网格和 8 条执行规则不能塞进这份 JSON。
