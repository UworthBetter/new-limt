# new_LiMT 干净流水线实验

本目录只服务预注册 `PREREGISTRATION_v0.5.md`（Cost-Aware Training for LiMT，clean pipeline）。
上游实现在 `D:\research\limt-repro`，不要整棵拷进这里。

Gate B 之前，不要打开 2023-01 至 2026-06-30 的新流水线模型输出。
2026-10-15 之前带标签日历错误的结果不能当证据。

## 文件放哪里

| 目录 | 放什么 | 不要放什么 |
|---|---|---|
| 根目录 | 本说明、预注册锁定稿 | 日志、checkpoint、临时表 |
| `protocol/` | 偏差日志、未决项、Gate 板、代码核对 | 训练产物 |
| `configs/draft/` | 从预注册和上游抄下的草稿配置 | 把草稿当成 Gate B 已冻结 |
| `src/` | 本轮必须新写的代码；目前只有上游对照 | 再复制一份 limt-repro |
| `tests/` | 预注册 6.5 节要求的断言，失败必须响 | 探索性扫描脚本 |
| `runs/` | 原始输出，键为 market / backbone / label / rule / seed | 论文用汇总表 |
| `results/` | 从 `runs/` 汇总出的表 | checkpoint、npz 原件 |
| `logs/` | 一次性命令日志 | 正式核对结论（结论放 `protocol/checks/`） |
| `prior_art/` | 第 16 节待核清单和阅读笔记 | 根目录堆 PDF |
| `paper/` | Gate D 之后的正文 | 现在不要在这里开写 |

## 当前进度

代码核对写在 `protocol/checks/verify_in_code_6.5.md`。
断言在 `tests/test_a_g0_a_lbl.py`（行为化重跑 5/5 通过，2026-10-17）。
预注册现行版本：`PREREGISTRATION_v0.6.md`（v0.5 保留不动）。
seed-1 已落盘：H（10-17）、G0/C0 tau 网格 6 跑（10-18）。两臂选 tau=0.5、全部 ep20；核验与决定见 `protocol/tau_grid_report_2026-10-18.md`。
2026-10-18 发射（链式，日志 `logs/hft_dose_score_20261018.log`）：H-FT 种子 1 → 探索性剂量臂（alpha_aux=0.03，deviation log 已先记）→ 全量 3τ 补打分。
正式配置还没冻结，Gate A / Gate B 都没过；2023+ 输出仍封存。
