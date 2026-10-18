# Gate B 就绪清单（2026-10-19 07:20）

状态：全部臂的配置已冻结、规则已选定；除 CSI500 H 种子 2-3（Kaggle `csi500-h-s23` 在飞，Gate C 素材，不阻塞 Gate B）外无未决项。**gate_b.json 未写**——等用户检查本文件后一次写入。

## 1. 哈希链（先记后跑均已满足）

| 事件 | commit | 时间（+08） |
|---|---|---|
| 基线快照 | `8a7a4c45` | 10-18 16:45:27 |
| 偏差日志五条（benchmark 重定位/印花税/TOST/option (b)/诊断） | `c8c6c27` | 10-18 16:45:28（先于其覆盖的一切新跑） |
| Gate A 填写（[FILL] 全填+§6.5 打勾+14/14 断言） | `841f00be` | 10-18 16:54:08 |
| Gate A 记录 | `8820fa5` | 10-18 16:54:51 |
| runner 批+σ_d(final)+seed-1 网格 | `abe58da` | 10-18 17:14 |
| benchmark 基建+夜间队列 | `6be7ae5` / `4869966` / `bb8174f` / `54245ec` | 10-18 晚 |
| H 打分路径修复 | `080a1c7` | 10-19 03:57 |
| benchmark 两处 bug 修复（Timestamp 索引 / n_jobs=0） | 见 10-19 log | 10-19 04:2x |

## 2. σ_d 与分档（§4.2/§4.3，机械执行）

- σ_d(prelim) = 6.78%（ridge_h5 vs H，上界型代理）
- **σ_d(final) = 1.189%（年化）**（C0−G0 seed1 τ0.5 默认规则执行账本，486 配对日；`checks/sigma_d_final_2026-10-18.json`）
- **MDE80 = 1.581 pp → Tier P1**（≤2.5pp）
- 伴随诊断：持仓重叠 96.12%、秩相关 0.99975、执行层拖累差 C0−G0 = **+0.019 pp/年（C0 略高，与软层符号相反，照录）**

## 3. 三种子训练（8/8 零失败，config hash 全程 `393b1993`）

| 种子 | H | H-FT / G0 / C0 验证窗软层（/天） |
|---|---|---|
| 1 | 11ep 选 ep3 | 0.00108742 / 0.00109438 / 0.00109494 |
| 2 | 11ep 选 ep3 | 0.00106132 / 0.00106691 / 0.00106799 |
| 3 | 11ep 选 ep3 | 0.00108067 / 0.00108432 / 0.00108599 |

跨种子：H 全部 11ep 早停选 ep3；微调全部 20/20 撞上限；软层 C0−G0 三种子全正（+5.6e-7/+1.1e-6/+1.7e-6 每日）。

## 4. 规则选择（2021–2022，种子平均，0.5pp 低换手平局；`runs/csi300_neural_selection.json`）

| 臂 | 选定规则 | 种子平均年化超额（前三见文件） |
|---|---|---|
| **G0** | **drop5-every1d（默认）** | 20.69%（第二名 drop2-every1d 18.28%，无平局） |
| **C0** | **drop5-every1d（默认）** | 20.23%（第二名 19.25%，无平局） |
| H | drop20-every5d | 18.33% |
| H-FT | drop2-every1d | 20.03%（0.2pp 内平局→低换手） |
| G0dose003* | drop5-every1d | exploratory |
| C0dose003* | drop2-every1d | exploratory |

**含义**：两主臂调优规则均收敛到默认规则 → Delta（主估计量）= 默认规则配对差；SC3 与之重合。t+2 断言已入各日志（terminal mark 2022-12-30 ≤ 2022-12-31，未读任何 2023 价格）。

## 5. Benchmark 臂（描述性；`runs/csi300_bench_selection.json`）

六模型，Qlib v0.9.6 官方默认超参逐字钉死（`configs/benchmarks_qlib_defaults_v1.json`，sha256 见 gate_b 草案），clean r1 标签，单跑 seed 1（Transformer 配置 seed 0）：

| 模型 | 训练时长 | 选定规则 |
|---|---|---|
| LSTM | 41 min | drop5-every1d |
| GRU | 41 min | drop10-every5d |
| ALSTM | 35 min | drop2-every5d |
| Transformer | 37 min | drop5-every5d |
| XGBoost | 10 s | drop2-every5d |
| DoubleEnsemble | 75 s | drop2-every1d |

已声明：Qlib 发表数字来自不同窗口与其自家 handler，比较须声明口径；MASTER 无 qlib 内建实现未入列；CSRankNorm 未重加；pyqlib 安装方式（cp312 wheel 换标签+_libs 桩）已披露。

## 6. 剂量诊断（验证窗 5 日 IC + soft vs top-k 2021–2022；`checks/dose_diag_2026-10-18.json` 全量版）

5 日 IC：H 0.0382 / H-FT 0.0556 / G0 0.0549 / C0 0.0548 / G0dose 0.0396 / C0dose 0.0435（三种子版本逐值在文件）。soft−topk：正式臂 −12～−13pp；剂量臂 G0dose −50pp（soft −29.7 vs topk +20.3）。

## 7. CSI500 H（Kaggle，Gate C 素材）

- seed 1：**已验收**（12ep 选 ep4，val_aux 0.9432；三锚全对：data_version `bae7115e`、config hash `393b1993`、trainer sha；checkpoint sha `06cd06e5` 本地复核一致；runs/csi500_H_kaggle/）
- 种子 2-3：`csi500-h-s23` kernel 在飞（~08:30 前后），轮询器自动收货后同流程验收
- 环境：T4×1、torch 2.11+cu128、python 3.13.15（与本地不同源，逐位轨迹不保证——预注册框架允许，运行环境已记录）

## 8. gate_b.json 草案要素（待用户点头后写入）

```json
{
  "config_hashes": {"neural_formal": "393b1993...", "benchmarks": "<sha256(benchmarks_qlib_defaults_v1.json)>", "trainer": "a960a995...", "data_version_csi300": "3c70f663...", "code_commit": "<HEAD at write time>"},
  "tau": 0.5,
  "tier": "P1", "sigma_d_final": 0.011894, "mde80_pp": 1.5811,
  "selected_rules": {"H": "drop20-every5d", "H-FT": "drop2-every1d", "G0": "drop5-every1d", "C0": "drop5-every1d"},
  "benchmark_rules": {"LSTM": "drop5-every1d", "GRU": "drop10-every5d", "ALSTM": "drop2-every5d", "Transformer": "drop5-every5d", "XGBoost": "drop2-every5d", "DoubleEnsemble": "drop2-every1d"},
  "seeds": {"neural": [1,2,3], "benchmarks": [1], "dose": [1]},
  "default_rule": "drop5-every1d (topk50 drop5 daily)",
  "tost": {"margin_pp": 2.0, "ci": "90% stationary bootstrap"},
  "stamp_duty_sensitivity": {"sell_fee": "10bp from 2023-08-28", "primary_unchanged": "sell 15bp throughout", "role": "reported alongside, never decisive"},
  "exploratory_arms": ["G0dose003_seed1", "C0dose003_seed1", "LSTM/GRU/ALSTM/Transformer/XGBoost/DoubleEnsemble (descriptive)"],
  "estimand_note": "both primary arms selected the default rule; Delta = default-rule paired difference; SC3 coincides"
}
```

## 9. 解封后的评价执行（gate_b.json 之后）

各臂评价窗打分（checkpoint 重载 / benchmark 按 8A 已冻结配置）→ 选定规则+默认规则 → 只写 `_sealed` 后统一读取汇总 → §7 机械判定留 Gate C。
