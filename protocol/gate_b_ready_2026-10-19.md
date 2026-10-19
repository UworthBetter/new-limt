# Gate B 就绪清单（修订版，2026-10-19 第二轮）

状态：全部臂配置冻结；种子扩展轮（benchmark 随机臂 seeds 2–3）按 DL 2026-10-19 条目在跑。**gate_b.json 未写**——等终审。

## 1. 哈希链（先记后跑均已满足）

| 事件 | commit | 时间（+08） |
|---|---|---|
| 基线快照 | `8a7a4c45` | 10-18 16:45:27 |
| 偏差日志五条 | `c8c6c27` | 10-18 16:45:28（先于其覆盖的一切新跑） |
| Gate A 填写 | `841f00be` | 10-18 16:54:08 |
| Gate A 记录 | `8820fa5` | 10-18 16:54:51 |
| 工程修复①H 打分路径（含时序披露） | `080a1c7` | 10-19 03:57:35 |
| 工程修复②benchmark Timestamp/n_jobs | `48a78d6` | 10-19 04:26:24 |
| CSI500 H 验收 + 就绪文档 v1 | `2e33391` | 10-19 |
| benchmark 取证 + DL 种子扩展条目（先于 seeds 2–3 一切运行） | `9a578bb` | 10-19 09:50:27 |

工程修复一律"engineering fix, no protocol change"，详见 `protocol/engineering_log.md`（含 H 分数文件早于 commit 22–47 秒的时序披露与证据链）。

## 2. σ_d 与分档（§4.2/§4.3，机械执行）

- σ_d(prelim) = 6.78%（ridge_h5 vs H，上界型代理）
- **σ_d(final) = 1.189%（年化）→ MDE80 = 1.581 pp → Tier P1**（`checks/sigma_d_final_2026-10-18.json`）
- 三诊断：持仓重叠 96.12%、秩相关 0.99975、执行层拖累差 C0−G0 = +0.019 pp/年（与软层符号相反，照录）

## 3. 三种子训练（8/8 零失败，config hash `393b1993`）

| 种子 | H | H-FT / G0 / C0 验证窗软层（/天） |
|---|---|---|
| 1 | 11ep 选 ep3 | 0.00108742 / 0.00109438 / 0.00109494 |
| 2 | 11ep 选 ep3 | 0.00106132 / 0.00106691 / 0.00106799 |
| 3 | 11ep 选 ep3 | 0.00108067 / 0.00108432 / 0.00108599 |

软层 C0−G0 三种子全正（+5.6e-7 / +1.1e-6 / +1.7e-6 每日，≈0.03–0.04 pp/年）。

## 4. 神经臂规则选择（2021–2022 种子平均；`runs/csi300_neural_selection.json`）

| 臂 | 选定规则 | 三种子平均年化超额 |
|---|---|---|
| **G0** | **drop5-every1d（默认）** | 20.69%（第二名差 2.4pp，无平局） |
| **C0** | **drop5-every1d（默认）** | 20.23%（第二名差 0.98pp，无平局） |
| H | drop20-every5d | 18.33% |
| H-FT | drop2-every1d | 20.03%（0.2pp 平局→低换手） |
| G0dose003* | drop5-every1d | exploratory |
| C0dose003* | drop2-every1d | exploratory |

**主估计量声明**：两主臂调优规则均收敛到默认规则 → **Delta = 默认规则配对差；SC3 与主对比完全重合**（gate_b.json 中明确写出；SC3 不重复进 Holm 族）。t+2 断言入各日志（terminal mark 2022-12-30 ≤ 2022-12-31）。

## 5. CPU 基线臂（§8 essential；10-17 已跑完，此处补入清单）

规则选择同口径（2021–2022 种子平均 + 0.5pp 低换手平局）；配置 = `src/cpu_baselines.py`（sha256 `4c585a0f…`，基线 commit `8a7a4c4` 起未改动）：ridge_alpha=10 确定性单跑；LightGBM 两臂 Qlib Alpha158 树参数（num_leaves 210 / lr 0.2 / 官方 L1L2）seeds 1–3、1000 轮上限 valid 早停 50。

| 臂 | 标签 | 种子 | 选定规则 |
|---|---|---|---|
| ridge | h5 | 1（确定性） | drop5-every5d |
| ridge | r1 | 1 | drop2-every1d |
| lgbm-reg | h5 | 1–3 | drop10-every5d |
| lgbm-reg | r1 | 1–3 | drop5-every1d |
| lgbm-rank | h5 | 1–3 | drop2-every1d |
| lgbm-rank | r1 | 1–3 | drop5-every5d |

## 6. Benchmark 臂（描述性；Qlib v0.9.6 官方默认参）

**措辞（按审阅修正）**："Qlib defaults, except label normalization: CSRankNorm not re-added"——除标签归一化外超参逐字取自官方 yaml；数据约定=干净面板 FilterCol 20 列 + z_r1、train 2008–2018 / early stop on valid 2019–2020（六个模型逐个确认：ts 四模型 early_stop 按 valid loss；XGBoost num_boost_round=1000 上限 + early_stopping_rounds=50 在 valid 上早停（yaml 的 n_estimators 在 qlib train 路径下不生效，按 qlib 自身默认行为执行并披露）；DoubleEnsemble 按配置固定 3 子模型×28 轮、无早停机制）。

**取证（`checks/benchmark_forensics.json`，回应"10s/75s 不可信"红旗）**：train 768,935 行 / valid 145,719 行，与 CPU 基线 768,338 差 0.08%（特征有限性口径差，无训练集缩水）；XGB 实跑 ~106 轮（best 56 + patience 50）、DE 84 轮；输入为 20 列（FilterCol）非 158 列——速度由此解释。验证窗 r1 IC：XGB 0.0323 / DE 0.0245 / LSTM 0.0197 / GRU 0.0246 / ALSTM 0.0211 / Transformer 0.0294。ts 四模型全部早停且最优在第 0–1 轮（LSTM/GRU/ALSTM 11 轮、Transformer 7 轮）——描述性事实照录。

**种子（按 DL 2026-10-19 条目冻结）**：LSTM/GRU/ALSTM/DoubleEnsemble seeds 1–3（随机臂，§8A 约定；seeds 2–3 运行中，选择将按种子平均）；XGBoost 与 Transformer 单跑（确定性：xgboost 默认 seed 0 + hist 单机；Transformer 官方 yaml 显式 seed 0）。

**环境披露**：pyqlib cp312 wheel 换标签安装；`qlib.data._libs` 桩模块**被调用即抛 NotImplementedError**（不存在静默路径——任何触达都会使运行显式失败）；表达式引擎不在本研究代码路径上。

## 7. 剂量诊断（`checks/dose_diag_2026-10-18.json` 全量）

5 日 IC：H 0.0382 / H-FT 0.0556 / G0 0.0549 / C0 0.0548 / G0dose 0.0396 / C0dose 0.0435。soft vs topk（2021–22）：正式臂 soft−topk ≈ −12～−13pp；**剂量臂 G0dose soft −29.7pp vs topk +20.3pp（差 −50pp）——验证窗软层增益在窗外执行层前已消失**（论文 surrogate-gap 一节的直接证据）。

## 8. CSI500 H（Kaggle，Gate C 素材）

三种子完成验收：seed1 12ep/ep4（`06cd06e5`）、seed2 13ep/ep5（`d6ef1297`）、seed3 14ep/ep6（`4d961d0b`）；三锚全过（data_version `bae7115e`、config hash `393b1993`、trainer sha）；`runs/csi500_H_kaggle/`。

**CSI500 G0/C0 前置（未启动，记录要求）**：baostock 成交账本须做分板块涨跌停带（主板 10%；创业板 2020-08-24 起 20%；科创板 20%；ST 5%；上市首日等无限制日）；baostock 与 RESSET 数据源不同须披露并抽查对账。

## 9. 解封范围

**仅 CSI300**。CSI500 的一切评价期输出保持封存，直至 CSI500 G0/C0 存在且其规则在 2021–2022 上选定之后。

## 10. gate_b.json 草案（修订版，待终审后写入）

```json
{
  "unseal_scope": "CSI300 only; CSI500 evaluation outputs stay sealed until CSI500 G0/C0 exist and their rules are selected on 2021-2022",
  "config_hashes": {
    "neural_formal": "393b1993...", "trainer": "a960a995...", "data_version_csi300": "3c70f663...",
    "execution_feed_resselt": "aff735c3...", "benchmark_h00300": "d6b21b61...",
    "cpu_baselines": "4c585a0f...", "benchmarks_spec": "da5b7425...",
    "code_commit": "<HEAD at write time>"
  },
  "fee_schedule": {"buy_bp": 5, "sell_bp": 15, "min_fee_cny": 5, "lot": 100, "capital_cny": 1e8,
                   "stamp_duty_sensitivity": "sell 10bp from 2023-08-28 (CT/137/23), reported alongside, never decisive"},
  "tau": 0.5,
  "tier": "P1", "sigma_d_final": 0.011894, "mde80_pp": 1.5811,
  "inference": {"bootstrap": "stationary, mean block length 10, 10000 resamples, rng_seed 20261019",
                "primary_bound": "one-sided 5% lower bound of Delta",
                "newey_west_lag": 5,
                "holm_family": ["SC1","SC2","SC4","SC5","SC6","SC7"],
                "sc3_note": "SC3 coincides with the primary (both arms selected the default rule); not double-counted"},
  "selected_rules": {"H": "drop20-every5d", "H-FT": "drop2-every1d", "G0": "drop5-every1d", "C0": "drop5-every1d",
                     "ridge_h5": "drop5-every5d", "ridge_r1": "drop2-every1d",
                     "lgbm-reg_h5": "drop10-every5d", "lgbm-reg_r1": "drop5-every1d",
                     "lgbm-rank_h5": "drop2-every1d", "lgbm-rank_r1": "drop5-every5d"},
  "benchmark_rules": {"<to be filled from the 3-seed selection once seeds 2-3 land>": "..."},
  "seeds": {"neural": [1,2,3], "cpu_lgbm": [1,2,3], "ridge": [1], "benchmarks_stochastic": [1,2,3], "benchmarks_deterministic": [1], "dose": [1]},
  "holdout_minimum_days": 120,
  "exploratory_arms": ["G0dose003_seed1", "C0dose003_seed1", "six Qlib-default benchmark arms (descriptive)"],
  "estimand_note": "Delta = default-rule paired difference (both arms selected the default rule)"
}
```

写入并 commit 后，其 sha256 记入 `protocol/gates.md`（gate_b.json 自身哈希）。

## 11. 解封后评价执行

各臂评价窗打分（选定+默认规则）→ 只写 `_sealed` → 统一读取汇总 → Gate C 机械判定。

> 注：本文件为内部工作稿；对外送审正本 = `protocol/gate_b_report_en.md`（英文、时间均为 America/Los_Angeles）。
