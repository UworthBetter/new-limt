# 未决项工作板

来源：预注册第 17 节，2026-10-17 抄到本目录。状态只在这里更新，不改预注册正文。

## 影响代码结构（先定）

| ID | 项 | 预注册状态 | 本目录状态 |
|---|---|---|---|
| S1 | G0 训练账本零费率，选 checkpoint 用真实费率监控账本 | 已确认，必须做 | **代码已改** 2026-10-17。微调选轮走 `monitoring_mean_log`（5/15 bp、最低 5 元、100 股）。零费率只记日志。见 `limt-repro/src/v2_w4_training.py` |
| S2 | tau 网格 {0.5, 0.75, 1.0} | 已批准 | **已跑完** 2026-10-18（runs/csi300_g0c0_seed1_tau_20261017，6 跑全 20 epoch）。两臂按监控指标均选 tau=0.5；正式冻结等 Gate B。§6.2 理由已修正（v0.6） |
| S3 | L0 listwise | 不做 | 不建该臂 |
| S4 | 干净 1 日 H 重训 | 不做 | 标签杠杆只在 ridge 与 LightGBM 上 |
| S5 | H-FT 只用监督损失、同一套 epoch 上限 | 待核对 | **已定** 2026-10-17。模式就是 `group=H-FT`，不另加旗标。选轮仍等 A-G0，正式 H-FT 先不启动。见 `structural_decisions.md` |
| S6 | 输出键 (market, backbone, label, rule, seed)，并保存毛收益与成本拆分 | 未做 | **已定** 2026-10-17。日频列和 `Delta_cost` 的符号已写死。代码还没按这个写 |
| S7 | Gate B 哈希出现之前挡住全部 2023+ 输出 | 未做 | **已定** 2026-10-17。`GateLocked`，CPU 与 GPU 都算。`gate_b.json` 还不存在，读取函数还没写 |
| S8 | 原文协议桥接阶梯 | 主矩阵不做 | 不建该实验 |
| S9 | CSI500 通过 W1–W2 | 未做 | **W2 已通过** 2026-10-17。验收在 `data/csi500_w1w2/results/v2/new_W0W2_acceptance.md`。端点错位 0。N6 已测：1 个 H epoch 510.4 秒 |
| S10 | 标签轴 h5/r1，以及八条 `drop*-every*d` 规则 | v0.5 未编号 | **已定** 2026-10-17。补的是代码接口，不是新实验。神经网络臂不做 r1 |

## 不影响代码结构

N6 已测量，数在 `checks/n6_csi500_epoch.json`：全股票池 1 个 H epoch 510.4 秒。预注册正文里的 `[FILL]` 还没改。
CSI300 CPU 基线已跑完，选择结果在 `runs/csi300_cpu/selection.json`。评价期日频在 `runs/_sealed/`，没有汇总。CSI500 基线未跑：没有和 1802 只股票对齐的成交账本。
其余 N 项仍空着。不要在缺测量时填数。

2026-10-18 增补：
- **C-T 种子 1 推迟**（记录在 v0.6 §8 与 tau_grid_report）：γ_to×中位换手 = 1.9e-4 与费用项同量级，而费用项在验证窗已证近惰性。剂量问题解决前不跑。
- **H-FT 种子 1**（矩阵内）2026-10-18 发射：src/run_hft_seed1.py，监控 τ=0.5。
- **探索性剂量臂**（G0/C0 @ α_aux=0.03, τ=0.5, 种子 1）：已在 deviation_log.md 预登记后发射（src/run_dose_seed1.py）。全部结果 exploratory。
- **N5 在填**：src/score_val_taus.py 把每个 checkpoint 在验证窗 3τ 的 N_eff、单名最大权重、监控账本指标写入 `checks/n5_neff_validation.json`。
- 实测每跑 20 epoch ≈ 1.5–1.6 小时（含双份份额账本回放）；H-FT ≈1.6h、剂量臂两跑 ≈3.1h。预注册 §10 的 Kaggle T4 数字不适用于本机。

2026-10-18 晚增补：
- **CSI500 H ×3 seeds 提前到 Kaggle 并行**（调度决定，非偏差：矩阵内工作、§6.2 配置已冻结）。理由：H 纯监督训练不需要 execution feed，也不产生/不读任何 2023+ 输出，不受 Gate 门禁约束；G0/C0 仍等 CSI500 对齐成交账本（本地数据工程）。发射包在 `kaggle_csi500_h/`（README_upload.md 有上传步骤与三项完整性锚：data_version bae7115e…、config hash 393b1993…、trainer sha a960a995…）。
