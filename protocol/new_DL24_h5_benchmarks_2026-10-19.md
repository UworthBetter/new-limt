# DL-24（新）h5 基准增补：LSTM / Transformer × {default-h5, tuned-h5}（r14，开封前；已签字）

实验助手，2026-10-19（PT）。**r14 版**：取代 `new_DL24_h5_benchmarks_r13_draft.md`（e280591f…，保留不改），只是把 09:29 卡片选项的原文补上。r13 的说明：r13 取代 r11（dbce5170…），只增加签字记录和 spec 定稿的 sha，正文不变。r11 的说明：**r11 版**：取代 `new_DL24_h5_benchmarks_r8_draft.md`（36f25734…，保留不改），内容不变，只把过时的文件引用改到最终版本（spec 草稿 r8 json、GPT 规格 r9、防护规格 v3.1d、重训清单 r9），并补上 r9 的 cfg1 参数断言和截止日两点。以下是 r8 的说明：**r8 版**，在 r7 的基础上按 Claude 第 8 轮（07:52 PDT 的回复和 `/workspace/uploads/claude_round8_reply.md`）修订，取代 `new_DL24_h5_benchmarks_r7_draft.md`（5c985568…，保留不改）；r7 是按第 7 轮裁决修订的，它取代的 `new_DL24_h5_benchmarks_draft.md`（3ee6cf36…）也保留不改。新文件，待 Claude 审。依据：Claude 的 h5 方案（`/workspace/uploads/claude_h5_plan_reply.md`）。所有时间均为 PDT。`[sha]`、`[时间]` 是待填占位。

**编号说明**：第 6 轮草稿 `new_DL_rows_round6_draft.md`（c45a7bdc…，保留不改）曾把 v3 → v3.1 编为 DL-24。按 Claude 07:33 PDT 的决定，**DL-24 = 本条（h5 基准）**，v3 → v3.1 改为 **DL-25**，见 `new_DL_rows_round6b_draft.md`。

---

- **谁、何时决定**：PI beelingz，2026-10-19 07:28 PDT 决定执行，当时任何评估期（2023 年以后）产出都没有打开过，2021–22 的规则选择结果也没看过（T7 产出包没解开）。方案由外部审阅（Claude）提出。
- **目的**：基准公平性，把标签效应和模型效应分开。现有 Qlib 默认基准用的是 r1 标签（DL-2，CSRankNorm(r1)），而 LiMT 用的是 h5 标签。同时换标签又调参，两种效应会混在一起，所以分两步、两条臂：

| 臂 | 标签 | 超参 | 状态 |
|---|---|---|---|
| default-r1 | CSRankNorm(r1) | 官方公布值 | 已跑（15 个 tag 里的 LSTM、Transformer） |
| default-h5 | CSRankNorm(h5) | 官方公布值 | 新增 |
| tuned-h5 | CSRankNorm(h5) | 下面固定网格中选出的一组 | 新增 |

  default-h5 − default-r1 = 标签效应；tuned-h5 − default-h5 = 调参效应。

- **地位**：描述性臂，**不进 Holm 族**。主臂（C0/G0）、主估计量和全部推断设置都不变。tuned 臂一律叫 "tuned-h5"，**任何地方都不得称为"官方默认"或"published defaults"**。default-h5 可以称为"官方默认超参（h5 标签）"。
- **模型**：LSTM（经典基线，r1 下 best epoch 是 0–1，最差情形）和 Transformer（唯一在第一轮后还在学的 NN 基准）。GBDT 不加：CPU 上的 LightGBM 已有 h5 和 r1 两组 Qlib 参数臂，树模型的标签效应已经测过。
- **标签**：h5 = C(t+6)/C(t+1) − 1（标签文件 `new_v2_labels_csi300.npz`，sha `7bf5633b…`，键 `z_h5`、`label_mask_h5`、`exit_idx_h5`、`entry_idx`）。每个交易日在 `label_mask_h5` 内做 CSRankNorm，即 (rank(pct) − 0.5) × 3.46，与 DL-2 相同。只用于训练和验证标签，r1 也是这样处理的。
  - **端点清洗：开启**（第 7 轮第 1 条）。训练和验证样本只要标签端点 `dates[t+6]` 晚于所在段末（训练 2018-12-31，验证 2020-12-31），就剔除。不清洗的话，训练标签会读到验证期价格，早停就不干净了。代价：按 Claude 的估计，训练约 2,500 天里少约 5 个边界日，验证约 485 天里少 4 天。两种做法都碰不到封存数据：验证端点最多到 2021 年初，不在封存范围内。
  - **不对称披露**：已跑的 default-r1 臂（v2）**没有清洗**。r1 端点是 t+2，每个边界重叠 1 天。
- **调参网格**（每个模型 8 组，现在固定）：

| 因子 | 水平 |
|---|---|
| 学习率 | 官方值；官方值 ÷ 10（LSTM 0.001 / 0.0001；Transformer 0.0001 / 0.00001） |
| 宽度：LSTM `hidden_size` / Transformer `d_model` | 官方值；2 × 官方值（64 / 128） |
| 早停指标 | 验证损失（官方）；验证期 5 日 IC |

  其余全部保持官方值，包括 epoch 上限（LSTM 200，Transformer 100）、早停 patience（LSTM 10，Transformer 5）、batch size、层数、dropout、损失函数和 n_jobs。官方值出处：spec v2 `configs/new_benchmarks_qlib_defaults_v2.json`（49010f9b…）的 `models.LSTM` / `models.Transformer` 原样复制；Transformer 的类默认值取自 qlib v0.9.6 `pytorch_transformer_ts.py:26-43`。cfg1–cfg8 的编号见 spec 草稿。cfg1 = 全部官方值，也就是 default-h5 的配置。**cfg1 seed 1 只跑一次**，同时算作 default-h5 seed 1 和调参池的成员（第 7 轮第 2 条：重跑同配置同种子只是 GPU 非确定性下的复制品，属伪重复）。
- **验证期 5 日 IC 的定义**：对每个验证信号日（2019–2020），取分数与 `z_h5` 的 Pearson 相关，掩码为 loss_mask_valid ∧ 端点不越过 2020-12-31 ∧ 上市合格 ∧ 两边都有限；再对各日取均值。与 limt-repro `v2_w5_results.daily_ic` 相同。
- **选择**：只用 seed 1。每个模型的 8 组配置，按回载后最佳 checkpoint 的验证期日均 5 日 IC 选，这和所有臂早停用的是同一窗口。**并列规则（精确顺序）**：
  1. best = 8 组中 IC 的最大值；
  2. 候选池 = IC ≥ best − 0.002 的配置；
  3. 池内先选非官方因子最少的，即偏向官方值；
  4. 再选 IC 最高的；
  5. 再选 cfg 编号最小的。

  任何一组失败或 IC 非有限，就停下报告，不做选择。

  **选择偏差披露**（第 7 轮第 3 条）：用 IC 早停的配置，选择时用的是它们早停时的同一指标、同一窗口，所以选择对这些配置偏乐观。这偏向 tuned 臂，也就是对基准更公平，对 LiMT 的结论是保守的。

  **若选中 cfg1**：tuned-h5 就是 default-h5。作为一项发现报告（"调参预算选回了官方配置"），不算两条臂，也不再加跑。只记录选中的配置名；每组的 IC 只写进文件，不在日志里打印。
- **种子**：tuned-h5 = 选中配置的 seed 1–3，其中 seed 1 **复用调参时那次运行**。这次运行参与了选择，在此披露。default-h5 = seed 1–3，其中 seed 1 就是 cfg1 的调参运行。
- **执行规则**：与其它所有臂相同，2021–2022 的 8 条规则网格、种子平均、0.5 pp 内取低换手。DE 不动。
- **运行数（现在声明，第 7 轮第 2 条）**：每个模型 7 个只用于调参的运行（cfg2–cfg8 seed 1）、3 个 default-h5（seed 1 兼作 cfg1）、2 个选中配置的 seed 2–3，共 12 个。两个模型合计 **24 个**。运行数按两个模型中有几个选中 cfg1 来定：**都没选中是 24 个，一个选中是 22 个，两个都选中是 20 个**（Claude 第 8 轮 (c) 确认）。Claude 方案原文的 26 个已作废。
- **算力**：Claude 估 A10 上 5–8 小时（按 26 个运行估的）。这是可选工作，**计入 60 GPU 小时上限**（不属于 DL-1/2 重跑豁免）。
  - **记法（第 7 轮第 5 条）**：GPU 实例只要开着就计时，包括空闲和排错，取实例挂钟时间；各 tag 的 device_hours 之和另记一份，仅供参考。实验助手的估算见文末附录。
- **spec 文件**：新建 `configs/new_benchmarks_h5_v3.json`（草稿 `/workspace/limt_gateb/new_benchmarks_h5_v3_r8_draft.json`，039ee1cb…；**status 字段在 beelingz 本人批准本条之前一律不写 FROZEN**；定稿 sha `77a2d8db5bac48e7db4a6fef597690add5b084bd26c16498e786954e7dadeae9`（见下方签字一节）），**v2 不改**。代码规格见 `/workspace/uploads/new_GPT_spec_h5_benchmarks_r9.md`（84f25f17…），防护规格见 `/workspace/uploads/new_GPT_spec_v3_1d_guard_fix.md`（7eea4ee1…）。代码由 GPT 写，新文件，不改 v2 代码。
- **在开封流程中的位置**（都在写 gate_b.json 之前，因为每条臂只能开封一次）：
  1. 先 commit DL-24 和 v3 spec（DL-24 里写明 spec 的 sha），**立即 push**。push 必须在任何运行（包括 T1）之前完成。上次的顺序问题见 DL-16 和 DL-26；
  2. 在新 runner 上做 T1 类检查：种子、h5 标签、IC 函数、RNG 隔离；
  3. 训练 24 / 22 / 20 个运行（两个模型中选中 cfg1 的有 0 / 1 / 2 个），在验证期上选配置，只记录配置名；
  4. 在 2021–22 跑 8 条规则网格，更新取证（只用验证期数值）；
  5. 接回原来的流程：§8（加入两条新臂）→ gate_b.json（加进 arms 列表）→ push → 开封。
- **评估期账本**：v3（203e6a45…）没有写死基准列表，只计算 C0/G0（`new_gate_b_analysis_v3.py:147-149, :397-412, :488`），h5 臂不进 v3.1，也不进 Holm（`HOLM_KEYS`，`:66`）。按第 7 轮第 4 条，所有描述性臂（含 h5 臂）的评估期账本由另一个冻结脚本 `src/new_bench_eval_v1.py` 产出。它和 v3.1 共用防护模块 `src/new_guarded_data_v1.py`，臂列表从 gate_b.json 读取，只出描述性指标，不出 p 值、不做 Holm。它的 sha 记进 gate_b.json，与 v3.1 在同一次开封运行里跑。规格见 GPT r7 规格 §7 和 v3.1c §7。
- **签字**（r13）：
  1. **09:29 PDT，点选卡片**：beelingz 点了一个选项。选项文字是实验助手写的，内容是附带条件的批准，原文是：“签：我批准 DL-24 r11，Claude 审核无修改才生效”。这个条件后来满足了：Claude 第 10 轮确认 DL-24 r11 不需要修改。这一次照实记录，**但不作为最终签字**。DL-3 也是同样的处理：先点卡片，之后再用本人的话追认。
  2. **09:38 PDT，本人打字签字**：在与 Claude 的对话中，beelingz 亲手输入：“我已审阅 DL-24 r11（4ce459c0）并同意，内容无改动。beelingz，日期10-19”。**披露**：句式沿用了 Claude 建议的大意，但文字是 beelingz 自己打的，不是点卡片。以这一次为准。
- **spec 定稿**：`configs/new_benchmarks_h5_v3.json` 的定稿是 `new_benchmarks_h5_v3_frozen_20261019.json`，sha `77a2d8db5bac48e7db4a6fef597690add5b084bd26c16498e786954e7dadeae9`。它在 039ee1cb（DRAFT r8，保留不改）的基础上只改了两处：status 改为 `"FROZEN: DL-24 approved by beelingz 2026-10-19 09:38 PDT"`，并加上 `r1_meta_sha256`（6 个 r1 meta 的 sha，与 DL-27 的允许清单逐条一致）。其余字段逐字节相同。已在盒子上用 GPT 的 `assert_cfg1` 核对：6/6 的 cfg1 kwargs 等于对应 r1 meta 的 kwargs；status 不含 draft。

---

## 附录：算力估计与并发方案（实验助手）

**已观测**（2026-10-19 A10 运行报告，只看拟合时间和轮数）：
- LSTM s1 单独跑：11 轮，264 s，约 24 s/轮。这个数包含 qlib 每轮对训练集、验证集各做的一次评估。
- 5 个并发时（05:16–06:00）：LSTM s2/s3 约 19–20 分钟（11–12 轮，约 100–105 s/轮）；Transformer 约 10–12 分钟（9–11 轮，约 60–65 s/轮）。
- 推论：5 个并发时 LSTM 的总吞吐只比单跑高约 15%（5/103 对 1/24 轮每秒），说明瓶颈在 CPU（8 核，每个进程 n_jobs=20 个数据进程），不在 GPU。
- Transformer 单独跑的速度没有观测过，按并发时与 LSTM 的比例推算约 15 s/轮，**这是估计，未验证**。

**假设**（未验证，要等实测）：
- 用 valid loss 早停、官方 lr 的配置：轮数与 r1 相近（LSTM 11–12，Transformer 6–11）。
- IC 早停或 lr ÷ 10 的配置：可能 11–40 轮。
- 2 倍宽度：模型小、数据加载是瓶颈，每轮时间估 ×1.0–1.5。
- 每个进程启动（BlindData 解压、建表、预测）约 2–4 分钟。

| 情形 | LSTM 13 个 | Transformer 13 个 | 启动开销 | 合计（串行等效） |
|---|---|---|---|---|
| 低（按原 26 个估，24 个约少 8%） | 13×11×24 s ≈ 0.95 h | 13×8×15 s ≈ 0.45 h | ≈ 0.9 h | ≈ 2.3 h |
| 高 | 13×30×30 s ≈ 3.3 h | 13×25×19 s ≈ 1.7 h | ≈ 1.7 h | ≈ 6.7 h |

Claude 的 5–8 h 落在高情形附近，可以作为预算上限。

**并发方案（一张 A10，8 核，32 GB 内存）**：
- 并发 K = 3，不用 5：并发主要是为了让一个进程的 CPU 段（解压、建表、预测）和另一个进程的 GPU 训练重叠，K = 5 的实测增益很小。n_jobs、batch_size 是冻结参数，不改。
- 同一时间最多 1 个 Transformer d_model=128 的运行：之前观测 Transformer 并发时约 12 GB 显存，宽度加倍后激活会变大。这是估计，先观察一次再决定。
- 阶段 A，与选择无关，共 20 个：16 个 seed 1 运行（2 模型 × 8 组，cfg1 即 default seed 1）和 4 个 default-h5 seed 2–3。按预计时长从长到短排，先跑 lr ÷ 10、2 倍宽度和 IC 早停的配置。
- 阶段 B：某个模型的 8 组跑完，先跑选择（只打印配置名），再跑 tuned 的 seed 2–3，2 模型共 4 个（选中 cfg1 的模型不跑）。
- 之后在 CPU 上跑规则网格和取证。这一步不占 GPU，可以关掉 GPU 实例，在 CPU 环境跑，平台指纹以训练产物的 meta 为准。
- 预计 wall-clock：K = 3 时约为串行等效的 0.75–0.9 倍，即约 2–6 h。
- GPU 小时按实例挂钟时间记（第 7 轮已定，含空闲和排错），各 tag `device_hours` 之和另记作参考。另外，实例空闲 1 小时会自动停机，长任务要用 nohup 跑。

## r8 增补（Claude 第 8 轮）

- **必须存 checkpoint**：h5 runner 每个运行都要把 checkpoint（回载后的最佳 epoch）存为 `new_<tag>.pt`，并把它的 sha256 写进 `<tag>.meta.json` 的 `artifact_sha256`。没存成或 sha 不一致就硬失败。全部 checkpoint 的 sha 都写进 gate_b.json 的 `descriptive_arms[*].score_source`。
- **checkpoint 的位置**：r7 的 cache 路径在实例上，没有打进封存包。r8 要求 checkpoint 和 meta 一起打包上传，在数据集里单独放一个子目录，不和分数包混在一起。
- **评估期打分**：只在开封那一步做，用 `new_bench_eval_v1`。开封前不出任何评估期预测，详见 **DL-28**。
- 已有基准臂缺模型的处理（重训和比对）见 DL-28 和 `new_retrain_list_r9_draft.md`。

## r9/r11 增补（Claude 第 9 轮）

- **cfg1 等于 default-r1**：两者只在标签和清洗（purge）上不同。runner 断言 cfg1 解析出的构造器 kwargs，与 15 个 r1 tag 的 meta 中同模型、同种子那一份的 `kwargs` 和 `effective_model_settings` 完全相等。Transformer 另外要核对 qlib v0.9.6 的类默认值表和源码 sha（GPT 规格 r9 §2.2a）。
- **截止日与防护**：h5 runner 在代码里写死 `PRE_UNSEAL_CUTOFF = 2022-12-31`。防护模块 `new_guarded_data_v1.py` 必须先冻结、commit，runner 按已提交的 sha 校验（GPT 规格 r9 §2.7，v3.1d §7）。
- **checkpoint**：算文件 sha，再用 `torch.load` 读回，逐个张量比对后记录（GPT 规格 r9 §2.6）。
- **平台**：魔搭 A10，与 15 个 tag 同一平台；规则网格和取证在魔搭 CPU 实例上跑。
