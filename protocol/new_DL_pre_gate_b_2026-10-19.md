# 偏离日志草稿（Pre-Gate-B，第 2 版，吸收外部审稿第三轮意见）

> 状态：**草稿，未提交，未写 gate_b.json，未跑任何训练**。修订于 2026-10-19 01:50 PDT（box），吸收外部审稿第 4 轮回复。之前的版本存档在 `old/`（r1_0047、r2_0137）。
> **代码基准**：
> - GitHub `UworthBetter/new-limt` 的 HEAD 仍是 `bc0dc13d848c8c7beeddaf9d4e0bca5c3a11aca4`（2026-10-18 23:03:26 PDT；01:0x PDT 时 `git pull` 无更新）。
> - **笔记本 `D:\research\new_LiMT` 领先 GitHub 1 个 commit**：本地 HEAD 是 `b7306e0d4b0d37c8456456264ed8c7a2a9eca10a`（2026-10-19 00:53:04 PDT，"frozen analysis v2"），`main [origin/main: ahead 1]`，尚未推送。
> - 工作区另有未提交改动：`kaggle_csi500_h/logs_pipeline_20261018.log` 已修改；未跟踪文件有 `kaggle_csi500_h/logs_pipeline3_20261018.log`、`logs_poll_split_20261018.log`、`protocol/checks/qlib_b3_master_verification_2026-10-19.md`。
> **分析产物**：
> - v1：`src/gate_b_analysis.py`，sha256 `5742585ac41a9a291b0eaefed0086cc337220ad9cdf73aee290f16d3430325b6`，commit `4f60ef436b271ab07b0c5d02fcf2f7c04a89c215`。
> - v2：同一路径（原地修改，v1 只留在 git 历史里），sha256 `7d90d218e661a242512e542e309ea49e4858c428da6f2ea85018ef22e6db64fa`（笔记本上用 Get-FileHash 复算），commit `b7306e0d4b0d37c8456456264ed8c7a2a9eca10a`。
> **对照源码**：microsoft/qlib tag v0.9.6（commit `5a84aaf1dc402232646ad2836dea33ff15886552`）。
> **盲审**：本草稿没有任何 2023-01 之后（eval/full-test）或 2021–2022 rule-select 窗口的指标数值。核查只看了代码、日期、形状、掩码、meta 字段、文件时间戳，以及 2019 年以前训练段特征的 min/max 边界。
> **v2 行号**：引用 v2 代码时的行号以 `git show b7306e0:src/gate_b_analysis.py` 为准。

---

## DL-1（B1）XGBoost 与 Transformer 改跑种子 1–3；种子 0 保留为描述性的公开配置复现（已定）

**变更**：取代 2026-10-19 那条 DL（commit `9a578bb0ccee6d33be2198b31cad51adfee1e46f`）中"XGBoost 与 Transformer 维持单跑（确定性）"的规定。
- XGBoost、Transformer 各跑种子 1、2、3，规则选择按 §5 对 3 个种子平均。
- 已有的种子 0 单跑（tag 是 `XGBoost_seed1` / `Transformer_seed1`，meta 里 `effective_seed=0`）保留，只作为"种子 0 公开配置复现"的描述性记录，不参与 3 种子规则选择。

**原因（代码证据）**：
- XGBoost 的 `subsample=0.8789`、`colsample_bytree=0.8879` 都 < 1（qlib v0.9.6 `examples/benchmarks/XGBoost/workflow_config_xgboost_Alpha158.yaml:37,41`；本仓库 `configs/benchmarks_qlib_defaults_v1.json:20`）。抽样由 xgboost 自己的 `seed` 参数驱动，所以"固定种子"不等于"跨种子确定"。
- 现行代码里种子没有进入 `xgb.train`：
  - qlib `qlib/contrib/model/xgboost.py:18-20` 把 kwargs 存进 `self._params`，`:58-67` 原样传给 `xgb.train(self._params, …)`；yaml 和 spec 里都没有 `seed`。
  - `src/run_benchmarks.py:146-148,177` 的全局 `random/np/torch` 种子对 xgboost 不起作用；`:143-144` 还禁止 XGBoost 用 seed≠1 运行。
  - 笔记本上实际运行的 meta：`XGBoost_seed1.meta.json` 的 `effective_seed=0`、`kwargs.nthread=20`。
- 合成数据验证（不是项目数据；box 上 xgboost 3.4.1，与笔记本 venv_bench 的 xgboost 3.4.1 相同；复刻 qlib 的调用方式）：
  - 不传 seed 与 seed=0 的预测逐位相同；
  - seed 1/2/3 两两预测不同；
  - 不传 seed 时，全局 `np.random.seed` 无论设成什么，预测都逐位相同。
  - 脚本：`/workspace/limt_gateb/new_xgb_seed_demo_synthetic.py`、`new_xgb_global_seed_check_synthetic.py`。
- Transformer：qlib `qlib/contrib/model/pytorch_transformer_ts.py:63-65` 在构建模型（`:67`）之前用 config 的 seed 重新设种子；`run_benchmarks.py:185-186` 的 `setdefault("seed", 0)` 加上 spec 里的 `"seed": 0`，使任何 `--seed` 都被重置为 0。必须显式写 `kwargs["seed"]=seed`。
- 附带披露：yaml 的 `n_estimators: 647` 在 qlib 的 `xgb.train` 路径下不被使用，实际是 `num_boost_round=1000`、`early_stopping_rounds=50`（`xgboost.py:26-27`）。

**跑前门槛**：
- XGB：确认 params 里有 seed，且两个种子的预测不同。
- 4 个 NN：同一种子两次构建，初始 state_dict 的哈希相同；不同种子则不同。
- 任一项不通过就停，不进入正式跑。

**受影响结果**：XGB 和 Transformer 的旧单跑降级为种子 0 复现。全部是描述性臂，不进 Holm 族。

---

## DL-2（B2(a)，已确认）LSTM / GRU / ALSTM / Transformer 加回 CSRankNorm，各跑 3 个种子

**变更**：这 4 个 NN 基准的训练标签改为 CSRankNorm(r1)，做法与 qlib `qlib/data/dataset/processor.py:331,360-362` 相同：每个交易日截面内 `rank(pct=True)`，减 0.5，再乘 3.46。与官方 yaml 的 `learn_processors: [DropnaLabel, CSRankNorm(label)]` 一致（`examples/benchmarks/{LSTM,GRU,ALSTM,Transformer}/workflow_config_*_Alpha158.yaml:27-31`）。
- 只作用于 train 和 valid 的学习标签；早停用的 valid loss 也按 CSRankNorm 标签计算；推理分数不做变换。
- 每个模型跑种子 1–3。Transformer 同时满足 DL-1 和 DL-2，只跑一套。

**对 z_r1 做 CSRankNorm，与对原始 r1 做完全相同**：
- z_r1 由 `v2_calendar_labels.py:47-53` 的 `cs_z` 生成：对每个交易日 i，只在掩码 `label_mask_r1 = universe & isfinite(raw_r1)`（`:98`）内做 `(v - mean_i) / (std_i + 1e-12)`。
- 这是同一天、同一集合上的正斜率仿射变换（std_i > 0），严格单调递增，所以截面排名（含并列）完全不变，`rank(pct=True)` 结果逐位相同。
- 因此 CSRankNorm(z_r1) ≡ CSRankNorm(raw r1) ≡ qlib 官方对原始 1 日收益做的 CSRankNorm（前提是排名集合相同）。

**排名集合**：每天所有标签有限的股票，即 `label_mask_r1[t]`（当天在成分股范围内、原始 r1 两端价格都存在）。对应 qlib 的 DropnaLabel 加上 handler 当天的成分股范围。
- **注意**：不能用 `isfinite(z_r1)` 来定义集合。`cs_z` 把掩码外的格子填成 0（`v2_calendar_labels.py:48`），所以 z_r1 处处有限；照 `run_benchmarks.py:168` 的 `mask_r1 & isfinite(z_r1)` 写法，必须以 `label_mask_r1` 为准。
- 集合也不能写成"flt 保留的样本内"：flt 还叠加了 train/valid 的窗口和其他过滤，以它为准会改变截面。

**XGB/DE 不改**：Alpha158 的默认 learn processors 是 `DropnaLabel + CSZScoreNorm(label)`（qlib `qlib/contrib/data/handler.py:37-40`，`:106`）。官方 XGB/DE 的 yaml 没有覆盖这一项，所以官方用的就是截面 z-score，与 z_r1 一致。

**现有 z-score 标签的结果**：
- 以下 tag 保留为"z-score 标签描述性对照"，不再用于 benchmark_rules：`LSTM_seed{1,2,3}`、`GRU_seed{1,2,3}`、`ALSTM_seed{1,2,3}`，以及 `Transformer_seed1`（effective_seed 0）。

**重跑之后**：
- 新建 forensics 记录；
- 按 §5 对 3 个种子平均，重新为 LSTM、GRU、ALSTM、Transformer、XGBoost 这 5 组选规则；
- DoubleEnsemble 的规则不变。

---

## DL-3（spec JSON 更正 + B3 特征处理披露）→ 新文件 `configs/new_benchmarks_qlib_defaults_v2.json`

v1 文件不覆盖。v2 的 sha 在定稿后填入 DL-8。要改的内容：

1. **"class-default CSRankNorm" 写错了**。位置在 `configs/benchmarks_qlib_defaults_v1.json:9`，原文："Qlib's own CSRankNorm label processor (explicit in the NN yamls, class-default for XGB/DE) is NOT re-applied; this is the only deviation from the published pipelines."
   - 更正为：NN 的 yaml 显式写了 CSRankNorm，v2 起加回（DL-2）；XGB/DE 的类默认是 CSZScoreNorm，与 z_r1 一致。
2. **删掉"only deviation"的说法**。同类措辞还出现在：`src/run_benchmarks.py:7-8`、`protocol/gate_b_report_en.md:84`（Rev2 §8）、`protocol/gate_b_ready_2026-10-19.md:64`。
3. **XGB/DE 特征处理披露**（原文在 v1 `:8`）。
   - **官方做法**：`Alpha158.__init__` 的 `infer_processors=[]`（qlib `handler.py:105`），官方 XGB/DE 的 yaml 没有覆盖，所以喂进去的是原始 Alpha158 特征：不归一化、不截断、NaN 保留。
   - **我们的做法**：所有臂用同一个干净面板，即 RobustZScoreNorm（fit 区间 2008-01-01..2018-12-31）、±3 截断、Fillna→0、VWAP0 置 0。
   - **证据（已核实）**：
     - 构建代码 `D:\research\limt-repro\src\v2_calendar_labels.py`，sha `3abd456f1f6f364e107a0c0799d85e78f0e963fc1b89d50f5595b12859198c66`。笔记本正式版、box 副本、W0W2 manifest 三处 sha 一致。
       - `:119-120`：`Alpha158(…, fit_start_time="2008-01-01", fit_end_time=train_end, infer_processors=[], learn_processors=[])`，先取原始特征；
       - `:124`：`RobustZScoreNorm("2008-01-01", train_end, clip_outlier=True)`；
       - `:126`：`.fillna(0)`；
       - `:127`：`f["VWAP0"] = 0.0`；
       - `:132`：再做一次 `np.nan_to_num`。
     - qlib 的 `clip_outlier` 是 `np.clip(X, -3, 3)`（v0.9.6 `processor.py:299-300`）。
     - 面板实测，只读 2007-06-01..2018-12-28 训练段的边界，不看分布：
       - 笔记本正式面板 `D:\research\new_LiMT\data\csi300_w1w2\data\v2\new_v2_panel_csi300.npz`（sha `a25879f4b94e60b6224bf52ec777d25e4f4511af4be3591711647992b601a62e`，形状 4707×882×158 float32）：全局 min −3.0、max +3.0；158 个特征里有 98 个正好顶到 +3、84 个正好顶到 −3；NaN 为 0；VWAP0 全为 0。
       - box 上魔搭 fix1 版面板（sha `cb774f596e67041b683e95a4b1a4000e7ad03e3b6b7b40eb79a7e184cbba2c9b`）结果完全相同。
   - **要披露的后果**：
     - 截断不可逆，树模型看不到 |z| > 3 范围内的排序；
     - 填 0 取代了 XGBoost/LightGBM 原生的缺失值分裂方向；
     - VWAP0 被整列置 0，对所有臂都是常数特征（新发现，补进 §8）；
     - 归一化的 fit 区间是 2008–2018，官方 NN yaml 是 2008–2014。
   - 不需要重跑。
   - NN 的 yaml 本身就是 FilterCol + RobustZScoreNorm(clip) + Fillna，与我们的处理基本一致；fit 区间不同仍需披露。
4. **面板的构建环境（Alpha158 用真 qlib 计算，`_libs` 是编译版，不是桩）**：
   - **笔记本正式面板**（a25879f4…，labels `7bf5633b7ad9d146e5154a120307d8dbf771a9d5a5c9852f9eb7aba0bd1064a5`，splits `0ab69b33d0c8e76c020ebd100791eaf5535cac0e4c4c075207b03890baec17ad`）的文件时间：labels 是 2026-10-17 05:48 PT，panel 是 05:54 PT。meta 记录 `workers=4`、`elapsed_s=400`。
     - **CSI300 这次构建没有留下构建日志，也没有 environment freeze**，这是证据缺口。
     - 同一会话稍早的 CSI500 构建日志（仓库 `data/csi500_w1w2/logs/v2/new_W1.log`，05:16–05:23 PT）显示：用的是 `D:\research\limt-repro\.venv\Scripts\python.exe -u v2_calendar_labels…`，`workers=4, affinity_cpus=16`，`data_path=D:/research/limt-repro/data_new/cn_data`。
     - 这个 venv 现在的状态（2026-10-19 只读查询）：Python 3.13.9、**pyqlib 0.9.8.dev34**（不是 0.9.6）、numpy 2.4.1、pandas 2.3.3；`qlib/data/_libs` 里有 `rolling.cp313-win_amd64.pyd`、`expanding.cp313-win_amd64.pyd`，实际 import 到的是 `rolling.cp313-win_amd64.pyd`，即编译好的 Cython 扩展，不是桩。
     - 查询的是"现在"的 venv，不能完全证明 10-17 05:48 时就是这个状态（缺 freeze）。
   - **box/魔搭 fix1 版**（labels `e15ed6f6…`，panel `cb774f59…`；manifest sha `28bb5b7506d19c79d7246df35f945c37a931004b13e4991d1892ec49941e92e5`，笔记本 `D:\research\limt-repro\results\v2\fix1\v2_w0w2_out_fix1\results\v2\` 下有同 sha 的副本）：
     - `new_environment_freeze.txt`（sha `3878b1a0f53d988c754cbf26907dc9a7d52b5e1e64b81700d6730e930923855b`）显示 Python 3.11.11、`pyqlib @ file:///tmp/qs/qlib-be725493eb1a6bbb42bf11b37aa7669f59610ff1`（microsoft/qlib main 分支 2026-09-16 的 commit，晚于 v0.9.6）、Cython 0.29.36，表明是从源码安装的。
   - **与 benchmark 运行环境的区别**：笔记本 `venv_bench` 是 Python 3.13.9、pyqlib 0.9.6、xgboost 3.4.1、torch 2.7.1+cu118；它的 `_libs` 只有 cp312 的 .pyd，Python 3.13 加载不了，所以实际走 .py 桩。这与 `run_benchmarks.py:16-17` 说"桩，调用就报错"一致。但 benchmark 不计算特征，只读面板，所以不冲突。
   - **措辞更正**：§8 应写成"Alpha158 特征由 qlib（0.9.8.dev34，编译版 `_libs`）在 limt-repro 环境计算；benchmark 训练在 pyqlib 0.9.6 环境中只读取该面板"，并披露两个 qlib 版本不同。v0.9.6 与 0.9.8.dev34 的 Alpha158 表达式：**已核实，相同**（2026-10-19 只读核对）。
     - **怎么比的**：在笔记本上，对 `venv_bench`（pyqlib 0.9.6）和 `D:\research\limt-repro\.venv`（0.9.8.dev34）两个环境，逐文件算 sha256 并做 difflib 比较。
     - **逐字节相同**：`qlib/contrib/data/loader.py`（Alpha158DL）`472f9dc8…`、`qlib/data/ops.py` `6f785624…`、`qlib/contrib/data/handler.py` `19722325…`、`qlib/data/base.py` `cc29103b…`、`qlib/data/_libs/rolling.pyx` `54824a8d…`、`expanding.pyx` `2aca0fbe…`。
     - **有差异的文件**：
       - `qlib/data/dataset/processor.py`，只有两处：
         - `Fillna` 的实现写法：numpy 掩码改为 `df[...].fillna`，语义相同。另外我们的构建在 `v2_calendar_labels.py:126` 直接调用 pandas 的 `fillna(0)`，根本不经过这个 processor。
         - `CSRankNorm` 的 `groupby(..., group_keys=False)`：只影响返回值的索引形状，面板构建也不用它。
         - 面板实际用到的 `RobustZScoreNorm` 没有差异。
       - `qlib/__init__.py`：版本号来源和 NFS 挂载代码不同，与特征计算无关。
     - **与官方 tag 源码的关系**：笔记本上这些文件和 box 上 v0.9.6 tag 源码（`5a84aaf1…`）的 sha 不同，**原因只是换行符**。把 tag 源码的 LF 改成 CRLF 后，7 个文件的 sha 全部与笔记本上的一致，例如 loader.py：LF `814b7f7a…` → CRLF `472f9dc8…`。Windows 上的 wheel 或安装过程把换行符写成了 CRLF，内容与 v0.9.6 tag 相同。
     - **结论**：Alpha158DL 及相关算子在 0.9.6 tag、venv_bench、0.9.8.dev34 三者之间内容相同。§8 可以写"Alpha158 表达式和算子与 Qlib v0.9.6 源码逐字相同（版本差异只在 processor 的 Fillna/CSRankNorm 实现写法和 `__init__`，面板构建都没用到）"。这样 §8 文字和"official defaults"说法不再被阻塞。
5. `seed_policy`（v1 `:11`）按 DL-1、DL-2 重写。
6. 新增字段：每个模型的 `label_processor`、`feature_processing_official` / `feature_processing_ours`、`spec_supersedes`（列出 da5b7425… 和 e189f6cb…）、`benchmark_note`（H00300 全收益 vs Qlib 的 SH000300 价格指数）。

---

## DL-4（A1 / Gate C 产物 / SC2）Holm 族对齐冻结代码 v2

**v2 现状**（sha `7d90d218e661a242512e542e309ea49e4858c428da6f2ea85018ef22e6db64fa`，commit `b7306e0d4b0d37c8456456264ed8c7a2a9eca10a`）：
- `:71` `HOLM_KEYS = ("SC1","SC2","SC5","SC6","SC7")`，与 v1 相同，没有 SC4。
- `:368-370` 中 `pvals` 的 SC2 位置仍写死为 `np.nan`；`:401-403` 过滤 NaN，所以实际 m=4（SC1/SC5/SC6/SC7）。
- **v2 的 docstring `:11-18` 没有更新**，仍写着"SC2 … (paired bootstrap, shared resample indices)"以及族 "[SC1, SC2, SC5, SC6, SC7]"，与实现不符（`:358` 的注释是对的）。需要在 DL 里注明以实现为准。下一次修订时更正，修订就会改变 sha。

**变更**：
1. gate_b.json 写 `holm_family_gate_b = ["SC1","SC5","SC6","SC7"]`，标"暂定 m=4"。
   - `sc2_note`：SC2 的 NaN 是结构性的。IR 是矩的比值，代码从不为它算 p 值，只报各种子的值和符号。
2. Gate C 族 = SC1、SC4、SC5、SC6、SC7（m=5）。由单独的 Gate C 产物计算（新文件 `src/new_gate_c_holm.py`，GPT 写），在 CSI500 解封前冻结、算哈希，并写进 gates.md。sha 待生成。
3. 可选断言（新测试）：gate_b.json 里的 Holm 族 = `HOLM_KEYS` 去掉结构性 NaN 键（SC2）。

---

## DL-5（B6，已确认：方案 (b)，见"用户签字记录"）留出期由"决定性"改为"描述性"

**变更**：
- 2026-06-30 之后的数据只作描述性报告，最多看两次：
  - (i) 投稿时一次中期打分，报告天数、Delta 加 CI、各种子符号，标"中期，描述性"；
  - (ii) 满 120 个交易日时一次最终打分，无论结果如何都在修订稿或附录中报告。
- gate_b.json 把 `"holdout_minimum_days": 120`（草案在 `gate_b_report_en.md:147`、`gate_b_ready_2026-10-19.md:112`）替换为 `holdout` 区块：`{"role":"descriptive","looks":2,"interim":"at submission","final_min_trading_days":120,"dl_ref":"<本条 DL 的 commit>"}`。
- 2027-01-15 保留为硬上限，只是最晚期限，可以提前投稿。注明其理由已与留出期脱钩（`PREREGISTRATION_v0.6.md:293`、`gate_b_report_en.md:155` 的理由要改）。

**引用留出期、需要一并改标的位置**：
- `PREREGISTRATION_v0.6.md:28`（§0 规则 3，新措辞见 DL-11）
- `:258-262`（§9）
- `:286`（Gate D）
- `:293`（§13）
- `:380`（N9）
- `protocol/gates.md:34`、`paper/README.md:4`、`plan.md:21`
- S1–S3（§7，`:168-197`）、成功标准、Gate C（`:285`）都没有引用留出期。

**完整性断言：v2 仍有缺口**。v2 中没有 `max_date_touched ≤ 2026-06-30` 一类的断言（v2 全文检索不到 `max_date`；`2026-06-30` 只出现在 `:273` 的 docstring）。现状如下：
- 标签：`eval_ic` 已改为用 `data.loss_masks[split][t]`（v2 `:278`），越界的 4 天被排除（见下面的"eval_ic 越界"）。但没有任何断言把"实际访问到的最晚标签终点"打印出来，也没有对它做断言。
- 价格：执行回放终点是 t+2，软组合 `portfolio_events` 最远用到 t+2（trainer `a960a9951ab52c5455c15f8e0c5d60a783a6beb4beef5b4003ee578f0c21ee43`）。eval 最后一个信号日 2026-06-26 的 t+2 是 2026-06-30，所以按代码逻辑不越界，但 v2 里没有断言。对比 `src/rule_grid_neural.py:71-77` 有现成的 `ASSERT no-price-after-window` 写法可以照搬。
- 基准：`load_bench(BENCH_CSV)`（v2 `:308`）会把 H00300 整份读入（到 2026-09-01），只在行日期处取值（`:264`），同样没有断言。
- `Data(...)`（`:306`）把完整日历（到 2026-10-13）的特征、标签都载入内存。
- **处理（外部审稿第 4 轮已定）**：guard 写进 analysis v3 内部，只用一个哈希，见 DL-12。它检查**实际访问到的**日期：回放用的价格、经 eval mask 过滤后的标签终点、被采样的基准行。日志记 `max_date_touched`，晚于 2026-06-30 就直接失败。不另建独立脚本。

**eval_ic 越界（已确认；v2 已修正标签部分）**：
- **核查证据**：
  - 笔记本正式数据：labels `7bf5633b…`、splits `0ab69b33…`。
  - 笔记本 `D:\research\limt-repro\src`：`v2_w4_data.py` `89638e96c6b6f6c54f0bb3c07cad3c3413bbe17e82b8219bd80008da72914a4a`、`v2_w6_policy.py` `6b0cd7ee952a0eb5ebc89f15bd88db0eb1e9693802c06b27e15a12382d9ca8ba`，与 box 副本逐字节相同。
  - box 副本数据：labels `e15ed6f6…`。
  - 两份数据结论一致。
- **机制**：
  - `v2_calendar_labels.py:37-39,97` 在完整日历上算 h5，终点是 t+6；日历到 2026-10-13，最后一个有价格的日期也是 2026-10-13。
  - `v2_w4_data.py:72-76` 只裁 `loss_masks`；`:42` 原样读入 `targets`。
  - v1 `gate_b_analysis.py:250-259` 的 `eval_ic` 只做 `isfinite` 过滤。
- **越界的信号日**：eval 信号日共 841 个（2023-01-03..2026-06-26）。越界的是 2026-06-23 / 06-24 / 06-25 / 06-26，h5 终点分别是 07-01 / 07-02 / 07-03 / 07-06。这 4 天各 300 只候选股全部是真实标签，最多越界 4 个交易日。
  - 笔记本 splits 文件里这 4 天的 `loss_mask_eval` 是 0；06-22 及以前（终点 ≤ 06-30）是 300。所以 v2 改用 loss mask 后，这 4 天被正确排除。
- **受影响范围**：只有 SC7 的 5 日 IC 对比和 S1.3 的 IC 护栏。
- **附带问题**：`cs_z` 把掩码外的格子填 0（z_h5 全表没有 NaN），v1 会把无标签股票当作标签 0 算进 IC。v2 用的 loss mask ⊆ label_mask（`v2_w4_data.py:64` 有这条断言），所以一并修正了。

---

## DL-6（B4）TOST 等价界不动，预先承诺额外报告（指向 v2）

- 等价界 ±2.0 pp/年 不变（原 DL：`protocol/deviation_log.md:14`，commit `c8c6c2799f4ea60a9ab6228a3b283490a7b4c72b`）。v2 的实现在 `:136-138` 的 `tost()` 和 `:141-156` 的 `conclusion()`。
- 预先承诺报告：
  - 90% CI 的半宽，以及它占实测成本拖累（SC6 的年化 fee drag，两臂各自的值和差值）的比例；
  - 时间顺序的披露：等价界是在种子 1 的验证结果出来之后才定的（DL commit 在 2026-10-18 01:45:28 PDT，晚于 seed-1 H、tau 网格和剂量组）。
- CI 只重抽交易日（v2 `bootstrap_indices`/`infer`，对种子平均后的日序列做平稳自助），不含种子间方差。种子合并方式是日组合收益先在种子间平均（v2 `:350-351`），并始终报告各种子的 Delta 和符号。
- 自助法为主，NW(lag 5) 作为敏感性。

---

## DL-7（C7）TOST 进入结论的判定表：已在 v2 实现

**冻结的 v0.6 §7 原本没有规定**。`PREREGISTRATION_v0.6.md` 全文没有出现 TOST；§7（`:168-197`）只用 Delta 阈值、下界、种子符号、IC 护栏、CSI500、SC3、SC6。

**判定规则（定稿）**：
- 优先级：**优效 > 劣效 > 等价 > 不确定**。
- 劣效和等价可以同时成立（CI 落在 (−2, 0) 内），所以两个 flag 同时设置。
- 优效与 TOST 等价也可以同时成立，同样同时设 flag。
- "不确定"在任何文字里都不得写成"无效果"，应写成"既不能排除 0，也不能排除 ≥2pp 的效应"。
- 记号：LB = 单侧 5% 自助下界 = 90% CI 下端（同一次自助）；UB = 90% CI 上端。
- 定义：优效 = LB > 0；劣效 = UB < 0；等价 = TOST（−2 < LB 且 UB < 2）；不确定 = 以上都不成立。
- 进入 S1–S3 的方式不变：仍按 §7 的 S1→S2→S3 顺序机械判定。判定表只决定措辞，不改阈值。
  - 优效但 Delta < 2pp：不算 S1，措辞为"为正但低于预注册阈值"；
  - 劣效：S3 中"Delta is negative"那一条；
  - 等价：一般落到 S3，S3 措辞升级为"有界零：差异在 ±2pp 以内"（若 S2 某条触发，则 S2 优先）。

**v2 `conclusion()`（`:141-156`）与上述规则的核对**：
- 一致：
  - `verdict` 的优先级是 superior > inferior > equivalent > uncertain；
  - `flags` 中 `superior`、`inferior`、`tost_equivalent` 三者相互独立，劣效加等价、优效加等价都能同时报告；
  - 字符串用的是 "uncertain"，没有出现 "no effect"。
- 差异和说明（不影响判定，建议记入 DL）：
  1. `conclusion()` 只是 `result["conclusion"]` 的一个输出（`:371-373`），没有接进 `s_mapping`/S1–S3，也没有自动生成措辞。"不确定 ≠ 无效果"这条禁令目前只是文字规定，代码不检查。
  2. 自测 `:430-434` 各例都令 lower5 ≠ CI 下端（如 `conclusion(0.4, [-0.5, 1.0], …)`：lower5 = 0.4 而 CI 下端 = −0.5），两者自相矛盾（真实运行中两者相等）。不影响逻辑，但这组测试数据不具代表性。
  3. 优效用 `lower5_pp > 0`，劣效用 `ci90_pp[1] < 0`。真实运行中 `lower5 == ci90[0]`（v2 `infer`），所以与定义一致。

---

## DL-8（A2）三个 spec 哈希的管辖表（逐个运行核实）

| spec 版本 | 完整 sha256 | 来源 commit（PDT） |
|---|---|---|
| v1-a | `da5b742579bf8c5a15e049ec03dd59e0ba6f39da59bd6582b1e338b5a883ecff` | `6be7ae583d4a4afb9ef3c6149d8ec8253424f3a7`（2026-10-18 02:48:37） |
| v1-b | `e189f6cbe3453d172fc407b6bf2117be15bcba7c72c698f53f125d92533e5165` | `5a8b9a42d6c734e8f03eed77bf4de8c5d08c1399`（2026-10-18 19:15:55）至今 |
| v2 | 待生成（新文件） | 待提交 |

runner 在启动时读取 spec（`run_benchmarks.py:139`），所以某次运行归哪一版，以该进程的启动时间为准。启动时间取日志文件的 CreationTime，结束时间取 meta 和 npz 的 mtime，均为笔记本时间（PT）。meta.json 里没有 spec 哈希字段，这是缺口，v2 runner 要补上。

| tag | 启动日志（PT） | meta/npz 时间（PT） | n_features / feature_set | effective_seed | 归属 spec |
|---|---|---|---|---|---|
| XGBoost/DE 20 列首轮（已作废删除） | bench_xgb 02:35:47、bench_de 02:40:44 | — | 20 | — | v1-a |
| LSTM_seed1、GRU_seed1、ALSTM_seed1、Transformer_seed1 | bench_gpu_fix2 13:26:24 | 14:09:53 / 14:52:33 / 15:29:23 / 16:07:55 | meta 无该字段（旧 runner），按 FilterCol-20 | meta 无字段（Transformer = config 0） | v1-a |
| LSTM_seed2 | bench_seed2 18:50:57（早于 19:15:55） | 19:34:38 | meta 无字段（旧 runner） | meta 无字段 | **v1-a** |
| XGBoost_seed1（158 列） | bench_tree158_cache 19:16:05 | 19:17:06 | 158 / all158 | 0（nthread=20，fit 40 s） | v1-b |
| DoubleEnsemble_seed1 | bench_de158 19:29:18 | 19:39:23 | 158 / all158 | 1 | v1-b |
| GRU_seed2 / ALSTM_seed2 | parA/parB 20:13 | 21:10:17 / 21:11:38 | 20 / filter20 | 2 | v1-b |
| GRU_seed3 / LSTM_seed3 / ALSTM_seed3 | 21:10–21:54 | 21:53:39 / 21:55:42 / 22:33:47 | 20 / filter20 | 3 | v1-b |
| DoubleEnsemble_seed3 | parA_de_s3 21:55:46 | 22:05:05 | **158** / all158 | 3 | v1-b |
| DoubleEnsemble_seed2 | parB_de_s2_158 22:33:51 | 22:43:13 | **158** / all158 | 2 | v1-b |
| 新的 15 个 tag | — | — | — | 1–3 | v2 |

- **DE 种子 2–3 的 `n_features=158` 已核实**（meta 字段）。三种子规则选择的最终日志 `rule_grid_bench_final2.log` 是 22:45:16–22:57:22，晚于所有 158 列 DE/XGB 产物，与 `runs/csi300_bench_selection.json` 的 mtime 22:57:22 一致。所以 XGB（drop10-every5d）和 DE（drop5-every1d）的规则来自 158 列模型。没有打开 selection json 的 grid 数值。
- v1-a 和 v1-b 的 `models.*.kwargs` 完全相同（diff 已核），超参在两版之间一致。
- Rev2 §8（`gate_b_report_en.md:84`）里的 `da5b7425…` 是过期引用；中文草案 `gate_b_ready_2026-10-19.md:94` 也是。

---

## DL-9（A3/A6）哈希链补全与剂量组勘误

| 事件 | 完整 commit | PDT |
|---|---|---|
| 基线快照（仓库首个 commit） | `8a7a4c45493b562712177688c9d52118a51b3fb3` | 2026-10-18 01:45:27 |
| 首批 DL 六行 | `c8c6c2799f4ea60a9ab6228a3b283490a7b4c72b` | 01:45:28 |
| Gate A 填写 | `841f00becdee4df72a052c102392266d7b9bf9e2` | 01:54:08 |
| Gate A 记录 | `8820fa5bef4a3b33fa3babe6e67a1525e23322c5` | 01:54:51 |
| 工程修复 1 | `080a1c73e33058aaaae90ee640ffee93724655e4` | 12:57:35 |
| 工程修复 2 | `48a78d6be16223e6d6c472bfe4b7cbc4609a5737` | 13:26:24 |
| DL：benchmark 种子扩展 + 取证 | `9a578bb0ccee6d33be2198b31cad51adfee1e46f` | 18:50:27 |
| `--seed` 参数 | `d3a97b62de76e698bc996dc705f841c79d11decb` | 18:50:57 |
| DL：XGB/DE 改用 158 列 | `5a8b9a42d6c734e8f03eed77bf4de8c5d08c1399` | 19:15:55 |
| 冻结分析 v1（5742585a…） | `4f60ef436b271ab07b0c5d02fcf2f7c04a89c215` | 19:44:05 |
| 工程日志：158 列结果 | `9430acff46ddb0a7b416c91bdc8890a3e1d34db5` | 20:00:13 |
| 关闭 N8（提议） | `caee9164a65a6026c7fb7016f925441db556ab87` | 20:02:10 |
| 3 种子 benchmark 规则 | `30ff5389c8289cddb5d54c09699132b80c0ad9a1` | 23:03:10 |
| 同上（补填） | `bc0dc13d848c8c7beeddaf9d4e0bca5c3a11aca4` | 23:03:26 |
| 冻结分析 v2（7d90d218…）（笔记本，未推送） | `b7306e0d4b0d37c8456456264ed8c7a2a9eca10a` | 2026-10-19 00:53:04 |
| 本轮 DL、v2 spec、重跑、§8 重写 | 待提交 | — |

**剂量组勘误（采用的原文，commit 换成完整 sha）**：

> 仓库首个 commit（`8a7a4c45493b562712177688c9d52118a51b3fb3`，PDT 01:45:27）和首批 DL（`c8c6c2799f4ea60a9ab6228a3b283490a7b4c72b`，01:45:28）都晚于剂量组运行，其结果在同一 commit 中。"先记后跑"（v0.6 §8 第 215 行；gate_a_record"早于其覆盖的每次运行"）对剂量组不成立，予以撤回；剂量组始终是事后探索性分析。c8c6c279 中其余每行 DL 涉及的运行或分析都开始于 01:45:28 之后，由哈希链验证。

**逐行核对最后一句：有一行不成立。** 证据是笔记本 `D:\research\new_LiMT\logs` 和 `runs` 的文件 CreationTime/LastWriteTime（本地时区为 Pacific）以及 git 记录；文件时间不防篡改，只作佐证。

| c8c6c279 中的 DL 行 | 涉及的运行/分析，以及最早的开始证据（PT） | 是否晚于 2026-10-18 01:45:28 |
|---|---|---|
| ① 探索性剂量组 | `runs/csi300_dose003_seed1_20261018` 创建于 **2026-10-17 20:26:28**，最后写入 21:47:32 | 否（即上面的勘误） |
| ② benchmark 改定位（6 个 Qlib 默认臂） | 最早的 benchmark 日志 `bench_xgb_20261018.log` 创建于 02:35:47；代码首次出现在 `6be7ae58…`（02:48:37） | 是 |
| ③ 印花税敏感性 | 代码首次出现在 `4f60ef4…`（19:44:05）；基线 `8a7a4c45` 的 src/tests 中检索不到 stamp 相关代码；目前还没有在 eval 上运行过 | 是 |
| ④ TOST ±2pp | 同上，代码首次出现在 `4f60ef4…`（19:44:05）；目前没有在 eval 上运行过 | 是（但界值是在 seed-1 验证结果之后定的，见 DL-6） |
| ⑤ 规则选择种子 option (b)（seed 2–3） | `seed2_chain_20261018.log` 创建于 02:02:31；`runs/csi300_H_seed2_20261018` 创建于 02:04:03 | 是 |
| ⑥ 诊断记录（梯度分解、"H-FT 解释 94%"、剂量组激活成本项）及披露 | `grad_decomp_c0_tau050_seed1_20261018.log` 是 **2026-10-17 23:50:13 – 2026-10-18 00:13:27**；`hft_dose_score_20261018.log` 是 **10-17 19:13:35 – 10-18 00:20:55**；`runs/csi300_HFT_seed1_20261018` 创建于 **10-17 19:18:37** | **否**：这一行记录的是已经完成的诊断结果，属于事后披露，不是先登记 |

**最后一句换成外部审稿第 4 轮给出的措辞**（逐字采用，英文原文）：

> "Rows ②–⑤ concern runs that started after 01:45:28 PDT (② and ⑤, by git history and file timestamps, which are corroborative and not tamper-proof) or analyses not yet run (③ and ④). Row ⑥ discloses diagnostic analyses completed before the commit (2026-10-17 19:13 to 10-18 00:20 PDT). It is a disclosure, not a registration, and those analyses are exploratory."

审稿意见：第⑥行是对已完成分析的披露，不是登记。它从未声称"先记后跑"，所以不需要撤回，只需要归类。原句"由哈希链验证"说过头了：②⑤靠的是文件时间戳，③④还没有运行。

勘误全文因此是：剂量组那一段（上面引文的前两句）加上这段替换句。

另外，基线快照与 DL 相差 1 秒（01:45:27 / 01:45:28 PDT），两者都晚于 seed-1 和剂量组的结果。v0.6 §8 第 215 行和 `protocol/checks/gate_a_record_2026-10-18.json` 的相应说法需要一并改正。

---

## DL-10（分析产物 v2）

- **旧**：`src/gate_b_analysis.py` v1，sha256 `5742585ac41a9a291b0eaefed0086cc337220ad9cdf73aee290f16d3430325b6`，commit `4f60ef436b271ab07b0c5d02fcf2f7c04a89c215`（2026-10-18 19:44:05 PDT）。
- **新**：同一路径 v2，sha256 `7d90d218e661a242512e542e309ea49e4858c428da6f2ea85018ef22e6db64fa`，commit `b7306e0d4b0d37c8456456264ed8c7a2a9eca10a`（2026-10-19 00:53:04 PDT；在笔记本上，未推送）。
- **Bug 修复**：`eval_ic` 改用 `data.loss_masks["eval"]`（v2 `:268-284`）。v1 有两个问题：(a) 最后 4 个 eval 信号日（06-23..06-26）的 h5 标签读到了 2026-06-30 之后的价格；(b) 把无标签股票当成标签 0（`cs_z` 填 0）算进 IC。影响 SC7 的 IC 和 S1.3 的护栏。
- **解封前的协议改动**（不是 bug）：
  - SC5 的 Holm p 值从"相关系数换算"改为平稳自助法（`sc5_pvalue`，`:287-297`）。预先指定 C0 臂进 Holm 族，两臂都报告。
  - 新增 `conclusion()` 四态判定（`:141-156`）。
- **SC5 p 值的实现与问题**：
  - 构造方式：`p = 2 · min(P*(ρ* ≤ 0), P*(ρ* ≥ 0))`，ρ* 是在种子平均后的配对日序列上做平稳自助得到的 Spearman 系数。这是**反推百分位 CI**（看 0 落在自助分布的哪个位置）的双侧 p，**不是**以原假设为中心的零分布。
  - **方向不一致（标出）**：SC5 的科学假设有方向（RQ4：软层增益能传导到执行层，即相关为正）。族内其他成员都用单侧 `p_one_sided_pos = P*(mean* ≤ 0)`（SC1、SC6、SC7，`:368-370`）。为了一致，SC5 应改为单侧 `P*(ρ* ≤ 0)`。v2 用的是双侧，当前偏保守，但与族内口径不一致。**外部审稿第 4 轮已定：在 v3 中改为单侧，见 DL-12。**
- **SC6 的符号（外部审稿第 4 轮确认是 bug，在 v3 中修复，见 DL-12）**。
  - v2 `:336` 定义 `drag = c_fee - g_fee`，`:360` 对它做 `infer`，Holm 用的是 `p_one_sided_pos`（检验均值 > 0，也就是 C0 的费用**高于** G0）。
  - 预注册 §3（`PREREGISTRATION_v0.6.md:70-72`）把 Delta_cost 定义为 "G0\* 与 C0\* 年化成本拖累之差"，并给出恒等式 Delta = Delta_gross + Delta_cost。
  - 代码里 `delta = c_ex − g_ex`，`gross = (c_net+c_fee) − (g_net+g_fee)`，所以 `delta = gross − drag`，即恒等式要求 **Delta_cost = −drag = g_fee − c_fee**。
  - 同时，S1 "supported" 要求 "SC6 lower bound > 0"，S2 写的是 "Delta_cost positive … costs fell"。照现在的写法，`SC6_delta_cost_pp`、`SC6_ci90_pp` 和 SC6 的 Holm p 都是反号的。
  - 修复方式：把 `drag` 改为 `g_fee − c_fee`，或者在报告层取负号并改用 `P*(mean* ≥ 0)`。修复要新 sha 并补一行 DL。这一点 v1 也一样。
- 声明：v1、v2 都没有以 `--run` 运行过；没有人看过任何评估窗口（2023-01 之后）的数值。v2 是在解封前修订的。
- gate_b.json 的 `analysis_sha256` 指向 **v3**，草稿里暂写"待 v3 sha"（见 DL-12）。v2 不会被运行。
- Gate C 的 Holm 产物（DL-4）待冻结。

---

## DL-11（§0 规则 3 改写，英文原文）

> Rule 3 (revised by DL-11, 2026-10-19, before any clean-pipeline evaluation output was opened). No window in this study is fully confirmatory. All decisions (S1–S3, the SC tests and the TOST verdict) are made mechanically by the frozen analysis artefact on the 2023-01-01 to 2026-06-30 CSI300 window, and on CSI500 at Gate C. Because this window was exposed before registration (rule 2), these are reported as preregistered analyses on a previously exposed window, not as confirmatory tests. The post-cutoff sample after 2026-06-30, which no version of any model or analysis has touched, is reported descriptively per DL-5: one interim look at submission and one final look at 120 or more trading days, reported regardless of outcome. The paper uses 'preregistered', never 'confirmatory'.

注意：原文里写的是 "DL-xx"，这里填成 DL-11。"which no version of any model or analysis has touched" 这句，在 v1 `eval_ic` 的 bug 修正之后才成立：v1 从未运行过，所以确实没有人碰过留出期的数据。DL-10 已说明这一点。

---

## DL-12（分析产物 v2 → v3）

- **旧**：`src/gate_b_analysis.py` v2，sha256 `7d90d218e661a242512e542e309ea49e4858c428da6f2ea85018ef22e6db64fa`，commit `b7306e0d4b0d37c8456456264ed8c7a2a9eca10a`。
- **新**：v3，文件 `src/new_gate_b_analysis_v3.py`（由 GPT 编写），sha256 和 commit **待 v3 sha**。
- 这是唯一一次 v2→v3 修订，下面 5 项合并在这一版里。需求和验收标准见 `/workspace/uploads/new_GPT_code_spec_gate_b.md` 第 9 节。
- gate_b.json 的 `analysis_sha256` 指向 v3（当前写"待 v3 sha"）。

1. **SC6 符号修复（bug 修复）**
   - **预注册依据（只读核实）**，`PREREGISTRATION_v0.6.md` @ bc0dc13：
     - `:70`："Delta_cost: the difference in annual cost drag … between G0* and C0*"；
     - `:72`："Delta = Delta_gross + Delta_cost (accounting identity)"；
     - `:176`：S1 supported 要求 "SC6 (Delta_cost) lower bound > 0"；
     - `:189`（S2）："Delta_cost (SC6) is positive with lower bound > 0 but Delta_net is not: costs fell but gross alpha fell with them"。
     - 结论：预注册的方向是成本下降为正，Δ_cost = g_fee − c_fee。
   - **推导**：net = gross − fee，所以 Δ = (gross_C0 − fee_C0) − (gross_G0 − fee_G0) = Δ_gross + (g_fee − c_fee) = Δ_gross + Δ_cost。代码中 `delta = gross − drag`（drag = c_fee − g_fee）是对的，只有 SC6 反了：v2 检验的是"C0 费用更高"。
   - **v3 的改法**：
     - 定义 `dcost = g_fee − c_fee`，单侧检验 P*(mean* ≤ 0)；
     - v2 中所有用到 drag 的地方都已改，即 `:14`（docstring）、`:336`、`:350-351`、`:360`、`:370`、`:391-392`；
     - `s_mapping`（`:170-178`）本身不读 Δ_cost，v3 新增并列的 `s2_conditions`，用修复后的 dcost；
     - 新增两个自测：(a) 合成数据中 C0 费用更低，SC6 必须 > 0；(b) Δ = Δ_gross + Δ_cost，误差 < 1e-12。
   - **披露**：
     - 修复时没有看过任何评估数据；
     - v1 有同样的 bug；
     - 论文中 Rev2 验证图的 "+0.019 pp/yr" 要明确标注为 c_fee − g_fee，即 Δ_cost = −0.019。
2. **SC5 改为单侧 P*(ρ* ≤ 0)**
   - 与 SC1、SC6、SC7 的构造一致，决定在看到任何数据之前做出。
   - 披露 v2 用的是双侧 `2·min(P*(ρ*≤0), P*(ρ*≥0))`，v3 保留该字段仅作披露。
   - **前提核实（只读）**：
     - `:69` 的 SC5 原文 "Rank correlation of arms/models across soft and TopkDropout layers." 本身没有写方向；
     - `:37` RQ4 "Does a soft-portfolio gain transfer to TopkDropout with full frictions? (rank correlation across layers)" 是有方向的问题，transfer 对应正相关。
     - 审稿的条件是 "provided its registered hypothesis is directional (positive association)"。这里方向只由 RQ4 隐含，**需要用户或审稿明确确认后才能生效**；未确认时 v3 保留双侧。
     - 另需披露：`:69` 说的是 arms/models 层面的排名相关，而 v2/v3 计算的是 C0 单臂软层与执行层日序列的 Spearman。口径不一致，算法不改，只做披露。
3. **日期 guard 写在 v3 内部**（取代 DL-5 中"另写 guard"的设想）
   - 检查实际访问到的日期：回放价格，包括 terminal 的 t+2；经 `loss_masks["eval"]` 过滤后的 h5 标签终点 t+6；被采样的 H00300 行。
   - 日志写 `max_date_touched`，晚于 2026-06-30 就直接失败，日期同时写进输出 json。
4. **修正 v2 docstring `:11-18`**
   - SC2 不进 Holm；
   - 族 = [SC1, SC5, SC6, SC7]；
   - SC5 单侧，SC6 用 dcost；
   - 写明 guard；
   - SC4 在 Gate C 进族。
5. **修正 v2 自测 `:430-434`**
   - 改用自洽的输入（lower5 == ci90[0]），并通过真实的 `tost()` 计算；
   - 四态各至少一例，另加"优效 + 等价""劣效 + 等价"两种同时成立的情形。
6. **conclusion()**
   - 不接进 `s_mapping`。DL-7 规定它只决定措辞，审稿认为这样可以接受。
   - 在输出 json 中与 `s1_conditions`、`s2_conditions` 并列，并写明 `conclusion_note`：两者是有意分开的。
- **声明**：v1、v2、v3 都没有以 `--run` 运行过，也没有看过任何评估窗口的数值。

## DL-13（算力豁免行，须在 T3 前提交）

外部审稿第 4 轮原文，逐字采用：

> "Reruns mandated by DL-1/DL-2 are exempt from the 60 GPU-h cap on any device, because their length is fixed by published defaults and cannot be truncated. Actual device-hours are recorded. The cap continues to bind all discretionary compute, including the CSI500 fine-tunes."

- **理由**：这个上限是为了约束自选实验。DL-1/DL-2 的重跑是审稿强制要求的，epoch 上限由公开默认值确定。审稿明确不按 CPU/GPU 设备来裁定，因为在已知最坏情况超限之后再排除 CPU 小时，看起来像是事后挪球门。
- NN 组在 GPU 上跑；先实测 LSTM 种子 1。
- 每次运行的 meta 记录 `device_hours`。

## DL-14（数据标识勘误）

- **正式输入**（CSI300，目录 `D:\research\new_LiMT\data\csi300_w1w2\data\v2\`）：

  | 文件 | sha256 |
  |---|---|
  | panel `new_v2_panel_csi300.npz` | `a25879f4b94e60b6224bf52ec777d25e4f4511af4be3591711647992b601a62e` |
  | labels `new_v2_labels_csi300.npz` | `7bf5633b7ad9d146e5154a120307d8dbf771a9d5a5c9852f9eb7aba0bd1064a5` |
  | splits `new_v2_splits_csi300.npz` | `0ab69b33d0c8e76c020ebd100791eaf5535cac0e4c4c075207b03890baec17ad` |

  - **sha 的来源**：2026-10-19 在笔记本上只读计算。用的是训练代码自己的 `sha256_file`，从 `src/run_seed_chain.py:43-48` 用 ast 抽取函数原文后执行，Python 环境是 limt-repro `.venv`。该目录下没有 manifest json。
  - 结果与此前 Get-FileHash 的结果、以及预注册头部（`PREREGISTRATION_v0.6.md:6`）的前缀一致。
- 预注册中的 "fix1"（`:6` "fix1 labels sha 7bf5633b..."）指的就是这三份文件。
- `e15ed6f6…`（labels）和 `cb774f59…`（panel）是魔搭重建版（manifest `28bb5b75…`）。在已做的检查上与正式输入一致：日历、形状、掩码计数、±3 截断、eval 越界 4 天。但**它们不是正式输入**。
- **data_version 重算**：
  - 方法：用训练代码自己的 `panel_version`（`src/run_seed_chain.py:51-58`，依次哈希 `ARTIFACTS` = labels、panel、splits，见 `:35`），对上述正式输入重算，只读，没有改任何文件；
  - 结果：`3c70f66384b8a954d18540a3b129da2103d13b04629897d5204e6292d97f7379`；
  - **与所有 checkpoint meta 的锚点 3c70f663…（例如 `protocol/checks/n5_neff_validation.json:7`）完全一致**。
  - 因此这一项不阻塞 gate_b.json。

## 算力预算（口径见 DL-13）

- **实验机器（已定，2026-10-19 改定）**：魔搭免费 GPU 环境，NVIDIA A10 24 GB（驱动 550.54.15），8 核 CPU（Intel Xeon Platinum 8369B），约 28–32 GB 内存，镜像 ubuntu22.04-cuda13.0.3-py312-torch2.13.0（torch 2.13.0+cu130，CUDA 13.0，Python 3.12.13）。原定的 MI300X 和 55 GB AMD 选项因资源紧张无法启动。剩余免费额度约 11 小时；不够时，未开跑的 tag 整个转到 Kaggle 或 Colab，单个 tag 不拆跨平台。用户要求资源拉满、尽量并行（并行方案见运行计划）。
- **单次运行的最坏情况**（来自笔记本 GPU 实测的每 epoch 秒数，换到 A10 后需要重新实测）：
  - LSTM/GRU/ALSTM：200 epoch × 约 225 s ≈ 12.5 h/次，9 次约 110 h；
  - Transformer：100 epoch × 约 318 s ≈ 8.8 h/次，3 次约 26.5 h；
  - XGB 可以忽略；
  - 串行合计最坏约 137 h，并行后墙钟时间会缩短，但受 8 vCPU 限制。
  - 早停通常会大幅缩短运行时间，但 CSRankNorm 会改变早停点。
- **流程**：先单独实测 LSTM 种子 1，得到每 epoch 秒数、实际 epoch 数、CPU/GPU 占用，再推算并行度。
- **不得砍 epoch 上限**。
- **60 GPU-h**：DL-1/DL-2 强制要求的重跑按 DL-13 豁免，但实际机时（`device_hours`）逐次记录。上限继续约束所有自选计算，包括 CSI500 微调。

## 平台（已定）

- 魔搭 A10 实例。15 个新 tag 优先全部在这台机器上跑；额度不够时按上面的规则整 tag 外溢，并逐 tag 记录平台。
- 外部审稿原本倾向笔记本 GPU，用户已定为魔搭。
- **要披露**：
  - 旧的 z-score 对照（以及种子 0 复现）来自笔记本 NVIDIA GPU / CUDA 11.8，新的 15 个 tag 来自 A10 / CUDA 13.0，结果不会逐位一致；
  - 同一种子在 CUDA 上重跑也不保证逐位一致，因为 cuDNN 的 RNN 和 atomic 运算不确定。
- **每次运行记录**：Python、pyqlib、xgboost、torch、CUDA 版本，GPU 型号（A10），XGB 的 nthread，`torch.get_num_threads()`，并发任务数。

## 其他（与本轮 DL 一起提交）

- **B7 MASTER**：不进基准（用户已确认）。理由收窄为"Qlib v0.9.6 无内建 MASTER 实现及官方 benchmark 配置"（v0.9.6 的 `examples/benchmarks/` 目录里没有 MASTER，已核）。论文中要有一句交代。
- **README.md** 仍写着只服务 v0.5，需要更新。
- **gate_b.json 草案里的占位符**：`<HEAD at write time>`（`gate_b_report_en.md:122`、`gate_b_ready_2026-10-19.md:95`）以及各种截断哈希，都要在写入时换成完整值。拼写错误 `execution_feed_resselt`（`gate_b_ready_2026-10-19.md:93`）也要改。
- **数据版本命名**：预注册头部写的 "fix1 labels sha 7bf5633b…" 是笔记本 new_LiMT 的这份重建版；魔搭 fix1 版是 `e15ed6f6…`。两者掩码计数等元数据一致，但不是同一个文件。§8 和数据版本记录里要说明正式运行用的是 7bf5633b / a25879f4 / 0ab69b33。已由 DL-14 勘误，data_version 已重算且一致。

---

## 用户签字记录

- 时间：2026-10-19 00:41 PDT
- 签字人：beelingz
- 本人原话："我已阅读 DL-3、DL-12～15，同意按此执行。beelingz，2026-10-19"（2026-10-19 04:37 PDT 在对话中选定默认措辞）
- 确认内容：
  1. Gate A（N8）：同意硬上限 2027-01-15（与留出期脱钩，只是最晚期限，可以提前投稿），同意 60 GPU 小时上限。对应 commit `caee9164a65a6026c7fb7016f925441db556ab87` 中的提议。`PREREGISTRATION_v0.6.md:293` 的理由一并改为"与留出期脱钩"。
  2. 留出期（B6）按方案 (b)：改为描述性，不决定成败；投稿时报告一次中期结果，满 120 个交易日报告一次最终结果（修订稿或附录）；最多看两次。
  3. B2 选 (a)。
  4. MASTER 不进基准。
- 60 GPU-h 的计入口径：按 DL-13 处理（豁免 DL-1/DL-2 强制重跑，待提交）。运行平台：原定魔搭 MI300X，因资源紧张，用户于 2026-10-19 改定为魔搭免费 GPU 环境 NVIDIA A10 24 GB（CUDA 13.0、torch 2.13.0+cu130）；T1 已于 2026-10-19 04:31 PDT 在 A10 上通过。
- 已定（2026-10-19 03:25 PDT，beelingz 在对话中确认）：SC5 假设为正相关，v3 使用单侧 P*(ρ* ≤ 0)（DL-12 第 2 项生效）。注意 :69 估计量（跨臂排序相关）与实现（C0 软层/执行层日度 Spearman）不一致，已在 v3 docstring 披露，算法未改。

## DL-15（草稿）CSRankNorm 排名基准的 float32 并列例外
- 上游先用 float64 原始收益算出 z_r1，再把 raw_r1 和 z_r1 分别存成 float32。逐日仿射单调变换在 float64 下保序，但两者各自舍入到 float32 后，可能一边出现并列、另一边没有，于是规范里"z_r1 排名与存档 raw_r1 排名逐位相同"这条验收在部分切片上不成立。
- 处理：仍按规范用 z_r1 排名（与基准实际使用的标签一致），不改算法、不改数据。
- 验收改为：(1) 只要有一处排名不同，对应的股票在 raw_r1 或 z_r1 的某一边必须属于 float32 并列组，否则判 FAIL；(2) 报告受影响的天数、股票-日数、最大排名差，以及与 raw_r1 排名的逐日 Spearman 最小值；(3) 排名差不得超过所在并列组的大小。
- 2026-10-19 03:28 PDT 记录，排名之前写入，没有看任何 2023 年以后的指标。

- DL-3 批准记录：beelingz 于 2026-10-19 04:23 PDT 在对话中确认 DL-3（Alpha158 默认配置及 0.9.6/0.9.8.dev34 版本差异）。v2 规格 configs/new_benchmarks_qlib_defaults_v2.json（sha 49010f9b…）第 204 行 "FROZEN: DL-3 approved by user" 与此一致。
