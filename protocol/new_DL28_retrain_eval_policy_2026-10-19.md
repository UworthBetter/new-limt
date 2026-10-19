# DL-28：缺失模型的重训与评估期打分政策（草稿 r13，开封前）

实验助手，2026-10-19，时间均为 PDT。新文件，取代 `new_DL28_retrain_eval_policy_r12_draft.md`（edcf2152…，保留不改）。r13 按 Claude 第 10 轮修改：第 16 条的 v2 spec 行尾改为按代码判定（CRLF 49010f9b）；XGB 改在 A10 实例的 CPU 上重训；新增第 17 条。r12 的说明：r12 取代 r11（3efd716d…）。r12 新增第 16 条（h5 冒烟产物的日期核对、v2 spec 的行尾、h5 平台）。下面是 r11 的说明，r11 取代 r10（8308174e…）。r11 按 Claude 08:30 PDT 的决定修改：第 9 条把 LGBM 的分歧标为已解决（RESOLVED），并补记 lightgbm 的当前安装版本；新增第 14 条（输入上传）和第 15 条（时间线）。r10 取代的是 `new_DL28_retrain_eval_policy_r9_draft.md`（a82ae2a9…）。第 1–5 条沿用 r8，第 6–12 条是 r9 新增，第 8 条和第 9 条在 r10 按 Claude 08:25 PDT 的批准改写，第 13 条是 r10 新增。**DE 的溯源结论放在本条（DL-28）第 8 条，不放 DL-26**：DL-26 只管 v2 spec 的溯源，DE 用的是 v1 spec 和 v1 runner，它的溯源只对本条的重训有意义。内容由外部审阅（Claude）第 8 轮、08:04 PDT 的回复和第 9 轮（`/workspace/uploads/claude_round9_reply.md`）提出；PI 批准 [待填]。配套规格：`/workspace/uploads/new_GPT_spec_h5_benchmarks_r9.md`（§2.6、§10、§11、§11a）、`/workspace/uploads/new_GPT_spec_v3_1d_guard_fix.md`（§1、§7、§8）。

## 中文

1. **确定性臂**（ridge；XGB，CPU hist，显式种子；LightGBM，固定线程数）：按冻结 spec 重训并存模型，只重算验证期和 2021–22 的分数，与现有文件比对。
   - 容差事先声明：NaN 位置一致，且 `max|Δscore| < 1e-6`。
   - 比对一致：保留已选规则，记下模型 sha。
   - 比对不一致：当作非确定性臂处理。
2. **非确定性核心臂**：重训，存 checkpoint，重算验证期和 2021–22，重选规则，替换该臂，加一条 DL 行。全部在写 gate_b.json 之前完成。
3. **对照臂**（z-score 标签、seed-0）：只报 2021–22，标注"无评估窗口结果（未存模型）"。
4. **评估期打分只在开封那一步做**，用 `src/new_bench_eval_v1.py`，或其他 import `src/new_guarded_data_v1.py` 的冻结打分脚本。所有 checkpoint 和模型的 sha 都写进 gate_b.json，开封时逐一校验。
5. **开封前，任何臂都不出评估期预测**，包括对重训模型"试打分"。打分脚本只用合成数据或验证期数据测试。
6. **NN checkpoint 核对结果（r9）**
   - 08:09:40–08:10:39，在魔搭实例上用 `new_nn_cache_check_v1.sh`（4e6a7dbf…）只读核对。脚本不加载任何模型，也不打分。
   - 12 个 tag 全部 PASS：`.pt` 的 sha256 与 `.pt.meta.json` 里的 `artifact_sha256` 一致。caches 下正好有 12 个 `.pt`。
   - 已上传到数据集 `new_LiMT/nn_ckpt_20261019/`，共 27 个文件，SHA256SUMS fb66a1fb…。
   - 盒子端只下载了 SHA256SUMS 和核对表，没下 `.pt`。复核结果：26 个文件的 sha 与服务端记录逐一相同。
   - **12 个 NN tag 都不用重训**，臂不变；12 个 sha 写进 gate_b.json。完整 sha 见 `new_arm_model_inventory_r9.md`。
7. **ridge 比对脚本（r9 冻结）**
   - 脚本：`/workspace/limt_gateb/new_compare_ridge_daily_v1.py`，sha256 07f4e1214b619468827bd44002b46cdf5c2c6f0f1b837e7b6faea9528349a9ca。
   - 用法：每组新旧 2021–22 daily.csv 先比 sha256，相同就判一致。sha 不同时，按以下顺序比：表头、行数、非数值格、NaN 位置，最后看数值格的 `max|Δ|` 是否小于 1e-6。
   - stdout 只打印一个布尔值 `True`/`False`，不打印任何数值、差值、列名或行。
   - 任一单元格出现 2023 年或以后的日期，就拒绝比对，打印 `False`，退出码 3。
   - 容差 1e-6 写死在代码里，不能用参数覆盖。可选的 `--record` 只记录脚本 sha、容差、文件 sha、行数和判定结果。
   - 合成自测通过：相同文件、差 4e-7、相同 sha 都判 True；差 2e-6、行数不同、NaN 换成数值都判 False；日期在 2023 年的拒绝。
8. **DE 训练脚本溯源（r10，已定，属推断）**
   - 3 个 DE meta 里任何层级都**没有** `script_sha256`。v1 runner `src/run_benchmarks.py:219-225` 写 meta 时本来就不记脚本哈希。
   - 08:25 PDT 经 Claude 批准，只读了 `n_features` 这一个键：DoubleEnsemble seed1、seed2、seed3 都是 **158**。
   - 5a8b9a4（2026-10-18 19:15:55 PDT）把 XGBoost 和 DE 的特征集改成了全部 158 列，之前是 filter20。所以这 3 个 DE 运行用的是 5a8b9a4 或之后的代码。
   - `src/run_benchmarks.py` 在 5a8b9a4 之后没有再提交过，工作区与提交一致，sha 是 824daa85536b85e81c28381ebc5b5b4474cd6506dccb03cc500e37e15e5daa8a。3 个 DE 分数文件写于 19:39:23、22:05:05、22:43:13 PDT，都在这次提交之后。
   - **结论（推断）**：DE 是用 `src/run_benchmarks.py` 824daa85 加 spec v1-b `configs/benchmarks_qlib_defaults_v1.json` e189f6cb 训练的。这个结论排除了 5a8b9a4 之前的 filter20 代码，但并没有哈希能直接证明，所以记为推断，不是哈希核实。
   - 重训用这一版代码和 spec（h5 r9 §10）。
9. **平台与指纹（r10，Claude 08:25 PDT 批准）**
   - **（r13 改）** XGB 在**魔搭 A10 实例的主机 CPU** 上重训（见第 17 条），预期可能 MATCH；DE 和 ridge 在**魔搭 CPU 实例**上重训，预期 MISMATCH。新产物替换旧产物，规则按第 10 条重选，**原产物和它们的 2021–22 结果保留**。容差不变，仍是 1e-6。
   - h5 在**魔搭 A10** 上跑，和 15 个 tag 同一平台。CPU 作业不计 GPU 小时。
   - **lgbm 不重训**：12 个 `model.txt` 已经存在，只记 sha。Claude 08:25 的消息曾把 LGBM 列进重训名单。**已解决（RESOLVED，Claude 08:30 PDT）**：那是 Claude 自己写错了，以 r9/r10 为准，LGBM 只记 sha，不重训。
   - 平台指纹，原平台与计划主机对照：

| 臂 | 原平台 | 计划主机 |
|---|---|---|
| XGBoost-seeded s1–3 | 魔搭 A10 实例主机：Intel Xeon Platinum 8369B，8 核，约 28–32 GB 内存；镜像 ubuntu22.04-cuda13.0.3-py312-torch2.13.0；Python 3.12.13，torch 2.13.0+cu130，pyqlib 0.9.6（来源：`new_DL_draft_pre_gate_b.md:411`，`verify_production_platform` `:120-133`）；**xgboost 3.4.1**（meta 键 `/versions/xgboost`，3 个种子一致；只读了这一个键）；nthread 20 | 魔搭 DSW-CPU：64 nproc，28 GB；CPU 型号、操作系统和库版本**待指纹脚本记录**（`new_cpu_instance_env_plan_r10.md`） |
| DoubleEnsemble s1–3 | 笔记本：AMD Ryzen 7 7735H，8 核 16 线程，Windows 11（10.0.26200）（现在的硬件，用系统信息查的）；运行时的 Python 和 lightgbm 版本**没有记录**：DE meta 里没有任何版本键（键名都列过，没有看值） | 同上 |
| ridge h5、r1 | 同一台笔记本；`ridge_{h5,r1}_grid.json` 里没有版本键，只有规则网格字段 | 同上 |
| lgbm（不重训） | 同一台笔记本；`grid.json` 里没有版本键 | — |

   - **lightgbm：只有当前安装版本，运行时版本没有记录。** 08:30 PDT 经批准，在笔记本上用一条命令只读查了各解释器，查的是 `importlib.metadata.version`，与 `pip show` 等价：
     - `D:\research\new_LiMT\venv_bench`：Python 3.13.9，**lightgbm 4.7.0**，pyqlib 0.9.6，xgboost 3.4.1。它是唯一同时装了 pyqlib 0.9.6 的环境，位置在仓库内，**很可能就是跑 DE（v1 runner 需要 qlib）的环境，这一点是推断**。
     - `D:\research\limt-repro\.venv`：Python 3.13.9，lightgbm 4.7.0，pyqlib 0.9.8.dev34。可能是跑 lgbm 和 ridge 的环境（`cpu_baselines.py`），也是推断。
     - 其他装了 lightgbm 的：PATH 上的 hermes venv（3.11.15，4.7.0），系统 Python313（3.13.9，4.7.0），`D:\research\dtaf-venv`（3.11.15，4.7.0）。
     - 结论：各环境当前装的都是 lightgbm 4.7.0。**这是当前安装版本，运行时版本没有记录**，只作记录，不作约束。DE 本来就预期 MISMATCH。
   - 新主机的指纹写进 `new_cpu_host_fingerprint.json`，它的 sha 写进每个重训产物的 meta 和本条。
10. **MISMATCH 臂的规则重选和取证（第 9 轮第 3 条）**：
    - v2 tag（XGB、DE）用 v2 规则网格和取证的参数化副本（`new_rule_grid_bench_v2p.py`、`new_bench_forensics_v2p.py`）。
    - CPU 臂（ridge）用 `cpu_baselines.py`（4c585a0f）的规则选择逻辑（`new_cpu_reselect_v1.py`）。
    - 结果都写进新文件，只打印规则名。每个被替换的臂各加一条 DL 行，写明新旧 sha、新规则名、平台指纹和 `expected_verdict`。
11. **防护模块与截止日（第 9 轮第 1、2 条）**：
    - 开封前的所有入口（h5 runner、重训、比对、规则重选、取证）都用 `PRE_UNSEAL_CUTOFF = 2022-12-31`，开封时用 `UNSEAL_CUTOFF = 2026-06-30`。截止日由入口在代码里选定，不从命令行或 JSON 读。
    - `new_guarded_data_v1.py` 必须在任何重训之前冻结并 commit。runner 拿它的 sha 和已提交的 `configs/new_guarded_data_v1.sha256` 比对。
    - 模块以后若有改动，必须另起一条 DL 行，列出它防护过的重训和 h5 产物。
12. **开封的失败处理（第 9 轮第 8 条）**：
    - 开封只做一次：先跑 v3.1，再跑 bench_eval。
    - 任何一步失败就停下，记录并提交日志。要重跑，先写 DL 行。
    - bench_eval 失败不影响 v3.1 的产出。

13. **meta 键读取记录（r10）**
    - DE meta：读了 `n_features` 一个键的值，并递归列出了键名（不含值）。
    - XGB meta（盒子上的 `new_allowed_members` 副本，sha 881f6eb4…、a1cb572e…、30fce705…）：递归列出了键名（不含值），然后只读了 `/versions/xgboost` 的值。XGB meta 里没有 lightgbm 键。
    - lgbm 和 ridge 的 `grid.json`、`csi300_cpu/selection.json`：只列了键名，没有读值。
    - 其他键一律没读。

14. **重训输入上传（r11，Claude 08:30 PDT 批准）**
    - 上传内容：`src/run_benchmarks.py`（824daa85）、`src/cpu_baselines.py`（4c585a0f），以及 16 个 ridge 规则回放的 `daily.csv`（`runs/csi300/ridge/{h5,r1}/*/seed1/`）。
    - 上传位置：私有数据集 `new_LiMT/cpu_retrain_inputs_20261019/`，附 SHA256SUMS（9218a988…）。
    - **没有上传** `runs/_sealed/` 下的同名文件。
    - daily.csv 只算哈希，**从来没有打开**。这 16 个文件属于 2021–22，是根据目录约定（另有 `_sealed` 副本）推断的，没有看内容核实。比对脚本遇到 2023 年及以后的日期会拒绝，这是兜底。
    - 下载核对：17/18 字节一致。`run_benchmarks.py` 被魔搭规范化成了 LF 版（60eac60a…），还原成 CRLF 后等于 824daa85。
    - 详表见 `new_cpu_retrain_inputs_upload_r11.md`。
15. **时间线（r11）**
    - 08:28 PDT 只读查了笔记本仓库：`main...origin/main [ahead 2]`。没有 push 的是 `b7306e0`（00:53:04 PDT，分析 v2）和 `7e0ffbd7`（04:40:11 PDT，DL pre-Gate-B）。
    - 这和 DL-16 的记载一致：两者都在开封步骤 7 推送，推送时间还空着待填。在这之前，任何"DL 已推送"的说法都不成立。
    - 08:29–08:36 PDT，GPT 的 r9 交付文件陆续写进笔记本 `src/`（防护模块、v3.1、h5 和重训相关脚本），都还没有跟踪，也还没经实验助手核对。

16. **时间线（r12，09:15 PDT 按 Claude 批准只读核对）**
    - **h5 冒烟产物**：笔记本 `runs\new_r9_smoke_scores\` 下有 LSTM-h5default 和 Transformer-h5default 两个 seed1 的 meta.json 和 .npz，`runs\new_r9_smoke_models\` 下有两个 .pt，`new_r9_smoke_models` 里没有 meta。
      - 写入时间（mtime）在 08:41:03–08:41:37 PT。
      - meta 里的 `started_at` 是 08:40:49 PT，`ended_at` 是 08:41:04（LSTM）和 08:41:29（Transformer）。
      - 时间范围整体为 08:40:49–08:41:37 PT。
    - **来源**：GPT r9 交付时的自检进程（`new_r9_delivery_check.py`、pytest、`--smoke`），**这是 Claude 的说法，实验助手没有打开 argv 核实**。
    - **数据日期**：只读了日期相关的键，没打开 score、npz 或其他键。
      - `guard_cutoff_name` 是 PRE_UNSEAL_CUTOFF，`guard_cutoff_value` 是 2022-12-31；
      - `max_date_touched`：features 是 2018-09-05，labels 是 2018-07-06，prices、marks 和 bench 都没有访问过（None）。两个 meta 相同。
      - **没有任何日期落在 2023-01-01 或之后**。
    - **怎么看待**：这是开发自检，不是 DL-24 的运行，不能用来选配置或作任何结论；DL-24 仍然等 beelingz 本人签字。
    - **（r13 改）v2 spec 的行尾：按代码判定，A10 读到的是 CRLF 原字节 49010f9b**。r13 的结论，取代 r12 里“推断用的是 LF 版 387af011”的说法。
      - 依据 1：`src/new_gate_b_common.py`（12632786）第 19–24 行 `sha256_file`：`with Path(path).open('rb') as stream: ... h.update(block)`，对文件的原始字节计算哈希，没有做任何规范化。
      - 依据 2：同一文件第 106–108 行 `Provenance.__init__`：`for key in ('spec', ...): self.hashes[key] = {'path': str(path), 'sha256': sha256_file(path)}`。`path = args.spec`，就是 runner 运行时实际读取的那个文件。第 152 行写入 meta：`spec_sha256=self.hashes['spec'].get('sha256')`。
      - 依据 3：`src/new_run_benchmarks_v2.py`（4a96e6eb）第 167 行 `Provenance(__file__, args, ...)`，第 164 行 `json.loads(args.spec.read_text(...))`，读取的和计算哈希的是同一个路径。第 102 行把 T1 记录与 `sha256_file(args.spec)` 比对，比的是运行时计算出的值；代码里**没有**写死的期望值，也没有和固定值比对的 assert（`49010f9b`、`387af011` 在两份代码里都搜不到）。
      - 所以 15 个运行 meta 里的 `spec_sha256 = 49010f9b…`（事实 3）证明：A10 读到的是 CRLF 原字节，**魔搭没有转换这个文件**。
      - **为什么和 run_benchmarks.py 的现象不矛盾**：数据集的 `.gitattributes` 把 `*.json`、`*.csv`、`*.npz`、`*.txt` 等设为 `filter=lfs ... -text`，走 LFS，字节原样保存；`*.py`、`*.sh`、`*.md` 不在 LFS 规则里，按普通 git 文本处理，服务器端把 CRLF 规范化成了 LF。这与已经看到的现象一致：`run_benchmarks.py` 被转成 LF（60eac60a），而 ridge 的 `daily.csv` 和 v1 spec 下载回来都逐字节相同。另外，盒子上从数据集下载的 04:17 副本（`new_prep_round5/ds/new_LiMT/code_20261019_0417/configs/`）也是 49010f9b。
      - 后果：入库时保留 49010f9b 的原字节，用 `.gitattributes -text`；LF 版 387af011 只作参考，不是运行时用的版本。
    - **h5 的平台**：Claude 09:15 决定，h5 仍在魔搭 A10 上跑。理由是它和 12 个 NN tag 用同一套硬件和 CUDA 栈，不需要另写平台 DL，也没有 ROCm 的确定性风险。

17. **Claude 第 10 轮的采纳项（r13）**
    - **XGB 改在 A10 实例的主机 CPU 上重训**：原运行用的就是这台主机（Xeon 8369B，8 核），xgboost 3.4.1，nthread 20，所以有可能逐位一致（MATCH）。MATCH 时保留原规则，不必替换这条臂、重选规则；MISMATCH 时仍按第 10 条处理。重训和比对用 `new_retrain_bench_v1.py` 和 `new_compare_scores_v1.py`，容差 1e-6 不变，并记录 A10 主机的指纹。DE 和 ridge 仍在 CPU 实例上跑。
    - **开封运行也要锁定 pandas 版本**：开封用的环境（v3.1 和 bench_eval）要锁定 pandas，版本写进开封指纹，不只记在 CPU 重训主机上。理由是盒子上 pandas 3.0.6 让一个逐位比较的测试出现了 1.1e-16 的差异。目标版本与 A10 一致，即 2.3.3。
    - **签字的时间点**：DL-25 到 DL-28 要由 beelingz **用自己的话**签字，时间点是写 gate_b.json 之前，不是 commit 之前。commit A 时签字栏还写着“待签”，这是允许的。
    - **`runs/new_r9_smoke_*` 不进任何提交**，下游也不读它们。

## English

1–5. Unchanged from r8:
   - Deterministic arms are retrained, saved and compared against the existing scores at a declared tolerance: identical NaN positions and `max|Δscore| < 1e-6`.
   - Non-deterministic core arms are retrained, saved, rule-reselected and replaced, each with its own DL row, before gate_b.json is written.
   - Controls report 2021–22 only, labelled "no evaluation-window result (no saved model)".
   - Evaluation-period scoring happens only at unseal, via `new_bench_eval_v1` or another frozen script that imports `new_guarded_data_v1`. All checkpoint and model shas go into gate_b.json.
   - No evaluation-period predictions before unseal, including trial scoring.
6. **NN checkpoint check (r9)**
   - Read-only check on the ModelScope instance, 08:09:40–08:10:39 PDT, using `new_nn_cache_check_v1.sh` (4e6a7dbf…). No model was loaded and nothing was scored.
   - All 12 tags PASS: each `.pt` sha256 matches `artifact_sha256` in its `.pt.meta.json`, and there are exactly 12 `.pt` files under the caches.
   - Uploaded to `new_LiMT/nn_ckpt_20261019/` in the dataset: 27 files, SHA256SUMS fb66a1fb….
   - Box-side cross-check: only SHA256SUMS and the check table were downloaded, no `.pt` files. All 26 file shas match the server-side records.
   - **None of the 12 NN tags need retraining.** The arms are unchanged, and the 12 shas go into gate_b.json.
7. **Ridge comparison script (frozen in r9)**
   - Script: `new_compare_ridge_daily_v1.py`, sha256 07f4e121….
   - Each pair of old and new 2021–22 daily.csv files is first compared by sha256; equal hashes mean a match. If the hashes differ, the script compares, in order: header, row count, non-numeric cells, NaN positions, and finally whether numeric `max|Δ| < 1e-6`.
   - It prints exactly one boolean and never any value, difference, column name or row.
   - It refuses any file containing a date in 2023 or later: it prints `False` and exits with code 3.
   - The tolerance is hard-coded and cannot be overridden by an argument.
   - Synthetic self-tests passed.
8. **DE provenance (r10; recorded here, not in DL-26, because DL-26 covers only the v2 spec).**
   - The DE metas have no `script_sha256` key at any depth.
   - With Claude's approval at 08:25 PDT, only the `n_features` key was read: **158 for seeds 1, 2 and 3**.
   - Commit 5a8b9a4 (2026-10-18 19:15:55 PDT) switched XGBoost and DE from filter20 to all 158 features, so these DE runs used code from 5a8b9a4 or later.
   - `run_benchmarks.py` has not changed since that commit (824daa85), and the DE scores were written after it.
   - **Conclusion (inference, not hash-verified):** DE was trained with `src/run_benchmarks.py` 824daa85 and spec v1-b e189f6cb. The retrain uses these.
9. **Platform and fingerprints (r10, approved by Claude at 08:25 PDT).**
   - **(r13 change)** XGB is retrained on the **ModelScope A10 instance's host CPU** (see item 17), where a MATCH is possible. DE and ridge are retrained on the ModelScope CPU instance, with MISMATCH expected. The new artifacts replace the old ones and rules are reselected under item 10. **The originals and their 2021–22 results are kept.** The tolerance is unchanged at 1e-6.
   - h5 runs on the A10.
   - **LGBM is not retrained:** its 12 models exist and are only hashed. Claude's 08:25 message listed LGBM among the retrains, which conflicts with r9/r10; this has been flagged to the parent agent.
   - The LGBM discrepancy is **RESOLVED (Claude, 08:30 PDT)**: listing LGBM among the retrains was Claude's own error, and r9/r10 govern. LGBM is hash-only.
   - **lightgbm: currently installed version only; the runtime version was not recorded.**
     - `D:\research\new_LiMT\venv_bench` has Python 3.13.9, lightgbm 4.7.0, pyqlib 0.9.6 and xgboost 3.4.1. It is the likely DE environment; this is an inference.
     - `D:\research\limt-repro\.venv` has Python 3.13.9, lightgbm 4.7.0 and pyqlib 0.9.8.dev34. It is possibly the lgbm/ridge environment; also an inference.
     - Every environment that has lightgbm has 4.7.0. This is recorded for reference only and is not a constraint.
   - Original fingerprints:
     - XGB: A10 host, Xeon Platinum 8369B, 8 cores, image ubuntu22.04-cuda13.0.3-py312-torch2.13.0, Python 3.12.13, torch 2.13.0+cu130, pyqlib 0.9.6, **xgboost 3.4.1** (meta key `/versions/xgboost`, the only key read), nthread 20.
     - DE, ridge and lgbm: the laptop, AMD Ryzen 7 7735H (8C/16T), Windows 11 10.0.26200. No version keys appear in any of their metas, and the lightgbm version is not recorded anywhere.
   - Planned host: ModelScope DSW-CPU, 64 nproc, 28 GB. The rest is to be recorded by the fingerprint step in `new_cpu_instance_env_plan_r10.md`.
10. **Rule reselection and forensics for MISMATCH arms (round 9, item 3).**
    - v2 tags (XGB, DE) go through parameterised copies of the v2 rule grid and forensics (`new_rule_grid_bench_v2p.py`, `new_bench_forensics_v2p.py`).
    - The CPU arm (ridge) goes through the `cpu_baselines.py` (4c585a0f) selection logic, via `new_cpu_reselect_v1.py`.
    - Outputs go to new files, and only rule names are printed. Each replaced arm gets its own DL row recording the old and new shas, the new rule name, the platform fingerprint and `expected_verdict`.
11. **Guard module and cutoffs (round 9, items 1–2).**
    - Every pre-unseal entry point (h5 runner, retraining, comparison, rule reselection, forensics) uses `PRE_UNSEAL_CUTOFF = 2022-12-31`; unseal uses `UNSEAL_CUTOFF = 2026-06-30`. Each entry point selects its cutoff in code, never from the CLI or JSON.
    - `new_guarded_data_v1.py` is frozen and committed before any retrain. Runners check its sha against the committed `configs/new_guarded_data_v1.sha256`.
    - Any later change to the module requires its own DL row listing the retrains and h5 artifacts it guarded.
12. **Unseal failure policy (round 9, item 8).**
    - Unseal runs once: v3.1 first, then bench_eval.
    - On any failure, stop, then record and commit the logs. A rerun needs a DL row first.
    - A bench_eval failure does not affect v3.1's outputs.
13. **Meta keys read (r10).**
    - DE metas: key names listed recursively without values; only the value of `n_features` was read.
    - XGB metas: key names listed without values; only the value of `/versions/xgboost` was read.
    - lgbm and ridge grid files: key names only, no values.
    - No other values were read.
14. **Retrain inputs uploaded (r11, approved by Claude at 08:30 PDT).**
    - Files: `run_benchmarks.py` (824daa85), `cpu_baselines.py` (4c585a0f) and 16 ridge `daily.csv` files from `runs/csi300/ridge/*/*/seed1/`.
    - Destination: `new_LiMT/cpu_retrain_inputs_20261019/`, with SHA256SUMS 9218a988….
    - `_sealed` copies were excluded. The daily.csv files were hashed only, never opened; that they cover 2021–22 is inferred from the directory convention.
    - Box-side check: 17/18 byte-identical. `run_benchmarks.py` was normalised to LF by ModelScope (60eac60a…); restoring CRLF gives 824daa85.
    - Details: `new_cpu_retrain_inputs_upload_r11.md`.
15. **Timeline (r11).**
    - At 08:28 PDT the laptop repo was `main...origin/main [ahead 2]`: `b7306e0` (00:53:04 PDT) and `7e0ffbd7` (04:40:11 PDT) are not yet pushed. This is consistent with DL-16, where the push time is still a placeholder.
    - Between 08:29 and 08:36 PDT, GPT's r9 deliverables appeared in the laptop `src/`. They are untracked and not yet checked by the assistant.
16. **Timeline (r12; read-only check at 09:15 PDT, approved by Claude).**
    - **h5 smoke artifacts.** The laptop `runs\new_r9_smoke_scores\` holds a seed1 meta.json and .npz for each of LSTM-h5default and Transformer-h5default. `runs\new_r9_smoke_models\` holds two .pt files and no metas.
      - File mtimes: 08:41:03–08:41:37 PT.
      - Meta `started_at`: 08:40:49 PT. `ended_at`: 08:41:04 (LSTM) and 08:41:29 (Transformer).
      - Overall span: 08:40:49–08:41:37 PT.
    - **Origin.** GPT's r9 delivery self-check process (`new_r9_delivery_check.py` / pytest / `--smoke`). **This is per Claude; the assistant did not open argv to verify it.**
    - **Data dates.** Only date-range keys were read; no scores, npz or other keys were opened.
      - `guard_cutoff_name` is PRE_UNSEAL_CUTOFF and `guard_cutoff_value` is 2022-12-31.
      - `max_date_touched`: features 2018-09-05, labels 2018-07-06; prices, marks and bench were never touched (None). Both metas are identical.
      - **No date is on or after 2023-01-01.**
    - **Status.** This was a development self-check, not a DL-24 run. It must not be used for config selection or any conclusion. DL-24 still awaits beelingz's personal sign-off.
    - **(r13 change) v2 spec line endings: from the code, the A10 read the CRLF bytes, 49010f9b.** This r13 conclusion replaces r12's "LF form 387af011 inferred".
      - Evidence 1: `src/new_gate_b_common.py` (12632786), lines 19–24, `sha256_file`: `with Path(path).open('rb') as stream: ... h.update(block)`. It hashes the raw bytes with no normalisation.
      - Evidence 2: same file, lines 106–108, `Provenance.__init__`: `for key in ('spec', ...): self.hashes[key] = {'path': str(path), 'sha256': sha256_file(path)}`, where `path = args.spec`, the file the runner actually reads. Line 152 writes it into the meta: `spec_sha256=self.hashes['spec'].get('sha256')`.
      - Evidence 3: `src/new_run_benchmarks_v2.py` (4a96e6eb), line 167, `Provenance(__file__, args, ...)`; line 164, `json.loads(args.spec.read_text(...))`. The same path is read and hashed. Line 102 compares the T1 record against `sha256_file(args.spec)`, a value computed at runtime. There is **no** hard-coded expected value and no assert against a pinned constant; neither `49010f9b` nor `387af011` appears in either file.
      - So `spec_sha256 = 49010f9b…` in all 15 run metas (fact 3) proves the A10 read the CRLF bytes: **ModelScope did not convert this file.**
      - **Why this is consistent with run_benchmarks.py.** The dataset's `.gitattributes` marks `*.json`, `*.csv`, `*.npz`, `*.txt` and others as `filter=lfs ... -text`, so they are stored byte-exact through LFS. `*.py`, `*.sh` and `*.md` have no LFS rule, so they are handled as ordinary git text and the server normalised CRLF to LF. That matches what was observed: `run_benchmarks.py` came back as LF (60eac60a), while the ridge `daily.csv` files and the v1 spec came back byte-identical. The box's 04:17 copy downloaded from the dataset (`new_prep_round5/ds/new_LiMT/code_20261019_0417/configs/`) is also 49010f9b.
      - Consequence: commit the original 49010f9b bytes, protected with `.gitattributes -text`. The LF form 387af011 is for reference only; it is not what the runs used.
    - **h5 platform.** Claude decided at 09:15 that h5 stays on the ModelScope A10. Reasons: same hardware and CUDA stack as the 12 NN tags, no extra platform DL, and no ROCm determinism risk.
17. **Adopted Claude round-10 items (r13).**
    - **XGB is retrained on the A10 instance's host CPU.** The original runs used this host (Xeon 8369B, 8 cores) with xgboost 3.4.1 and nthread 20, so a bit-identical MATCH is possible. On a MATCH the original rule is kept and the arm is neither replaced nor reselected; on a MISMATCH, item 10 applies. Retrain and compare with `new_retrain_bench_v1.py` and `new_compare_scores_v1.py`, keep the tolerance at 1e-6, and record the A10 host fingerprint. DE and ridge stay on the CPU instance.
    - **Pin pandas for the unseal run too.** The unseal environment (v3.1 and bench_eval) pins pandas and records the version in the unseal fingerprint, not only on the CPU retrain host. Reason: on the box, pandas 3.0.6 caused a 1.1e-16 difference in an exact-equality test. The target is 2.3.3, as on the A10.
    - **Sign-off timing.** DL-25 to DL-28 need beelingz's sign-off **in his own words** before gate_b.json is written, not before the commits. Committing A while the sign-off fields still read "pending" is fine.
    - **`runs/new_r9_smoke_*` stays out of every commit**, and nothing downstream reads it.
