# 重训清单 r13（草稿）

实验助手，2026-10-19（PT）。取代 `new_retrain_list_r9_draft.md`（2aa086fb…，保留不改）。r13 按 Claude 第 10 轮修改：XGB 改在 A10 实例的主机 CPU 上重训（DL-28 第 17 条）。r9 的说明：r9 取代 r8（698fd5ef…）。

r9 相对 r8 的变化：
- NN checkpoint 核对通过，12 个 NN tag 免重训；
- 按 beelingz 08:17、08:18 PDT 的规则，**所有运行都在魔搭上**；
- 按 Claude 第 9 轮修订；
- 重算了算力。

**平台规则**
- CPU 作业在**魔搭 CPU 实例**（DSW-CPU：64 nproc，28 GB，额度不限）上跑，不计 GPU 小时。
- GPU 作业在**魔搭 A10** 上跑，只在确实有较大的 GPU 作业时才开。
- 两个实例共享 `/mnt/workspace`，输入数据、cache 和环境只需准备一次。

**规则**：容差 `max|Δscore| < 1e-6`，并且 NaN 位置一致。容差写死，**不放宽**。开封前不对任何臂做评估期预测（DL-28）。

## 1. 不用重训，只记 sha（写进 gate_b.json）

| 臂 | 依据 |
|---|---|
| **LSTM / GRU / ALSTM / Transformer-csrank，共 12 个 tag** | 08:09:40–08:10:39 在实例上核对，PASS 12/12。已上传到数据集 `new_LiMT/nn_ckpt_20261019/`（SHA256SUMS 的 sha 为 fb66a1fb…），盒子端复核 26/26 一致。sha 见 `new_arm_model_inventory_r9.md` |
| lgbm-reg、lgbm-rank × {h5, r1} × s1–3，共 12 个 | 笔记本上已有 `model.txt`。注意：原运行**没有设 `deterministic=True`**，以后如需重训，按第 2 节在 CPU 实例上做，预期结果为 MISMATCH |
| H、H-FT、G0、C0、剂量臂 | 笔记本上已有 `checkpoint.pt` |

## 2. 在 CPU 实例上重训、存模型、比对（**预期 MISMATCH，走替换路径**）

| 臂 | 原平台 | 本次平台 | 冻结 spec 和代码 | 比对工具 | 预期结果 | CPU 时间（估计） |
|---|---|---|---|---|---|---|
| XGBoost-seeded s1–3 | 魔搭 A10 实例的主机 CPU（Xeon 8369B） | **魔搭 A10 实例的主机 CPU（r13 改）** | spec v2 49010f9b（CRLF 原字节），runner 4a96e6eb | `new_compare_scores_v1.py` | 同一台主机、同版本、nthread 20，**有可能 MATCH**；MATCH 时保留原规则 | 约 0.2–0.5 h |
| DoubleEnsemble s1–3 | 笔记本 CPU（Windows） | 魔搭 CPU 实例 | spec v1-b e189f6cb，`run_benchmarks.py` 824daa85（这个对应关系是推断的，见 DL-28 第 8 条） | 同上 | 指纹不同；底座 LightGBM 没设 deterministic=True，**预期 MISMATCH** | 约 0.5 h |
| ridge h5、r1 | 笔记本 CPU（Windows） | 魔搭 CPU 实例 | `cpu_baselines.py` 4c585a0f，alpha 10 | `new_compare_ridge_daily_v1.py`（07f4e121…）：比 2021–22 的 daily.csv，只打印一个布尔值 | 指纹不同，**预期 MISMATCH**（闭式解，也可能碰巧一致） | 几分钟 |

**为什么预期 MISMATCH（第 9 轮第 4 条）**：重训前先比较平台指纹和原 meta。这三个臂的原平台都不是 CPU 实例，所以在 meta 里写 `expected_verdict: MISMATCH_EXPECTED`，照样训练和比对。比对 MISMATCH 时用新模型替换旧臂，**不放宽容差**。如果碰巧 MATCH，就保留原来选定的规则，只记新模型的 sha。

**MISMATCH 之后（第 9 轮第 3 条，h5 r9 §11a）**：
- XGB 和 DE 用 v2 规则网格的参数化副本 `new_rule_grid_bench_v2p.py` 重选规则，用 `new_bench_forensics_v2p.py` 做取证；
- ridge 用 `new_cpu_reselect_v1.py`，它调用 cpu_baselines 的规则选择逻辑；
- 结果写进新文件，只打印规则名；
- 每个被替换的臂登记一条 DL 行；
- 这一步也在 CPU 实例上做，约需几分钟。

**CPU 实例要先准备的东西**
1. **环境**：要和原平台锁定的版本一致，包括 pyqlib 0.9.6、xgboost、lightgbm、numpy、pandas、scipy、scikit-learn 和 Python。
   - 因为共享 `/mnt/workspace`，A10 上的 venv 也许能直接用，但要先核实 CPU 架构和指令集是否兼容：在 CPU 实例上 import 全部依赖，跑一次合成冒烟测试，记录 CPU flags。
   - DE 和 ridge 原来在 Windows 笔记本上跑，包版本可能和 A10 的不同，要按各自原 meta 或锁文件分别锁定。读这些版本键需要授权。
   - 版本装不成一致，就在 DL 里登记；这种情况本来也预期 MISMATCH。
2. **输入**：从私有数据集取代码、spec 和 BlindData，逐个核对 sha。
3. **防护模块**：`new_guarded_data_v1.py` 必须先冻结并 commit，runner 核对它和已提交的 sha 一致（v3.1d §7）。
4. **平台指纹**：写入 `new_cpu_host_fingerprint.json`。
5. **线程数**：照原值固定为 20。
6. **产物上传**：上传到数据集 `new_LiMT/cpu_retrain_20261019/`，附 SHA256SUMS。

## 3. h5 臂（魔搭 A10，与 15 个 tag 同平台）

- 运行数是 24 / 22 / 20，取决于有 0、1 还是 2 个模型选中 cfg1。必须存 checkpoint，按 h5 r9 §2.6 逐张量核对。cfg1 的构造参数要与 r1 meta 一致（§2.2a）。
- 平台还是 A10，**不需要平台 DL**，也不需要断点续跑。
- **准备工作放到 CPU 实例上做**，包括取输入、建 cache、跑合成自测，以缩短 A10 的挂钟时间。要在 GPU 上做的只有 T1 和训练。
- 规则网格和取证也在 CPU 实例上做，平台指纹以训练产物的 meta 为准。

## 4. 对照臂

z-score 对照和 seed-0 对照只报 2021–22 的结果，输出写 `"no evaluation-window result (model not saved)"`；中文 DL 写"无评估窗口结果（未存模型）"。

## 5. 算力合计（r9，**估计**）

| 项目 | GPU 小时（A10 挂钟） | CPU 实例（不计入 GPU 小时） |
|---|---|---|
| A10，08:06 开机到关机（NN 核对用了 08:09:40–08:10:39） | 08:06 到关机 **[待填]**；08:20 前关机约 0.25 h | — |
| h5 训练，K = 3 并发（A10 上观测值外推） | 串行等效 2.1–6.2 h，挂钟约 2–6 h；加上开机、T1 和收尾约 0.3–0.5 h，共约 **2.3–6.5 h** | 准备、规则网格和取证约 0.5 h |
| XGB、DE、ridge 重训和比对 | 0 | 约 1 h |
| MISMATCH 后的规则重选和取证 | 0 | 约 0.2 h |
| CPU 实例的环境准备和核实 | 0 | 约 0.5–1 h |
| 12 个 NN 重训 | **0（不需要）** | — |
| **剩余合计** | **约 2.5–6.8 GPU-h** | 约 2–3 h |
| 已用（今天的 Gate B） | 约 5.9 h | — |
| **累计** | **约 8.4–12.7 GPU-h，低于 60 h 上限** | — |

## 6. 顺序

1. **A10**：NN 核对已经做完。XGB 重训（r13）要在 A10 主机 CPU 上跑，可以和 h5 用同一次开机。
2. GPT 交付 `new_guarded_data_v1.py` 并通过自测，然后冻结、commit、push（v3.1d §7）。
3. CPU 实例：准备环境和输入，记录平台指纹；重训 DE、ridge 并比对（XGB 在 A10 上跑，r13）；预期 MISMATCH 的臂进行规则重选和取证，登记 DL 行。
4. 先 commit 并 push h5 spec 和 DL-24，再开 A10 跑 T1 和 h5 训练，每个运行都存 checkpoint。之后回到 CPU 实例做选择、规则网格和取证。
5. 汇总所有 sha，起草 gate_b.json，交 Claude 审，然后 push，开封。开封顺序是先跑 v3.1、再跑 bench_eval，任何一步失败都停下。
