# DL-27 封存包内允许成员的解包记录（草稿 r8，开封前）

实验助手，2026-10-19，时间均为 PDT。新文件；按 Claude 第 8 轮 (e) 单独成行，不并进 DL-26，因为两者内容无关。

## 中文

- **什么**：把 Gate B 产出封存包里的一部分允许成员解到盒子上的 `/workspace/new_prep_round5/` 下。
- **哪个包**：`new_gateb_outputs_20261019T134432Z.tar.gz`
  - sha256 `d0f04dd2…`，24,922,469 字节。
  - 由 `/workspace/new_prep_round5/new_dl_archive.py`（写于 07:05:55）从魔搭数据集 `UworthBetter/new-limt-gateb-inputs` 的 `outputs_20261019/` 下载，07:06:16 落盘，日志 `new_dl_archive.log` 时间为 07:06:26。
- **哪一步、谁做的**：第 5/6 轮准备 worker 做的 RNN 自检（Claude 第 6 轮第 2 条）。实验助手的子代理执行，不是本 worker。
- **解了什么、怎么解的**
  1. `new_extract_allowed.py`（07:07:40）按正则只解：
     - `runs/new_scores_v2/*.npz`
     - `runs/new_scores_v2/*.json`
     - `protocol/checks/new_benchmark_forensics_v2.json`

     排除 `runs/new_csi300_t7/` 和规则网格。结果放在 `new_allowed_members/`，目录时间 07:07:40。
  2. 15 个运行日志 `logs/*.log` 放在 `new_allowed_logs/logs/`，目录时间 07:07:49。用的哪条命令**没有记录**，记为 [未核实]。
  3. 成员文件的 mtime 保留了实例上的原值（05:02–06:44）。
- **文件清单和 sha256**：共 46 个文件，即 15 个分数 npz、15 个 meta.json、1 个取证 json 和 15 个日志。完整清单见 `/workspace/limt_gateb/new_allowed_members_sha256_r8.txt`（sha256 `7c9d58f0bac17c1d0e301cca2d5608e5a88dcc721db715e1a16d42d7ef8db3ac`），只用 `sha256sum` 算，没有打开文件。
- **内容属于哪一段**：npz 里只有验证期（2019–2020）和规则选择期（2021–22）的分数，没有评估期（2023 年以后）。这一点由 runner 的设计保证（`new_run_benchmarks_v2.py` 只写这两段），不是看文件得出的。
- **读没读**
  - 本 worker：**没有打开** npz、日志和取证 json。只读过 15 个 meta 的 `spec_sha256`、`inputs.spec` 和 `kwargs` 键，用于 DL-26 的溯源。
  - 更早的 worker 留下了笔记 `new_b1b3_raw.txt`（07:08:32）和 `new_b_forensics_valid.txt`（07:08:52），看起来读过 meta 和取证的验证期字段。本 worker 没有打开这两份笔记，内容 [未核实]。
  - 2021–22 规则选择结果（T7）**没有解**，也没有读。
- **影响**：只涉及验证期和 2021–22 分数，不碰评估期，盲态不受影响。gate_b.json 写入封存包 sha 和本清单 sha。

## English

- **What**: A subset of allowed members of the Gate B output archive was extracted on the box under `/workspace/new_prep_round5/`.
- **Archive**: `new_gateb_outputs_20261019T134432Z.tar.gz`
  - sha256 `d0f04dd2…`, 24,922,469 bytes.
  - Downloaded from `outputs_20261019/` in the ModelScope dataset `UworthBetter/new-limt-gateb-inputs` by `/workspace/new_prep_round5/new_dl_archive.py` (written 07:05:55). It landed at 07:06:16; the log `new_dl_archive.log` is timestamped 07:06:26.
- **Step and actor**: The RNN self-check of the round-5/6 prep worker (Claude's round 6, item 2), run by an assistant subagent, not by this worker.
- **What was extracted, and how**
  1. `new_extract_allowed.py` (07:07:40) extracted only members matching:
     - `runs/new_scores_v2/*.npz`
     - `runs/new_scores_v2/*.json`
     - `protocol/checks/new_benchmark_forensics_v2.json`

     It excluded `runs/new_csi300_t7/` and the rule grid. Output went to `new_allowed_members/`, directory time 07:07:40.
  2. The 15 run logs `logs/*.log` went to `new_allowed_logs/logs/`, directory time 07:07:49. The command used was **not recorded** [unverified].
  3. Member mtimes keep their instance values (05:02–06:44).
- **File list and sha256**: 46 files, namely 15 score npz, 15 meta.json, 1 forensics json and 15 logs. The full list is in `/workspace/limt_gateb/new_allowed_members_sha256_r8.txt` (sha256 `7c9d58f0bac17c1d0e301cca2d5608e5a88dcc721db715e1a16d42d7ef8db3ac`), computed with `sha256sum` only, without opening any file.
- **Which periods the content covers**: The npz files hold validation (2019–2020) and rule-selection (2021–22) scores only, with nothing from the evaluation period (2023 onward). This follows from the runner's design (`new_run_benchmarks_v2.py` writes only those two segments), not from inspecting the files.
- **Who read what**
  - This worker did **not open** the npz files, the logs or the forensics json. It read only the `spec_sha256`, `inputs.spec` and `kwargs` keys of the 15 metas, for the DL-26 provenance.
  - An earlier worker left the notes `new_b1b3_raw.txt` (07:08:32) and `new_b_forensics_valid.txt` (07:08:52), which suggest it read meta and forensics validation fields. This worker did not open those notes; their content is [unverified].
  - The 2021–22 rule-selection output (T7) was **not extracted** and not read.
- **Impact**: Only validation and 2021–22 scores are involved and the evaluation period is untouched, so blinding holds. gate_b.json will record the archive sha and the sha of this list.
