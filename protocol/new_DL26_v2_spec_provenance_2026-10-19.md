# DL-26（新）spec v2 的来源记录：2a65bd42 与 49010f9b，以及 spec 未在运行前入库（草稿 r13，开封前）

实验助手，2026-10-19（PT）。**r9 版**，取代 `new_DL26_v2_spec_provenance_r8_draft.md`（7f29cf62…，保留不改）。按 Claude 08:04 PDT 的回复改写状态字段的披露，并填上 04:23 的占位。披露措辞已经 Claude 批准。所有时间为 PDT。这一条只用到文件哈希、修改时间、文件内容，以及 15 个运行 meta 里的 `spec_sha256`、`inputs.spec`、`kwargs` 这几个键；没有读任何指标。

## 事实（已核对）

1. **2a65bd42 已找回，原样保存。**
   - 来源：盒子上 03:45 的代码包副本 `/workspace/new_limt_gateb_code_20261019_0345/configs/new_benchmarks_qlib_defaults_v2.json`，8,036 字节，盒子 mtime 03:46:02，sha256 `2a65bd42b7e05c72f7225d21a8fdb2da2f06fcc1974fb49ad873fd78de95f890`。
   - 存档：逐字节拷贝到 `/workspace/limt_gateb/new_benchmarks_qlib_defaults_v2_2a65bd42_recovered.json`，sha 不变。
2. **与 49010f9b 的差异只有一行，在描述字段。**
   - 第 204 行 `status`：`"DRAFT: DL-3 pending; not frozen or approved for production"` → `"FROZEN: DL-3 approved by user"`。
   - 字节数差 29，正好是两个字符串的长度差。diff 文件：`/workspace/limt_gateb/new_v2_spec_2a65bd42_vs_49010f9b.diff`（`0b5ca0e2…`）。
   - 逐键递归比较：六个模型（LSTM、GRU、ALSTM、Transformer、XGBoost、DoubleEnsemble）的 `module`、`class`、`kwargs` 完全相同；`filter_col_20`、`feature_sets`、`data_convention`、`step_len_ts` 也完全相同。
   - **结论：差异不涉及实质内容，不影响任何运行。**
3. **15 个运行实际用的是 49010f9b，不是 2a65bd42。之前"运行 meta 里记录的是 2a65bd42"这个前提是错的。** 2a65bd42 只出现在合成审计记录里（见第 4 点），没有出现在任何一个运行 meta 里。
   - 15 个运行 meta（LSTM、GRU、ALSTM、Transformer-csrank 和 XGBoost-seeded，各 seed 1–3）的 `spec_sha256` 和 `inputs.spec.sha256` 全部是 `49010f9b…`。
   - 每个 meta 的 `kwargs` 都等于 49010f9b 里对应模型的 `kwargs`，只把 `seed` 换成该运行的种子，15/15 一致。
   - runner 的 `verify_t1` 还要求 T1 记录里的 spec 哈希等于运行时的 spec 文件哈希（`new_run_benchmarks_v2.py:102`），所以 T1 记录对应的也是 49010f9b。T1 记录文件在 protocol/checks 里，我没有打开。
4. **2a65bd42 只出现在一处：** `configs/new_benchmarks_qlib_defaults_v2.meta.json`（f726b4a4…）。
   - 这是 `new_gate_b_delivery_audit.py` 的**合成数据**交付审计记录（`synthetic: true`，`status: "DL-3 draft"`），时间 02:42:55–02:43:37。
   - 笔记本上这个文件的 LastWriteTime 是 02:43:37，审计跑的是当时的草稿 2a65bd42。
   - 文件一直没更新，所以记录的哈希与后来冻结的文件不同。它不是运行的来源记录。
5. **入库状态：** 笔记本 `D:\research\new_LiMT` 在 HEAD `7e0ffbd7` 下，`git status` 显示 `configs/new_benchmarks_qlib_defaults_v2.json` 和 `.meta.json` 都是 `??`，即**从未入库**。
   - 笔记本上 49010f9b 的 LastWriteTime 是 03:59:45，CreationTime 是 01:47:46。
   - 其它副本：盒子上 04:17:58 的代码包（49010f9b）；04:19:21 的魔搭上传副本（49010f9b）。

## 偏差与披露

- **spec 没有在 T2（训练）之前入库**，到现在也没有入库，这是偏差。
  - 能证明 15 个运行用的是哪个 spec 的，只有：(a) 每个运行 meta 里记录的 `spec_sha256`；(b) 04:17 和 04:19 的盒子副本及魔搭副本，都早于 04:31 的 T1；(c) 现在笔记本工作区里的文件。这三者哈希一致，都是 49010f9b。
- **状态字段是预填的。** 经过如下（事实来自父代理核对的对话记录，08:04 PDT 转达）：
  1. **03:49 PDT**：实验助手给 beelingz 转发 GPT 用的修改清单，第 4 项写着"去掉 DRAFT: DL-3 pending 标记，DL-3 已经定了"。**当时 DL-3 还没有得到任何批准。**
  2. **03:59:45 PDT**：GPT 按这份清单，把 `status = "FROZEN: DL-3 approved by user"` 写进 49010f9b，也就是笔记本上的文件时间。
  3. **04:21 PDT**：卡片询问 DL-3 是否确认。
  4. **04:23 PDT**：beelingz 在卡片上**点选**了「确认，DL-3 我批准」（这是卡片选项的文字，属于卡片点选；对话记录里没有打字消息）。这是 DL-3 的实际批准。
  5. **08:04 PDT**：beelingz 在对话中用自己的话追认这次点选："是，04:23 我点了「确认，DL-3 我批准」"。beelingz 另外也向外部审阅（Claude）确认过 04:23 这次卡片点选（父代理 08:18 PDT 转达）。
  6. 另有 **07:28 PDT** 的运行后书面签字："我已审阅 DL-3、DL-12 至 DL-15 并同意。这几条是在 Gate B 运行之后签的，内容与本地提交 7e0ffbd7 相同，没有改动。beelingz，2026-10-19"（`new_gates_signoff_section_draft.md:29-31`）。
  - **结论**：49010f9b 的 FROZEN 状态是在批准前 24 分钟预先写进去的，起因是实验助手 03:49 的清单。实际批准时间是 04:23。全部运行（从 04:31 的 T1 起）都在批准之后。49010f9b 与批准前草稿 2a65bd42 的唯一差别就是这一个状态字段（第 204 行），超参数和数据约定都没有变。

## 处理（开封前，要 Claude 同意后由实验助手执行）

1. 两版都入库（命名已定，Claude 07:52 PDT）：
   - **运行版 49010f9b** 用原名 `configs/new_benchmarks_qlib_defaults_v2.json` 入库，字节不改。15 个 meta 和 gate_b.json 引用的就是这个名字和哈希。**不建 `_asrun` 副本。**
   - **DL-3 批准前的草稿 2a65bd42** 存为 `configs/new_benchmarks_qlib_defaults_v2_predl3_draft.json`，字节与 `/workspace/limt_gateb/new_benchmarks_qlib_defaults_v2_2a65bd42_recovered.json` 相同，并附 diff 文件 `new_v2_spec_2a65bd42_vs_49010f9b.diff`（0b5ca0e2…）。
   - `configs/new_benchmarks_qlib_defaults_v2.meta.json`（f726b4a4…）原样入库，本条注明它是合成审计记录。
2. gate_b.json 里登记的 spec v2 哈希保持 `49010f9b…`，与 DL-23 一致。
3. 不需要重跑任何东西，也没有受影响的运行。

## 签字

[待 beelingz 签字，注明时间]


## 时间线补记（r11，2026-10-19 08:28 PDT；取代 r9 `new_DL26_v2_spec_provenance_r9_draft.md`，a93844cf…，r9 保留不改）

- 只读查了笔记本仓库，状态为 `main...origin/main [ahead 2]`。没有 push 的两个提交是 `b7306e0`（00:53:04 PDT，分析 v2）和 `7e0ffbd7`（04:40:11 PDT，DL pre-Gate-B，DL-3 的签字文本就在这个提交里）。
- v2 spec 文件 49010f9b 和 2a65bd42 都还没有被跟踪（untracked）。
- 这和 DL-16 一致，DL-16 里的推送时间还空着待填。在真正推送之前，"DL 已推送"这句话不成立。
- 提交计划见 `new_commit_push_plan_r11.md`。

## 补记（r13，09:4x PDT；取代 r12 `new_DL26_v2_spec_provenance_r12_draft.md`，r12 保留不改）

- 按 Claude 第 10 轮第 3 条修改。r12 补记里关于 h5 的内容（冒烟产物、h5 平台）与 v2 spec 的来源无关，**已移到 DL-28 第 16 条**，这里只保留这条交叉引用。
- **v2 spec 的行尾：按代码判定，A10 读到的是 CRLF 原字节 49010f9b**。r13 的结论，取代 r12 里“推断用的是 LF 版 387af011”的说法。
  - 依据 1：`src/new_gate_b_common.py`（12632786）第 19–24 行 `sha256_file`：`with Path(path).open('rb') as stream: ... h.update(block)`，对文件的原始字节计算哈希，没有做任何规范化。
  - 依据 2：同一文件第 106–108 行 `Provenance.__init__`：`for key in ('spec', ...): self.hashes[key] = {'path': str(path), 'sha256': sha256_file(path)}`。`path = args.spec`，就是 runner 运行时实际读取的那个文件。第 152 行写入 meta：`spec_sha256=self.hashes['spec'].get('sha256')`。
  - 依据 3：`src/new_run_benchmarks_v2.py`（4a96e6eb）第 167 行 `Provenance(__file__, args, ...)`，第 164 行 `json.loads(args.spec.read_text(...))`，读取的和计算哈希的是同一个路径。第 102 行把 T1 记录与 `sha256_file(args.spec)` 比对，比的是运行时计算出的值；代码里**没有**写死的期望值，也没有和固定值比对的 assert（`49010f9b`、`387af011` 在两份代码里都搜不到）。
  - 所以 15 个运行 meta 里的 `spec_sha256 = 49010f9b…`（事实 3）证明：A10 读到的是 CRLF 原字节，**魔搭没有转换这个文件**。
  - **为什么和 run_benchmarks.py 的现象不矛盾**：数据集的 `.gitattributes` 把 `*.json`、`*.csv`、`*.npz`、`*.txt` 等设为 `filter=lfs ... -text`，走 LFS，字节原样保存；`*.py`、`*.sh`、`*.md` 不在 LFS 规则里，按普通 git 文本处理，服务器端把 CRLF 规范化成了 LF。这与已经看到的现象一致：`run_benchmarks.py` 被转成 LF（60eac60a），而 ridge 的 `daily.csv` 和 v1 spec 下载回来都逐字节相同。另外，盒子上从数据集下载的 04:17 副本（`new_prep_round5/ds/new_LiMT/code_20261019_0417/configs/`）也是 49010f9b。
  - 后果：入库时保留 49010f9b 的原字节，用 `.gitattributes -text`；LF 版 387af011 只作参考，不是运行时用的版本。
