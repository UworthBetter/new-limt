# DL-25 分析产物 v3 → v3.1（草稿 r13，开封前）

实验助手，2026-10-19，时间均为 PDT。新文件，取代 `new_DL25_analysis_v3_1_r11_draft.md`（c29a8d74…，保留不改）。r13 填上了 sha：GPT 09:18 交付完成后，实验助手核对过，并在盒子上重跑了自测。r11 的说明：r11 取代 `new_DL_rows_round6b_draft.md`（4115ae1b…）里的 DL-25 一节。r11 只改了一处：引用的规格从 v3.1b 换成 **v3.1d**，即 `/workspace/uploads/new_GPT_spec_v3_1d_guard_fix.md`（sha 7eea4ee10387797d0241e782f7c3df4267e678ef30ee170b84daf81e097b2a41）。该规格配套 h5 规格 r9 `new_GPT_spec_h5_benchmarks_r9.md`（84f25f17…）。DL-12a 是 DL-12 的附录行，与本条互相引用。编号沿用 07:33 PDT 的决定：DL-24 是 h5 基准，v3 → v3.1 是本条。

- **起因**：外部审阅第 5、6 轮要求，防护要覆盖所有价格、标记和特征的访问。读代码核实的结果：
  - v3 的 `portfolio_events` 路径实际只通过已记账的 `relatives()` 和 `stale` 读价格，没有越界；
  - 但 `PriceGuardData.__getattr__` 会把 `close`、`marks`、`features` 原样转发出去；
  - 打分路径 `predict` 用的是未包装的数据；
  - 自测没有覆盖防护。
- **v3.1 的改动**（v3.1d 规格 §1–§3、§7、§8）：
  1. 改用白名单式的守卫对象，`run()` 里所有数据访问都要经过它。
  2. 按五个流分别记录 max_date_touched：prices、marks、features、labels、bench。任何一个流晚于截止日就硬失败；任何一个流从未被访问也失败。后一条是实验助手加的。
  3. 自测加入阳性对照：故意越界访问，必须报错。
  4. 运行时记录 sys.modules 的导入闭包、各文件的 sha256，以及第三方包版本（torch、pyqlib、xgboost、lightgbm 等）。
  5. **防护类独立成模块** `src/new_guarded_data_v1.py`，v3.1 和 `new_bench_eval_v1.py` 共用。模块里定义两个常量：`UNSEAL_CUTOFF = "2026-06-30"` 给开封用，`PRE_UNSEAL_CUTOFF = "2022-12-31"` 给开封前的重训和 h5 用。由各个入口在代码里选用，命令行和 JSON 都不能改。两个常量各有阳性对照。
  6. **先冻结防护模块**：任何重训或 h5 运行之前，模块就要 commit；开封前各 runner 按已提交的 `configs/new_guarded_data_v1.sha256` 校验。之后模块若有改动，要另起 DL 行。
  7. **开封的顺序和失败处理**：先跑 v3.1，再跑 bench_eval。任一步失败就停下、记录、commit 日志；要重跑，先写 DL 行。bench_eval 失败不影响 v3.1 的产出。
- **不改的部分**：算法、常数（TAU、BLOCK_MEAN、N_BOOT、RNG_SEED、NW_LAG、TOST_MARGIN_PP、RULE、STAMP_CUTOFF、SEEDS、HOLM_KEYS）、统计口径、gate_b.json 的校验逻辑。
- **sha**（09:2x PDT 核对）：
  - v3：`203e6a455796f4eca90fa5a9a0038459eec2324279bb13c72d5622339dd8f75d`
  - **v3.1** `src/new_gate_b_analysis_v3_1.py`：`4135981bbaf7844bc270a26b0a4de7cbe06983342e4ad9a62d51590d273e4b88`
  - **防护模块** `src/new_guarded_data_v1.py`：`5de86785d2725a48b03ceb4c865926fd3c6516157c23f4780bd86844cbd7efa3`。`configs/new_guarded_data_v1.sha256` 的内容就是这个值，文件本身的 sha 是 f45bc04b…。08:36 的快照 6ae87320 已被取代。
  - **描述性评估** `src/new_bench_eval_v1.py`：`cc5c12faa2f350a22d4ca441e89dd713ab8bc2cff212b800e65b6e832e2967fd`
  - **一次性开封包装** `src/new_unseal_once_v1.py`：`b62b44575269dce50b2b6728288ad709e525bedbf2b4c23df8f71e9ee8e6acad`
  - 笔记本、交付 zip、GPT 的 code manifest 三处的 sha 完全一致。
- **自测输出**（只用合成数据，所有阳性对照都触发了）：
  **主证据是 GPT 在笔记本上的输出**，目录 `D:\research\new_LiMT\protocol\checks\new_r9_delivery_20261019_release\`。这个目录的 SHA256SUMS 是 3e131896…，与交付 zip 里的副本逐字节一致，55 个文件全部 OK。
  - `new_pytest.txt`（5182073448644846…）：**41 passed，0 failed**
  - `new_bench_eval_final_pytest.txt`（c13f6f3a2de5edcf…）：**14 passed，0 failed**；测试时的源码 sha 是 cc5c12fa，测试期间源码没有变动
  - `new_guard_self_test.txt`（c97d54fa…）：PASS，阳性对照 28 个全部触发
  - `new_analysis_self_test.txt`（3c98ec1c…）：PASS，含 guard 和统计两部分
  - `new_t1_smoke.txt`（3a0999ac…）、`new_runner_smoke.txt`（4be3ddbe…）、`new_smoke_schema.txt`（6c6edc37…）：都是 PASS
  - `new_acceptance.json`：passed=true，synthetic_only=true，7 项检查的返回码都是 0，protected_unchanged=true，14 个受保护基线文件都相同
  - 盒子上的重跑（下表）是在接到"不必重跑"的指示之前做的，只作旁证。第一次用 pandas 3.0.6 时，`test_original_ridge_fit_roundtrip` 因 1.1e-16 的浮点差异 FAIL；换成 pandas 2.2.3 后，41/41 和 14/14 都通过。

  | 输出 | GPT 交付（笔记本） | 盒子重跑（py3.12、torch 2.13.0+cpu、pyqlib 0.9.6、pandas 2.2.3） |
  |---|---|---|
  | 防护自测 `--self-test` | c97d54fab44f205916d9439907ba116bb553bf3976d8eab881f8a5e5d73b6699 | 63a8c5e582599b47e46b37376cc639312fe67d08e8f9922f46fe63a1db2bccbd |
  | v3.1 自测 `--self-test` | 3c98ec1c168595936434129421a3bb31675b11c901526c5bee7ee8ee3a66fc59 | 7fb074b62e6488fa54df53720e4019b9e63381ac7d3b0074b65fc8b2d9e6a0fa |

  两边都 PASS。日志 sha 不同是因为日志里记录了运行时路径和版本，不代表结果不同。盒子上的日志在 `/workspace/limt_gateb/new_r9check_root/new_LiMT/protocol/checks/new_r9_box_check_20261019_pd223/`。
- **开封顺序**（Claude 第 6 轮）：v3.1 到位并通过自测（含阳性对照）→ 记录 sha → PI 签字 → 继续开封流程。gate_b.json 的 `analysis_sha256` 填 v3.1 的 sha，`guarded_data_sha256` 填防护模块的 sha。
- 这次修改是在没有查看任何 2023 年以后数据的情况下完成的。
- **签字**：[待 beelingz 签字，注明时间]
