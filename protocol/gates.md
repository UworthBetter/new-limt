# Gate 板

四道门都还没过。过门条件以预注册第 12 节为准，这里只记本目录做到哪。

## Gate A（现在）

- [x] 第 6.5 节核对项全部打勾
- [x] A-G0-1 至 A-G0-4、A-LBL-1 断言通过
- [x] 验证窗 2019–2020 的 sigma_d(prelim) 已测量
- [x] 预注册里的 `[FILL]` / 仍开放的 `[DECISION]` 已填或明确标成不做
- [x] 预注册文件已哈希并提交

2026-10-17：只完成了代码核对和断言文件。断言没有全部通过，所以 Gate A 不过。
2026-10-17 深夜更正：`tests/test_a_g0_a_lbl.py` 行为化重写后 5/5 通过（A-G0-1..4 + A-LBL-1，见 experiment_report_2026-10-17.md）。断言条件已满足；Gate A 其余项（6.5 打勾、sigma_d(prelim)、[FILL]、哈希提交）仍未做。
2026-10-18：seed-1 H 与 G0/C0 tau 网格 6 跑完成（runs/csi300_H_seed1_20261017、runs/csi300_g0c0_seed1_tau_20261017）。两臂监控指标均选 tau=0.5、全部选 ep20（见 tau_grid_report_2026-10-18.md）。Gate B 前置项（tau 冻结、N5、sigma_d(final)、分档）在做；H-FT 与探索性剂量臂已登记后发射。

**Gate A 落定（2026-10-18 16:54，诚实声明：晚于 seed-1 各跑完成，不倒签）**：
- 诚实时序：断言 2026-10-17 首过、2026-10-18 在当前 trainer（sha a960a995…）复跑 14/14（`checks/pytest_gate_a_2026-10-18.txt`）；§6.5 打勾、[FILL] 填写与 sigma_d(prelim) 均在 seed-1 H（10-17）、tau 网格/H-FT/剂量臂（10-18）之后完成。
- sigma_d(prelim) = **6.78%（年化）**，MDE80 = 9.0 pp（ridge_h5 vs H seed1，默认规则，执行账本，485 配对日；`checks/sigma_d_prelim_2026-10-18.json`）。注意这是"两个不同模型"的上界型代理；确认性配对 C0−G0 预计远小。
- 明确标开的项：§11 作者仓库回测配置核对（N10，未查——按"未指明"处理并报敏感性）；§13 硬性写作截止日（用户决定，Gate C 前定）；§16.5 先验文献补查未做（限制 8 已披露）。
- 哈希链：基线 commit `8a7a4c45493b562712177688c9d52118a51b3fb3`；**偏差日志五条 commit `c8c6c2799f4ea60a9ab6228a3b283490a7b4c72b`（2026-10-18T16:45:28+08:00，先于其覆盖的一切新跑）**；**Gate A commit `841f00becdee4df72a052c102392266d7b9bf9e2`（16:54:08+08:00）**；细节在 `checks/gate_a_record_2026-10-18.json`。

## Gate B

tau 网格和 seed-1 验证结果出来之后，先冻结配置和 tau，再算 sigma_d(final) 并套用第 4.3 节分档。
在此之前不要打开 seed-1 的 2023+ 输出。

## Gate C

CSI300 三个种子齐了之后，按第 7 节机械判定 S1 / S2 / S3。种子 1 单独不能判。

## Gate D

CSI500 和截止日后样本都出来之后，再定标题、摘要和投稿。
`paper/` 在这之前不写正文。
