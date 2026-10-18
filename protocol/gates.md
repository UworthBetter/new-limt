# Gate 板

四道门都还没过。过门条件以预注册第 12 节为准，这里只记本目录做到哪。

## Gate A（现在）

- [ ] 第 6.5 节核对项全部打勾
- [ ] A-G0-1 至 A-G0-4、A-LBL-1 断言通过
- [ ] 验证窗 2019–2020 的 sigma_d(prelim) 已测量
- [ ] 预注册里的 `[FILL]` / 仍开放的 `[DECISION]` 已填或明确标成不做
- [ ] 预注册文件已哈希并提交

2026-10-17：只完成了代码核对和断言文件。断言没有全部通过，所以 Gate A 不过。
2026-10-17 深夜更正：`tests/test_a_g0_a_lbl.py` 行为化重写后 5/5 通过（A-G0-1..4 + A-LBL-1，见 experiment_report_2026-10-17.md）。断言条件已满足；Gate A 其余项（6.5 打勾、sigma_d(prelim)、[FILL]、哈希提交）仍未做。
2026-10-18：seed-1 H 与 G0/C0 tau 网格 6 跑完成（runs/csi300_H_seed1_20261017、runs/csi300_g0c0_seed1_tau_20261017）。两臂监控指标均选 tau=0.5、全部选 ep20（见 tau_grid_report_2026-10-18.md）。Gate B 前置项（tau 冻结、N5、sigma_d(final)、分档）在做；H-FT 与探索性剂量臂已登记后发射。

## Gate B

tau 网格和 seed-1 验证结果出来之后，先冻结配置和 tau，再算 sigma_d(final) 并套用第 4.3 节分档。
在此之前不要打开 seed-1 的 2023+ 输出。

## Gate C

CSI300 三个种子齐了之后，按第 7 节机械判定 S1 / S2 / S3。种子 1 单独不能判。

## Gate D

CSI500 和截止日后样本都出来之后，再定标题、摘要和投稿。
`paper/` 在这之前不写正文。
