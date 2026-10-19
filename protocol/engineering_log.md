# 工程修复记录（非协议变更）

准则：只记实现层修复，不动预注册任何阈值/网格/判据。每条注明"engineering fix, no protocol change"与提交哈希。与偏差日志（协议层）分开维护。

| 日期* | 修复 | 提交 | 性质 | 影响评估 |
|---|---|---|---|---|
| 2026-10-19 | H 种子 2/3 打分路径写死 `_20261017` 目录导致 SKIP（H-FT/G0/C0 不受影响）；改为日期无关 glob + 唯一性断言 | `080a1c7` | engineering fix, no protocol change | 修复前日志明确打印 SKIP 与旧路径（logs/score_seed23_20261018.log），修复后两个 tag 正常产出；meta.json 记录正确 checkpoint 路径与 sha。**时序披露**：代码编辑先于打分执行，commit 时间戳（03:57:35）晚于两个分数文件落盘（03:56:48 / 03:57:13）22–47 秒；一切 H 规则选择产物（csi300_neural_selection.json，04:12:46）严格晚于 commit |
| 2026-10-19 | benchmark 适配层两处：TSDataSampler 要求 master 索引为真 Timestamp（字符串索引与 Timestamp 比较崩溃）；Windows 多进程 spawn 无法 pickle 内存数据集 → n_jobs=0（DataLoader 工程参数，非超参；其余 kwargs 保持 yaml 原文） | `48a78d6` | engineering fix, no protocol change | 崩溃发生在训练开始前（无任何产出）；修复后 6/6 模型正常训练。取证（checks/benchmark_forensics.json）：train 768,935 行 / valid 145,719 行（与 CPU 基线 768,338 差 0.08%=特征有限性口径差），无训练集缩水；XGB 早停 ~106 轮、DE 84 轮（3×28），验证窗 r1 IC 全部在 0.020–0.032 区间 |
| 2026-10-19 | run_benchmarks 增 `--seed`（全局种子与 tag 后缀），服务 DL 2026-10-19 种子扩展条目 | 见当日 log | engineering fix, no protocol change | 仅新增 CLI 参数；seed 1 行为不变 |

*日期按本仓库既有约定（本地时区）记录；对外产物中的时间按用户指示另行换算。

| 日期* | 修复/事件 | 提交 | 性质 | 影响评估 |
|---|---|---|---|---|
| 2026-10-19 | XGB/DE 特征集修正执行：重跑于全 158 列（DL 条目先行）。结果：XGBoost 40s、验证窗 r1 IC **0.0445**（20 列版 0.0323 作废）；DoubleEnsemble 574s、IC **0.0401**（原 0.0245 作废）。对照：CPU lgbm-reg r1（全 158 特征）IC 0.0352——修正后 GBDT 双双超过 LightGBM，排序合理 | 见 DL 2026-10-19 第二条 + 当日 log | spec correction toward official defaults | 20 列版低 IC 确系特征维度枷锁；外部审阅判断成立 |
| 2026-10-19 | H seed2/3 分数重放核验：以修复后代码重打分，并经 rule_grid_neural 幂等分支对 2021–2022 全部 8 规则×2 种子的已存 daily.csv 逐字符比对——16/16 零差异（重放层逐位一致） | — | verification, no change | 080a1c7 时序披露的补充证据 |
| 2026-10-19 | 事故记录：H 核验的网格调用意外以 2-tag 部分运行覆写了 14 臂 selection json；随后全量 14-tag 幂等重跑恢复（全部走比对分支，数值不变） | 见当日 log | operational incident, no data change | 选择产物恢复；教训=部分运行也写汇总文件 |
| 2026-10-19 | 种子扩展完成 + 作废文件替换：seeds 2-3 双流水线并行落地（13/13 训练零失败）；XGB/DE r1 的 20 列版 daily.csv 按 DL 作废条目删除并由 158 列版重新生成（幂等校验正确拦截了差异、按 supersession 处理）；三种子 benchmark 正式选择出炉 | — | execution of registered entries | runs/csi300_bench_selection.json 终版 |
| 2026-10-19 | 冻结分析件修订 v2（解封前，四项核验驱动）：①eval_ic 改用 loss_masks["eval"]——修掉最后 4 个信号日（06-23/24/25/26）h5 出场价越过 2026-06-30 的读数（W2 loss mask 本已排除，原脚本未用它）；②SC5 的 Holm p 从相关系数换算改为真平稳 bootstrap 双侧 p（C0 臂为族内成员，预指定；双臂都报）；③新增 TOST 四态判定表（superior/inferior/equivalent/uncertain，方向优先于等价、共现以 flags 报告）；④自测扩充（判定表优先级、SC5 p 两端行为）。新 sha 7d90d218… | 见当日 commit | analysis elaboration before unsealing | SC1/SC5/SC6/SC7 族构成不变（m=4，SC2 按 moment 之比逐种子报告、SC4 留 Gate C——均在文件头声明） |
