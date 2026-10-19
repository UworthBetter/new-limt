# 工程修复记录（非协议变更）

准则：只记实现层修复，不动预注册任何阈值/网格/判据。每条注明"engineering fix, no protocol change"与提交哈希。与偏差日志（协议层）分开维护。

| 日期* | 修复 | 提交 | 性质 | 影响评估 |
|---|---|---|---|---|
| 2026-10-19 | H 种子 2/3 打分路径写死 `_20261017` 目录导致 SKIP（H-FT/G0/C0 不受影响）；改为日期无关 glob + 唯一性断言 | `080a1c7` | engineering fix, no protocol change | 修复前日志明确打印 SKIP 与旧路径（logs/score_seed23_20261018.log），修复后两个 tag 正常产出；meta.json 记录正确 checkpoint 路径与 sha。**时序披露**：代码编辑先于打分执行，commit 时间戳（03:57:35）晚于两个分数文件落盘（03:56:48 / 03:57:13）22–47 秒；一切 H 规则选择产物（csi300_neural_selection.json，04:12:46）严格晚于 commit |
| 2026-10-19 | benchmark 适配层两处：TSDataSampler 要求 master 索引为真 Timestamp（字符串索引与 Timestamp 比较崩溃）；Windows 多进程 spawn 无法 pickle 内存数据集 → n_jobs=0（DataLoader 工程参数，非超参；其余 kwargs 保持 yaml 原文） | `48a78d6` | engineering fix, no protocol change | 崩溃发生在训练开始前（无任何产出）；修复后 6/6 模型正常训练。取证（checks/benchmark_forensics.json）：train 768,935 行 / valid 145,719 行（与 CPU 基线 768,338 差 0.08%=特征有限性口径差），无训练集缩水；XGB 早停 ~106 轮、DE 84 轮（3×28），验证窗 r1 IC 全部在 0.020–0.032 区间 |
| 2026-10-19 | run_benchmarks 增 `--seed`（全局种子与 tag 后缀），服务 DL 2026-10-19 种子扩展条目 | 见当日 log | engineering fix, no protocol change | 仅新增 CLI 参数；seed 1 行为不变 |

*日期按本仓库既有约定（本地时区）记录；对外产物中的时间按用户指示另行换算。
