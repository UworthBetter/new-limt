# Kaggle 卸载方案：CSI500 H × 3 seeds

日期：2026-10-18。目的：把 CSI500 的 H 训练搬到 Kaggle GPU，与本地 4050 的 CSI300 种子链并行。

## 为什么搬这个（而不是别的）

- **不动正在跑的东西**：本地种子链（seeds 2-3）和夜间 benchmark 队列照旧。
- **CSI500 H 是矩阵内、配置已冻结的工作**（预注册 §8 GPU per market/seed: H/G0/C0；§6.2 CSI500 复用 CSI300 配置、不调参）。训练本身不受 Gate 门禁——门禁管的是"打开 2023+ 输出"，H 训练只写 checkpoint 与验证窗 history，不产生也不读任何 2023+ 输出。
- **H 不需要 execution feed**（纯监督训练）；G0/C0 需要 CSI500 对齐的成交账本，那个是本地数据工程（baostock 断点续传下载 1802 只），与 Kaggle 无关，另行推进。
- 预计 Kaggle 用量：早停同 CSI300（~11-15 epoch）≈ 每 seed 1.6-2.2h，3 seeds ≈ 5-7h（一个 12h 会话内）；最坏 50 epoch 不早停 ≈ 21h → 第二个会话续跑（脚本按 seed 断点续传、每个 seed 完成后重新打包 zip）。

## 上传方式（ZCode 已于 2026-10-18 晚用 API 代办）

数据集 `beelingz/csi500-newlimt` 已创建：v1 = code（11 个文件），v2 = +三个 CSI500 npz（5.7GB）。布局**扁平**（kaggle CLI 跳过子目录）：所有文件在数据集根，`kaggle_run.sh` 按此取路径。若需手工重传，staging 在 `kaggle_csi500_h/upload/`（npz 为硬链接，不占额外磁盘）。

手动路径（备查）：Kaggle → Datasets → New Dataset → 命名 `csi500-newlimt` → 上传 `upload/` 下全部文件 → Notebooks → Import `csi500_H_kaggle.ipynb` → Add Input 挂载数据集 → GPU T4/P100 → Run All。

## 结果回收

- 跑完在 `/kaggle/working/csi500_H_kaggle_results.zip` 下载，放到 `D:\research\new_LiMT\runs\kaggle_incoming\`，然后告诉 ZCode 做验收（checkpoint sha + config hash + data_version + history 完整性核验后并入 runs/）。

## 完整性核对值（ZCode 本地预计算，2026-10-18）

| 项 | 期望值 |
|---|---|
| CSI500 data_version | `bae7115ee7b0b2d05f22f12ecef5c6e37c13cc697dd15a98524cfcf273f29ec1` |
| config hash（与 CSI300 正式跑逐位同） | `393b19932ea61cdb542cf970cf23a0e82741a3b59f7a8b76be7f9083ac3230d9` |
| trainer sha256（v2_w4_training.py） | `a960a9951ab52c5455c15f8e0c5d60a783a6beb4beef5b4003ee578f0c21ee43` |

三项全部由脚本在 Kaggle 上自检/打印；data_version 不一致 = 上传损坏，直接停。

## 注意

- Kaggle 的 torch/驱动版本与本地不同：同一 seed 在不同 GPU 上不保证逐位一致轨迹（这在预注册框架内本来就允许——每个 run 一次成型、记录环境；本地 run 同样记录）。
- 若 12h 会话中断：重新 Run All 即可，已完成的 seed 会被跳过。
- 第二阶段（CSI500 G0/C0）等本地 execution feed 建好后再议：届时 payload 还需加 feed npz，且 6 个微调 × 4-6h 超周配额，需要 Kaggle/本地分摊。
