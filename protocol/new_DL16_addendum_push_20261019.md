# DL-16 附录：Gate B 提交与推送记录（2026-10-19）

- 仓库：D:\research\new_LiMT → origin https://github.com/UworthBetter/new-limt.git（main）
- 执行依据：commit plan r15（new_commit_push_plan_r15.md 6f813e65…；manifest new_commit_manifest_r15.json e671ca2c…）
- 本仓库 core.autocrlf=false（local）；.gitattributes 对全部 85 个路径 -text，git check-attr text = unset（85/85）

## 提交（每个提交前逐个核对暂存 blob sha256 = manifest，全部一致）
| 步骤 | commit | 文件数 |
|---|---|---|
| 0 .gitattributes | 301e8e27d1cf1d25ca88fa08c0b6dcdb49bad160 | 1 |
| D 已运行代码+v2 spec | 56d6b080a60be415bdec2cadd1c2665e777a029a | 16 |
| A DL-25..28 | 0870348f9534b0a268b410d7df603ae6ca9de512 | 12 |
| B guard 冻结 | d393a74c77687874eb86d953a40955932a2ad989 | 2 |
| B2 r9 交付 | 2e445d876b93ce6f1f1c5d4d5bf2a2c4c466811a | 43 |
| C DL-24 + 冻结 h5 spec 77a2d8db | 0e9b4485e56e4ad21a6c8ba955360ce0f5631e8d | 11 |

此前未推送的 b7306e0、7e0ffbd 同批推送。

## 推送
- 时间：2026-10-19 09:50:41–09:50:53 PDT；结果 bc0dc13..0e9b448 main -> main，非强推，未需交互凭据。
- origin/main = 0e9b4485e56e4ad21a6c8ba955360ce0f5631e8d

## 重新克隆核验（D:\research\new_LiMT_verify_clone_20261019_095105，-c core.autocrlf=false）
- manifest 85/85 sha 一致。
- GPT release new_SHA256SUMS.txt（55 条，路径相对仓库根）：53 条一致；2 条为预期例外：
  - configs/new_benchmarks_h5_v3.json：SUMS 记 039ee1cb（GPT 原稿），仓库为批准后的冻结版 77a2d8db（DL-24）；原 039ee1cb 备份于工作区 configs/old/new_benchmarks_h5_v3.json.039ee1cb.bak（未提交）。
  - protocol/new_DL25_guard_r9_implementation_draft.md（784ee584）：不在提交计划内，未提交（笔记本工作区仍有该文件，未追踪）。

## 偏差说明
- B2 中 4 个文件（release/new_smoke_models/*.pt 2 个，release/new_smoke_scores/*.npz 2 个）被 .gitignore 忽略，使用 `git add -f` 加上这 4 个明确路径（仍为显式路径，sha 已核对）。
- commit D 的 2 个文件（predl3 草稿 2a65bd42、diff 0b5ca0e2）漏放在暂存 zip 中，已按计划从盒子补拷后再执行；未提交前发现并停止，无错误提交。
- 未纳入：runs/new_r9_smoke_*、kaggle_csi500_h/logs_pipeline_20261018.log（仍为已修改未暂存）、new_smoke_cache/*_fit.pt。
