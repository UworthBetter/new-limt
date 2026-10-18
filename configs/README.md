# 配置

`draft/` 里的文件是 2026-10-17 从预注册和上游抄下来的草稿。
`gate_b_frozen` 都是 false。Gate B 之前不要把它们复制成正式运行配置。

| 文件 | 含义 |
|---|---|
| `draft/preregistration_target.json` | 预注册要的实验，不是 `v2_w6_run.py` 能直接吃的配置 |
| `draft/resolved_upstream_formal.json` | `configs/w4w6_formal.json` 盖到上游 `DEFAULT` 上之后，代码实际会跑的那一份 |
| `draft/DIFF.md` | 上面两份差在哪 |

差没补上之前，不要启动正式 GPU 训练。
