# 断言

`test_preregistration_assertions.py` 对照 `D:\research\limt-repro` 检查预注册第 6.5 节。

在 `new_LiMT` 下运行。默认 `python` 没有 torch，用上游虚拟环境：

```
D:\research\limt-repro\.venv\Scripts\python.exe -m pytest tests/test_preregistration_assertions.py -q --tb=line
```

`test_a_g0_a_lbl.py` 是 A-G0-1 至 A-G0-4 和 A-LBL-1。
A-LBL-1 读 fix1 的 CSI300 标签，不读模型输出。
A-G0 四条在监控账本写进 `v2_w4_training.py` 之前应当失败。

`test_a_g0_selection_uses_separate_real_fee_monitoring_ledger` 在上游补上监控账本之前应当失败。
失败是预期信号，不要改成跳过。
其他断言通过，只说明那几条已经和当前代码一致。

输出副本放在 `protocol/checks/`，不要只留在终端里。
