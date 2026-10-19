# CPU 重训输入上传记录 r12

新文件，取代 `new_cpu_retrain_inputs_upload_r11.md`（b5182264…，保留不改），r11 全文附在后面。

## r12 新增（Claude 08:43 批准）

| 笔记本文件 | 上传位置 | 上传前 sha256 | 下载到盒子后 | 结果 |
|---|---|---|---|---|
| `configs/benchmarks_qlib_defaults_v1.json`（4233 字节，LF，crlf=0） | `UworthBetter/new-limt-gateb-inputs`：`new_LiMT/cpu_retrain_inputs_20261019/configs/benchmarks_qlib_defaults_v1.json` | e189f6cbe3453d172fc407b6bf2117be15bcba7c72c698f53f125d92533e5165 | e189f6cbe3453d172fc407b6bf2117be15bcba7c72c698f53f125d92533e5165 | **MATCH**，不需要处理 CRLF |

- 用笔记本的 venv `D:\tools\new_ms-venv` 上传，用的是缓存的登录，没有打印 token。
- 盒子上的下载副本在 `/workspace/limt_gateb/new_cpu_inputs_verify_r12/`。
- 已有的 SHA256SUMS（9218a988…）**不包含**这个文件，也没有改动；以本记录为准。

---

# CPU 重训输入上传记录（r11）

实验助手，2026-10-19，时间均为 PDT。新文件。依据 Claude 08:30 PDT 批准的第 7 条。

- **上传方式**：笔记本 venv `D:\tools\new_ms-venv`（Python 3.13.9，用缓存的登录，没有打印 token）→ 私有数据集 `UworthBetter/new-limt-gateb-inputs`，目录 `new_LiMT/cpu_retrain_inputs_20261019/`。
- **上传时间**：08:3x PDT。服务端提交时间 1792423988–1792423989，即约 08:33 PDT。
- **共 19 个文件**：
  - 2 个代码文件；
  - 16 个 ridge 2021–22 规则回放的 `daily.csv`（`runs/csi300/ridge/{h5,r1}/<8 条规则>/seed1/`）；
  - 1 个 SHA256SUMS。
- **没有上传 `runs/_sealed/`**：那里有一套同名的 `daily.csv`，体积更大（约 105–137 KB，开封前的副本约 60–78 KB），看起来含封存期数据。脚本断言过路径里不含 `_sealed`。
- **daily.csv 从头到尾只算过哈希**，笔记本和盒子两边都没有打开或打印内容。
- **核对方式**：盒子把这 19 个文件全部下载下来，只用 `sha256sum -c` 比对。
- **结果**：17/18 字节一致。`run_benchmarks.py` 不一致，原因是行尾符：笔记本上的原文件是 CRLF（12210 字节，246 行）；数据集里存的是 LF 版（11964 字节），服务端记录的 Sha256 也是 LF 版的 60eac60a…。把 LF 还原成 CRLF 后，sha 正好等于 824daa85。这和第 5 轮记录过的魔搭 CRLF 规范化现象一致（`new_prep_round5/new_19_crlf_compare.txt`）。**所以 CPU 实例要先把它还原成 CRLF（`sed 's/$/\r/'`），再按 824daa85 校验；或者 runner 改成按 LF 版的 60eac60a 校验，并在 DL 里注明。**

| 路径（相对 `new_LiMT/cpu_retrain_inputs_20261019/`） | 笔记本上传前 sha256 | 盒子下载后 | 结论 |
|---|---|---|---|
| src/run_benchmarks.py | 824daa85536b85e81c28381ebc5b5b4474cd6506dccb03cc500e37e15e5daa8a | 60eac60ac4090bd2a8c7ef4d679d6dbdc632da36f8a7e3f06dc29dc6f570c556（LF）；还原 CRLF 后为 824daa85… | 只差行尾符 |
| src/cpu_baselines.py | 4c585a0fda0d93055aae78e5c2099cfe08ae701d52132803ac77460456e940da | 一致 | OK |
| runs/csi300/ridge/h5/drop10-every1d/seed1/daily.csv | f81907963427de1a8a885fac9f04d382251dbed65ffa9a61d068d0ccb7c6c1fa | 一致 | OK |
| runs/csi300/ridge/h5/drop10-every5d/seed1/daily.csv | 1206f8b48d9c26fef3fb1d31ce94b118819612abd9eca226673ebd1b45e2195f | 一致 | OK |
| runs/csi300/ridge/h5/drop2-every1d/seed1/daily.csv | 8141b04bfaef7215fa3ec6fc09a2fb735514b838045f943f2753d1bc1eb23563 | 一致 | OK |
| runs/csi300/ridge/h5/drop2-every5d/seed1/daily.csv | c5b9affdca923a6a40745c9e8a4eab03d0dc2b74699b71efaaa37f6448428624 | 一致 | OK |
| runs/csi300/ridge/h5/drop20-every1d/seed1/daily.csv | d73712df0814dfb9114bd277f899676d6bd174a7c3293e80b9c18e8accc96d84 | 一致 | OK |
| runs/csi300/ridge/h5/drop20-every5d/seed1/daily.csv | 82c90c116f473ebd5cda652474b3e877b0ce696a1a14d1d4c51d950233ccf7fa | 一致 | OK |
| runs/csi300/ridge/h5/drop5-every1d/seed1/daily.csv | 0ea1d48a085315cd76e74162159a5f36370d7852b210b3f1ccb0292c6005a22c | 一致 | OK |
| runs/csi300/ridge/h5/drop5-every5d/seed1/daily.csv | 1ed21cd6d6dff7ac35bbd05848f60eae1fa18f947e6e935c1e344d85a9c5aeca | 一致 | OK |
| runs/csi300/ridge/r1/drop10-every1d/seed1/daily.csv | 02f9724a921308c50e9d0b1ed4e95c74d5485f4d9dc903cf2d75b2e3bb0cc444 | 一致 | OK |
| runs/csi300/ridge/r1/drop10-every5d/seed1/daily.csv | 17b1703fbf56e9abc686703f345ddac41202e71f6aabec137d274c2aa45e6414 | 一致 | OK |
| runs/csi300/ridge/r1/drop2-every1d/seed1/daily.csv | d75d91ac2fdb7ba947e4ab6f5e6a9dbd175031ed30c838303250f3006dfd49b2 | 一致 | OK |
| runs/csi300/ridge/r1/drop2-every5d/seed1/daily.csv | 6c88567753ad8a291339ba8b76ce6bd8d9f0ab6b95764ae822cb254fe130415f | 一致 | OK |
| runs/csi300/ridge/r1/drop20-every1d/seed1/daily.csv | 60b9a1f0d363a95678e0b377b5771ec3bf536b30b0ec71a03649ed7de4304dbd | 一致 | OK |
| runs/csi300/ridge/r1/drop20-every5d/seed1/daily.csv | 36d76150350ed693e238945bbbab63b840a62281f1af307708d99266aa041fe0 | 一致 | OK |
| runs/csi300/ridge/r1/drop5-every1d/seed1/daily.csv | ab27208f6d2aee683a0993527abcb5810ff9ddd86b55848618ca62276222eea4 | 一致 | OK |
| runs/csi300/ridge/r1/drop5-every5d/seed1/daily.csv | c9e7ecc6a0075904415f59e396ead2fd9f0e2f1dd546157fa6cfcc5717330da1 | 一致 | OK |
| SHA256SUMS | 9218a988d04e95cdae3b22efd5193d4ecf0d5f1d743d4f1ff8e7ba7630bb8385 | 一致 | OK |

盒子上的副本在 `/workspace/limt_gateb/new_cpu_inputs_verify_r11/`，只用来算哈希，不打开。
