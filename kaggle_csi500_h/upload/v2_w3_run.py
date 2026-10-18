"""Fail-closed W3 CPU acceptance runner; never overwrites an existing run directory."""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def verify_fix1(root):
    """Verify exact W1/W2 report AND all five data artifacts against its manifest."""
    root = Path(root)
    manifest = root / "results/v2/new_run_manifest_instance.txt"
    contents = manifest.read_text(encoding="utf-8")
    if not re.search(r"^status=SUCCESS stage=complete exit_code=0(?: |$)", contents, re.M):
        raise ValueError("W1/W2 manifest has no successful completion")
    hashes = {}
    for line in contents.splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})\s+(.+)", line)
        if match:
            filename = match[2].replace("\\", "/").split("/")[-1]
            if filename in hashes:
                raise ValueError(f"ambiguous duplicate manifest file: {filename}")
            hashes[filename] = match[1]
    names = ["data/v2/new_v2_labels_csi300.npz", "data/v2/new_v2_panel_csi300.npz",
             "data/v2/new_v2_splits_csi300.npz", "data/v2/new_v2_feature_probe_csi300.npz",
             "data/v2/new_v2_W1_meta_csi300.json", "results/v2/new_W0W2_acceptance.json",
             "results/v2/new_W0W2_acceptance.md", "results/v2/new_environment_freeze.txt"]
    checked = {}
    for relative in names:
        path = root / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing/empty W1/W2 artifact: {path}")
        print(f"verify sha256 {relative}", flush=True)
        actual = digest(path)
        if actual != hashes.get(path.name):
            raise ValueError(f"W1/W2 hash mismatch: {path}")
        checked[relative] = actual
    report = json.loads((root / names[5]).read_text(encoding="utf-8"))
    required = ["0_array_contract", "1_SZ002074_2018-12-26", "1_SH600549_2019-06-26",
                "2_endpoint_misalignment", "4_purge", "5_masks"]
    if any(report.get(k, {}).get("status") != "PASS" for k in required):
        raise ValueError("W1/W2 mandatory check missing or failed")
    if any(v.get("status") == "FAIL" for v in report.values()):
        raise ValueError("W1/W2 report contains a failure")
    metadata = json.loads((root / names[4]).read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 2 or metadata.get("perturb", {}).get("status") != "PASS":
        raise ValueError("requires W1 schema v2 and a passed physical rebuild")
    return dict(status="PASS", root=str(root.resolve()), manifest_sha256=digest(manifest), artifacts=checked)


def fixture_replays(out):
    """Export auditable deterministic reference/proxy and constrained execution paths."""
    import csv
    import numpy as np
    import torch
    from v2_w3_ledger import replay_daily, continuous_step
    dates = np.array(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09"])
    # Currency/share synthetic quotes. Day zero only supplies the signal timestamp.
    prices = np.array([[10., 20.], [10., 20.], [11., 20.], [11., 18.], [12., 19.], [12.5, 19.5]])
    states = np.array([[0, 0], [0, 0], [1, 0], [3, 0], [0, 2], [0, 0]])
    targets = [np.array([0., .5, .5]), np.array([0., .2, .8]), np.array([0., 0., 1.]), None]
    _, discrete = replay_daily(dates, prices, states, targets)
    w, wealth, rows, fills = torch.tensor([1., 0., 0.], dtype=torch.float64), 1_000_000., [], []
    for k, item in enumerate(discrete):
        i = k + 1
        y = np.r_[1., prices[i] / prices[i - 1]] if k else np.ones(3)
        target = targets[k] if k < len(targets) else None
        step = continuous_step(w, torch.tensor(y.tolist(), dtype=torch.float64),
                               None if target is None else torch.tensor(target.tolist(), dtype=torch.float64))
        prior = wealth
        wealth *= float(step["gross"] * step["mu"])
        proxy_fee = prior * float(step["gross"] * step["fee"])
        w = step["weights"]
        row = {key: value for key, value in item.items() if key != "orders"}
        row.update(proxy_wealth=wealth, proxy_fee=proxy_fee,
                   proxy_turnover=float(step["turnover"]), proxy_target_penalty=float(step["target_penalty"]),
                   wealth_gap=item["wealth"] - wealth)
        rows.append(row)
        for order in item["orders"]:
            fills.append(dict(signal_date=item["signal_date"], execution_date=item["execution_date"], **order))
    for name, data in [("daily_comparison.csv", rows), ("executed_orders.csv", fills)]:
        with (out / name).open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    with (out / "synthetic_fixture.json").open("x", encoding="utf-8") as stream:
        json.dump(dict(dates=dates.tolist(), unadjusted_prices=prices.tolist(), execution_states=states.tolist(),
                       signal_targets=[x.tolist() if x is not None else None for x in targets]), stream, indent=2)
    return dict(scope="synthetic accounting fixture, not investment performance",
                events=len(rows), trades=len(fills), terminal_liquidation=False,
                execution_final_wealth=rows[-1]["wealth"], proxy_final_wealth=wealth,
                execution_fees=sum(row["fees"] for row in rows))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--w0w2-root")
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    if args.threads < 1 or (not args.synthetic_only and not args.w0w2_root):
        parser.error("positive threads and --w0w2-root required (or explicit --synthetic-only)")
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="RUNNING", stage="setup", start=now(), python=platform.python_version(),
                    scope="W3_SYNTHETIC_ONLY" if args.synthetic_only else "W3_CPU_ACCEPTANCE_AND_FIX1_HASH_GATE",
                    configuration=dict(buy_rate=.0005, sell_rate=.0015, min_fee=5., lot=100,
                        terminal="mark without liquidation", valuation="last known mark; missing quote forbids fill",
                        share_units="unadjusted price with corporate actions reflected in shares"))
    def save():
        temporary = out / ".manifest.tmp"
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(out / "run_manifest.json")
    save()
    try:
        available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
        threads = min(args.threads, available)
        for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "W3_THREADS"]:
            os.environ[key] = str(threads)
        import numpy as np
        import torch
        torch.set_num_threads(threads)
        manifest.update(numpy=np.__version__, torch=torch.__version__, threads=threads)
        source = Path(__file__).resolve().parent
        test = source.parent / "tests/test_v2_w3_ledger.py"
        tracked = [source / "v2_w3_ledger.py", Path(__file__).resolve(), source / "run_w3.sh", test]
        manifest["code_sha256"] = {str(p): digest(p) for p in tracked}
        with (out / "environment_freeze.txt").open("x", encoding="utf-8") as log:
            subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=log, stderr=subprocess.STDOUT, check=True)
        manifest["stage"] = "unit_tests"; save()
        command = [sys.executable, "-m", "pytest", str(test), "-q", "-p", "no:cacheprovider", f"--junitxml={out / 'tests.xml'}"]
        manifest["test_command"] = command
        print("W3 deterministic accounting/execution/gradient tests", flush=True)
        with (out / "tests.log").open("x", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"pytest failed with exit code {result.returncode}; see tests.log")
        tree = ET.parse(out / "tests.xml")
        cases = tree.findall(".//testcase")
        counts = dict(total=len(cases), failures=len(tree.findall(".//failure")), errors=len(tree.findall(".//error")), skipped=len(tree.findall(".//skipped")))
        if counts["total"] == 0 or any(counts[k] for k in ["failures", "errors", "skipped"]):
            raise RuntimeError(f"incomplete acceptance: {counts}")
        manifest["tests"] = counts
        manifest["stage"] = "fix1_inputs"; save()
        manifest["fix1_inputs"] = dict(status="SKIPPED_EXPLICIT_SYNTHETIC_ONLY") if args.synthetic_only else verify_fix1(args.w0w2_root)
        manifest["stage"] = "fixture_replay"; save()
        manifest["replay"] = fixture_replays(out)
        summary = ("# W3 CPU 验收结果\n\n"
            f"范围：{manifest['scope']}。合成数据用于检查账本与撮合，不作为策略收益证据。\n\n"
            f"- 程序检查：{counts['total']} 项通过，失败 0，跳过 0。\n"
            f"- fix1 产物哈希：{manifest['fix1_inputs']['status']}。\n"
            "- daily_comparison.csv：连续比例费用代理与股数执行层的逐日比较。\n"
            "- executed_orders.csv：每笔实际成交数量、价格、成交额和费用。\n"
            "- 最后一个事件只估值，保留仓位；末次持仓收益计入财富。\n\n"
            "通过本验收后可接入 W4 训练；真实数据执行回测还需未复权报价、公司行动处理和可靠的交易状态。\n")
        (out / "acceptance.md").write_text(summary, encoding="utf-8")
        manifest["output_sha256"] = {p.name: digest(p) for p in out.iterdir() if p.is_file() and p.name != "run_manifest.json"}
        manifest.update(status="SUCCESS", stage="complete", exit_code=0, end=now()); save()
        print(f"SUCCESS: {counts['total']} checks, output {out}", flush=True)
        return 0
    except Exception as error:
        manifest.update(status="FAILED", error=f"{type(error).__name__}: {error}", exit_code=1, end=now())
        save(); traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
