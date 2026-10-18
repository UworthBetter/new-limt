"""Serial W4-W6 orchestration: verified data -> ridge/H/fine-tunes -> W5 artifacts."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from v2_w6_policy import DEFAULT_EVALUATION_END, splits_for

DEFAULT = dict(mode="smoke", seeds=[1], dimension=32, lookback=5, batch_days=4, max_symbols=32,
    train_days=12, valid_days=8, report_days=8, epochs_h=1, epochs_finetune=1, patience=0,
    lr_h=1e-4, lr_finetune=1e-5, tau=.5, alpha_aux=.3, gamma_to=.001, beta_risk=0.,
    buy_rate=.0005, sell_rate=.0015, clip_norm=1., amp=False, threads=8,
    ridge_alpha=10., rules=["soft", "topk-reference"], topk=50, drop=5,
    capital=100000000., min_fee=5., lot=100, device="cpu", st_policy="limits_only",
    evaluation_end=DEFAULT_EVALUATION_END)


def validate_config(config):
    if set(config) != set(DEFAULT):
        raise ValueError(f"unknown/missing configuration keys: {set(config) ^ set(DEFAULT)}")
    splits_for(config['evaluation_end'])
    if config["mode"] not in ("smoke", "formal") or config["beta_risk"] != 0:
        raise ValueError("mode must be smoke/formal and beta_risk must be zero")
    if config['st_policy'] not in ('limits_only','exclude_signal'):
        raise ValueError('ST policy must be limits_only/exclude_signal')
    positive = ["dimension", "lookback", "batch_days", "epochs_h", "epochs_finetune", "lr_h", "lr_finetune", "tau", "clip_norm", "threads", "ridge_alpha", "topk", "capital", "lot"]
    import math
    if any(not math.isfinite(config[k]) or config[k] <= 0 for k in positive):
        raise ValueError("nonpositive/nonfinite hyperparameter")
    integers = ["dimension", "lookback", "batch_days", "epochs_h", "epochs_finetune", "threads", "topk", "lot", "drop", "patience", "max_symbols", "train_days", "valid_days", "report_days"]
    if any(type(config[k]) is not int or config[k] < 0 for k in integers):
        raise ValueError("invalid integer hyperparameter")
    if config["dimension"] % 4 or config["batch_days"] < 2 or config["drop"] > config["topk"]:
        raise ValueError("dimension/batch/topk contract violated")
    if any(not math.isfinite(config[k]) or config[k] < 0 for k in ["alpha_aux", "gamma_to", "min_fee"]):
        raise ValueError("invalid auxiliary/turnover/minimum-fee coefficient")
    if any(not 0 <= config[k] < 1 for k in ["buy_rate", "sell_rate"]):
        raise ValueError("invalid fees")
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]) or any(type(s) is not int or s < 0 for s in config["seeds"]):
        raise ValueError("seeds must be distinct nonnegative integers")
    if not config["rules"] or len(set(config["rules"])) != len(config["rules"]) or set(config["rules"]) - {"soft", "topk-reference", "topk-execution"}:
        raise ValueError("invalid rules")
    if config["device"] not in ("cpu", "cuda", "auto") or type(config["amp"]) is not bool:
        raise ValueError("invalid device/AMP flag")
    if config["mode"] == "formal" and any(config[k] for k in ["max_symbols", "train_days", "valid_days", "report_days"]):
        raise ValueError("formal runs cannot silently use a smoke subset")
    return config


def report_days(data, name, config):
    if name == "full-test":
        import numpy as np
        start = data.days("rule_select")[0]; end = data.days("eval")[-1]
        days = np.arange(start, end + 1)
        return days[:config["report_days"]] if config["report_days"] else days
    return data.days(name, limit=config["report_days"])


def run_pipeline(data, config, out, identity, run_id, bench=None, execution_feed=None, on_progress=None):
    import numpy as np
    import torch
    from v2_w3_run import digest
    from v2_w4_training import train, predict, ridge_fit, ridge_predict, portfolio_events
    from v2_w5_results import canonical, json_write, daily_ic, write_daily, summary, actual_execution, load_execution_feed
    if data.splits!=splits_for(config['evaluation_end']):
        raise ValueError('data/run evaluation_end mismatch')
    out = Path(out)
    if execution_feed is not None:
        execution_feed = load_execution_feed(data, execution_feed)
    windows = ["rule_select", "eval", "full-test"]
    # Canonical output identifiers use hyphens, including window names.
    models = ["ridge-h5", "H", "H-FT", "G0", "C0", "C-T"]
    cells = [dict(model_id=m, seed=s, window=w.replace("_", "-"), rule_id=r, status="pending", error="")
        for s in config["seeds"] for m in models for w in windows for r in config["rules"]]
    ic_rows, produced = [], {}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    directory = out / "cells"; directory.mkdir()
    checkpoints = out / "models"; checkpoints.mkdir()
    devices = "cuda" if config["device"] == "auto" and torch.cuda.is_available() else config["device"]
    device = "cpu" if devices == "auto" else devices
    for seed in config["seeds"]:
        clean_h = None
        for group in models:
            location = checkpoints / f"{group}__s{seed}"; location.mkdir()
            model = None; coefficients = None; start = time.perf_counter()
            these_cells = [c for c in cells if c["model_id"] == group and c["seed"] == seed]
            for cell in these_cells:
                cell["status"] = "running"
            if on_progress:
                on_progress(group, seed, "training", cells)
            try:
                if group == "ridge-h5":
                    coefficients, count = ridge_fit(data, config)
                    with (location / "ridge.npz").open("xb") as stream:
                        np.savez(stream, coefficients=coefficients)
                    json_write(location / "fit.json", dict(data_version=identity, alpha=config["ridge_alpha"], samples=count))
                    checkpoint = location / "ridge.npz"
                elif group == "C-T" and config["gamma_to"] == 0:
                    # No duplicate fitting for an exactly degenerate group.
                    checkpoint = checkpoints / f"C0__s{seed}" / "checkpoint.pt"
                    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
                    from v2_w4_training import make_model
                    model = make_model(data, config).to(device); model.load_state_dict(state["state_dict"])
                    json_write(location / "reuse.json", dict(reused_from=str(checkpoint.relative_to(out)), sha256=digest(checkpoint)))
                else:
                    model, state = train(data, config, group, seed, location, device, identity, clean_h, execution_feed)
                    checkpoint = location / "checkpoint.pt"
                    if group == "H":
                        clean_h = checkpoint
                produced[f"{group}/s{seed}"] = dict(checkpoint=str(checkpoint.relative_to(out)), checkpoint_hash=digest(checkpoint),
                    training_seconds=time.perf_counter() - start)
                for window_name in windows:
                    days = report_days(data, window_name, config)
                    predictions = ridge_predict(data, days, coefficients) if model is None else predict(model, data, days, config, device)
                    window_id = window_name.replace("_", "-")
                    name = canonical(run_id, group, seed, "predictions", window_id)
                    with (directory / f"{name}.npz").open("xb") as stream:
                        np.savez_compressed(stream, dates=data.dates[days].astype(str), symbols=data.symbols,
                            scores=predictions[..., 0], candidate=np.stack([data.candidate(t) for t in days]))
                    json_write(directory / f"{name}.meta.json", dict(data_version=identity, config_hash=config_hash,
                        checkpoint_hash=digest(checkpoint), sha256=digest(directory / f"{name}.npz"), seed=seed, model_id=group))
                    ic = daily_ic(data, days, predictions[..., 0], window_name)
                    json_write(directory / f"{name}.ic.json", ic)
                    ic_rows.append(dict(model_id=group, seed=seed, window=window_id, **{k: v for k, v in ic.items() if k != "records"}))
                    for cell in [c for c in these_cells if c["window"] == window_id]:
                        try:
                            rule = cell["rule_id"]
                            if rule == "topk-execution":
                                if execution_feed is None:
                                    raise ValueError("raw execution feed required")
                                rows = actual_execution(data, days, predictions[..., 0], execution_feed, config)
                            else:
                                rows = portfolio_events(data, days, predictions[..., 0], config["tau"], config["buy_rate"], config["sell_rate"], rule,
                                                        config["topk"], config["drop"])
                            meta = dict(run_id=run_id, model_id=group, seed=seed, rule_id=rule, data_version=identity,
                                config_hash=config_hash, checkpoint_hash=digest(checkpoint), producer="v2_w6_run",
                                objective_family="supervised-h5" if group in ("H", "H-FT", "ridge-h5") else "log-growth",
                                alpha_aux=config["alpha_aux"] if group in ("G0", "C0", "C-T") else None,
                                gamma_to=config["gamma_to"] if group == "C-T" else 0., beta_risk=0.,
                                fee_convention="fees = actual notional * rate; min fee only in share executor",
                                state_approximation="TBPTT; all-cash epoch/window start; no terminal liquidation",
                                execution_layer="share-execution" if rule == "topk-execution" else "continuous-proportional-reference",
                                scope="engineering_smoke" if config["mode"] == "smoke" else "historical_exploratory",
                                benchmark_id=None if bench is None else bench["identity"])
                            meta.update(evaluation_end=config['evaluation_end'],unknown_state_policy='block_both_keep_code_4',
                                delisted_holdings_policy='freeze_last_mark_no_synthetic_sale_or_conversion')
                            base = canonical(run_id, group, seed, rule, window_id)
                            path = directory / f"{base}.csv"
                            bounds = (data.splits["rule_select"][0], data.splits["eval"][1]) if window_name == "full-test" else data.splits[window_name]
                            measured = write_daily(path, rows, meta, bounds, bench)
                            cell.update(status="success", path=str(path.relative_to(out)), metrics=measured)
                        except Exception as error:
                            cell.update(status="failed", error=f"{type(error).__name__}: {error}")
                if on_progress:
                    on_progress(group, seed, "finished", cells)
            except Exception as error:
                (location / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
                for cell in these_cells:
                    if cell["status"] in ("running", "pending"):
                        cell.update(status="failed", error=f"{type(error).__name__}: {error}")
                if on_progress:
                    on_progress(group, seed, "failed", cells)
            finally:
                del model
                if device == "cuda":
                    torch.cuda.empty_cache()
    json_write(out / "checkpoint_index.json", produced)
    return summary(out, cells, ic_rows), cells


def main(argv=None):
    from v2_w3_run import now, digest, verify_fix1
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True); parser.add_argument("--out-dir", required=True)
    parser.add_argument("--w0w2-root"); parser.add_argument("--benchmark-csv"); parser.add_argument("--execution-feed")
    parser.add_argument("--synthetic-demo", action="store_true")
    args = parser.parse_args(argv)
    supplied = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    if set(supplied) - set(DEFAULT):
        parser.error("unknown configuration keys")
    config = validate_config(dict(DEFAULT, **supplied))
    if args.synthetic_demo and config["mode"] != "smoke":
        parser.error("synthetic data may only use smoke mode")
    if not args.synthetic_demo and not args.w0w2_root:
        parser.error("--w0w2-root required for real data")
    out = Path(args.out_dir).resolve(); out.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="RUNNING", stage="preflight", start=now(), config=config,
        scope="SYNTHETIC_ENGINEERING_DEMO" if args.synthetic_demo else config["mode"],
        phase="W4-W6", benchmark="not requested in smoke" if not args.benchmark_csv else args.benchmark_csv)
    def save():
        temporary = out / ".manifest.tmp"
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(out / "run_manifest.json")
    save()
    start = time.perf_counter()
    try:
        affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
        threads = min(config["threads"], affinity)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "W3_THREADS"):
            os.environ[key] = str(threads)
        import numpy as np
        import torch
        torch.set_num_threads(threads)
        device = "cuda" if config["device"] == "auto" and torch.cuda.is_available() else config["device"]
        if config["mode"] == "formal" and (device != "cuda" or not torch.cuda.is_available()):
            raise ValueError("formal full-universe configuration requires CUDA; use smoke on CPU")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        from v2_w5_results import benchmark, json_write
        if config["mode"] == "formal" and not args.benchmark_csv:
            raise ValueError("formal evaluation requires H00300 total-return CSV + provenance sidecar")
        bench = benchmark(args.benchmark_csv) if args.benchmark_csv else None
        if "topk-execution" in config["rules"]:
            if not args.execution_feed:
                raise ValueError("topk-execution requires an explicit raw-price/corporate-action feed")
            meta = json.loads(Path(args.execution_feed + ".meta.json").read_text(encoding="utf-8"))
            if meta.get("sha256") != digest(args.execution_feed) or meta.get("price_kind") != "unadjusted" or meta.get("corporate_actions") != "explicit" or not meta.get("source"):
                raise ValueError("invalid execution feed provenance")
            manifest["execution_feed"] = meta
            if config['mode']=='formal' and meta.get('market_state_schema')!='historical_states_v1':
                raise ValueError('formal execution requires verified historical market state provenance')
        source = Path(__file__).resolve().parent
        tracked = [*source.glob("v2_w[3456]*.py"), source / "model.py", source / "run_w4w6.sh",
                   source / "v2_market_states.py", source / "build_market_state_matrix.py",
                   source / "download_execution_baostock.py",
                   source.parent / "tests/test_v2_w3_ledger.py", source.parent / "tests/test_v2_w4w6.py",
                   source.parent / "tests/test_market_states.py", source.parent / "tests/test_v2_run_policy.py", Path(args.config)]
        manifest.update(code_sha256={str(p): digest(p) for p in tracked}, python=sys.version,
            numpy=np.__version__, torch=torch.__version__, threads=threads, actual_device="cpu" if device == "auto" else device)
        with (out / "environment_freeze.txt").open("x", encoding="utf-8") as stream:
            subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=stream, stderr=subprocess.STDOUT, check=True)
        manifest["stage"] = "unit_checks"; save()
        with (out / "checks.log").open("x", encoding="utf-8") as stream:
            subprocess.run([sys.executable, "-m", "pytest", str(source.parent / "tests/test_v2_w3_ledger.py"),
                str(source.parent / "tests/test_v2_w4w6.py"), str(source.parent / "tests/test_market_states.py"),
                str(source.parent / "tests/test_v2_run_policy.py"),
                "-q", "-p", "no:cacheprovider", f"--junitxml={out / 'checks.xml'}"],
                stdout=stream, stderr=subprocess.STDOUT, check=True)
        import xml.etree.ElementTree as ET
        tree = ET.parse(out / "checks.xml")
        if not tree.findall(".//testcase") or any(tree.findall(".//" + k) for k in ["failure", "error", "skipped"]):
            raise ValueError("incomplete/skipped acceptance checks")
        manifest["checks_passed"] = len(tree.findall(".//testcase"))
        manifest["stage"] = "data_gate"; save()
        from v2_w4_data import Data, write_contract
        if args.synthetic_demo:
            from v2_w6_demo import make_demo
            data = make_demo(out / "synthetic_inputs", out / "cache", config)
            identity = "SYNTHETIC-" + digest(out / "synthetic_inputs/data/v2/new_v2_panel_csi300.npz")
            manifest["upstream"] = "explicit synthetic demo; no W0-W2 success claim"
        else:
            gate = verify_fix1(args.w0w2_root); manifest["upstream"] = gate
            identity = gate["manifest_sha256"]
            data = Data(args.w0w2_root, out / "cache", config["lookback"], config["max_symbols"],config['evaluation_end'])
        if config['mode']=='formal' and data.dates[-1]<np.datetime64(config['evaluation_end']):
            raise ValueError('evaluation_end exceeds available source calendar')
        manifest["data_version"] = identity
        write_contract(data, out / "data_contract.json")
        if bench:
            # Check all dates BEFORE expensive fitting; benchmark never imputed.
            for name in ["valid", "rule_select", "eval", "full-test"]:
                days = data.days("valid", limit=config["valid_days"]) if name == "valid" else report_days(data, name, config)
                needed = data.dates[np.r_[days + 1, days[-1] + 2]]
                if any(str(d) not in bench["values"] for d in needed):
                    raise ValueError(f"benchmark coverage incomplete for {name}")
            manifest["benchmark_provenance"] = bench["provenance"]
        manifest["stage"] = "training_and_reporting"; save()
        def progress(group, seed, status, cells):
            manifest.update(current_model=group, current_seed=seed, model_status=status,
                successful_cells=sum(c["status"] == "success" for c in cells), planned_cells=len(cells))
            (out / "cell_status.live.json").write_text(json.dumps(cells, ensure_ascii=False, indent=1), encoding="utf-8")
            save()
        complete, cells = run_pipeline(data, config, out, identity, out.name.replace("_", "-"), bench, args.execution_feed, progress)
        manifest.update(status="SUCCESS" if complete else "FAILED", stage="complete" if complete else "partial_cells",
            exit_code=0 if complete else 1, end=now(), wall_seconds=time.perf_counter() - start,
            successful_cells=sum(c["status"] == "success" for c in cells), planned_cells=len(cells),
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else 0)
        try:
            import resource
            manifest["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        except ImportError:
            manifest["process_peak_rss_kib"] = None
        manifest["output_sha256"] = {str(p.relative_to(out)): digest(p) for p in out.rglob("*") if p.is_file() and "cache" not in p.relative_to(out).parts and p.name != "run_manifest.json"}
        save(); print(f"{manifest['status']}: {manifest['successful_cells']}/{manifest['planned_cells']} cells -> {out}", flush=True)
        return manifest["exit_code"]
    except Exception as error:
        manifest.update(status="FAILED", error=f"{type(error).__name__}: {error}", exit_code=1, end=now(), wall_seconds=time.perf_counter() - start)
        save(); traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
