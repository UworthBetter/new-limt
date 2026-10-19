"""Synthetic T1: seeds/settings, labels/purge, IC, RNG and best-weight reload."""
import argparse
import importlib.util
import json
import logging
from pathlib import Path
import platform
import sys
import tempfile
import numpy as np
from new_gate_b_common import sha256_file,write_json
from new_h5_common_v3 import resolve_config,effective_settings
from new_run_benchmarks_h5_v3 import construct
from new_verify_seeds import state_hash
from new_guarded_data_v1 import third_party_versions


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--spec',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');args=p.parse_args()
    import torch
    torch.set_num_threads(1);logging.disable(logging.CRITICAL)
    location=args.root/'tests/new_test_h5_benchmarks.py'
    s=importlib.util.spec_from_file_location('new_h5_acceptance',location)
    test=importlib.util.module_from_spec(s);s.loader.exec_module(test)
    spec=json.loads(args.spec.read_text()) if args.spec else test.synthetic_spec() if args.smoke else None
    if spec is None:raise SystemExit('production T1 requires finalized --spec')
    checks={};passed=True
    for name in ('LSTM','Transformer'):
        for k in range(1,9):
            key=f'{name}/cfg{k}'
            try:
                hashes=[];settings=[]
                for seed in (1,1,2):
                    _,kw,metric,_=resolve_config(spec,name,f'cfg{k}',seed)
                    m=construct(name,kw,metric);hashes.append(state_hash(m));settings.append(effective_settings(m))
                    for field,value in kw.items():
                        if field in settings[-1] and settings[-1][field]!=value:raise AssertionError('setting mismatch')
                ok=hashes[0]==hashes[1] and hashes[0]!=hashes[2]
                checks[key]=dict(passed=ok,hashes=hashes,settings=settings)
            except Exception as exc:
                checks[key]=dict(passed=False,error=type(exc).__name__)
            passed &= checks[key]['passed']
    extra={}
    for fn in ('test_rank_and_purge','test_ic_known_answers','test_cfg1_and_request_rejection','test_selection_ties'):
        try:getattr(test,fn)();extra[fn]=True
        except Exception:extra[fn]=False
    for name in ('LSTM','Transformer'):
        try:test.test_training_rng_and_best(name);extra['rng/'+name]=True
        except Exception as exc:extra['rng/'+name]=False
    passed=bool(passed and all(extra.values()))
    dependencies={n:sha256_file(args.root/'src'/n) for n in (
        'new_guarded_data_v1.py','new_h5_common_v3.py','new_h5_models_v3.py',
        'new_bench_dataset.py','new_gate_b_common.py','new_bench_replay.py','new_run_benchmarks_v2.py')}
    write_json(args.out,dict(passed=passed,checks=checks,extra_checks=extra,synthetic=True,
        engineering_fixture=args.spec is None,spec_sha256=sha256_file(args.spec) if args.spec else None,
        runner_sha256=sha256_file(Path(__file__).with_name('new_run_benchmarks_h5_v3.py')),
        verifier_sha256=sha256_file(__file__),dependencies=dependencies,
        runtime=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda),
        third_party=third_party_versions()))
    print('ASSERT T1 '+('PASS' if passed else 'FAIL'))
    raise SystemExit(0 if passed else 1)


if __name__=='__main__':main()
