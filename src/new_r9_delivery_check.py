"""Reproducible synthetic-only delivery evidence; never opens real observations."""
import ast
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
DELIVERY=ROOT/'protocol/checks/new_r9_delivery_20261019'
FILES=('new_guarded_data_v1.py','new_gate_b_analysis_v3_1.py','new_h5_common_v3.py',
 'new_h5_models_v3.py','new_run_benchmarks_h5_v3.py','new_select_h5_config_v3.py',
 'new_verify_seeds_h5_v3.py','new_rule_grid_bench_h5_v3.py','new_bench_forensics_h5_v3.py',
 'new_bench_eval_v1.py','new_retrain_bench_v1.py','new_compare_scores_v1.py',
 'new_compare_ridge_daily_v1.py','new_rule_grid_bench_v2p.py','new_bench_forensics_v2p.py',
 'new_cpu_reselect_v1.py','new_benchmark_manifest_v1.py','new_model_artifacts_v1.py',
 'new_unseal_once_v1.py','new_r9_delivery_check.py')
PROTECTED={
 'src/new_run_benchmarks_v2.py':'4a96e6ebbdb6686ed243b162fc03575004e98d0e0ba81e3c10c8c9d8f3789fe6',
 'src/new_verify_seeds.py':'919eb74116d79cda3cab123323ff03bfc63d72907e0cc10b84eebe541a283cf8',
 'src/new_rule_grid_bench_v2.py':'ad4a8b1e48a00124c815d8a033ebea0ce45a1a42d6132f8d7aa768fed320d7e0',
 'src/new_bench_forensics_v2.py':'6731849040e341b21d6527c05c46961df2240f35b7c246fa4aa7e1632d0bb65b',
 'src/new_gate_b_common.py':'12632786f8801d5f233f9293b108e5f6fba5afe75a0b088bd5e8aa5d197ba849',
 'src/new_bench_dataset.py':'2bd555f8c7052624bcfa43b067f1f81964e8f894c66918531bbd9879e8528616',
 'src/new_bench_replay.py':'ba32c089c864a9989cb7f12af8c8ff46c0db0f9c73131c16304f03d5e54406f5',
 'src/new_gate_b_analysis_v3.py':'203e6a455796f4eca90fa5a9a0038459eec2324279bb13c72d5622339dd8f75d',
 'configs/new_benchmarks_qlib_defaults_v2.json':'49010f9b891316468a4c48fef1a5078a850deb38f571e8ac8e3be751e1f41f90',
 'configs/benchmarks_qlib_defaults_v1.json':'e189f6cbe3453d172fc407b6bf2117be15bcba7c72c698f53f125d92533e5165',
 'src/cpu_baselines.py':'4c585a0fda0d93055aae78e5c2099cfe08ae701d52132803ac77460456e940da',
 'src/run_benchmarks.py':'824daa85536b85e81c28381ebc5b5b4474cd6506dccb03cc500e37e15e5daa8a',
 '../limt-repro/src/v2_w4_training.py':'a960a9951ab52c5455c15f8e0c5d60a783a6beb4beef5b4003ee578f0c21ee43',
 '../limt-repro/src/v2_w4_data.py':'89638e96c6b6f6c54f0bb3c07cad3c3413bbe17e82b8219bd80008da72914a4a'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def diff(old,new,target):
    result=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='frozen-original',tofile='new-implementation'))
    (DELIVERY/target).write_text(result,encoding='utf-8',newline='\n')


def main():
    global DELIVERY
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,default=DELIVERY)
    DELIVERY=parser.parse_args().out_dir.resolve()
    DELIVERY.mkdir(parents=True,exist_ok=False)
    baseline={name:dict(expected=h,actual=sha(ROOT/name),passed=sha(ROOT/name)==h) for name,h in PROTECTED.items()}
    if not all(r['passed'] for r in baseline.values()):raise SystemExit('protected baseline hash mismatch')
    (DELIVERY/'new_protected_baseline.json').write_text(json.dumps(baseline,indent=2),encoding='utf-8')
    diff((ROOT/'src/new_gate_b_analysis_v3.py').read_text(encoding='utf-8'),
         (ROOT/'src/new_gate_b_analysis_v3_1.py').read_text(encoding='utf-8'),'new_analysis_v3_to_v3_1.diff')
    # The fit copies differ only by diagnostic dispatch, history and exposed best_epoch.
    copies=(ROOT/'src/new_h5_models_v3.py').read_text(encoding='utf-8');tree=ast.parse(copies)
    import inspect,importlib
    for name,module,cls in (('lstm','pytorch_lstm_ts','LSTM'),('transformer','pytorch_transformer_ts','TransformerModel')):
        origin=getattr(importlib.import_module('qlib.contrib.model.'+module),cls)
        node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==('GuardedLSTM' if name=='lstm' else 'GuardedTransformer'))
        fn=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='fit')
        text='\n'.join(copies.splitlines()[fn.lineno-1:fn.end_lineno])+'\n'
        diff(inspect.getsource(origin.fit),text,'new_'+name+'_fit.diff')
    for name,path in (('xgb','new_run_benchmarks_v2.py'),('de','run_benchmarks.py'),('ridge','cpu_baselines.py')):
        diff((ROOT/'src'/path).read_text(encoding='utf-8'),
             (ROOT/'src/new_retrain_bench_v1.py').read_text(encoding='utf-8'),
             'new_'+name+'_retrain_path.diff')
    commands=[
      ('new_pytest.txt',[sys.executable,'-m','pytest',*[str(ROOT/'tests'/n) for n in (
          'new_test_h5_benchmarks.py','new_test_bench_eval_v1.py','new_test_retrain_compare_v1.py')],'-q','--disable-warnings']),
      ('new_guard_self_test.txt',[sys.executable,str(ROOT/'src/new_guarded_data_v1.py'),'--self-test','--limt-src',str(ROOT.parent/'limt-repro/src')]),
      ('new_analysis_self_test.txt',[sys.executable,str(ROOT/'src/new_gate_b_analysis_v3_1.py'),'--self-test','--limt-src',str(ROOT.parent/'limt-repro/src')]),
      ('new_t1_smoke.txt',[sys.executable,str(ROOT/'src/new_verify_seeds_h5_v3.py'),'--smoke','--spec',str(ROOT/'configs/new_benchmarks_h5_v3.json'),'--out',str(DELIVERY/'new_t1_record.json')]),
      ('new_runner_smoke.txt',[sys.executable,str(ROOT/'src/new_run_benchmarks_h5_v3.py'),'--root',str(ROOT),
        '--limt-src',str(ROOT.parent/'limt-repro/src'),'--spec',str(ROOT/'configs/new_benchmarks_h5_v3.json'),
        '--cache',str(DELIVERY/'new_smoke_cache'),'--models','LSTM','Transformer','--arm','default',
        '--models-out',str(DELIVERY/'new_smoke_models'),'--out-dir',str(DELIVERY/'new_smoke_scores'),'--threads','1','--smoke'])]
    checks=[]
    for name,command in commands:
        with (DELIVERY/name).open('x',encoding='utf-8') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,text=True)
        checks.append(dict(log=name,returncode=result.returncode,sha256=sha(DELIVERY/name)))
        print('ASSERT '+name+(' PASS' if result.returncode==0 else ' FAIL'),flush=True)
        if result.returncode:break
    if len(checks)==len(commands) and all(x['returncode']==0 for x in checks):
        required={'kwargs','effective_seed','seed_passed_to_model','feature_set','n_features',
            'effective_model_settings','fit_defaults','fit_seconds','training','model_started_at',
            'model_wall_seconds','device_hours','expected_container_image','seed_verification_sha256',
            'arm','cfg','grid_factors','label','label_norm','rank_set','purge_h5','early_stop_metric',
            'best_epoch','best_valid_ic5','selection_role','spec_sha256','t1_record_sha256',
            'artifact_sha256','checkpoint_path','checkpoint_bytes','n_tensors','score_sha256'}
        ok=True
        for name in ('LSTM','Transformer'):
            tag=name+'-h5default_seed1'
            meta=json.loads((DELIVERY/'new_smoke_scores'/(tag+'.meta.json')).read_text(encoding='utf-8'))
            checkpoint=Path(meta['checkpoint_path'])
            ok &= required<=set(meta) and sha(checkpoint)==meta['artifact_sha256']
            ok &= sha(DELIVERY/'new_smoke_scores'/(tag+'.npz'))==meta['score_sha256']
            ok &= meta['training']['actual_epochs']==len(meta['training']['per_epoch'])==2
            ok &= meta['device_hours']==0 and meta['purge_h5'] is True
        log=DELIVERY/'new_smoke_schema.txt'
        log.write_text('ASSERT smoke metadata and checkpoint schema '+('PASS' if ok else 'FAIL')+'\n',encoding='utf-8')
        checks.append(dict(log=log.name,returncode=0 if ok else 1,sha256=sha(log)))
        print('ASSERT '+log.name+(' PASS' if ok else ' FAIL'),flush=True)
    tests=[ROOT/'tests'/n for n in ('new_test_h5_benchmarks.py','new_test_bench_eval_v1.py','new_test_retrain_compare_v1.py')]
    files=[ROOT/'src'/n for n in FILES]+tests+[ROOT/'configs/new_benchmarks_h5_v3.json',ROOT/'configs/new_guarded_data_v1.sha256']
    manifest={str(p.relative_to(ROOT)):sha(p) for p in files}
    (DELIVERY/'new_code_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    result=dict(passed=len(checks)==len(commands)+1 and all(x['returncode']==0 for x in checks),
        synthetic_only=True,checks=checks,protected_unchanged=all(sha(ROOT/n)==v for n,v in PROTECTED.items()),
        production_blockers=['Reviewed h5 draft still declares DRAFT; experiment assistant must finalize and freeze it.',
            'Final spec must add six r1 metadata SHA256 pins in r1_meta_sha256 (two models x three seeds).',
            'Guard, checksum, finalized spec and DL entries must be committed and pushed before real runs.',
            'Production T1 and h5 require ModelScope A10 / pinned runtime; local synthetic evidence is not that runtime.'])
    (DELIVERY/'new_acceptance.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    raise SystemExit(0 if result['passed'] and result['protected_unchanged'] else 1)


if __name__=='__main__':main()
