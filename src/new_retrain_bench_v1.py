"""r9 CPU retraining adapter. No evaluation-period predictions exist here.

Original training calls: new_run_benchmarks_v2.construct_model / XGBModel.fit;
run_benchmarks.TreeDataset/build_master / DEnsembleModel.fit; and
cpu_baselines.fit_ridge:266, predict_ridge:239. None is modified or copied.
Only IO, guarding, model readback and platform records are added.
"""
import argparse
from contextlib import redirect_stdout,redirect_stderr
from datetime import datetime,timezone
import importlib
import io
import json
import logging
import os
from pathlib import Path
import platform
import random
import time
import numpy as np
import pandas as pd
from new_gate_b_common import path_args,resolve_paths,preflight,sha256_file,write_json,Provenance
from new_guarded_data_v1 import (DateGuard,PRE_UNSEAL_CUTOFF,GuardedData,verify_guard,
    third_party_versions,import_closure)
from new_h5_common_v3 import H5BlindData,persistent_models_path,purge_days
from new_model_artifacts_v1 import save_model

FROZEN={
 'XGBoost':('49010f9b891316468a4c48fef1a5078a850deb38f571e8ac8e3be751e1f41f90','new_run_benchmarks_v2.py','4a96e6ebbdb6686ed243b162fc03575004e98d0e0ba81e3c10c8c9d8f3789fe6'),
 'DoubleEnsemble':('e189f6cbe3453d172fc407b6bf2117be15bcba7c72c698f53f125d92533e5165','run_benchmarks.py','824daa85536b85e81c28381ebc5b5b4474cd6506dccb03cc500e37e15e5daa8a'),
 'ridge':(None,'cpu_baselines.py','4c585a0fda0d93055aae78e5c2099cfe08ae701d52132803ac77460456e940da')}


def fingerprint():
    import psutil
    from threadpoolctl import threadpool_info
    info=Path('/proc/cpuinfo').read_text() if Path('/proc/cpuinfo').exists() else ''
    def field(key):return next((s.split(':',1)[1].strip() for s in info.splitlines() if s.startswith(key)),None)
    return dict(cpu=field('model name') or platform.processor(),cpu_flags=field('flags'),
        nproc=os.cpu_count(),memory_bytes=psutil.virtual_memory().total,os=platform.platform(),
        kernel=platform.release(),python=platform.python_version(),
        third_party=third_party_versions(),blas_openmp=threadpool_info())


def platform_verdict(original,current):
    prior=original.get('cpu_host_fingerprint')
    matched=prior is not None and prior==current
    return dict(platform_match=matched,original_platform_status='known' if prior else 'unknown',
                expected_verdict='MATCH_EXPECTED' if matched else 'MISMATCH_EXPECTED')


def main():
    p=argparse.ArgumentParser(description=__doc__);path_args(p)
    p.add_argument('--tag',required=True);p.add_argument('--seed',type=int,choices=(1,2,3),required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--models-out',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--original-meta',type=Path,required=True)
    args=p.parse_args()
    if args.threads is not None:raise SystemExit('thread override forbidden')
    resolve_paths(args);verify_guard(args.root)
    kind='XGBoost' if args.tag.startswith('XGBoost-seeded_') else 'DoubleEnsemble' if args.tag.startswith('DoubleEnsemble_') else 'ridge' if args.tag.startswith('ridge') else None
    if kind is None:raise SystemExit('only missing XGB/DE/ridge models may be retrained')
    digest,filename,code_sha=FROZEN[kind]
    if sha256_file(args.root/'src'/filename)!=code_sha or (digest and sha256_file(args.spec)!=digest):
        raise SystemExit('frozen training source/spec mismatch')
    args.models_out=persistent_models_path(args.models_out)
    preflight([args.out,args.cache]);args.out.mkdir(parents=True)
    import torch
    torch.manual_seed(args.seed);random.seed(args.seed);np.random.seed(args.seed)
    original=json.loads(args.original_meta.read_text(encoding='utf-8'))
    if original.get('tag',args.tag)!=args.tag or original.get('effective_seed',args.seed)!=args.seed:
        raise SystemExit('original tag/seed differs')
    fp=fingerprint();fp_path=args.out/'new_cpu_host_fingerprint.json';write_json(fp_path,fp)
    provenance=Provenance(__file__,args);guard=DateGuard(PRE_UNSEAL_CUTOFF)
    # Import original functions before closure capture; imports do not run their main().
    if kind=='ridge':
        from cpu_baselines import fit_ridge,predict_ridge,DEFAULT
        if DEFAULT['ridge_alpha']!=10:raise SystemExit('ridge alpha mismatch')
    else:
        original_module=importlib.import_module(filename[:-3])
        frozen_spec=json.loads(args.spec.read_text(encoding='utf-8'))
        with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
            importlib.import_module(frozen_spec['models'][kind]['module'])
        third_party_versions(('torch','numpy','pandas','pyqlib',
                              'xgboost' if kind=='XGBoost' else 'lightgbm'))
    closure_start=import_closure(args.root,args.limt_src)
    data=GuardedData(H5BlindData(args),guard)
    valid_days,select_days=data.days('valid'),data.days('rule_select')
    start=time.perf_counter();wall_start=datetime.now(timezone.utc).isoformat();history={}
    columns=list(data.feature_names)
    if kind=='ridge':
        label='h5' if 'h5' in args.tag else 'r1' if 'r1' in args.tag else None
        if label is None:raise SystemExit('ridge tag must specify h5 or r1')
        horizon=6 if label=='h5' else 2
        days=purge_days(data.dates,data.days('train'),'2018-12-31',horizon)
        values=data.z_h5 if label=='h5' else data.z_r1
        mask=data.loss_masks['train'] if label=='h5' else data.mask_r1
        model,count=fit_ridge(data,days,values,mask)
        def predict(days):return predict_ridge(data,days,model)
        kwargs=dict(ridge_alpha=10)
    else:
        spec=json.loads(args.spec.read_text(encoding='utf-8'));entry=spec['models'][kind]
        kwargs=dict(entry['kwargs'])
        if kwargs.get('nthread' if kind=='XGBoost' else 'num_threads')!=20:
            raise SystemExit('original thread count must equal 20')
        hi=int(valid_days[-1])+2
        labels=np.full((hi+1,len(data.symbols)),np.nan,np.float32)
        for t in range(hi+1):labels[t]=data.z_r1[t]
        master=original_module.build_master(data,labels,0,hi,columns)
        train_span=('2008-01-01','2018-12-31');valid_span=(str(data.dates[valid_days[0]]),str(data.dates[valid_days[-1]]))
        masks={}
        for split,span in (('train',train_span),('valid',valid_span)):
            keep=np.zeros(labels.shape,bool)
            for t in np.flatnonzero((data.dates[:hi+1]>=np.datetime64(span[0]))&(data.dates[:hi+1]<=np.datetime64(span[1]))):
                keep[t]=data.mask_r1[t]&np.isfinite(labels[t])
            masks[split]=pd.Series(keep.reshape(-1),index=master.index)
        masks['test']=None
        dataset=original_module.TreeDataset(master,dict(train=train_span,valid=valid_span,test=valid_span),masks,columns)
        logging.disable(logging.CRITICAL)
        with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
            if kind=='XGBoost':
                model,kwargs=original_module.construct_model(entry,args.seed)
                model.fit(dataset,evals_result=history,verbose_eval=False)
            else:
                kwargs['n_jobs']=0
                model=getattr(importlib.import_module(entry['module']),entry['class'])(**kwargs)
                model.fit(dataset)
        prediction_master=original_module.build_master(data,np.zeros((len(data.dates),len(data.symbols)),np.float32),max(0,int(valid_days[0])-25),int(select_days[-1]),columns)
        from new_bench_dataset import score_matrix
        def predict(days):
            dates=[str(d) for d in data.dates[days]]
            ds=original_module.TreeDataset(prediction_master,{'test':(dates[0],dates[-1])},{'test':None},columns)
            return score_matrix(model.predict(ds),dates,list(data.symbols))
    artifact=save_model(model,kind,args.models_out,args.tag,columns)
    values={}
    for split,days in (('valid',valid_days),('rule_select',select_days)):
        values['scores_'+split]=predict(days);values['dates_'+split]=data.dates[days].astype(str)
        values['candidate_'+split]=np.stack([data.candidate(int(t)) for t in days])
    target=args.out/(args.tag+'.npz')
    with target.open('xb') as f:np.savez_compressed(f,**values)
    meta=dict(original)
    meta.update(provenance.finish(dict(tag=args.tag,kwargs=kwargs,**platform_verdict(original,fp))))
    meta.update(artifact);meta.update(guard.metadata(('features','labels')))
    old_scores=args.original_meta.with_name(args.original_meta.name.removesuffix('.meta.json')+'.npz')
    meta.update(retrain_of=dict(path=str(args.original_meta),sha256=sha256_file(args.original_meta),
            scores_path=str(old_scores),scores_sha256=sha256_file(old_scores) if old_scores.is_file() else None),
        fingerprint_sha256=sha256_file(fp_path),cpu_host_fingerprint=fp,score_sha256=sha256_file(target),
        fit_seconds=time.perf_counter()-start,train_seconds=time.perf_counter()-start,
        wall_start=wall_start,wall_end=datetime.now(timezone.utc).isoformat(),
        import_closure=dict(start=closure_start,end=import_closure(args.root,args.limt_src,closure_start),third_party=third_party_versions()),
        training_code_sha256=code_sha,de_provenance='inferred, not hash-verified' if kind=='DoubleEnsemble' else None)
    write_json(target.with_suffix('.meta.json'),meta);print('WROTE '+str(target)+' '+sha256_file(target))


if __name__=='__main__':main()
