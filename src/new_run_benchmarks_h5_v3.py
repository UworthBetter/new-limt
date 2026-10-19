"""h5 r9 runner. Only synthetic smoke may relax platform/runtime/training size.

All raw observation loading occurs in H5BlindData construction, then GuardedData
wraps it. Training reads labels only for purged signal days. Prediction master
frames use dummy labels. PRE_UNSEAL_CUTOFF is fixed in this source.
"""
import argparse
from contextlib import redirect_stdout
from datetime import datetime,timezone
import io
import importlib
import json
import logging
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import pandas as pd

from new_gate_b_common import (path_args, resolve_paths, preflight, write_json,
                               sha256_file, Provenance)
from new_bench_dataset import BenchDataset, build_master, score_matrix
from new_run_benchmarks_v2 import verify_production_platform,PRODUCTION_IMAGE
from new_guarded_data_v1 import (DateGuard, PRE_UNSEAL_CUTOFF, GuardedData,
    verify_guard, import_closure, third_party_versions)
from new_h5_common_v3 import (H5BlindData, csrank_h5, daily_ic, purge_days,
    resolve_config, assert_cfg1, verify_qlib_source, effective_settings,
    persistent_models_path, save_checkpoint, tag_for)


def validate_request(args, selection=None):
    if args.arm=='default':
        if args.cfg:
            raise SystemExit('default cannot specify --cfg')
        return ['cfg1']
    if not args.cfg or 'cfg1' in args.cfg or len(set(args.cfg))!=len(args.cfg):
        raise SystemExit('tuned requires distinct cfg2..cfg8; cfg1 reuses default')
    if args.seed!=1:
        if selection is None:
            raise SystemExit('final tuned seeds require --selection')
        for name in args.models:
            if args.cfg != [selection[name]['selected_cfg']]:
                raise SystemExit('final seeds must use selected config')
    return args.cfg


def verify_t1(args):
    import torch
    if not args.seed_verification:
        raise SystemExit('production requires --seed-verification')
    record=json.loads(args.seed_verification.read_text(encoding='utf-8'))
    expected={f'{name}/cfg{k}' for name in ('LSTM','Transformer') for k in range(1,9)}
    if (not record.get('passed') or set(record.get('checks',{}))!=expected
        or not all(x.get('passed') for x in record['checks'].values())
        or record.get('runner_sha256')!=sha256_file(__file__)
        or record.get('verifier_sha256')!=sha256_file(Path(__file__).with_name('new_verify_seeds_h5_v3.py'))
        or record.get('spec_sha256')!=sha256_file(args.spec)):
        raise SystemExit('failed or stale T1 record')
    if record.get('runtime')!=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda):
        raise SystemExit('T1 runtime differs')
    for name,digest in record['dependencies'].items():
        if sha256_file(Path(__file__).with_name(name))!=digest:
            raise SystemExit('T1 dependency changed')


def synthetic_h5(spec):
    from types import SimpleNamespace
    rng=np.random.default_rng(120)
    dates=pd.bdate_range('2018-01-01',periods=180).values.astype('datetime64[D]')
    symbols=np.array([f'S{i:04d}' for i in range(32)])
    columns=spec['filter_col_20']
    x=rng.normal(size=(180,len(symbols),len(columns))).astype(np.float32)
    y=x[...,0]+.1*rng.normal(size=x.shape[:2])
    mask=np.ones(y.shape,bool)
    return SimpleNamespace(dates=dates,symbols=symbols,features=x,feature_names=np.array(columns),
        z_h5=y,z_r1=y,mask_h5=mask,mask_r1=mask,feature_mask=mask,
        universe=mask,listing_eligible=mask,lookback=21,
        loss_masks={k:mask.copy() for k in ('train','valid','rule_select')},
        candidate=lambda t:mask[t])


def prepare_training(data, spec, name, train_span, valid_days, train_end, valid_end):
    # No validation target whose endpoint crosses the segment is ever read.
    hi=int(valid_days[-1])+6
    if hi>=len(data.dates):raise SystemExit('insufficient h5 endpoint calendar')
    y=np.full((hi+1,len(data.symbols)),np.nan,np.float32)
    masks={}
    for split,span,end in (('train',train_span,train_end),
                           ('valid',(str(data.dates[valid_days[0]]),str(data.dates[valid_days[-1]])),valid_end)):
        days=np.flatnonzero((data.dates>=np.datetime64(span[0])) & (data.dates<=np.datetime64(span[1])))
        days=purge_days(data.dates,days,end)
        keep=np.zeros(y.shape,bool)
        for t in days:
            mask=data.mask_h5[t]
            raw=data.z_h5[t, np.flatnonzero(mask)]
            if not np.isfinite(raw).all():
                raise SystemExit('nonfinite h5 label-mask member')
            row=np.full((1,len(data.symbols)),np.nan)
            row[0,mask]=raw
            y[t]=csrank_h5(row,mask[None,:],0,0)[0]
            keep[t]=mask
        masks[split]=keep
    master=build_master(data,y,0,hi,spec['filter_col_20'])
    spans=dict(train=train_span,valid=(str(data.dates[valid_days[0]]),str(data.dates[valid_days[-1]])))
    spans['test']=spans['valid']
    filters={k:pd.Series(v.reshape(-1),index=master.index) for k,v in masks.items()}
    filters['test']=None
    return BenchDataset(master,spans,filters,spec['step_len_ts'])


def construct(name, kwargs, metric):
    import torch
    with redirect_stdout(io.StringIO()):
        from new_h5_models_v3 import GuardedLSTM,GuardedTransformer
    seed=kwargs['seed']; random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model=(GuardedLSTM if name=='LSTM' else GuardedTransformer)(**kwargs)
    model.early_stop_metric=metric
    return model


def main():
    p=argparse.ArgumentParser(description=__doc__);path_args(p)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--models',nargs='+',choices=('LSTM','Transformer'),required=True)
    p.add_argument('--arm',choices=('default','tuned'),required=True)
    p.add_argument('--cfg',nargs='+',choices=[f'cfg{k}' for k in range(1,9)])
    p.add_argument('--seed',type=int,choices=(1,2,3),default=1)
    p.add_argument('--out-dir',type=Path)
    p.add_argument('--models-out',type=Path,required=True)
    p.add_argument('--selection',type=Path)
    p.add_argument('--seed-verification',type=Path)
    p.add_argument('--smoke',action='store_true');args=p.parse_args()
    args.root=args.root or Path(__file__).resolve().parents[1]
    args.spec=args.spec or args.root/'configs/new_benchmarks_h5_v3.json'
    selection=json.loads(args.selection.read_text()) if args.selection else None
    cfgs=validate_request(args,selection)
    resolve_paths(args,('root','spec') if args.smoke else ('root','spec','limt_src','w0w2','labels','panel','splits'))
    args.out_dir=args.out_dir or args.root/'runs/new_scores_h5_v3'
    if len(set(args.models))!=len(args.models):raise SystemExit('duplicate models')
    for name in ('scores','new_scores_v2'):
        if args.out_dir.resolve().is_relative_to((args.root/'runs'/name).resolve()):
            raise SystemExit('protected output directory')
    if args.smoke and 'smoke' not in args.out_dir.name.lower():raise SystemExit('smoke needs separate named directory')
    args.models_out=persistent_models_path(args.models_out,args.smoke)
    spec=json.loads(args.spec.read_text(encoding='utf-8'))
    if not args.smoke:
        if 'draft' in spec.get('status','').lower():raise SystemExit('production requires finalized committed spec')
        verify_guard(args.root);verify_t1(args);verify_production_platform()
    guard=DateGuard(PRE_UNSEAL_CUTOFF)
    import torch
    with redirect_stdout(io.StringIO()):
        from new_h5_models_v3 import GuardedLSTM
    closure_start=import_closure(args.root,args.limt_src or args.root.parent/'limt-repro/src')
    targets=[args.out_dir/(tag_for(n,c,args.seed)+ext) for n in args.models for c in cfgs for ext in ('.npz','.meta.json')]
    preflight(targets+[args.cache]+[args.models_out/f'new_{tag_for(n,c,args.seed)}.pt' for n in args.models for c in cfgs])
    provenance=Provenance(__file__,args,synthetic=args.smoke)
    if args.smoke:
        args.cache.mkdir(parents=True); raw=synthetic_h5(spec)
        valid_days,select_days=np.arange(90,135),np.arange(135,178)
        train_span=(str(raw.dates[20]),str(raw.dates[89])); train_end=train_span[1];valid_end=str(raw.dates[134])
    else:
        raw=H5BlindData(args);valid_days=raw.days('valid');select_days=raw.days('rule_select')
        train_span=('2008-01-01','2018-12-31');train_end='2018-12-31';valid_end='2020-12-31'
    data=GuardedData(raw,guard);del raw
    logging.disable(logging.CRITICAL)
    for name in args.models:
        for cfg in cfgs:
            model_clock=time.perf_counter();model_started_at=datetime.now(timezone.utc).isoformat()
            entry,kwargs,metric,factors=resolve_config(spec,name,cfg,args.seed)
            if not args.smoke:
                verify_qlib_source(entry,name)
                _,default_kwargs,_,_=resolve_config(spec,name,'cfg1',args.seed)
                assert_cfg1(args.root,spec,name,args.seed,default_kwargs)
            if args.smoke:kwargs=dict(kwargs,n_epochs=2,n_jobs=0,batch_size=64,GPU=-1)
            model=construct(name,kwargs,metric)
            if not args.smoke:
                default_model=construct(name,default_kwargs,'valid_loss')
                assert_cfg1(args.root,spec,name,args.seed,default_kwargs,default_model);del default_model
                model=construct(name,kwargs,metric)
            dataset=prepare_training(data,spec,name,train_span,valid_days,train_end,valid_end)
            dates=[str(d) for d in data.dates[valid_days]]
            dummy=np.zeros((len(data.dates),len(data.symbols)),np.float32)
            master=build_master(data,dummy,max(0,int(valid_days[0])-25),int(select_days[-1]),spec['filter_col_20'])
            def predict(span):
                return model.predict(BenchDataset(master,{'test':span},{'test':None},spec['step_len_ts']))
            span=(dates[0],dates[-1])
            def diagnostic(m,ds):
                return daily_ic(data,valid_days,score_matrix(predict(span),dates,list(data.symbols)),end=valid_end)
            model.ic_callback=diagnostic
            tag=tag_for(name,cfg,args.seed);temporary=args.cache/f'new_{tag}_fit.pt'
            start=time.perf_counter();history={}
            model.fit(dataset,evals_result=history,save_path=str(temporary))
            fit_seconds=time.perf_counter()-start
            values={}
            for split,days in (('valid',valid_days),('rule_select',select_days)):
                ds=[str(d) for d in data.dates[days]]
                values['dates_'+split]=np.array(ds)
                values['scores_'+split]=score_matrix(predict((ds[0],ds[-1])),ds,list(data.symbols))
                values['candidate_'+split]=np.stack([data.candidate(int(t)) for t in days])
            ic=daily_ic(data,valid_days,values['scores_valid'],end=valid_end)
            if ic is None or not np.isfinite(ic):raise SystemExit('undefined checkpoint IC')
            checkpoint=save_checkpoint(args.models_out/f'new_{tag}.pt',model)
            target=args.out_dir/f'{tag}.npz';target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as f:np.savez_compressed(f,**values)
            training=dict(per_epoch=model.per_epoch,actual_epochs=len(model.per_epoch),best_epoch=model.best_epoch)
            for split in ('train','valid'):
                losses=np.array([r[split+'_loss'] for r in model.per_epoch])
                if not np.isfinite(losses).all():raise SystemExit('nonfinite per-epoch loss')
                best=int(np.argmin(losses))
                training[split]=dict(actual_rounds=len(losses),best_iteration=best,minimum_loss=float(losses[best]))
            model_elapsed=time.perf_counter()-model_clock
            verification_sha=sha256_file(args.seed_verification) if args.seed_verification else None
            meta=provenance.finish(dict(tag=tag,model=entry['module']+'.'+entry['class'],kwargs=kwargs,
                effective_model_settings=effective_settings(model),arm=args.arm,cfg=cfg,grid_factors=factors,
                effective_seed=args.seed,seed_passed_to_model=args.seed,
                feature_set='filter20',n_features=len(spec['filter_col_20']),
                seed_policy=entry.get('seed_policy','explicit model seed 1,2,3'),
                model_started_at=model_started_at,model_wall_seconds=model_elapsed,
                device_hours=model_elapsed/3600 if model.use_gpu else 0.,fit_defaults={},
                expected_container_image=PRODUCTION_IMAGE if not args.smoke else None,
                seed_verification_sha256=verification_sha,
                label='h5',label_norm='csrank_h5',rank_set='label_mask_h5',purge_h5=True,
                purge_asymmetry='r1 runner has no purge; h5 purge is mandatory',
                ic_overlap_note='overlapping labels affect variance; only mean IC is selected',
                early_stop_metric=metric,training=training,
                best_epoch=model.best_epoch,best_valid_ic5=ic,fit_seconds=fit_seconds,
                selection_role='default' if cfg=='cfg1' else 'tuning' if args.seed==1 else 'final',
                smoke=args.smoke,descriptive=True,
                t1_record_sha256=verification_sha,
                checkpoint=checkpoint,**guard.metadata(('features','labels'))))
            meta.update(checkpoint);meta['score_sha256']=sha256_file(target)
            meta['import_closure']=dict(start=closure_start,end=import_closure(args.root,args.limt_src or args.root.parent/'limt-repro/src',closure_start),third_party=third_party_versions())
            write_json(target.with_suffix('.meta.json'),meta)
            print('BENCH DONE '+tag,flush=True)


if __name__=='__main__':main()
