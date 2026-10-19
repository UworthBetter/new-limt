"""Synthetic-only acceptance checks for labels, RNG, patience and artifacts."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from new_h5_common_v3 import (csrank_h5,purge_days,daily_ic,isolated_rng,
    save_checkpoint,validate_checkpoint,resolve_config,network)
from new_run_benchmarks_h5_v3 import validate_request,construct
from new_bench_dataset import BenchDataset
from new_verify_seeds import state_hash


def synthetic_spec():
    """Engineering fixture, NEVER a substitute for the reviewed production spec."""
    entries={
        'LSTM':dict(module='qlib.contrib.model.pytorch_lstm_ts',class_='LSTM',
            kwargs=dict(d_feat=2,hidden_size=8,num_layers=1,dropout=0.,n_epochs=2,
                        lr=.001,early_stop=2,batch_size=8,metric='loss',loss='mse',n_jobs=0,GPU=-1)),
        'Transformer':dict(module='qlib.contrib.model.pytorch_transformer_ts',class_='TransformerModel',
            kwargs=dict(d_feat=2,d_model=8,num_layers=1,dropout=0.,n_epochs=2,
                        lr=.001,early_stop=2,batch_size=8,metric='loss',loss='mse',n_jobs=0,GPU=-1))}
    for e in entries.values():e['class']=e.pop('class_')
    resolved={n:{f'cfg{k}':dict(kwargs={},early_stop_metric='valid_ic5' if k%2==0 else 'valid_loss')
                 for k in range(1,9)} for n in entries}
    return dict(official_entry_v2=entries,tuning_grid=dict(resolved=resolved),
        filter_col_20=['f0','f1'],step_len_ts=3,selection=dict(tie_rule='synthetic fixture'))


def tiny_dataset():
    rng=np.random.default_rng(123)
    index=pd.MultiIndex.from_product([pd.bdate_range('2019-01-01',periods=20),['A','B','C','D']],names=['datetime','instrument'])
    x=rng.normal(size=(len(index),3)).astype(np.float32)
    frame=pd.DataFrame(x,index=index,columns=['f0','f1','LABEL0'])
    return BenchDataset(frame,dict(train=('2019-01-03','2019-01-14'),valid=('2019-01-15','2019-01-28'),test=('2019-01-15','2019-01-28')),
        dict(train=None,valid=None,test=None),3)


def test_rank_and_purge():
    z=np.array([[1.,2.,2.,100.]])
    mask=np.array([[True,True,True,False]])
    actual=csrank_h5(z,mask,0,0)
    np.testing.assert_allclose(actual[0,:3],(np.array([1/3,5/6,5/6])-.5)*3.46,atol=1e-7)
    assert np.isnan(actual[0,3])
    z[0,0]=np.inf
    with pytest.raises(ValueError):csrank_h5(z,mask,0,0)
    dates=np.arange(np.datetime64('2020-12-20'),np.datetime64('2021-01-10'))
    np.testing.assert_array_equal(purge_days(dates,[5,6],'2020-12-31'),[5])


def test_ic_known_answers():
    dates=np.arange(np.datetime64('2020-01-01'),np.datetime64('2020-01-10'))
    z=np.tile([1.,2.,3.,1000.],(9,1));mask=np.tile([True,True,True,False],(9,1))
    d=SimpleNamespace(dates=dates,symbols=list('ABCD'),z_h5=z,loss_masks={'valid':mask},listing_eligible=np.ones_like(mask))
    assert abs(daily_ic(d,[0],np.array([[1.,2.,3.,-10000.]]))-1)<1e-12
    assert abs(daily_ic(d,[0],np.array([[-1.,-2.,-3.,10000.]]))+1)<1e-12


def test_cfg1_and_request_rejection():
    spec=synthetic_spec();spec['tuning_grid']['resolved']['LSTM']['cfg1']['kwargs']['lr']=.5
    with pytest.raises(SystemExit):resolve_config(spec,'LSTM','cfg1',1)
    for arm,cfg,seed in [('default',['cfg2'],1),('tuned',['cfg1'],1),('tuned',['cfg2'],2)]:
        with pytest.raises(SystemExit):validate_request(SimpleNamespace(arm=arm,cfg=cfg,seed=seed,models=['LSTM']))


@pytest.mark.parametrize('name',['LSTM','Transformer'])
def test_training_rng_and_best(name):
    import torch
    torch.set_num_threads(1)
    spec=synthetic_spec();_,kw,_,_=resolve_config(spec,name,'cfg1',1)
    digests=[]
    with tempfile.TemporaryDirectory() as tmp:
        for enabled in (False,True):
            model=construct(name,kw,'valid_loss');model.diagnostics_enabled=enabled
            def diag(m,ds):
                # Use real complete prediction DataLoader, including its RNG base seed.
                prediction=m.predict(ds)
                return float(np.corrcoef(np.arange(len(prediction)),prediction.values)[0,1])
            model.ic_callback=diag
            model.fit(tiny_dataset(),evals_result={},save_path=str(Path(tmp)/f'{enabled}.pt'))
            digests.append(state_hash(model))
        assert digests[0]==digests[1]
        # Compare to the original unmodified qlib fit as well.
        import importlib
        entry=spec['official_entry_v2'][name]
        original=getattr(importlib.import_module(entry['module']),entry['class'])(**kw)
        original.fit(tiny_dataset(),evals_result={},save_path=str(Path(tmp)/'original.pt'))
        assert state_hash(original)==digests[0]
        model=construct(name,kw,'valid_ic5');states=[]
        def diagnostic(m,ds):
            m.predict(ds)
            states.append(state_hash(m))
            return 1. if len(states)==1 else -1.
        model.ic_callback=diagnostic
        model.fit(tiny_dataset(),evals_result={},save_path=str(Path(tmp)/'best.pt'))
        assert state_hash(model)==states[0] and model.best_epoch==0


def test_checkpoint_corruption(tmp_path):
    import torch
    from new_gate_b_common import sha256_file
    spec=synthetic_spec();_,kw,_,_=resolve_config(spec,'LSTM','cfg1',1)
    model=construct('LSTM',kw,'valid_loss');path=tmp_path/'new_model.pt'
    meta=save_checkpoint(path,model)
    assert meta['artifact_sha256']==sha256_file(path)
    state=network(model).state_dict();altered={k:v.clone() for k,v in state.items()}
    altered[next(iter(altered))].reshape(-1)[0]+=1
    with pytest.raises(SystemExit):validate_checkpoint(path,altered)
    path.write_bytes(path.read_bytes()[:20])
    with pytest.raises(SystemExit):validate_checkpoint(path,state)


def test_selection_ties():
    from new_select_h5_config_v3 import select_config
    ics={f'cfg{k}':.01 for k in range(1,9)};ics['cfg2']=.011
    counts={c:1 for c in ics};counts['cfg1']=0
    assert select_config(ics,counts)[0]=='cfg1'
    ics['cfg8']=.02;assert select_config(ics,counts)[0]=='cfg8'


def test_models_out_required():
    import subprocess
    r=subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'src/new_run_benchmarks_h5_v3.py'),
        '--cache','unused','--models','LSTM','--arm','default'],capture_output=True,text=True)
    assert r.returncode!=0 and '--models-out' in r.stderr


def test_supplied_spec_schema():
    root=Path(__file__).resolve().parents[1]
    spec=json.loads((root/'configs/new_benchmarks_h5_v3.json').read_text(encoding='utf-8'))
    _,kw,metric,_=resolve_config(spec,'Transformer','cfg1',2)
    assert kw==dict(seed=2,n_jobs=20) and metric=='valid_loss'
    for name in ('LSTM','Transformer'):
        width='hidden_size' if name=='LSTM' else 'd_model'
        for k in range(2,9):
            _,kw,metric,_=resolve_config(spec,name,f'cfg{k}',1)
            row=spec['tuning_grid']['resolved'][name][f'cfg{k}']
            assert kw['lr']==row['lr'] and kw[width]==row[width]
            assert metric==row['early_stop_metric']
