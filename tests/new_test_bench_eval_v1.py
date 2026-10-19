"""Synthetic descriptive parsing, replay diagonal, guards and one-shot failure."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT.parent/'limt-repro/src'))
from new_bench_eval_v1 import parse_arms,assert_descriptive,configure_replay
from new_guarded_data_v1 import DateGuard,UNSEAL_CUTOFF,GuardedData,guard_execution_feed,self_test
from new_gate_b_common import sha256_file
from new_h5_common_v3 import daily_ic


def test_manifest_validation(tmp_path):
    model=tmp_path/'new_model.npz';np.savez(model,coef=np.ones(3))
    arm=dict(arm_id='ridge',family='cpu_baseline',label='h5',seeds=[1],default_rule='drop5-every1d',
        selected_rule='drop2-every5d',descriptive=True,note='',score_source={'1':dict(path=str(model),sha256=sha256_file(model),adapter='ridge')})
    assert parse_arms({'descriptive_arms':[arm]})==[arm]
    for broken in ({k:v for k,v in arm.items() if k!='family'},dict(arm,descriptive=False)):
        with pytest.raises(SystemExit):parse_arms({'descriptive_arms':[broken]})
    with pytest.raises(SystemExit):parse_arms({'descriptive_arms':[arm,arm]})
    model.write_bytes(b'corrupt')
    with pytest.raises(SystemExit):parse_arms({'descriptive_arms':[arm]})
    duplicate=dict(arm,note='identical_to_default')
    assert parse_arms({'descriptive_arms':[duplicate]})==[duplicate]


def test_forbidden_inference_fields():
    assert_descriptive(dict(arms={'ridge':dict(mean_turnover=.1)}))
    for name in ('p_value','holm','conclusion','s_mapping'):
        with pytest.raises(SystemExit):assert_descriptive({'nested':{name:0}})


def test_r1_mask_and_endpoint():
    dates=np.arange(np.datetime64('2026-06-20'),np.datetime64('2026-07-10'))
    z=np.tile([1.,2.,3.,0.],(len(dates),1));mask=np.tile([True,True,True,False],(len(dates),1))
    raw=SimpleNamespace(dates=dates,symbols=list('ABCD'),z_r1=z,mask_r1=mask)
    guard=DateGuard(UNSEAL_CUTOFF);data=GuardedData(raw,guard)
    scores=np.array([[1.,2.,3.,1000000.]])
    assert abs(daily_ic(data,[0],scores,horizon=2,end=UNSEAL_CUTOFF)-1)<1e-12
    wrong=np.corrcoef(scores[0],z[0])[0,1];assert abs(wrong-1)>.1
    t=int(np.where(dates==np.datetime64('2026-06-29'))[0][0])
    assert daily_ic(data,[t],scores,horizon=2,end=UNSEAL_CUTOFF) is None
    with pytest.raises(SystemExit):data.z_r1[t]


def test_guard_controls():
    self_test(ROOT.parent/'limt-repro/src')


def test_real_replay_diagonal():
    replay,_=configure_replay()
    dates=np.arange(np.datetime64('2019-01-01'),np.datetime64('2019-01-15'))
    n=60;raw=SimpleNamespace(dates=dates,symbols=np.array([f'S{i}' for i in range(n)]),candidate=lambda t:np.ones(n,bool))
    guard=DateGuard(UNSEAL_CUTOFF);data=GuardedData(raw,guard);replay.GUARD=guard
    values={'raw_close':np.tile(np.arange(n)+20.,(len(dates),1)),
            'state':np.zeros((len(dates),n),np.int8),'share_multiplier':np.ones((len(dates),n)),
            'cash_dividend_per_old_share':np.zeros((len(dates),n))}
    feed=guard_execution_feed(values,dates,guard);days=np.arange(2,10)
    scores=np.tile(np.arange(n,dtype=float),(len(days),1))
    assert replay.replay_diagonal(data,days,scores,feed)=='PASS'


def test_unseal_failure_preserves_analysis(tmp_path):
    from new_unseal_once_v1 import run_once
    target=tmp_path/'new_analysis.json';target.write_text('{}')
    before=sha256_file(target)
    journal=run_once([('analysis',[sys.executable,'-c','pass']),
        ('bench_eval',[sys.executable,'-c','import sys;sys.stderr.write("synthetic failure");sys.exit(7)'])],
        tmp_path/'new_attempt',target)
    assert not journal['passed'] and journal['stages'][-1]['returncode']==7
    assert journal['stages'][-1]['analysis_output_unchanged'] and sha256_file(target)==before
    assert (tmp_path/'new_attempt/new_unseal_log.json').is_file()
    with pytest.raises(FileExistsError):run_once([],tmp_path/'new_attempt',target)


def test_descriptive_run_all_families_synthetic(tmp_path,monkeypatch):
    import new_bench_eval_v1 as evaluator
    from new_guarded_data_v1 import GuardedBench
    dates=np.r_[np.arange(np.datetime64('2019-01-01'),np.datetime64('2019-01-15')),
                np.arange(np.datetime64('2026-06-15'),np.datetime64('2026-07-10'))]
    n=60;features=np.random.default_rng(15).normal(size=(len(dates),n,3)).astype(np.float32)
    mask=np.ones((len(dates),n),bool);targets=np.stack([features[...,0]]*3,axis=-1)
    valid=np.arange(2,10);evaluation=np.arange(14,28)
    loss=mask.copy();loss[24:]=False
    raw=SimpleNamespace(dates=dates,symbols=np.array([f'S{i}' for i in range(n)]),
        feature_names=np.array(['a','b','c']),features=features,targets=targets,
        z_h5=features[...,0],z_r1=features[...,1],mask_r1=mask,listing_eligible=mask,
        loss_masks={'eval':loss,'valid':mask},candidate=lambda t:mask[t],
        stale=np.zeros_like(mask),relatives=lambda m,p:np.ones(n+1),
        days=lambda split:evaluation if split=='eval' else valid)
    feed=dict(raw_close=np.full(mask.shape,20.),state=np.zeros(mask.shape,np.int8),
              share_multiplier=np.ones(mask.shape),cash_dividend_per_old_share=np.zeros(mask.shape))
    def loader(args,guard):
        return GuardedData(raw,guard),guard_execution_feed(feed,dates,guard),GuardedBench({str(d):0. for d in dates},guard)
    model=tmp_path/'new_ridge.npz';np.savez(model,coef=np.array([0.,1.,0.,0.]))
    source=dict(path=str(model),sha256=sha256_file(model),adapter='ridge',columns=['a','b','c'])
    arms=[]
    for family in evaluator.FAMILIES:
        arm=dict(arm_id=family,family=family,label='r1',score_source={'1':source},seeds=[1],
                 selected_rule='drop2-every5d',default_rule='drop5-every1d',descriptive=True,note='synthetic')
        if family in evaluator.CONTROLS:arm['rule_select_result']=source
        if family=='qlib_h5':
            arm.update(identical_to_default=True,score_source={})
        if family=='limt_ablation':arm['soft_portfolio']=True
        arms.append(arm)
    gate=tmp_path/'new_gate.json';gate.write_text(json.dumps(dict(bench_eval_sha256=sha256_file(evaluator.__file__),descriptive_arms=arms)))
    monkeypatch.setattr(evaluator,'load_evaluation_data',loader)
    monkeypatch.setattr(evaluator,'verify_guard',lambda *args:None)
    args=SimpleNamespace(out=tmp_path/'new_eval.json',cache=tmp_path/'new_cache',gate=gate,
                         root=ROOT,limt_src=ROOT.parent/'limt-repro/src')
    evaluator.run(args)
    result=json.loads(args.out.read_text());assert_descriptive(result)
    assert set(result['arms'])==set(evaluator.FAMILIES)
    assert len(result['soft_marks_max_per_call'])==1
    assert result['arms']['qlib_h5']['eval']=='identical_to_default'
    assert all(v is not None for v in result['max_date_touched'].values())


@pytest.mark.parametrize('name',['LSTM','GRU','ALSTM','Transformer'])
def test_saved_ts_adapter(tmp_path,name):
    import torch,importlib
    import new_bench_eval_v1 as evaluator
    torch.set_num_threads(1)
    module,clsname=evaluator.TS_CLASSES[name]
    cls=getattr(importlib.import_module('qlib.contrib.model.'+module),clsname)
    kw=dict(d_feat=2,n_jobs=0,GPU=-1,batch_size=8)
    kw['d_model' if name=='Transformer' else 'hidden_size']=8
    model=cls(**kw)
    nets=[v for v in vars(model).values() if isinstance(v,torch.nn.Module)]
    assert len(nets)==1
    path=tmp_path/('new_'+name+'.pt');torch.save(nets[0].state_dict(),path)
    dates=np.arange(np.datetime64('2019-01-01'),np.datetime64('2019-01-21'))
    raw=SimpleNamespace(dates=dates,symbols=np.array(list('ABCD')),feature_names=np.array(['a','b']),
        features=np.random.default_rng(12).normal(size=(len(dates),4,2)).astype(np.float32))
    guard=DateGuard(UNSEAL_CUTOFF);data=GuardedData(raw,guard);days=np.arange(3,10)
    source=dict(path=str(path),sha256=sha256_file(path),adapter='qlib_ts',model=name,
                kwargs=kw,columns=['a','b'],step_len=3)
    prediction=evaluator.predict_source(source,data,days)
    assert prediction.shape==(7,4) and np.isfinite(prediction).all()
    assert guard.maximum['features']==str(dates[9]) and guard.maximum['labels'] is None


def test_saved_limt_adapter(tmp_path):
    import torch
    from v2_w4_training import make_model
    from v2_w6_run import DEFAULT,validate_config
    from new_bench_eval_v1 import predict_source
    torch.set_num_threads(1)
    dates=np.arange(np.datetime64('2019-01-01'),np.datetime64('2019-01-21'))
    features=np.random.default_rng(13).normal(size=(len(dates),4,2)).astype(np.float32)
    raw=SimpleNamespace(dates=dates,symbols=np.array(list('ABCD')),feature_names=np.array(['a','b']),
        features=features,feature_mask=np.ones((len(dates),4),bool),lookback=3,
        windows=lambda days:np.stack([features[t-2:t+1].transpose(1,0,2) for t in days]))
    config=validate_config(dict(DEFAULT,dimension=16,batch_days=2,amp=False))
    model=make_model(raw,config)
    path=tmp_path/'new_limt.pt';torch.save(dict(configuration=config,state_dict=model.state_dict()),path)
    guard=DateGuard(UNSEAL_CUTOFF);data=GuardedData(raw,guard)
    prediction=predict_source(dict(path=str(path),adapter='limt'),data,np.arange(3,10))
    assert prediction.shape==(7,4) and np.isfinite(prediction).all()
    assert guard.maximum['features']==str(dates[9]) and guard.maximum['labels'] is None


def test_unseal_preflight_failure_is_logged(tmp_path):
    from new_unseal_once_v1 import run_once
    input_file=tmp_path/'new_input.txt';input_file.write_text('synthetic')
    result=run_once([],tmp_path/'new_attempt',tmp_path/'new_output.json',
                    [dict(path=str(input_file),sha256='0'*64)])
    assert result['passed'] is False and result['stages']==[]
    assert (tmp_path/'new_attempt/new_unseal_log.json').is_file()


def test_undefined_seed_ic_is_order_independent():
    from new_bench_eval_v1 import seed_average
    assert seed_average([dict(ic=.1),dict(ic=None)])==dict(ic=None)
    assert seed_average([dict(ic=None),dict(ic=.1)])==dict(ic=None)
