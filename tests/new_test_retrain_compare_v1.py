"""Strict synthetic comparisons, original CPU selection and model persistence."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT.parent/'limt-repro/src'))
from new_compare_scores_v1 import compare,TOL
from new_guarded_data_v1 import DateGuard,PRE_UNSEAL_CUTOFF
from new_retrain_bench_v1 import platform_verdict


def payload():
    return dict(scores_valid=np.array([[1.,np.nan,3.],[2.,4.,6.]]),
        dates_valid=np.array(['2019-01-02','2019-01-03']),candidate_valid=np.ones((2,3),bool),symbols=np.array(list('ABC')))


@pytest.mark.parametrize('change,expected',[(None,'MATCH'),('value','MISMATCH'),('nan','MISMATCH'),('order','MISMATCH'),('shape','MISMATCH'),('symbols','MISMATCH'),('tolerance','MISMATCH')])
def test_score_comparison(change,expected):
    a=payload();b=copy.deepcopy(a)
    if change=='value':b['scores_valid'][0,0]+=1e-5
    if change=='nan':b['scores_valid'][0,1]=0
    if change=='order':b['dates_valid']=b['dates_valid'][::-1]
    if change=='shape':b['scores_valid']=b['scores_valid'][:1]
    if change=='symbols':b['symbols']=b['symbols'][::-1]
    if change=='tolerance':b['scores_valid'][0,0]+=2*TOL
    r=compare(a,b,['scores_valid'],DateGuard(PRE_UNSEAL_CUTOFF))
    assert r['verdict']==expected


def test_2023_refused():
    a=payload();a['dates_valid'][0]='2023-01-03'
    with pytest.raises(SystemExit):compare(a,a,['scores_valid'],DateGuard(PRE_UNSEAL_CUTOFF))


def test_platform_unknown_and_mismatch():
    assert platform_verdict({},dict(cpu='A'))['expected_verdict']=='MISMATCH_EXPECTED'
    assert not platform_verdict({'cpu_host_fingerprint':dict(cpu='A')},dict(cpu='B'))['platform_match']
    assert platform_verdict({'cpu_host_fingerprint':dict(cpu='A')},dict(cpu='A'))['platform_match']


def test_original_ridge_fit_roundtrip(tmp_path):
    from cpu_baselines import fit_ridge,predict_ridge
    from new_guarded_data_v1 import GuardedData
    from new_model_artifacts_v1 import save_model,load_predict_tabular
    rng=np.random.default_rng(30);x=rng.normal(size=(10,8,3)).astype(np.float32)
    dates=np.arange(np.datetime64('2017-01-01'),np.datetime64('2017-01-11'))
    raw=SimpleNamespace(features=x,dates=dates,symbols=list('ABCDEFGH'),feature_names=['a','b','c'],candidate=lambda t:np.ones(8,bool))
    data=GuardedData(raw,DateGuard(PRE_UNSEAL_CUTOFF));y=x[...,0]+x[...,1];mask=np.ones_like(y,bool)
    one,_=fit_ridge(data,np.arange(10),y,mask);two,_=fit_ridge(data,np.arange(10),y,mask)
    np.testing.assert_array_equal(predict_ridge(data,np.arange(10),one),predict_ridge(data,np.arange(10),two))
    artifact=save_model(one,'ridge',tmp_path,'ridge',['a','b','c'])
    np.testing.assert_array_equal(load_predict_tabular(artifact,x[0]),np.c_[np.ones(8),x[0]]@one)


def test_cpu_rule_choice_same_as_frozen_copy():
    from cpu_baselines import choose_arm as original
    from new_bench_replay import choose_arm,RULES
    grids=[[dict(rule=r,annual_excess=.1-i*.0001,mean_turnover=1/(i+1)) for i,(r,_,_) in enumerate(RULES)] for s in (1,2,3)]
    assert choose_arm(grids)==original(grids)


def test_ridge_frozen_comparator(tmp_path):
    from new_compare_ridge_daily_v1 import same_content,rows,Refuse
    a=tmp_path/'new_old.csv';b=tmp_path/'new_new.csv'
    a.write_text('date,net\n2021-01-04,0.1\n');b.write_text('date,net\n2021-01-04,0.1000004\n')
    assert same_content(a,b)[0]
    b.write_text('date,net\n2021-01-04,0.100002\n');assert not same_content(a,b)[0]
    b.write_text('date,net\n2023-01-03,0.1\n')
    with pytest.raises(Refuse):rows(b)


def test_xgb_and_de_roundtrip(tmp_path):
    import pandas as pd
    import random
    from new_bench_dataset import TreeDataset
    from new_run_benchmarks_v2 import construct_model
    from new_model_artifacts_v1 import save_model,load_predict_tabular,probe
    from qlib.contrib.model.double_ensemble import DEnsembleModel
    rng=np.random.default_rng(25);x=rng.normal(size=(240,3)).astype(np.float32)
    index=pd.MultiIndex.from_product([pd.bdate_range('2017-01-02',periods=30),list('ABCDEFGH')],names=['datetime','instrument'])
    frame=pd.DataFrame(np.c_[x,x[:,0]*.5+x[:,1]],index=index,columns=['a','b','c','LABEL0'])
    ds=TreeDataset(frame,dict(train=('2017-01-02','2017-01-27'),valid=('2017-01-30','2017-02-10'),test=('2017-01-30','2017-02-10')),
                   dict(train=None,valid=None,test=None),['a','b','c'])
    entry=dict(module='qlib.contrib.model.xgboost',class_='XGBModel',kwargs=dict(nthread=20,max_depth=2,eta=.1,subsample=.8))
    entry['class']=entry.pop('class_')
    predictions=[]
    for attempt in (1,2):
        model,_=construct_model(entry,1)
        model.fit(ds,num_boost_round=3,early_stopping_rounds=2,evals_result={},verbose_eval=False)
        predictions.append(model.predict(ds).values)
    np.testing.assert_array_equal(predictions[0],predictions[1])
    artifact=save_model(model,'XGBoost',tmp_path,'XGBoost',['a','b','c'])
    source=dict(artifact,adapter='xgboost')
    from xgboost import DMatrix
    np.testing.assert_array_equal(load_predict_tabular(source,probe(['a','b','c']).values),model.model.predict(DMatrix(probe(['a','b','c']))))
    np.random.seed(1);random.seed(1)
    de=DEnsembleModel(num_models=2,epochs=3,enable_fs=False,enable_sr=False,
                     num_threads=20,verbosity=-1,min_data_in_leaf=5)
    de.fit(ds)
    artifact=save_model(de,'DoubleEnsemble',tmp_path,'DE',['a','b','c'])
    expected=sum(m.predict(probe(['a','b','c'])[list(cols)])*w
                 for m,cols,w in zip(de.ensemble,de.sub_features,de.sub_weights))/sum(de.sub_weights)
    np.testing.assert_array_equal(load_predict_tabular(artifact,probe(['a','b','c']).values),expected)


def test_frozen_analysis_algorithms_unchanged():
    import ast
    before=ast.parse((ROOT/'src/new_gate_b_analysis_v3.py').read_text(encoding='utf-8'))
    after=ast.parse((ROOT/'src/new_gate_b_analysis_v3_1.py').read_text(encoding='utf-8'))
    functions=('bootstrap_indices','infer','paired_apply','newey_west','tost','conclusion',
               'holm','s_mapping','ir','spearman','share_replay_fee','replay_diagonal',
               'neural_scores','rows_to_series','eval_ic','sc5_pvalue')
    for name in functions:
        a=next(n for n in before.body if isinstance(n,ast.FunctionDef) and n.name==name)
        b=next(n for n in after.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(a)==ast.dump(b),name


def test_parameterized_grid_same_synthetic_books(tmp_path,monkeypatch):
    """Actual main() paths; daily CSV bytes and numerical JSON match v2."""
    import json
    import new_rule_grid_bench_v2 as old
    import new_rule_grid_bench_v2p as new
    from new_gate_b_common import sha256_file
    import new_bench_replay
    dates=np.arange(np.datetime64('2021-01-01'),np.datetime64('2021-01-15'))
    days=np.arange(2,10);n=60
    raw=SimpleNamespace(dates=dates,symbols=np.array([f'S{i}' for i in range(n)]),
        candidate=lambda t:np.ones(n,bool),days=lambda split:days)
    feed=dict(raw_close=np.full((len(dates),n),20.),state=np.zeros((len(dates),n),np.int8),
              share_multiplier=np.ones((len(dates),n)),cash_dividend_per_old_share=np.zeros((len(dates),n)))
    spec=tmp_path/'new_spec.json';spec.write_text('{}')
    source_dir=tmp_path/'new_sources';source_dir.mkdir();manifest=dict(kinds=list(old.KINDS),label='r1csrank',tags=[])
    for kind in old.KINDS:
        for seed in (1,2,3):
            tag=f'{kind}_seed{seed}';path=source_dir/(tag+'.npz')
            np.savez(path,dates_rule_select=dates[days].astype(str),scores_rule_select=np.tile(np.arange(n),(len(days),1)))
            path.with_suffix('.meta.json').write_text(json.dumps(dict(tag=tag,artifact_sha256=sha256_file(path),spec_sha256=sha256_file(spec))))
            manifest['tags'].append(dict(tag=tag,kind=kind,seed=seed,path=str(path)))
    mp=tmp_path/'new_manifest.json';mp.write_text(json.dumps(manifest))
    class SyntheticProvenance:
        def __init__(self,*a,**k):pass
        def finish(self,*a,**k):return {'synthetic':True}
    for module,name in ((old,'old'),(new,'new')):
        monkeypatch.setattr(module,'resolve_paths',lambda *a:None)
        monkeypatch.setattr(module,'BlindData',lambda args:raw)
        monkeypatch.setattr(module,'blind_feed',lambda *a:feed)
        monkeypatch.setattr(module,'blind_benchmark',lambda *a:dict(values={str(d):0. for d in dates}))
        monkeypatch.setattr(module,'Provenance',SyntheticProvenance)
        if module is new:monkeypatch.setattr(module,'verify_guard',lambda *a:None)
        argv=['grid','--root',str(tmp_path),'--spec',str(spec),'--cache',str(tmp_path/('new_cache_'+name)),
            '--scores-dir',str(source_dir),'--out',str(tmp_path/('new_'+name+'.json')),
            '--out-dir',str(tmp_path/('new_books_'+name))]
        if module is new:argv+=['--manifest',str(mp)]
        monkeypatch.setattr(sys,'argv',argv);module.main()
    a=json.loads((tmp_path/'new_old.json').read_text());b=json.loads((tmp_path/'new_new.json').read_text())
    assert a==b
    old_files=list((tmp_path/'new_books_old').rglob('daily.csv'))
    assert len(old_files)==18*8
    for path in old_files:
        assert path.read_bytes()==(tmp_path/'new_books_new'/path.relative_to(tmp_path/'new_books_old')).read_bytes()


def test_compare_files_missing_identity_is_mismatch(tmp_path):
    import json
    from new_compare_scores_v1 import compare_files
    from new_gate_b_common import sha256_file
    value=payload();value.pop('symbols')
    a=tmp_path/'new_old.npz';b=tmp_path/'new_new.npz';np.savez(a,**value);np.savez(b,**value)
    a.with_suffix('.meta.json').write_text('{}');b.with_suffix('.meta.json').write_text('{}')
    result=compare_files(a,b,sha256_file(a),sha256_file(b),['scores_valid'],DateGuard(PRE_UNSEAL_CUTOFF))
    assert result['verdict']=='MISMATCH' and not result['index_match']
    with pytest.raises(SystemExit):compare_files(a,b,'0'*64,sha256_file(b),['scores_valid'],DateGuard(PRE_UNSEAL_CUTOFF))


def test_guard_checksum_file(tmp_path):
    from new_guarded_data_v1 import verify_guard
    verify_guard(ROOT,committed=False)
    (tmp_path/'configs').mkdir();(tmp_path/'configs/new_guarded_data_v1.sha256').write_text('0'*64+'\n')
    with pytest.raises(SystemExit):verify_guard(tmp_path,committed=False)


def test_parameterized_forensics_same_synthetic_output(tmp_path,monkeypatch):
    import json
    import new_bench_forensics_v2 as old
    import new_bench_forensics_v2p as new
    from new_gate_b_common import sha256_file
    dates=np.r_[np.arange(np.datetime64('2019-01-01'),np.datetime64('2019-01-19')),
                np.arange(np.datetime64('2021-01-01'),np.datetime64('2021-01-11'))]
    days=np.arange(2,10);n=8;rng=np.random.default_rng(200)
    raw=SimpleNamespace(dates=dates,symbols=np.array(list('ABCDEFGH')),z_r1=rng.normal(size=(len(dates),n)),
        mask_r1=np.ones((len(dates),n),bool),days=lambda split:days)
    source_dir=tmp_path/'new_sources';source_dir.mkdir()
    kinds=['LSTM-csrank','GRU-csrank','ALSTM-csrank','Transformer-csrank','XGBoost-seeded']
    manifest=dict(kinds=kinds,label='r1csrank',tags=[])
    for kind in kinds:
        for seed in (1,2,3):
            tag=f'{kind}_seed{seed}';path=source_dir/(tag+'.npz')
            np.savez(path,dates_valid=dates[days].astype(str),scores_valid=rng.normal(size=(len(days),n)))
            path.with_suffix('.meta.json').write_text(json.dumps(dict(tag=tag,fit_seconds=.1,training=None,artifact_sha256=sha256_file(path))))
            manifest['tags'].append(dict(tag=tag,kind=kind,seed=seed,path=str(path)))
    mp=tmp_path/'new_manifest.json';mp.write_text(json.dumps(manifest))
    class SyntheticProvenance:
        def __init__(self,*a,**k):pass
        def finish(self,*a,**k):return {'synthetic':True}
    for module,name in ((old,'old'),(new,'new')):
        monkeypatch.setattr(module,'resolve_paths',lambda *a:None)
        monkeypatch.setattr(module,'BlindData',lambda args:raw)
        monkeypatch.setattr(module,'Provenance',SyntheticProvenance)
        if module is new:monkeypatch.setattr(module,'verify_guard',lambda *a:None)
        argv=['forensics','--root',str(tmp_path),'--cache',str(tmp_path/('new_cache_'+name)),
              '--scores-dir',str(source_dir),'--out',str(tmp_path/('new_'+name+'.json'))]
        if module is new:argv+=['--manifest',str(mp)]
        monkeypatch.setattr(sys,'argv',argv);module.main()
    assert (tmp_path/'new_old.json').read_bytes()==(tmp_path/'new_new.json').read_bytes()
