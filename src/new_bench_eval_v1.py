"""Descriptive evaluation at unseal only; arm list is exclusively gate_b.json.

Expected streams: prices (execution books), features (prediction/candidate),
bench (excess returns), labels (IC); marks additionally iff any arm requests a
soft portfolio. Every expected stream must be touched, and no other stream may
be touched. Control and identical_to_default entries never generate predictions.
No inferential statistics, significance fields, conclusion or ranking are emitted.
score_source maps seed strings to {path,sha256,adapter,columns,...}. TS sources
add model (LSTM/GRU/ALSTM/Transformer), kwargs and step_len. limt sources contain
their configuration in the weights_only checkpoint. Additional ensemble files
are explicitly listed in files, each with path and sha256.
"""
import argparse
from contextlib import redirect_stdout,redirect_stderr
import importlib
import io
import json
import logging
from pathlib import Path
import numpy as np

from new_gate_b_common import path_args,resolve_paths,preflight,sha256_file,write_json
from new_guarded_data_v1 import (DateGuard,UNSEAL_CUTOFF,verify_guard,load_evaluation_data,
    guarded_portfolio_events,import_closure,third_party_versions)
from new_h5_common_v3 import daily_ic,network
from new_model_artifacts_v1 import load_predict_tabular

FAMILIES=frozenset(('qlib_default_r1','qlib_h5','doubleensemble','cpu_baseline',
                   'limt_ablation','dose_exploratory','zscore_control','seed0_control'))
CONTROLS=frozenset(('zscore_control','seed0_control'))
TS_CLASSES={'LSTM':('pytorch_lstm_ts','LSTM'),'GRU':('pytorch_gru_ts','GRU'),
            'ALSTM':('pytorch_alstm_ts','ALSTM'),'Transformer':('pytorch_transformer_ts','TransformerModel')}


def sources_for(arm):
    sources=arm['score_source']
    if isinstance(sources,list):
        if len({str(x['seed']) for x in sources})!=len(sources):raise SystemExit('duplicate source seed')
        sources={str(x['seed']):x for x in sources}
    if not isinstance(sources,dict) or set(sources)!={str(s) for s in arm['seeds']}:
        raise SystemExit('score sources must match seeds exactly')
    return sources


def verify_source(source,tag):
    for file in [source]+source.get('files',[]):
        path=Path(file['path'])
        if not path.is_file() or sha256_file(path)!=file['sha256']:
            raise SystemExit('missing/mismatched model for '+tag)
    if source.get('adapter')=='doubleensemble':
        ensemble=json.loads(Path(source['path']).read_text(encoding='utf-8'))
        declared={(str(Path(f['path']).resolve()),f['sha256']) for f in source.get('files',[])}
        actual={(str(Path(f['path']).resolve()),f['sha256']) for f in ensemble['files']}
        if declared!=actual:raise SystemExit('undeclared ensemble dependency')


def parse_arms(gate):
    arms=gate.get('descriptive_arms');seen=set()
    if not isinstance(arms,list) or not arms:raise SystemExit('missing descriptive_arms')
    for arm in arms:
        required={'arm_id','family','label','score_source','seeds','selected_rule','default_rule','descriptive','note'}
        if required-set(arm) or arm['arm_id'] in seen or arm['family'] not in FAMILIES or arm['descriptive'] is not True:
            raise SystemExit('incomplete/duplicate/invalid descriptive arm')
        seen.add(arm['arm_id'])
        if not arm['seeds'] or len(set(arm['seeds']))!=len(arm['seeds']):raise SystemExit('invalid arm seeds')
        rule_parameters(arm['selected_rule']);rule_parameters(arm['default_rule'])
        if arm['default_rule']!='drop5-every1d':raise SystemExit('incorrect default rule')
        if arm['family'] in CONTROLS:
            result=arm.get('rule_select_result')
            if not result:raise SystemExit('controls require frozen rule_select_result path/sha')
            verify_source(result,arm['arm_id'])
        elif arm.get('identical_to_default') or 'identical_to_default' in arm['note']:
            continue
        else:
            for source in sources_for(arm).values():verify_source(source,arm['arm_id'])
    return arms


def rule_parameters(name):
    import re
    m=re.fullmatch(r'drop(2|5|10|20)-every(1|5)d',name)
    if m is None:raise SystemExit('invalid frozen rule')
    return int(m[1]),int(m[2])


def configure_replay():
    import new_gate_b_analysis_v3_1 as replay
    from new_bench_replay import DEFAULT,share_replay
    from v2_w3_ledger import Account,execute_close
    from v2_w4_training import topk_target,portfolio_events
    from v2_w6_policy import listing_eligible
    replay.CB_DEFAULT=DEFAULT;replay.share_replay=share_replay;replay.Account=Account
    replay.execute_close=execute_close;replay.topk_target=topk_target;replay.listing_eligible=listing_eligible
    replay.portfolio_events=portfolio_events
    return replay,portfolio_events


def predict_source(source,data,days):
    """Predict from saved weights; this function has no fit/train branch."""
    adapter=source['adapter'];columns=source.get('columns',list(data.feature_names))
    if adapter=='limt':
        import torch
        from v2_w4_training import make_model,predict
        from v2_w6_run import DEFAULT,validate_config
        ckpt=torch.load(source['path'],map_location='cpu',weights_only=True)
        config=validate_config(dict(DEFAULT,**ckpt['configuration']))
        device='cuda' if torch.cuda.is_available() else 'cpu'
        model=make_model(data,config).to(device);model.load_state_dict(ckpt['state_dict'])
        return predict(model,data,days,config,device)[...,0]
    if adapter=='qlib_ts':
        import torch
        from new_bench_dataset import BenchDataset,build_master,score_matrix
        name=source['model']
        if name not in TS_CLASSES:raise SystemExit('unknown qlib TS model')
        module,clsname=TS_CLASSES[name]
        cls=getattr(importlib.import_module('qlib.contrib.model.'+module),clsname)
        kwargs=dict(source['kwargs'])
        model=cls(**kwargs)
        state=torch.load(source['path'],map_location='cpu',weights_only=True)
        # The four official TS models expose distinct torch-module attribute names.
        nets=[v for v in vars(model).values() if isinstance(v,torch.nn.Module)]
        if len(nets)!=1:raise SystemExit('ambiguous qlib network')
        nets[0].load_state_dict(state);model.fitted=True
        dates=[str(d) for d in data.dates[days]]
        dummy=np.zeros((len(data.dates),len(data.symbols)),np.float32)
        master=build_master(data,dummy,max(0,int(days[0])-source['step_len']),int(days[-1]),columns)
        ds=BenchDataset(master,{'test':(dates[0],dates[-1])},{'test':None},source['step_len'])
        return score_matrix(model.predict(ds),dates,list(data.symbols))
    if adapter not in ('ridge','xgboost','lightgbm','doubleensemble'):raise SystemExit('unknown model adapter')
    positions=[list(data.feature_names).index(c) for c in columns]
    output=[]
    for t in days:
        x=data.features[int(t)][:,positions]
        ok=np.isfinite(x).all(axis=1);row=np.full(len(data.symbols),np.nan)
        if ok.any():row[ok]=load_predict_tabular(source,x[ok])
        output.append(row)
    return np.stack(output)


def descriptive_book(rows,bench):
    net=np.array([r['net'] for r in rows]);fee=np.array([r['fee_fraction'] for r in rows])
    baseline=np.array([bench[r['date']] for r in rows]);scale=252*100
    return dict(annual_gross_excess_pp=float((net+fee-baseline).mean()*scale),
        annual_net_excess_pp=float((net-baseline).mean()*scale),
        cost_drag_pp=float(fee.mean()*scale),mean_turnover=float(np.mean([r['turnover'] for r in rows])))


def assert_descriptive(value):
    if isinstance(value,dict):
        for k,v in value.items():
            if k.startswith('p_') or 'holm' in k.lower() or k in ('conclusion','s_mapping'):
                raise SystemExit('inferential field in descriptive output')
            assert_descriptive(v)
    elif isinstance(value,list):
        for v in value:assert_descriptive(v)


def seed_average(items):
    first=items[0]
    if any(x is None for x in items):return None
    if isinstance(first,dict):return {k:seed_average([x[k] for x in items]) for k in first}
    if isinstance(first,str):return first
    return float(np.mean(items))


def run(args):
    preflight([args.out,args.cache])
    gate=json.loads(args.gate.read_text(encoding='utf-8'))
    if gate.get('bench_eval_sha256')!=sha256_file(__file__):raise SystemExit('bench_eval sha mismatch')
    verify_guard(args.root,gate);arms=parse_arms(gate)
    if {a['family'] for a in arms}!=FAMILIES:raise SystemExit('gate must explicitly cover all declared families')
    guard=DateGuard(UNSEAL_CUTOFF);replay,portfolio=configure_replay();replay.GUARD=guard
    from v2_w4_data import Data
    from v2_w5_results import benchmark,load_execution_feed
    from v2_w6_run import DEFAULT,validate_config
    # Preimport supported adapter modules before the start closure, without loading data/models.
    for a in arms:
        if a['family'] not in CONTROLS and not a.get('identical_to_default') and 'identical_to_default' not in a['note']:
            for s in sources_for(a).values():
                if s['adapter']=='qlib_ts':
                    with redirect_stdout(io.StringIO()):
                        importlib.import_module('qlib.contrib.model.'+TS_CLASSES[s['model']][0])
                elif s['adapter']=='xgboost':
                    importlib.import_module('xgboost')
                elif s['adapter'] in ('lightgbm','doubleensemble'):
                    importlib.import_module('lightgbm')
    start=import_closure(args.root,args.limt_src)
    data,feed,bench=load_evaluation_data(args,guard);days=data.days('eval')
    expected={'prices','features','labels','bench'}
    if any(a.get('soft_portfolio') for a in arms):expected.add('marks')
    output={};soft_records=[];diagonal_done=False
    for arm in arms:
        name=arm['arm_id']
        if arm['family'] in CONTROLS:
            output[name]=dict(eval='no evaluation-window result (model not saved)',rule_select_result=arm['rule_select_result']);continue
        if arm.get('identical_to_default') or 'identical_to_default' in arm['note']:
            output[name]=dict(eval='identical_to_default',note='the tuning budget selected the published configuration');continue
        per_seed={}
        for seed,source in sources_for(arm).items():
            verify_source(source,name)
            with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):scores=predict_source(source,data,days)
            if scores.shape!=(len(days),len(data.symbols)):raise SystemExit('adapter output shape mismatch')
            if not diagonal_done:
                vd=data.days('valid')
                with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):probe=predict_source(source,data,vd)
                replay.replay_diagonal(data,vd,probe,feed);diagonal_done=True
            record=dict(ic_h5=replay.eval_ic(data,days,scores,split='eval'),rules={})
            if arm['label'] in ('r1','r1csrank','csrank','z_r1'):
                record['ic_r1']=daily_ic(data,days,scores,horizon=2,end=UNSEAL_CUTOFF)
            for role in ('default_rule','selected_rule'):
                rule=arm[role];drop,every=rule_parameters(rule)
                book=replay.share_replay_fee(data,days,scores,feed,drop,every,
                    replay.CB_DEFAULT['sell_rate'],replay.CB_DEFAULT['sell_rate'],'9999-01-01')
                stamp=replay.share_replay_fee(data,days,scores,feed,drop,every,
                    replay.CB_DEFAULT['sell_rate'],.0010,replay.STAMP_CUTOFF)
                record['rules'][role]=dict(rule=rule,base=descriptive_book(book,bench),stamp_sensitivity=descriptive_book(stamp,bench))
            if arm.get('soft_portfolio'):
                call_marks=[]
                rows=guarded_portfolio_events(portfolio,data,days,scores,guard,call_marks,
                    tau=replay.TAU,buy=replay.CB_DEFAULT['buy_rate'],sell=replay.CB_DEFAULT['sell_rate'],rule='soft')
                record['soft_marks_max']=call_marks[-1]
                soft_records.append(dict(arm=name,seed=seed,maximum=call_marks[-1]))
            per_seed[seed]=record
        output[name]=dict(family=arm['family'],label=arm['label'],descriptive=True,note=arm['note'],
                          per_seed=per_seed,seed_mean=seed_average(list(per_seed.values())))
    result=dict(arms=output,bench_eval_sha256=sha256_file(__file__),gate_sha256=sha256_file(args.gate),
        **guard.metadata(expected),soft_marks_max_per_call=soft_records,
        import_closure=dict(start=start,end=import_closure(args.root,args.limt_src,start),third_party=third_party_versions(('torch','numpy'))))
    assert_descriptive(result);write_json(args.out,result);print('WROTE '+str(args.out))


def main():
    p=argparse.ArgumentParser(description=__doc__);action=p.add_mutually_exclusive_group(required=True)
    action.add_argument('--run',action='store_true');action.add_argument('--self-test',action='store_true')
    path_args(p);p.add_argument('--gate',type=Path);p.add_argument('--cache',type=Path);p.add_argument('--out',type=Path)
    args=p.parse_args()
    if args.self_test:
        import subprocess,sys
        raise SystemExit(subprocess.call([sys.executable,'-m','pytest',str(Path(__file__).resolve().parents[1]/'tests/new_test_bench_eval_v1.py'),'-q']))
    resolve_paths(args,('root','limt_src','w0w2','labels','panel','splits','feed','bench'))
    args.gate=args.gate or args.root/'protocol/gate_b.json';args.out=args.out or args.root/'runs/new_bench_eval_v1.json'
    if args.cache is None:p.error('--run requires fresh --cache')
    logging.disable(logging.CRITICAL);run(args)


if __name__=='__main__':main()
