"""Validation-only configuration selection; no rule-select arrays are read."""
import argparse
import json
from pathlib import Path
import numpy as np
from new_gate_b_common import (path_args, resolve_paths, write_json, preflight,
                               sha256_file, benchmark_platform_fingerprint)
from new_guarded_data_v1 import DateGuard, PRE_UNSEAL_CUTOFF, GuardedData, verify_guard
from new_h5_common_v3 import H5BlindData, daily_ic, tag_for, resolve_config, BIAS_NOTE


def select_config(ics, factor_counts):
    if set(ics)!={f'cfg{k}' for k in range(1,9)} or not all(np.isfinite(v) for v in ics.values()):
        raise SystemExit('selection requires all eight finite configurations')
    best=max(ics.values());pool=[c for c,v in ics.items() if v>=best-.002]
    return min(pool,key=lambda c:(factor_counts[c],-ics[c],int(c[3:]))),pool


def main():
    p=argparse.ArgumentParser(description=__doc__);path_args(p)
    p.add_argument('--scores-dir',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--out',type=Path);args=p.parse_args()
    args.root=args.root or Path(__file__).resolve().parents[1]
    args.spec=args.spec or args.root/'configs/new_benchmarks_h5_v3.json'
    resolve_paths(args);verify_guard(args.root)
    args.out=args.out or args.root/'protocol/checks/new_h5_selection_v3.json';preflight([args.out,args.cache])
    spec=json.loads(args.spec.read_text(encoding='utf-8'))
    guard=DateGuard(PRE_UNSEAL_CUTOFF);data=GuardedData(H5BlindData(args),guard);days=data.days('valid')
    result={};platforms=set()
    for name in ('LSTM','Transformer'):
        ics={};counts={};inputs={}
        _,official,_,_=resolve_config(spec,name,'cfg1',1)
        for k in range(1,9):
            cfg=f'cfg{k}';tag=tag_for(name,cfg,1);path=args.scores_dir/(tag+'.npz')
            mp=path.with_suffix('.meta.json');meta=json.loads(mp.read_text(encoding='utf-8'))
            if (meta.get('tag')!=tag or meta.get('smoke') or meta.get('spec_sha256')!=sha256_file(args.spec)
                or meta.get('score_sha256')!=sha256_file(path)
                or meta.get('artifact_sha256')!=sha256_file(meta['checkpoint_path'])):
                raise SystemExit('invalid selection input provenance')
            platforms.add(json.dumps(benchmark_platform_fingerprint(meta),sort_keys=True))
            with np.load(path,allow_pickle=False) as z:
                if not np.array_equal(z['dates_valid'].astype('datetime64[D]'),data.dates[days]):
                    raise SystemExit('selection validation dates mismatch')
                ic=daily_ic(data,days,z['scores_valid'])
            if ic is None or abs(ic-meta['best_valid_ic5'])>1e-9:raise SystemExit('independent IC disagrees')
            ics[cfg]=ic
            _,kw,metric,_=resolve_config(spec,name,cfg,1)
            width='hidden_size' if name=='LSTM' else 'd_model'
            # Width defaults absent in Transformer's official kwargs are explicit in resolved grid.
            default_width=official.get(width,64);default_lr=official.get('lr',.0001)
            counts[cfg]=int(kw.get(width,default_width)!=default_width)+int(kw.get('lr',default_lr)!=default_lr)+int(metric!='valid_loss')
            inputs[tag]=dict(score_sha256=sha256_file(path),meta_sha256=sha256_file(mp),checkpoint_sha256=meta['artifact_sha256'])
        selected,pool=select_config(ics,counts)
        result[name]=dict(ics=ics,pool=pool,selected_cfg=selected,nonofficial_factors=counts,
            tuned_arm='identical_to_default' if selected=='cfg1' else 'tuned',
            note='the tuning budget selected the published configuration' if selected=='cfg1' else 'tuned-h5',
            seed1_selection_role='final_seed1_reused',bias_note=BIAS_NOTE,
            tie_rule=spec['selection']['tie_rule'],inputs=inputs)
    if len(platforms)!=1:raise SystemExit('selection platforms differ')
    result['spec_sha256']=sha256_file(args.spec);result.update(guard.metadata(('labels','features')))
    write_json(args.out,result)
    for name in ('LSTM','Transformer'):print('SELECTED '+name+' '+result[name]['selected_cfg'])


if __name__=='__main__':main()
