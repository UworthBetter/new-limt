"""All h5 tuning/final runs: validation IC only, with purge row counts."""
import argparse
import json
from pathlib import Path
import numpy as np
from new_gate_b_common import path_args,resolve_paths,preflight,write_json,sha256_file,Provenance
from new_guarded_data_v1 import DateGuard,PRE_UNSEAL_CUTOFF,GuardedData,verify_guard
from new_h5_common_v3 import H5BlindData,daily_ic,purge_days,selected_tags
from new_bench_forensics_v2 import daily_r1_ic


def main():
    p=argparse.ArgumentParser(description=__doc__);path_args(p)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--scores-dir',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path)
    args=p.parse_args();args.root=args.root or Path(__file__).resolve().parents[1]
    args.spec=args.spec or args.root/'configs/new_benchmarks_h5_v3.json'
    resolve_paths(args);verify_guard(args.root)
    args.out=args.out or args.root/'protocol/checks/new_benchmark_forensics_h5_v3.json';preflight([args.out,args.cache])
    provenance=Provenance(__file__,args)
    guard=DateGuard(PRE_UNSEAL_CUTOFF);data=GuardedData(H5BlindData(args),guard)
    selection=json.loads(args.selection.read_text(encoding='utf-8'));tags=sum(selected_tags(selection,True).values(),[])
    counts={}
    for split,lo,end in (('train','2008-01-01','2018-12-31'),('valid','2019-01-01','2020-12-31')):
        days=np.flatnonzero((data.dates>=np.datetime64(lo))&(data.dates<=np.datetime64(end)))
        purged=set(purge_days(data.dates,days,end).tolist());before=after=0
        for t in days:
            count=int(data.mask_h5[t].sum());before+=count
            if int(t) in purged:after+=count
        counts[split]=dict(before_purge=before,after_purge=after)
    days=data.days('valid');results={}
    for tag in tags:
        path=args.scores_dir/(tag+'.npz');mp=path.with_suffix('.meta.json')
        meta=json.loads(mp.read_text(encoding='utf-8'))
        if meta.get('tag')!=tag or meta.get('smoke') or meta.get('score_sha256')!=sha256_file(path):
            raise SystemExit('invalid forensic score provenance')
        with np.load(path,allow_pickle=False) as z:
            if not np.array_equal(z['dates_valid'].astype('datetime64[D]'),data.dates[days]):raise SystemExit('validation dates mismatch')
            scores=z['scores_valid']
        r1=daily_r1_ic(data,days,scores,data.z_r1,data.mask_r1)
        results[tag]=dict(mean_valid_r1_ic=float(np.mean(r1)) if r1 else None,
            mean_valid_h5_ic=daily_ic(data,days,scores),fit_seconds=meta['fit_seconds'],training=meta['training'],
            score_sha256=sha256_file(path),meta_sha256=sha256_file(mp))
    write_json(args.out,dict(row_counts=counts,benchmarks=results,selection_sha256=sha256_file(args.selection),
        meta=provenance.finish(guard.metadata(('labels','features')))))
    print('FORENSICS DONE (validation only)')


if __name__=='__main__':main()
