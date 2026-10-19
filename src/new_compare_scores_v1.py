"""r9 score comparison. Strict index equality, fixed exclusive tolerance."""
import argparse
import json
from pathlib import Path
import numpy as np
from new_gate_b_common import sha256_file,write_json
from new_guarded_data_v1 import DateGuard,PRE_UNSEAL_CUTOFF,verify_guard

TOL=1e-6


def compare(old,new,keys,guard):
    n_cells=n_nan=n_over=0;maximum=0.;index_match=True;nonfinite_match=True
    for key in keys:
        split=key.removeprefix('scores_')
        for index in ('dates_'+split,'candidate_'+split):
            if index not in old or index not in new or not np.array_equal(old[index],new[index]):index_match=False
        for item in (old,new):
            for d in item['dates_'+split].astype('datetime64[D]'):guard.touch('features',d)
        # Optional identifiers must match in both directions. Legacy v2 fixes instrument order
        # through label/panel input hashes, checked by compare_files below.
        for index in ('symbols','instruments'):
            if (index in old)!=(index in new) or (index in old and not np.array_equal(old[index],new[index])):index_match=False
        a,b=old[key],new[key]
        if a.shape!=b.shape:index_match=False;continue
        n_cells+=a.size;n_nan+=int(np.count_nonzero(np.isnan(a)!=np.isnan(b)))
        if not np.array_equal(np.isposinf(a),np.isposinf(b)) or not np.array_equal(np.isneginf(a),np.isneginf(b)):
            nonfinite_match=False
        keep=np.isfinite(a)&np.isfinite(b);diff=np.abs(a[keep].astype(np.float64)-b[keep].astype(np.float64))
        n_over+=int(np.count_nonzero(diff>=TOL));maximum=max(maximum,float(diff.max()) if diff.size else 0.)
    return dict(verdict='MATCH' if index_match and nonfinite_match and not n_nan and not n_over else 'MISMATCH',
        index_match=index_match,nonfinite_match=nonfinite_match,n_cells=n_cells,n_nan_mismatch=n_nan,
        n_over_tol=n_over,max_abs_diff=maximum,tol=TOL)


def compare_files(old,new,old_sha,new_sha,keys,guard):
    if sha256_file(old)!=old_sha or sha256_file(new)!=new_sha:raise SystemExit('score file sha mismatch')
    with np.load(old,allow_pickle=False) as a,np.load(new,allow_pickle=False) as b:
        index_provenance=True
        # Without explicit symbols require frozen upstream identifier provenance.
        if 'symbols' not in a or 'symbols' not in b:
            ma=json.loads(Path(old).with_suffix('.meta.json').read_text(encoding='utf-8'))
            mb=json.loads(Path(new).with_suffix('.meta.json').read_text(encoding='utf-8'))
            if any(ma.get('inputs',{}).get(k,{}).get('sha256') is None or
                   ma['inputs'][k]['sha256']!=mb.get('inputs',{}).get(k,{}).get('sha256') for k in ('labels','panel')):
                index_provenance=False
        result=compare(a,b,keys,guard)
        if not index_provenance:
            # Legacy DE metadata does not identify its column order. Never call
            # that a MATCH merely because two unlabelled matrices look alike.
            result.update(verdict='MISMATCH',index_match=False,
                          index_provenance='missing/different instrument-order provenance')
    result.update(old_sha256=old_sha,new_sha256=new_sha,script_sha256=sha256_file(__file__))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('old','new'):p.add_argument('--'+name,type=Path,required=True);p.add_argument('--'+name+'-sha',required=True)
    p.add_argument('--keys',default='scores_valid,scores_rule_select');p.add_argument('--tag',required=True)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--out',type=Path)
    args=p.parse_args();verify_guard(args.root);guard=DateGuard(PRE_UNSEAL_CUTOFF)
    keys=args.keys.split(',')
    if set(keys)-{'scores_valid','scores_rule_select'}:raise SystemExit('unsupported score keys')
    result=compare_files(args.old,args.new,args.old_sha,args.new_sha,keys,guard)
    result.update(guard.metadata(('features',)))
    write_json(args.out or args.root/'protocol/checks'/f'new_compare_{args.tag}.json',result)
    print(result['verdict']+' '+args.tag+(f" n_over_tol={result['n_over_tol']}" if result['verdict']=='MISMATCH' else ''))


if __name__=='__main__':main()
