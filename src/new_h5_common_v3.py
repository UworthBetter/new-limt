"""h5 labels, purging, provenance and artifact contracts shared by new runners.

The externally finalized spec is required; this module never invents grid values.
official_entry_v2 maps model names to {module,class,kwargs}; tuning_grid.resolved
maps model names to cfg1..cfg8, each containing kwargs and early_stop_metric.
An equivalent cfg-first resolved mapping is also accepted.
"""
from contextlib import contextmanager
import importlib
import inspect
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd

from new_gate_b_common import BlindData, preflight, sha256_file
from new_guarded_data_v1 import DateGuard, PRE_UNSEAL_CUTOFF, GuardedData

QLIB_SHA = {'LSTM': '4496a8bd0f2b8812523676f17bf50fdf18268ab66a5869e8f9b44c5cb4b4e230',
            'Transformer': '053bed36b9d712275fd1cd4735cc897a0adde196e2646f83b7773257d040d79f'}
SETTINGS = ('n_epochs','early_stop','batch_size','n_jobs','device','d_feat',
            'lr','hidden_size','d_model','num_layers','dropout','optimizer','loss')
BIAS_NOTE = ('valid_ic5-early-stopped configurations are selected on the same metric and window '
             'they stopped on; selection is optimistic toward them, which favours the tuned arm '
             '(conservative for the LiMT claim).')


def csrank_h5(z, mask, lo, hi):
    out = np.full((hi-lo+1, z.shape[1]), np.nan, np.float32)
    for i,t in enumerate(range(lo, hi+1)):
        keep = mask[t]
        values = z[t, keep]
        if not np.isfinite(values).all():
            raise ValueError('nonfinite member of label_mask_h5')
        if keep.any():
            out[i,keep] = (pd.Series(values).rank(method='average',pct=True).to_numpy()-.5)*3.46
    return out


def purge_days(dates, days, end, horizon=6):
    days = np.asarray(days, int)
    valid = days + horizon < len(dates)
    return days[valid & (np.asarray(dates)[np.minimum(days+horizon,len(dates)-1)].astype('datetime64[D]')
                         <= np.datetime64(end))]


def daily_ic(data, days, scores, split='valid', horizon=6, end='2020-12-31'):
    """h5: split loss mask AND listing eligibility, without candidate masking."""
    if scores.shape != (len(days),len(data.symbols)):
        raise SystemExit('IC score shape mismatch')
    values = []
    admissible = set(purge_days(data.dates, days, end, horizon).tolist())
    for i,t in enumerate(days):
        t = int(t)
        if t not in admissible:
            continue
        if horizon == 6:
            keep = data.loss_masks[split][t] & data.listing_eligible[t] & np.isfinite(scores[i])
            label = data.z_h5
        else:
            keep = data.mask_r1[t] & np.isfinite(scores[i])
            label = data.z_r1
        selected = np.flatnonzero(keep)
        if not len(selected):
            continue
        y = label[t, selected]
        x = scores[i, selected]
        ok = np.isfinite(y)
        x,y = x[ok],y[ok]
        if len(x)>1 and x.std()>0 and y.std()>0:
            values.append(float(np.corrcoef(x,y)[0,1]))
    return float(np.mean(values)) if values else None


class H5BlindData(BlindData):
    def __init__(self,args):
        super().__init__(args)
        n = len(self.dates)
        with np.load(args.labels,allow_pickle=False) as z:
            self.z_h5 = z['z_h5'][:n]
            self.mask_h5 = z['label_mask_h5'][:n]
            if not np.array_equal(z['exit_idx_h5'][:n],np.arange(n)+6):
                raise SystemExit('invalid h5 exit indices')
            if not np.array_equal(z['entry_idx'][:n],np.arange(n)+1):
                raise SystemExit('invalid entry indices')
        with np.load(args.splits,allow_pickle=False) as z:
            self.loss_masks = {s:z['loss_mask_'+s][:n] for s in ('train','valid','rule_select')}
        from v2_w6_policy import listing_eligible
        self.listing_eligible = listing_eligible(self.dates,self.symbols)


@contextmanager
def isolated_rng():
    import torch
    py,npstate,cpu = random.getstate(),np.random.get_state(),torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(py); np.random.set_state(npstate); torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def network(model):
    return model.LSTM_model if hasattr(model,'LSTM_model') else model.model


def effective_settings(model):
    return {k:str(getattr(model,k)) if k=='device' else getattr(model,k)
            for k in SETTINGS if hasattr(model,k)}


def resolve_config(spec, name, cfg, seed):
    entry = (spec['official_entry_v2'][name] if 'official_entry_v2' in spec
             else spec['models'][name]['official_entry_v2'])
    kwargs = dict(entry['kwargs'],seed=seed)
    resolved = spec['tuning_grid']['resolved']
    row = resolved[name][cfg] if name in resolved else resolved[cfg][name]
    width = 'hidden_size' if name=='LSTM' else 'd_model'
    if set(row)-{'kwargs','early_stop_metric','grid_factors','lr',width}:
        raise SystemExit('unknown resolved configuration field')
    changes = row.get('kwargs',{k:row[k] for k in ('lr',width) if k in row})
    allowed = {'lr',width}
    if set(changes)-allowed:
        # Complete resolved kwargs are supported only when frozen fields agree.
        if any(kwargs.get(k) != v for k,v in changes.items() if k not in allowed):
            raise SystemExit('grid changes a frozen constructor kwarg')
    if cfg=='cfg1':
        defaults = {'lr':.0001,'d_model':64} if name=='Transformer' else {}
        if any(v!=kwargs.get(k,defaults.get(k)) for k,v in changes.items()):
            raise SystemExit('cfg1 must equal official constructor kwargs')
        result=kwargs.copy()
    else:
        result = dict(kwargs,**changes)
    metric = row.get('early_stop_metric','valid_loss')
    if metric not in ('valid_loss','valid_ic5'):
        raise SystemExit('unknown early stop metric')
    # metric_fn itself must stay a supported qlib loss metric.
    if result.get('metric','') not in ('','loss'):
        raise SystemExit('qlib metric must remain loss; select IC via early_stop_metric')
    if cfg=='cfg1' and (result != kwargs or metric!='valid_loss'):
        raise SystemExit('cfg1 must equal official constructor kwargs')
    factors=row.get('grid_factors',next((r for r in spec['tuning_grid'].get('configs',[]) if r['cfg']==cfg),{}))
    return entry,result,metric,factors


def assert_cfg1(root, spec, name, seed, kwargs, model=None):
    path = Path(root)/'runs/new_scores_v2'/f'{name}-csrank_seed{seed}.meta.json'
    records = spec.get('r1_meta_sha256')
    if not isinstance(records,dict):
        raise SystemExit('finalized spec must pin r1_meta_sha256 for cfg1 checks')
    expected = records.get(path.name,records.get(f'{name}-csrank_seed{seed}'))
    if not expected:raise SystemExit('missing r1 metadata sha in finalized spec')
    if sha256_file(path)!=expected:
        raise SystemExit('r1 metadata sha mismatch')
    meta = json.loads(path.read_text(encoding='utf-8'))
    if kwargs!=meta['kwargs']:
        raise SystemExit('cfg1 differs from r1 constructor kwargs')
    if model is not None:
        effective = effective_settings(model)
        for k,v in meta['effective_model_settings'].items():
            if k in SETTINGS and effective.get(k)!=v:
                raise SystemExit('cfg1 differs from r1 effective settings: '+k)


def verify_qlib_source(entry,name):
    cls = getattr(importlib.import_module(entry['module']),entry['class'])
    if sha256_file(inspect.getfile(cls)) != QLIB_SHA[name]:
        raise SystemExit('qlib model source hash mismatch')
    if name=='Transformer':
        expected = dict(d_feat=20,d_model=64,batch_size=8192,nhead=2,num_layers=2,
            dropout=0,n_epochs=100,lr=.0001,metric='',early_stop=5,loss='mse',
            optimizer='adam',reg=.001,n_jobs=10,GPU=0,seed=None)
        actual={k:v.default for k,v in inspect.signature(cls.__init__).parameters.items()
                if k not in ('self','kwargs')}
        if actual!=expected:
            raise SystemExit('Transformer defaults differ from v0.9.6')
    return cls


def validate_checkpoint(path, state, expected_sha=None):
    import torch
    path=Path(path)
    if not path.is_file() or path.stat().st_size==0:
        raise SystemExit('missing/empty checkpoint')
    digest=sha256_file(path)
    if expected_sha and digest!=expected_sha:
        raise SystemExit('checkpoint file sha mismatch')
    try:
        loaded=torch.load(path,map_location='cpu',weights_only=True)
        if set(loaded)!=set(state) or any(loaded[k].dtype!=v.dtype or loaded[k].shape!=v.shape
            or not torch.equal(loaded[k],v.detach().cpu()) for k,v in state.items()):
            raise SystemExit('checkpoint tensor mismatch')
    except Exception as e:
        raise SystemExit('checkpoint readback failed') from e
    return dict(artifact_sha256=digest,checkpoint_path=str(path),
                checkpoint_bytes=path.stat().st_size,n_tensors=len(state))


def save_checkpoint(path, model):
    import torch
    preflight([path]); Path(path).parent.mkdir(parents=True,exist_ok=True)
    state=network(model).state_dict()
    with Path(path).open('xb') as f:
        torch.save(state,f)
    return validate_checkpoint(path,state)


def persistent_models_path(path, smoke=False):
    path=Path(path).resolve()
    if not smoke and not path.is_relative_to(Path('/mnt/workspace').resolve()):
        raise SystemExit('--models-out must be under /mnt/workspace')
    return path


def tag_for(name,cfg,seed):
    return f'{name}-h5default_seed{seed}' if cfg=='cfg1' else f'{name}-h5tuned_{cfg}_seed{seed}'


def selected_tags(selection, all_tuning=False):
    result={}
    for name in ('LSTM','Transformer'):
        cfg=selection[name]['selected_cfg']
        if cfg not in {f'cfg{i}' for i in range(1,9)}:
            raise SystemExit('invalid selected cfg')
        result[name+'-h5default']=[tag_for(name,'cfg1',s) for s in (1,2,3)]
        if cfg!='cfg1':
            result[name+'-h5tuned']=[tag_for(name,cfg,s) for s in (1,2,3)]
        if all_tuning:
            chosen=set(sum(result.values(),[]))
            result[name+'-tuning']=[tag_for(name,f'cfg{k}',1) for k in range(2,9)
                                    if tag_for(name,f'cfg{k}',1) not in chosen]
    return result
