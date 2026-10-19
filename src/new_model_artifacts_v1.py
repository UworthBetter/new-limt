"""Lossless model save/readback contracts on a fixed synthetic probe matrix."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from new_gate_b_common import preflight,sha256_file,write_json


def probe(columns):
    return pd.DataFrame(np.random.default_rng(20261019).normal(size=(17,len(columns))),columns=columns)


def exact(a,b):
    if not np.array_equal(np.asarray(a),np.asarray(b),equal_nan=True):
        raise SystemExit('model readback prediction mismatch')


def save_model(model,kind,directory,tag,columns):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    x=probe(columns);files=[]
    if kind=='ridge':
        path=directory/f'new_{tag}.npz';preflight([path])
        coef=np.asarray(model)
        with path.open('xb') as f:np.savez(f,coef=coef,columns=np.array(columns))
        with np.load(path,allow_pickle=False) as z:loaded=z['coef']
        exact(np.c_[np.ones(len(x)),x.values]@coef,np.c_[np.ones(len(x)),x.values]@loaded)
        adapter='ridge'
    elif kind=='XGBoost':
        import xgboost as xgb
        booster=model.model;path=directory/f'new_{tag}.ubj';preflight([path]);booster.save_model(path)
        loaded=xgb.Booster();loaded.load_model(path)
        dm=xgb.DMatrix(x)
        exact(booster.predict(dm),loaded.predict(dm));adapter='xgboost'
    elif kind=='DoubleEnsemble':
        import lightgbm as lgb
        for i,booster in enumerate(model.ensemble):
            path=directory/f'new_{tag}_sub{i}.txt';preflight([path]);booster.save_model(str(path))
            loaded=lgb.Booster(model_file=str(path));features=list(model.sub_features[i])
            exact(booster.predict(x[features]),loaded.predict(x[features]))
            files.append(dict(path=str(path),sha256=sha256_file(path),columns=features))
        path=directory/f'new_{tag}_ensemble.json'
        write_json(path,dict(files=files,weights=list(model.sub_weights),columns=list(columns)))
        adapter='doubleensemble'
    else:
        import lightgbm as lgb
        path=directory/f'new_{tag}.txt';preflight([path]);model.save_model(str(path))
        loaded=lgb.Booster(model_file=str(path));exact(model.predict(x),loaded.predict(x));adapter='lightgbm'
    if not path.is_file() or not path.stat().st_size:raise SystemExit('missing model artifact')
    return dict(adapter=adapter,path=str(path),checkpoint_path=str(path),artifact_sha256=sha256_file(path),
                checkpoint_bytes=path.stat().st_size,files=files,columns=list(columns))


def load_predict_tabular(source,x):
    kind=source['adapter'];path=source['path']
    columns=source['columns'];frame=pd.DataFrame(x,columns=columns)
    if kind=='ridge':
        with np.load(path,allow_pickle=False) as z:coef=z['coef']
        return np.c_[np.ones(len(frame)),frame.values]@coef
    if kind=='xgboost':
        import xgboost as xgb
        model=xgb.Booster();model.load_model(path)
        return model.predict(xgb.DMatrix(frame))
    if kind=='lightgbm':
        import lightgbm as lgb
        return lgb.Booster(model_file=path).predict(frame,num_iteration=source.get('best_iteration',-1))
    if kind=='doubleensemble':
        import lightgbm as lgb
        ensemble=json.loads(Path(path).read_text(encoding='utf-8'))
        result=np.zeros(len(frame));weights=ensemble['weights']
        for f,w in zip(ensemble['files'],weights):
            if sha256_file(f['path'])!=f['sha256']:raise SystemExit('ensemble submodel sha mismatch')
            result+=w*lgb.Booster(model_file=f['path']).predict(frame[f['columns']])
        return result/np.sum(weights)
    raise SystemExit('unsupported tabular adapter')
