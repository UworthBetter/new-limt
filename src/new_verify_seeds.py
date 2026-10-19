"""T1: synthetic-only model-seed verification; failure is persisted and exits 1."""
import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from new_gate_b_common import Provenance, path_args, preflight, resolve_paths, sha256_file, write_json
from new_run_benchmarks_v2 import TS_MODELS, construct_model
from new_bench_dataset import TreeDataset


def state_hash(model):
    import torch
    nets = [(k, v) for k, v in vars(model).items() if isinstance(v, torch.nn.Module)]
    if len(nets) != 1:
        raise ValueError('expected exactly one torch model')
    h = hashlib.sha256()
    for key, tensor in sorted(nets[0][1].state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        h.update(key.encode())
        h.update(str(value.dtype).encode())
        h.update(str(value.shape).encode())
        h.update(value.tobytes())
    return h.hexdigest()


def xgb_check(entry):
    rng = np.random.default_rng(21)
    x = rng.normal(size=(1600, 20))
    y = x[:, 0] + .5 * x[:, 1] + rng.normal(size=1600)
    dates = pd.bdate_range('2017-01-02', periods=40)
    index = pd.MultiIndex.from_product([dates, [f'S{i:03d}' for i in range(40)]],
                                      names=['datetime', 'instrument'])
    columns = [f'f{i}' for i in range(20)]
    frame = pd.DataFrame(np.c_[x, y], index=index, columns=columns + ['LABEL0'])
    dataset = TreeDataset(frame, {'train': (dates[0], dates[29]), 'valid': (dates[30], dates[-1]),
                                  'test': (dates[30], dates[-1])}, {}, columns)
    predictions, params, threads, rounds = [], [], [], []
    for seed in (1, 1, 2):
        model, kwargs = construct_model(entry, seed)
        assert model._params['seed'] == seed == kwargs['seed']
        model.fit(dataset, evals_result={}, verbose_eval=False)
        predictions.append(model.predict(dataset).to_numpy())
        params.append(dict(model._params))
        config = json.loads(model.model.save_config())
        threads.append(int(config['learner']['generic_param']['nthread']))
        rounds.append(model.model.num_boosted_rounds())
    same = np.array_equal(predictions[0], predictions[1])
    different = not np.array_equal(predictions[0], predictions[2])
    return dict(passed=bool(same and different), same_seed_identical=bool(same),
                different_seed_different=bool(different), params=params,
                prediction_sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in predictions],
                actual_rounds=rounds, actual_nthread=threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    path_args(parser)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    resolve_paths(args, ('root', 'spec'))
    args.out = args.out or args.root / 'protocol' / 'checks' / f'new_seed_verification_{datetime.now().date()}.json'
    preflight([args.out])
    spec = json.loads(args.spec.read_text(encoding='utf-8'))
    provenance = Provenance(__file__, args, synthetic=True)
    checks = {}
    try:
        checks['XGBoost'] = xgb_check(spec['models']['XGBoost'])
    except Exception as error:
        checks['XGBoost'] = dict(passed=False, error=repr(error))
    for name in sorted(TS_MODELS):
        try:
            hashes, kwargs_used = [], []
            for seed in (1, 1, 2):
                model, kwargs = construct_model(spec['models'][name], seed)
                assert kwargs['seed'] == seed == model.seed
                hashes.append(state_hash(model))
                kwargs_used.append(kwargs)
                del model
            checks[name] = dict(passed=hashes[0] == hashes[1] and hashes[0] != hashes[2],
                                state_dict_sha256=hashes, kwargs=kwargs_used)
        except Exception as error:
            checks[name] = dict(passed=False, error=repr(error))
    passed = all(c['passed'] for c in checks.values())
    meta = provenance.finish(dict(xgb_actual_nthread=checks['XGBoost'].get('actual_nthread')))
    write_json(args.out, dict(passed=passed, checks=checks, meta=meta,
                             verifier_sha256=sha256_file(__file__),
                             runner_sha256=sha256_file(Path(__file__).with_name('new_run_benchmarks_v2.py'))))
    print('T1 ' + ('PASS' if passed else 'FAIL: ' + ', '.join(k for k, v in checks.items() if not v['passed'])))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
