"""Gate C Holm family: SC1/SC4/SC5/SC6/SC7, m=5; structural SC2 excluded."""
import argparse
import json
import math
from pathlib import Path

from new_gate_b_analysis_v3 import HOLM_KEYS, holm
from new_gate_b_common import Provenance, path_args, preflight, resolve_paths, sha256_file, write_json

GATE_C_KEYS = ('SC1', 'SC4', 'SC5', 'SC6', 'SC7')


def adjust_family(pvalues):
    members = {key: float(pvalues[key]) for key in GATE_C_KEYS}
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in members.values()):
        raise ValueError('Gate C requires five finite p values in [0,1]')
    adjusted = holm(list(members.values()))
    return {key: dict(p_raw=value, p_holm=adj) for (key, value), adj in zip(members.items(), adjusted)}


def self_test():
    values = dict(SC1=.04, SC2=float('nan'), SC4=.001, SC5=.01, SC6=.03, SC7=.2)
    result = adjust_family(values)
    assert list(result) == list(GATE_C_KEYS) and 'SC2' not in result
    ordered = sorted(result.values(), key=lambda v: v['p_raw'])
    adjusted = [v['p_holm'] for v in ordered]
    assert adjusted == sorted(adjusted) and all(0 <= p <= 1 for p in adjusted)
    assert adjust_family(dict.fromkeys(GATE_C_KEYS, 1))['SC7']['p_holm'] == 1
    print('GATE C SELF-TEST PASS')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    path_args(parser)
    parser.add_argument('--analysis', type=Path)
    parser.add_argument('--sc4-p', type=float)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    resolve_paths(args, ('root', 'spec'))
    if args.analysis is None or args.sc4_p is None:
        parser.error('--analysis and --sc4-p required')
    args.out = args.out or args.root / 'runs' / 'new_gate_c_holm.json'
    preflight([args.out])
    provenance = Provenance(__file__, args)
    analysis = json.loads(args.analysis.read_text(encoding='utf-8'))
    if analysis.get('analysis_sha256') != sha256_file(Path(__file__).with_name('new_gate_b_analysis_v3.py')):
        raise SystemExit('Gate C requires hash-matched v3 analysis')
    if analysis.get('holm_family_gate_b') != list(HOLM_KEYS):
        raise SystemExit('unexpected Gate B family')
    if not analysis.get('SC5_sidedness', '').startswith('one-sided P*(rho*<=0)'):
        raise SystemExit('Gate C requires the confirmed v3 SC5 sidedness')
    if analysis.get('SC6_definition') != 'g_fee - c_fee (annualized pp; >0 = C0 pays lower fees)':
        raise SystemExit('unexpected SC6 definition')
    pvalues = {key: analysis['holm'][key]['p_raw'] for key in HOLM_KEYS}
    pvalues['SC4'] = args.sc4_p
    write_json(args.out, dict(holm_family_gate_c=list(GATE_C_KEYS), m=5,
                             holm=adjust_family(pvalues), SC4_p_supplied=args.sc4_p,
                             analysis_sha256=analysis['analysis_sha256'],
                             meta=provenance.finish(dict(analysis_input_sha256=sha256_file(args.analysis)))))
    print('GATE C HOLM DONE')


if __name__ == '__main__':
    main()
