#!/usr/bin/env python3
"""new_compare_ridge_daily_v1.py -- frozen comparison of ridge 2021-22 rule-replay daily.csv files (DL-28 r9).

Usage: new_compare_ridge_daily_v1.py --pair OLD.csv NEW.csv [--pair ...] [--record out.json]
stdout: exactly ONE line, "True" or "False".
  True  = every pair is byte-identical (sha256 equal), OR every pair has identical header,
          identical row count, identical non-numeric cells, identical NaN/empty positions, and
          max|delta| over numeric cells < TOL.
  False = anything else.
Never prints any value, difference, column name or row. Exit codes: 0 compared (True/False printed),
2 usage, 3 refused (a date >= 2023-01-01 found; prints "False"), 4 internal error (prints "False").
--record writes only: script sha, TOL, per-pair file shas, sha_equal flag, row counts, verdict.
TOL is fixed by DL-28 and cannot be overridden.
"""
import argparse, csv, hashlib, json, math, re, sys
from pathlib import Path

TOL = 1e-6
DATE_RE = re.compile(r'^(\d{4})-\d{2}-\d{2}')
EVAL_START_YEAR = 2023


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def num(x):
    try:
        v = float(x)
    except ValueError:
        return None
    return v


class Refuse(Exception):
    pass


def rows(p):
    with open(p, newline='', encoding='utf-8') as f:
        r = list(csv.reader(f))
    for row in r[1:]:
        for c in row:
            m = DATE_RE.match(c.strip())
            if m and int(m.group(1)) >= EVAL_START_YEAR:
                raise Refuse()
    return r


def same_content(a, b):
    ra, rb = rows(a), rows(b)
    if not ra or not rb or ra[0] != rb[0] or len(ra) != len(rb):
        return False, len(ra), len(rb)
    worst = 0.0
    for x, y in zip(ra[1:], rb[1:]):
        if len(x) != len(y):
            return False, len(ra), len(rb)
        for cx, cy in zip(x, y):
            vx, vy = num(cx), num(cy)
            if vx is None or vy is None:
                if (vx is None) != (vy is None) or cx != cy:
                    return False, len(ra), len(rb)
                continue
            if math.isnan(vx) or math.isnan(vy):
                if not (math.isnan(vx) and math.isnan(vy)):
                    return False, len(ra), len(rb)
                continue
            if math.isinf(vx) or math.isinf(vy):
                if vx != vy:
                    return False, len(ra), len(rb)
                continue
            worst = max(worst, abs(vx - vy))
    return worst < TOL, len(ra), len(rb)


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument('--pair', nargs=2, action='append', required=True, metavar=('OLD', 'NEW'))
    ap.add_argument('--record')
    a = ap.parse_args()
    rec = dict(script='new_compare_ridge_daily_v1.py', script_sha256=sha(__file__), tol=TOL, pairs=[])
    verdict = True
    for old, new in a.pair:
        so, sn = sha(old), sha(new)
        e = dict(old=old, new=new, old_sha256=so, new_sha256=sn, sha_equal=so == sn)
        if so == sn:
            rows(old)  # still enforce the date guard
            e['within_tol'] = True
        else:
            ok, na, nb = same_content(old, new)
            e.update(within_tol=ok, rows_old=na, rows_new=nb)
            verdict &= ok
        rec['pairs'].append(e)
    rec['verdict'] = verdict
    if a.record:
        p = Path(a.record)
        with p.open('x') as f:
            json.dump(rec, f, indent=1)
    print('True' if verdict else 'False')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Refuse:
        print('False'); sys.exit(3)
    except Exception as ex:  # never echo values
        print('False'); print(type(ex).__name__, file=sys.stderr); sys.exit(4)
