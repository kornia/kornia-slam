#!/usr/bin/env python3
"""score.py [--crop-to TEMPLATE] RUNDIR...: one JSON line per run directory written by run.sh.

Reads arm, seq, data, start and rep from RUNDIR/meta.txt. Status is "not done" (no done file),
"invalid" (header guard tripped) or "ok". For an ok run, scores with evaluate.py:
  traj.txt, traj_smoothed.txt, traj_vio.txt   body poses (tum), with RPE over 1 s and 10 s
  kornia_slam_raw.csv                         camera centres, 5 ms association
  kornia_slam_raw.csv@nearest                 the same with nearest ground truth (the app's printout)
--crop-to TEMPLATE (with {seq}, e.g. results/orbslam3/{seq}/f_run.txt) adds <file>@crop, scored only on
the frames that trajectory covers.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import evaluate, lines  # noqa: E402

TUM_FILES = ('traj.txt', 'traj_smoothed.txt', 'traj_vio.txt')
KORNIA_FILE = 'kornia_slam_raw.csv'
ALLOWED = TUM_FILES + (KORNIA_FILE,)


def read_meta(d):
    meta = {}
    for line in lines(os.path.join(d, 'meta.txt')):
        key, sep, value = line.strip().partition('=')
        if sep:
            meta[key] = value
    return meta


def crop_stamps(path):
    return sorted(int(float(l.split()[0])) for l in lines(path) if l.strip() and not l.startswith('#'))


def ev(run_dir, data, name, crop=None, **kw):
    if name not in ALLOWED:
        sys.exit(f'score.py: refusing {name}: only {ALLOWED} are scored')
    path = os.path.join(run_dir, name)
    if name == KORNIA_FILE:
        kw.update(fmt='kornia', log=os.path.join(run_dir, 'err.log'))
    else:
        kw.update(fmt='tum', rpe_deltas='1,10')
    return evaluate(data, path, keep=crop, **kw)


def summary(log):
    s = {}
    for m in re.finditer(r'^\[summary\] .*', log, re.M):
        s.update({k: float(v) for k, v in re.findall(r'(\w+)=(-?[\d.]+)', m.group(0))})
    # wall can include a laptop suspend; the per-frame time sum does not
    s['frame_ms_sum'] = sum(float(x) for x in re.findall(r'^\[ *\d+\] \w+.*?\s([\d.]+)ms fe=', log, re.M))
    s['loops_accepted'] = len(re.findall(r'\[loop-closure\] accepted', log))
    s['loops_applied'] = len(re.findall(r'\[loop-closure\] accepted.*applied=true', log))
    s['pgo_failed'] = len(re.findall(r'^\[pgo\] failed', log, re.M))
    return s


def one(d, crop_to=None):
    meta = read_meta(d)
    r = dict(run=os.path.basename(d.rstrip('/')), arm=meta.get('arm'), seq=meta.get('seq'),
             start=int(meta.get('start', 0)), rep=int(meta.get('rep', 0)))
    if not os.path.exists(os.path.join(d, 'done')):
        r['status'] = 'invalid' if os.path.exists(os.path.join(d, 'invalid')) else 'not done'
        return r
    dn = '\n'.join(lines(os.path.join(d, 'done')))
    r['rc'] = int(re.search(r'rc=(\d+)', dn).group(1))
    r['wall'] = int(re.search(r'wall=(\d+)', dn).group(1))
    r['meta'] = meta
    r['s'] = summary('\n'.join(lines(os.path.join(d, 'err.log'))))
    crop = crop_stamps(crop_to.format(seq=meta['seq'])) if crop_to else None
    for f in ALLOWED:
        p = os.path.join(d, f)
        if not (os.path.exists(p) and os.path.getsize(p)):
            continue
        r[f] = ev(d, meta['data'], f)
        if f == KORNIA_FILE:
            r[f + '@nearest'] = ev(d, meta['data'], f, nearest_gt=True)
        if crop is not None:
            r[f + '@crop'] = ev(d, meta['data'], f, crop=crop)
    r['status'] = 'ok'
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--crop-to', default=None, help='trajectory whose stamps limit the @crop scores; {seq} expands')
    ap.add_argument('runs', nargs='+')
    a = ap.parse_args()
    for d in a.runs:
        print(json.dumps(one(d, a.crop_to)), flush=True)


if __name__ == '__main__':
    main()
