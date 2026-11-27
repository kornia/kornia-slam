#!/usr/bin/env python3
"""score.py [--crop-to TEMPLATE] RUNDIR...: one JSON line per run directory written by run.sh.

Reads arm, seq, data, start and rep from RUNDIR/meta.txt. Status is "not done" (no done file),
"invalid" (header guard tripped), "failed" (done with rc != 0), "error" (the directory could not be
read; the reason is in "error") or "ok". For an ok or failed run, scores with evaluate.py:
  traj.txt, traj_smoothed.txt, traj_vio.txt   body poses (tum), with RPE over 1 s and 10 s
  kornia_slam_raw.csv                         camera centres, 5 ms association
  kornia_slam_raw.csv@nearest                 the same with nearest ground truth (the app's printout)
--crop-to TEMPLATE (with {seq}, e.g. results/orbslam3/{seq}/f_run.txt) adds <file>@crop, scored only on
the frames that trajectory covers. A file that cannot be scored gets {"error": reason}.
Exits 1 when any directory has status "error", after printing every line.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import BadInput, evaluate, lines  # noqa: E402

TUM_FILES = ('traj.txt', 'traj_smoothed.txt', 'traj_vio.txt')
KORNIA_FILE = 'kornia_slam_raw.csv'
ALLOWED = TUM_FILES + (KORNIA_FILE,)


def read_meta(d):
    path = os.path.join(d, 'meta.txt')
    if not os.path.exists(path):
        raise BadInput(f'{d}: no meta.txt, so not a run directory run.sh started')
    meta = {}
    for line in lines(path):
        key, sep, value = line.strip().partition('=')
        if sep:
            meta[key] = value
    missing = [k for k in ('arm', 'seq', 'data') if k not in meta]
    if missing:
        raise BadInput(f'{path} lacks {", ".join(missing)}')
    return meta


def crop_stamps(path):
    """Stamps (ns) of a trajectory file; seconds or nanoseconds, autodetected as in load_tum."""
    t = [float(l.replace(',', ' ').split()[0]) for l in lines(path) if l.strip() and not l.startswith('#')]
    if not t:
        raise BadInput(f'{path}: no stamps to crop to')
    scale = 1 if min(t) > 1e12 else 1e9
    return sorted(int(round(x * scale)) for x in t)


def ev(run_dir, data, name, crop=None, **kw):
    if name not in ALLOWED:
        sys.exit(f'score.py: refusing {name}: only {ALLOWED} are scored')
    path = os.path.join(run_dir, name)
    if name == KORNIA_FILE:
        kw.update(fmt='kornia', log=os.path.join(run_dir, 'err.log'))
    else:
        kw.update(fmt='tum', rpe_deltas='1,10')
    try:
        return evaluate(data, path, keep=crop, **kw)
    except BadInput as e:
        return {'error': e.reason}
    except Exception as e:  # a malformed file must not stop the other files and run directories
        return {'error': f'{type(e).__name__}: {e}'}


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
    rc, wall = re.search(r'rc=(\d+)', dn), re.search(r'wall=(\d+)', dn)
    if not (rc and wall):
        raise BadInput(f'{d}/done has no rc= and wall=')
    r['rc'], r['wall'] = int(rc.group(1)), int(wall.group(1))
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
    # A timeout (124) or crash leaves a partial trajectory: scored for diagnosis, kept out of tables.
    r['status'] = 'ok' if r['rc'] == 0 else 'failed'
    return r


def one_or_error(d, crop_to=None):
    try:
        return one(d, crop_to)
    except (BadInput, Exception) as e:  # BadInput is a SystemExit, not an Exception
        reason = e.reason if isinstance(e, BadInput) else f'{type(e).__name__}: {e}'
        return dict(run=os.path.basename(d.rstrip('/')), status='error', error=reason)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--crop-to', default=None, help='trajectory whose stamps limit the @crop scores; {seq} expands')
    ap.add_argument('runs', nargs='+')
    a = ap.parse_args()
    errors = 0
    for d in a.runs:
        r = one_or_error(d, a.crop_to)
        errors += r['status'] == 'error'
        print(json.dumps(r), flush=True)
    if errors:
        sys.exit(f'score.py: {errors} run directories could not be scored (status "error")')


if __name__ == '__main__':
    main()
