"""jumps.py RUN_DIR SEQ_DIR [SESSION] [--body C.txt]: the single-frame pose jumps before each primary loss and the
speed error of the coast that follows (card AC-2b, G48).

Reuses lossev.py's GT loader, frame classes, episodes and local rotation fit. A primary loss is a coasted episode
(lossev episodes) that starts in a segment at least 3 s old (lossev's primary age) and lasts at least 0.5 s, so its
first 0.5 s bin exists; `give_up` marks an episode a session segment starts in. Times are from the run's first
frame (a window's start_frame).

Per primary loss it prints:
  steps:  every single-frame camera-centre step over 0.5 m from 5 s before the onset to the end of the coast, with the
          frame class, the step (m), the rotation step (deg) and its time from the onset (negative: before it).
  speed:  SLAM and GT speeds over 0.25 s (5 frames), SLAM displacement taken into GT by the local rotation of the
          10 s before the loss; the median speed error over the first 0.5 s, and the slope of the speed error over
          the coast (m/s^2, least squares), with the SLAM speed at the start and the end.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lossev import Run, angle  # noqa: E402

STEP_M = 0.5
BEFORE_SEC = 5.0
MIN_AGE_SEC = 3.0
MIN_COAST_FRAMES = 10
SPEED_FRAMES = 5


def steps(r, lo, hi, onset):
    out = []
    for f in range(max(lo, r.lo + 1), hi):
        a, b = r.ts[f - 1], r.ts[f]
        if a not in r.traj or b not in r.traj: continue
        (pa, Ra), (pb, Rb) = r.traj[a], r.traj[b]
        d = float(np.linalg.norm(pb - pa))
        if d > STEP_M:
            out.append((f, r.cls.get(f, "?"), r.tsec[f] - r.tsec[r.lo], r.tsec[f] - r.tsec[onset], d, angle(Ra.T @ Rb)))
    return out


def speeds(r, i, j, Rm):
    rows = []
    for f in range(i, j):
        g = f - SPEED_FRAMES
        if r.ts[f] not in r.traj or r.ts[g] not in r.traj: continue
        p1, p0 = r.gt.pos(r.tsec[f]), r.gt.pos(r.tsec[g])
        if p1 is None or p0 is None: continue
        dt = r.tsec[f] - r.tsec[g]
        vs = float(np.linalg.norm(Rm @ (r.traj[r.ts[f]][0] - r.traj[r.ts[g]][0]))) / dt
        vg = float(np.linalg.norm(p1 - p0)) / dt
        rows.append((r.tsec[f] - r.tsec[i], vs, vg))
    return np.array(rows).reshape(-1, 3)


def report(r):
    print(f"body: GT-to-trajectory body rotation C {angle(r.C):.1f} deg")
    n = 0
    for i, j in r.episodes():
        k = np.searchsorted(r.seg_t, r.tsec[i]) - 1
        if k < 0 or r.tsec[i] - r.seg_t[k] < MIN_AGE_SEC or j - i < MIN_COAST_FRAMES: continue
        n += 1
        give_up = bool(((r.seg_t > r.tsec[i]) & (r.seg_t <= r.tsec[min(j, len(r.tsec) - 1)])).any())
        print(f"loss {n}: onset frame {i} t {r.tsec[i] - r.tsec[r.lo]:.2f} s, coast {r.tsec[min(j, len(r.tsec) - 1)] - r.tsec[i]:.2f} s, "
              f"segment age {r.tsec[i] - r.seg_t[k]:.1f} s, give_up {int(give_up)}")
        lo = int(np.searchsorted(r.tsec, r.tsec[i] - BEFORE_SEC))
        print("  steps: frame cls t_s from_onset_s step_m step_rot_deg")
        for f, c, t, dt, d, rot in steps(r, lo, j, i):
            print(f"    {f} {c} {t:.2f} {dt:+.2f} {d:.2f} {rot:.1f}")
        Rm = r.local_rotation(i - 1, r.seg_t[k])
        if Rm is None or not r.gt.ok:
            print("  speed: no GT or no local rotation fit")
            continue
        s = speeds(r, i, j, Rm)
        if len(s) < 2:
            print("  speed: too few GT frames")
            continue
        err = s[:, 1] - s[:, 2]
        first = err[s[:, 0] < 0.5]
        slope = np.polyfit(s[:, 0], err, 1)[0]
        print(f"  speed: first 0.5 s error med {np.median(first) if len(first) else np.nan:.2f} m/s "
              f"(slam {np.median(s[s[:, 0] < 0.5, 1]) if len(first) else np.nan:.2f}, "
              f"gt {np.median(s[s[:, 0] < 0.5, 2]) if len(first) else np.nan:.2f}); "
              f"slope {slope:+.2f} m/s^2 over {s[-1, 0]:.2f} s; slam speed {s[0, 1]:.2f} -> {s[-1, 1]:.2f} m/s")
    print(f"primary losses: {n}")


if __name__ == "__main__":
    args = sys.argv[1:]
    body = None
    if "--body" in args:
        i = args.index("--body"); body = np.loadtxt(args[i + 1]); del args[i:i + 2]
    if len(args) < 2:
        print(__doc__, file=sys.stderr); sys.exit(2)
    run = Run(args[0], args[1], args[2] if len(args) > 2 else f"{args[0]}/session.cbor")
    run.C = body if body is not None else (run.body_alignment()[0] if run.gt.ok else np.eye(3))
    report(run)
