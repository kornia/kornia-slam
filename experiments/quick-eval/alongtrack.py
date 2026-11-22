"""alongtrack.py GT TRAJ [TRAJ ...] [--t0 15]: along-track error and chord scale per run, and the
correlation of the 1 s along-track error series between runs.

Per run (Sim3 over poses from t0 on): SE3 and Sim3 RMS, scale, the 1 s steps of the aligned
trajectory minus the GT steps split into along-track (horizontal GT direction), cross-track and
vertical, their SD, the accumulated along/cross/vertical error over non-overlapping 1 s steps
(RMS about its mean), and the ratio of 100-pose chords (mean, SD). Then the correlation matrix of
the along-track series on a common GT grid (every 20th GT pose from t0), each run Sim3-aligned
from t0. For the 50 ms pseudo-dense GT of R_01/R_03 (K = 20 poses = 1 s).
Labels are the last two directories of each path.
"""
import argparse, os, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("gt"); ap.add_argument("traj", nargs="+"); ap.add_argument("--t0", type=float, default=15.0)
a = ap.parse_args()


def q2R(q):  # xyzw rows
    x, y, z, w = (q / np.linalg.norm(q, axis=1)[:, None]).T
    return np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w), 2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], 1).reshape(-1, 3, 3)


def load(p):
    t = np.loadtxt(p, comments="#", ndmin=2); return t[:, 0].astype(np.int64), t[:, 1:4], t[:, 4:8]


def assoc(te, tg):
    idx = np.searchsorted(tg, te).clip(1, len(tg) - 1)
    idx = np.where(np.abs(tg[idx-1] - te) < np.abs(tg[idx] - te), idx - 1, idx)
    return idx, np.abs(tg[idx] - te) < 10_000_000


def umeyama(A, B, scale):
    ma, mb = A.mean(0), B.mean(0); X, Y = A - ma, B - mb
    U, D, Vt = np.linalg.svd(Y.T @ X / len(A)); S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0: S[2, 2] = -1
    R = U @ S @ Vt; s = (D*np.diag(S)).sum() / X.var(0).sum() if scale else 1.0
    return s, R, mb - s*R@ma


def track_frame(dB):  # horizontal unit direction of each GT step and its left normal
    z = np.array([0, 0, 1.0]); h = dB - (dB @ z)[:, None]*z
    h /= np.maximum(np.linalg.norm(h, axis=1), 1e-9)[:, None]; return h, np.cross(z, h)


tg, pg, _ = load(a.gt); tgs = (tg - tg[0]) / 1e9; K = 20
names = ["/".join(os.path.abspath(p).split(os.sep)[-3:-1]) for p in a.traj]
rms0 = lambda v: np.sqrt(((v - v.mean())**2).mean())
for name, path in zip(names, a.traj):
    te, pe, _ = load(path); idx, ok = assoc(te, tg)
    A, B, t = pe[ok], pg[idx[ok]], tgs[idx[ok]]; m0 = t >= a.t0; A, B = A[m0], B[m0]
    s, R, tt = umeyama(A, B, True); P = (s*(R@A.T)).T + tt
    _, R3, t3 = umeyama(A, B, False); e = P - B; e3 = (R3@A.T).T + t3 - B
    dB = B[K:] - B[:-K]; de = (P[K:] - P[:-K]) - dB; h, c = track_frame(dB)
    al, cr, ve = (de*h).sum(1), (de*c).sum(1), de[:, 2]; sel = np.arange(0, len(de), K)
    k = 100; ce = np.linalg.norm(P[k:] - P[:-k], axis=1); cg = np.linalg.norm(B[k:] - B[:-k], axis=1); mm = cg > 1
    print(f"{name:20s} SE3 {np.sqrt((e3**2).sum(1).mean()):.3f} Sim3 {np.sqrt((e**2).sum(1).mean()):.3f} s {s:.4f} | "
          f"accum along {rms0(np.cumsum(al[sel])):.3f} cross {rms0(np.cumsum(cr[sel])):.3f} vert {rms0(np.cumsum(ve[sel])):.3f} | "
          f"1s-step along mean {al.mean()*1000:+.1f} mm sd {al.std()*1000:.1f}, cross sd {cr.std()*1000:.1f}, vert sd {ve.std()*1000:.1f} | "
          f"chord100 ratio mean {np.mean(ce[mm]/cg[mm]):.3f} sd {np.std(ce[mm]/cg[mm]):.3f}")

g0 = int(np.searchsorted(tgs, a.t0)); grid = np.arange(g0, len(tg) - K, K)
dB = pg[grid+K] - pg[grid]; h, _ = track_frame(dB); AL = []
for path in a.traj:
    te, pe, _ = load(path); idx, ok = assoc(te, tg); full = np.full((len(tg), 3), np.nan)
    A, B = pe[ok], pg[idx[ok]]; m = tgs[idx[ok]] >= a.t0
    s, R, t = umeyama(A[m], B[m], True); full[idx[ok]] = (s*(R@A.T)).T + t
    AL.append((((full[grid+K] - full[grid]) - dB)*h).sum(1))
M = np.array(AL); ok = ~np.isnan(M).any(0)
if len(a.traj) > 1:
    C = np.corrcoef(M[:, ok]); w = max(len(n) for n in names)
    print(f"corr of 1 s along-track error between runs ({ok.sum()} steps)")
    for i, n in enumerate(names): print(f"{n:{w}s} " + " ".join(f"{C[i, j]:5.2f}" for j in range(len(names))))
