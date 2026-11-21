"""steps.py EST GT [FROM TO]: step-length ratio ours/GT, per 25-pose bin and over poses FROM..TO (default 50..250).

Both files hold `t x y z qx qy qz qw` rows (t in ns); poses are associated within 10 ms.
A ratio of 1.0 means the estimated trajectory has the right scale locally; card J/K use it
to see a map at the wrong scale before the inertial initialisation fixes it.
"""
import sys, numpy as np


def load(p):
    a = np.loadtxt(p, comments="#", ndmin=2)
    return a[:, 0].astype(np.int64), a[:, 1:4]


te, pe = load(sys.argv[1])
tg, pg = load(sys.argv[2])
lo, hi = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (50, 250)
idx = np.searchsorted(tg, te).clip(1, len(tg) - 1)
idx = np.where(np.abs(tg[idx - 1] - te) < np.abs(tg[idx] - te), idx - 1, idx)
ok = np.abs(tg[idx] - te) < 10_000_000
pe, pg = pe[ok], pg[idx[ok]]
se = np.linalg.norm(np.diff(pe, axis=0), axis=1)
sg = np.linalg.norm(np.diff(pg, axis=0), axis=1)
bins = []
for b in range(0, len(se), 25):
    g = sg[b : b + 25].sum()
    bins.append(f"{b}:{se[b:b + 25].sum() / g:.2f}" if g > 1e-3 else f"{b}:-")
g = sg[lo:hi].sum()
window = se[lo:hi].sum() / g if g > 1e-3 else float("nan")
print(f"ratio[{lo},{hi})={window:.3f}  bins " + " ".join(bins[:16]))
