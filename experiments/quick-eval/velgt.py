"""velgt.py RUN_DIR GT NFRAMES [--frames A:B]: error of the `[vel]` frame velocity against ground truth (card U).

`v` is the IMU-body velocity in the SLAM world, at the previous frame's stamp `t`. Ground truth is the central
difference of the GT positions at the stamp nearest `t` (10 ms association), and `v` is rotated into the GT frame
by the SE3 Umeyama alignment of RUN_DIR/traj.txt. GT follows the device, not the IMU, so a lever-arm term that does
not grow with age remains: judge the growth with age, not the absolute error.

Age bins count frames since the reference keyframe: the first frame after a keyframe (age_s 0) is bin 1, as in the
card's [predgap] table. Rows: median error (m/s) and n per bin, split by ref_lost and src, then the least-squares
slope of the error against age_s (m/s per s). Frames with a reference that is not the last keyframe (age_s nan)
are counted under "nan".
"""
import re, sys, numpy as np

# Copied from ate.py, which runs at import.
def load(p):
    a = np.loadtxt(p, comments="#", ndmin=2); return a[:, 0].astype(np.int64), a[:, 1:4]
def umeyama_r(A, B):
    ma, mb = A.mean(0), B.mean(0); X, Y = A - ma, B - mb
    U, D, Vt = np.linalg.svd(Y.T @ X / len(A)); S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0: S[2, 2] = -1
    return U @ S @ Vt
def nearest(tg, t):
    idx = np.searchsorted(tg, t).clip(1, len(tg) - 1)
    idx = np.where(np.abs(tg[idx - 1] - t) < np.abs(tg[idx] - t), idx - 1, idx)
    return idx, np.abs(tg[idx] - t) < 10_000_000

run, gt_path, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
lo, hi = 0, n
if "--frames" in sys.argv:
    lo, hi = map(int, sys.argv[sys.argv.index("--frames") + 1].split(":"))
te, pe = load(f"{run}/traj.txt"); tg, pg = load(gt_path)
idx, ok = nearest(tg, te)
R = umeyama_r(pe[ok], pg[idx[ok]])

pat = re.compile(r"\[vel\] frame=(\d+) ref_kf=\d+ ref_lost=(\d) age_s=(\S+) v=\(([^,]+),([^,]+),([^)]+)\) t=(\S+) src=(\w+)")
rows = []
for line in open(f"{run}/stderr.txt"):
    m = pat.match(line)
    if not m or not lo <= int(m.group(1)) < hi: continue
    rows.append((int(m.group(2)), float(m.group(3)), np.array([float(m.group(i)) for i in (4, 5, 6)]),
                 int(round(float(m.group(7)) * 1e9)), m.group(8)))
if not rows: sys.exit(f"no [vel] lines in {run}/stderr.txt")
t = np.array([r[3] for r in rows]); gi, gok = nearest(tg, t)
# The central difference needs a neighbour on both sides: clip the index, then re-check the 10 ms association
# against the clipped sample, so a stamp past either end of the GT is dropped, not matched to a moved sample.
gi = gi.clip(1, len(tg) - 2)
gok = gok & (np.abs(tg[gi] - t) < 10_000_000)
vgt = (pg[gi + 1] - pg[gi - 1]) / ((tg[gi + 1] - tg[gi - 1])[:, None] * 1e-9)

def bin_of(age_s):
    if np.isnan(age_s): return "nan"
    k = int(round(age_s / 0.05)) + 1
    return "1" if k <= 1 else "2" if k == 2 else "3-4" if k <= 4 else "5+"

groups = {}
for (lost, age_s, v, _, src), g, good in zip(rows, vgt, gok):
    if not good: continue
    groups.setdefault((lost, src), []).append((bin_of(age_s), age_s, np.linalg.norm(R @ v - g)))
print(f"== {run.rstrip('/').split('/')[-1]} frames {lo}:{hi}, {len(rows)} [vel] lines, {int((~gok).sum())} without GT within 10 ms")
print("ref_lost src     " + "".join(f"{b:>16}" for b in ("1", "2", "3-4", "5+", "nan")) + "   slope (m/s per s)")
for (lost, src), items in sorted(groups.items()):
    cells = []
    for b in ("1", "2", "3-4", "5+", "nan"):
        e = [x[2] for x in items if x[0] == b]
        cells.append(f"{np.median(e):.3f} (n={len(e)})" if e else "-")
    fit = [(a, e) for _, a, e in items if not np.isnan(a)]
    slope = np.polyfit(*zip(*fit), 1)[0] if len({a for a, _ in fit}) > 1 else float("nan")
    print(f"{lost:>8} {src:<8}" + "".join(f"{c:>16}" for c in cells) + f"   {slope:.2f}")
