"""oracle.py SEQ SPLITS OUT TRAJ [DATA]: CP oracle (card AC scoring item 2).

Rewrites TRAJ with each piece between the split times (s from the first camera frame, comma-separated; "" for one
piece) Sim3-aligned to the pseudo-dense GT: the score a perfect link of the pieces would reach. Splits at the
restarts give "restarts linked"; splits every 60 s give "60 s pieces". Poses outside GT coverage keep their piece's
transform. Score OUT with evalcp.sh. MODE=se3|scale|rot restricts the per-piece transform; CW=1 for T_cw rotations.
Copied from diagV/drift/oracle.py (2026-11-22) with rot.py's helpers inlined.
"""
import os, sys, numpy as np


def q2R(q):
    x, y, z, w = (np.asarray(q, float) / np.linalg.norm(q))
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R2q(R):
    w = np.sqrt(max(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2; x = np.sqrt(max(0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = np.sqrt(max(0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2; z = np.sqrt(max(0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    x = np.copysign(x, R[2, 1] - R[1, 2]); y = np.copysign(y, R[0, 2] - R[2, 0]); z = np.copysign(z, R[1, 0] - R[0, 1])
    return np.array([x, y, z, w])


def assoc(te, tg):
    idx = np.searchsorted(tg, te).clip(1, len(tg) - 1)
    idx = np.where(np.abs(tg[idx - 1] - te) < np.abs(tg[idx] - te), idx - 1, idx)
    return idx, np.abs(tg[idx] - te) < 10_000_000


def umeyama(A, B, scale):
    ma, mb = A.mean(0), B.mean(0); X, Y = A - ma, B - mb
    U, D, Vt = np.linalg.svd(Y.T @ X / len(A)); S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0: S[2, 2] = -1
    R = U @ S @ Vt; s = (D * np.diag(S)).sum() / X.var(0).sum() if scale else 1.0
    return s, R, mb - s * R @ ma


SEQ, SPL, OUT, TRAJ = sys.argv[1], [float(x) for x in sys.argv[2].split(",") if x], sys.argv[3], sys.argv[4]
D = sys.argv[5] if len(sys.argv) > 5 else "/home/edgar/dev/.flux-connect-scratch/bench/lamaria/data"
a = np.loadtxt(TRAJ); te = a[:, 0].astype(np.int64); pe = a[:, 1:4]; qe = a[:, 4:8]
g = np.loadtxt(f"{D}/{SEQ}/ground_truth/pseudo_dense/{SEQ}.txt"); tg = g[:, 0].astype(np.int64); pg = g[:, 1:4]
cam = np.loadtxt(f"{D}/{SEQ}/asl/{SEQ}/aria/cam0/data.csv", delimiter=",", comments="#", usecols=0, dtype=np.int64)
t = (te - cam[0]) / 1e9; idx, ok = assoc(te, tg)
edges = [-1e9] + SPL + [1e9]; out = []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (t >= lo) & (t < hi); mo = m & ok
    if mo.sum() < 5: print(f"piece [{lo},{hi}) too few GT", mo.sum()); continue
    s, R, tr = umeyama(pe[mo], pg[idx[mo]], True)
    if os.environ.get("MODE") == "scale": R = np.eye(3); tr = np.zeros(3)
    if os.environ.get("MODE") == "se3": s, R, tr = umeyama(pe[mo], pg[idx[mo]], False)
    if os.environ.get("MODE") == "rot": s = 1.0; tr = np.zeros(3)
    for i in np.nonzero(m)[0]:
        p = s * R @ pe[i] + tr; Rq = (q2R(qe[i]) @ R.T) if os.environ.get("CW") else (R @ q2R(qe[i]))
        out.append((te[i], *p, *R2q(Rq)))
    print(f"piece [{lo:.0f},{hi:.0f}) n {m.sum()} scale {s:.3f}")
out.sort(key=lambda r: r[0])
with open(OUT, "w") as f:
    for r in out: f.write(f"{int(r[0])} " + " ".join(f"{v:.9f}" for v in r[1:]) + "\n")
