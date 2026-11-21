"""ate.py EST GT NFRAMES: SE3/Sim3 Umeyama ATE (m) of `t x y z qx qy qz qw` rows (t in ns) vs GT, 10 ms association."""
import sys, numpy as np
def load(p):
    a = np.loadtxt(p, comments="#", ndmin=2); return a[:, 0].astype(np.int64), a[:, 1:4]
te, pe = load(sys.argv[1]); tg, pg = load(sys.argv[2]); n = int(sys.argv[3])
idx = np.searchsorted(tg, te).clip(1, len(tg) - 1)
idx = np.where(np.abs(tg[idx - 1] - te) < np.abs(tg[idx] - te), idx - 1, idx)
ok = np.abs(tg[idx] - te) < 10_000_000
A, B = pe[ok], pg[idx[ok]]
def umeyama(A, B, scale):
    ma, mb = A.mean(0), B.mean(0); X, Y = A - ma, B - mb
    U, D, Vt = np.linalg.svd(Y.T @ X / len(A)); S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0: S[2, 2] = -1
    R = U @ S @ Vt; s = (D * np.diag(S)).sum() / X.var(0).sum() if scale else 1.0
    return np.sqrt(((s * (R @ X.T).T + mb - B) ** 2).sum(1).mean()), s
se3, _ = umeyama(A, B, False); sim3, s = umeyama(A, B, True)
path = np.linalg.norm(np.diff(B, axis=0), axis=1).sum()
print(f"poses={ok.sum()}/{n} tracked={ok.sum()/n:.1%} path={path:.1f}m ate_se3={se3:.3f}m ate_sim3={sim3:.3f}m scale={s:.3f}")
