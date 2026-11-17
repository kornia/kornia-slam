"""Shared EuRoC trajectory evaluator — one scoring path for every system.

Trajectories go in as TUM text (`timestamp tx ty tz qx qy qz qw`), are lifted
into the IMU/body frame that ground truth is expressed in, associated by
timestamp, and aligned with the same rigid Umeyama fit. Scoring both systems
here rather than trusting each one's own reporting is what makes the numbers
comparable.

Metrics, and why each is present:

- ``ate_rmse``       translation RMSE after a *rigid* fit. Deliberately not
                     Sim(3): a scale term would absorb trajectory-length error
                     and hide it.
- ``ate_rot_rmse``   attitude error after removing the constant frame offset a
                     camera-frame estimate carries against body-frame truth.
- ``rpe_*``          relative pose error over a fixed interval. Local error that
                     no global alignment can flatter — and, measured at several
                     intervals, what separates per-frame noise (error that
                     plateaus) from drift (error that grows).
- ``path_ratio``     estimated trajectory length over ground truth. A robust
                     stand-in for scale error: the Sim(3) scale stops meaning
                     anything once a trajectory diverges.
"""
import os
import re
from pathlib import Path

import numpy as np

def _euroc_root():
    """Directory holding the EuRoC sequence folders (MH_01_easy, ...).

    Taken from KORNIA_BENCH_EUROC, else `data/euroc` under the repo root.
    Resolved without touching the filesystem so importing this module never
    fails; `require_euroc()` is what reports a missing dataset.
    """
    env = os.environ.get("KORNIA_BENCH_EUROC")
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parents[4]/"data"/"euroc"


EUROC = _euroc_root()


def require_euroc():
    """Fail with something actionable rather than a bare FileNotFoundError."""
    if not EUROC.is_dir():
        raise SystemExit(
            f"EuRoC dataset not found at {EUROC}\n"
            "Set KORNIA_BENCH_EUROC to the directory holding the sequence "
            "folders, e.g.\n"
            "  export KORNIA_BENCH_EUROC=/path/to/kornia-slam-datasets/euroc")
    return EUROC

SEQS = ["MH_01_easy","MH_02_easy","MH_03_medium","MH_04_difficult","MH_05_difficult",
        "V1_01_easy","V1_02_medium","V1_03_difficult",
        "V2_01_easy","V2_02_medium","V2_03_difficult"]
SHORT = {s: s.split("_")[0]+"_"+s.split("_")[1] for s in SEQS}


def quat_to_R(q):
    """(N,4) xyzw -> (N,3,3)."""
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q.T
    return np.stack([
        np.stack([1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)], -1),
        np.stack([2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)], -1),
        np.stack([2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)], -1),
    ], -2)


def load_tum(path):
    d = np.loadtxt(path)
    d = d[np.argsort(d[:, 0])]
    t = d[:, 0]
    if t[0] > 1e12:          # ORB-SLAM3 EuRoC writes nanoseconds
        t = t / 1e9
    return t, d[:, 1:4], d[:, 4:8]


def load_gt(seq):
    csv = EUROC/seq/"mav0"/"state_groundtruth_estimate0"/"data.csv"
    d = np.loadtxt(csv, delimiter=",", skiprows=1)
    t = d[:, 0]/1e9
    p = d[:, 1:4]
    q = d[:, [5, 6, 7, 4]]   # wxyz -> xyzw
    return t, p, q


def load_T_BC(seq):
    txt = (EUROC/seq/"mav0"/"cam0"/"sensor.yaml").read_text()
    body = txt.split("T_BS:")[1].split("data:")[1].split("]")[0] + "]"
    vals = [float(v) for v in re.findall(r"-?\d+\.?\d*(?:e-?\d+)?", body)]
    return np.array(vals[:16]).reshape(4, 4)


def cam_to_body(p, q, T_BC):
    """Camera-frame poses -> body-frame poses using the rig extrinsic."""
    R_WC = quat_to_R(q)
    R_CB, t_CB = np.linalg.inv(T_BC)[:3, :3], np.linalg.inv(T_BC)[:3, 3]
    p_B = p + np.einsum("nij,j->ni", R_WC, t_CB)
    R_WB = np.einsum("nij,jk->nik", R_WC, R_CB)
    return p_B, R_WB


def rot_angle_deg(R):
    """Geodesic angle of a rotation (or a stack of them), in degrees."""
    tr = np.trace(R, axis1=-2, axis2=-1)
    return np.degrees(np.arccos(np.clip((tr-1)/2, -1.0, 1.0)))


def mean_rotation(Rs):
    """Chordal L2 mean of a stack of rotations (SVD projection)."""
    U, _, Vt = np.linalg.svd(Rs.sum(axis=0))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    return U @ S @ Vt


def rpe(t, R_e, p_e, R_g, p_g, delta_sec=1.0):
    """Relative pose error over a fixed time interval, TUM-style.

    Returns per-pair translation error (m) and rotation error (deg) for the
    relative motion between poses `delta_sec` apart — a local drift measure
    that no global alignment can flatter.
    """
    j = np.searchsorted(t, t + delta_sec)
    ok = j < len(t)
    i = np.where(ok)[0]
    j = j[ok]
    if len(i) == 0:
        return np.array([]), np.array([])
    # Relative motion in each trajectory's own frame.
    dR_e = np.einsum("nji,njk->nik", R_e[i], R_e[j])
    dt_e = np.einsum("nji,nj->ni", R_e[i], p_e[j]-p_e[i])
    dR_g = np.einsum("nji,njk->nik", R_g[i], R_g[j])
    dt_g = np.einsum("nji,nj->ni", R_g[i], p_g[j]-p_g[i])
    return (np.linalg.norm(dt_e-dt_g, axis=1),
            rot_angle_deg(np.einsum("nji,njk->nik", dR_g, dR_e)))


def associate(t_est, t_gt, max_dt=0.02):
    idx = np.searchsorted(t_gt, t_est).clip(1, len(t_gt)-1)
    pick = np.where(np.abs(t_gt[idx-1]-t_est) < np.abs(t_gt[idx]-t_est), idx-1, idx)
    ok = np.abs(t_gt[pick]-t_est) < max_dt
    return np.where(ok)[0], pick[ok]


def umeyama(src, dst, with_scale):
    """Fit dst ~= s*R*src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src-mu_s, dst-mu_d
    C = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(C)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = (D*np.diag(S)).sum()/ (xs**2).sum()*len(src) if with_scale else 1.0
    return s, R, mu_d - s*R@mu_s


def evaluate(traj_path, seq, frame="camera"):
    """Score one run. Returns metrics plus aligned tracks for plotting."""
    t_e, p_e, q_e = load_tum(traj_path)
    if frame == "camera":
        p_e, R_e = cam_to_body(p_e, q_e, load_T_BC(seq))
    else:
        R_e = quat_to_R(q_e)
    t_g, p_g, q_g = load_gt(seq)
    R_g = quat_to_R(q_g)

    ie, ig = associate(t_e, t_g)
    if len(ie) < 20:
        return None
    src, dst, ts = p_e[ie], p_g[ig], t_e[ie]

    s, R, tr = umeyama(src, dst, with_scale=False)
    al = (R @ src.T).T + tr
    err = np.linalg.norm(al-dst, axis=1)

    s_sim, _, _ = umeyama(src, dst, with_scale=True)

    # The Sim(3) scale stops being meaningful once a trajectory diverges: a poorly
    # correlated estimate drives the least-squares scale far from any real size
    # error. Path length is a geometric ratio that survives that, so both are
    # reported and `scale_trustworthy` says whether the Sim(3) figure can be read
    # as a scale error at all.
    path_e = np.linalg.norm(np.diff(src, axis=0), axis=1).sum()
    path_g = np.linalg.norm(np.diff(dst, axis=0), axis=1).sum()
    path_ratio = float(path_e/path_g) if path_g > 0 else float("nan")
    extent = float(np.linalg.norm(dst.max(0)-dst.min(0)))

    # Orientation: rotate the estimate into the ground-truth frame, then take
    # the geodesic angle between the two attitudes.
    Re, Rg = R_e[ie], R_g[ig]
    Re_al = np.einsum("ij,njk->nik", R, Re)

    # A camera-frame estimate carries a constant attitude offset against the
    # body-frame ground truth (for kornia-slam, the rectifying rotation, which
    # T_BS alone does not capture). Solve for that constant offset and report
    # the residual, so attitude is comparable across systems. For a system
    # already in the body frame the offset comes out near identity.
    R_off = mean_rotation(np.einsum("nji,njk->nik", Re_al, Rg))
    ate_rot_raw = rot_angle_deg(np.einsum("nji,njk->nik", Re_al, Rg))
    ate_rot = rot_angle_deg(np.einsum("ji,njk->nik", R_off,
                                      np.einsum("nji,njk->nik", Re_al, Rg)))

    rpe_t, rpe_r = rpe(ts, Re_al, al, Rg, dst, delta_sec=1.0)

    # Coverage: fraction of the GT time span the estimate actually spans.
    gt_span = t_g[-1]-t_g[0]
    cov = (ts[-1]-ts[0])/gt_span if gt_span > 0 else 0.0

    rms = lambda v: float(np.sqrt((v**2).mean())) if len(v) else float("nan")
    ate_now = float(np.sqrt((err**2).mean()))
    trustworthy = bool(extent > 0 and ate_now/extent < 0.15)
    return dict(seq=seq, n=len(ie), ate_rmse=float(np.sqrt((err**2).mean())),
                ate_mean=float(err.mean()), ate_median=float(np.median(err)),
                ate_max=float(err.max()), scale=float(s_sim),
                path_ratio=path_ratio, scale_trustworthy=trustworthy,
                gt_extent=extent,
                ate_rot_rmse=rms(ate_rot), ate_rot_median=float(np.median(ate_rot)),
                ate_rot_raw_rmse=rms(ate_rot_raw),
                attitude_offset_deg=float(rot_angle_deg(R_off)),
                rpe_trans_rmse=rms(rpe_t), rpe_rot_rmse=rms(rpe_r),
                coverage=float(cov), t=ts, err=err, rot_err=ate_rot,
                rpe_t=rpe_t, rpe_r=rpe_r, est=al, gt=dst,
                est_raw=src, t_gt_all=t_g, gt_all=p_g)
