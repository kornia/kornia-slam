"""clockscan.py SEQ_ASL_DIR GT.txt [--factory F.json] [--apply-ns N] [--max-dt-s 0.31]: camera-IMU clock offset.

Scans "IMU time = cam time + d" for d from -2 to 10 ms in 0.25 ms steps. For each d, the gyro of
SEQ_ASL_DIR/aria/imu0/data.csv is integrated on SO(3) and compared with the relative rotation of
consecutive ground-truth poses (20 ms < dt <= max-dt, from 5 s after the first pose) after a fixed
rotation (Kabsch; the pseudo-dense GT of R_11 and the challenge sequences is in a body frame about
105 deg from the IMU) and a constant gyro bias fit. A parabola on the squared RMS gives the minimum,
for the whole sequence and its thirds.

Two integration modes: `mid` treats samples as instantaneous (linear rate between them; ORB-SLAM3's
midpoint), `hold` holds each sample over the interval that ends at its stamp, as the app's default
`SampleInterpolation::Hold` (kornia-sensors/src/imu.rs:184-221), which hides about half a sample
period. The value to set is `imu_time_offset_ns = -round(d_hold x 1e6)`: the app adds it to every IMU
stamp (apps/kornia-slam-app/src/datasets/lamaria.rs:127-131). --apply-ns N shifts the IMU stamps by N
first, so a correct offset scans to d_hold ~ 0. --factory prints d_mid minus the record's imu-right
TimeOffsetSec_Device_Gyro.
"""
import argparse, json, os, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("seq"); ap.add_argument("gt")
ap.add_argument("--factory"); ap.add_argument("--apply-ns", type=int, default=0)
ap.add_argument("--max-dt-s", type=float, default=0.31)
a = ap.parse_args()


def q2R(q):  # xyzw rows
    x, y, z, w = (q / np.linalg.norm(q, axis=1)[:, None]).T
    return np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w), 2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], 1).reshape(-1, 3, 3)


def exp(v):  # rows of rotation vectors -> rotations
    th = np.linalg.norm(v, axis=1); K = np.zeros((len(v), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -v[:, 2], v[:, 1], -v[:, 0]
    K[:, 1, 0], K[:, 2, 0], K[:, 2, 1] = v[:, 2], -v[:, 1], v[:, 0]
    s = np.where(th > 1e-9, np.sin(th) / np.maximum(th, 1e-12), 1.0)
    c = np.where(th > 1e-9, (1 - np.cos(th)) / np.maximum(th, 1e-12)**2, 0.5)
    return np.eye(3) + s[:, None, None]*K + c[:, None, None]*K@K


def log(R):
    c = np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1); th = np.arccos(c)
    v = np.stack([R[:, 2, 1]-R[:, 1, 2], R[:, 0, 2]-R[:, 2, 0], R[:, 1, 0]-R[:, 0, 1]], 1)
    s = np.sin(th); return v * np.where(s > 1e-9, th / (2*np.maximum(s, 1e-12)), 0.5)[:, None]


def kabsch(P, Q):  # R with R p ~ q
    U, _, Vt = np.linalg.svd(Q.T @ P); S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0: S[2, 2] = -1
    return U @ S @ Vt


I = np.loadtxt(os.path.join(a.seq, "aria", "imu0", "data.csv"), delimiter=",", comments="#")
ti = I[:, 0].astype(np.int64) + a.apply_ns; gy = I[:, 1:4]; dti = np.diff(ti) / 1e9
G = np.loadtxt(a.gt, comments="#", ndmin=2); tg = G[:, 0].astype(np.int64); Rg = q2R(G[:, 4:8])


def cumulate(steps):  # orientation at every IMU stamp, re-orthonormalised every 1000 products
    C = np.empty((len(steps) + 1, 3, 3)); C[0] = np.eye(3)
    for k, E in enumerate(steps):
        C[k+1] = C[k] @ E
        if k % 1000 == 999: U, _, Vt = np.linalg.svd(C[k+1]); C[k+1] = U @ Vt
    return C


# hold: sample i over (t_{i-1}, t_i]; mid: rate linear between samples, step uses the mean.
C = {"hold": cumulate(exp(gy[1:] * dti[:, None])), "mid": cumulate(exp(0.5*(gy[1:] + gy[:-1]) * dti[:, None]))}


def orient(mode, t):  # IMU orientation at IMU times t (ns), inside the IMU span
    i = np.searchsorted(ti, t).clip(1, len(ti) - 1); h = (t - ti[i-1]) / 1e9
    if mode == "hold": part = gy[i] * h[:, None]
    else:
        f = (h / dti[i-1])[:, None]; part = (gy[i-1] + 0.5*f*(gy[i] - gy[i-1])) * h[:, None]
    return C[mode][i-1] @ exp(part)


j = np.arange(len(tg) - 1); dt = (tg[1:] - tg[:-1]) / 1e9
j = j[(dt > 0.02) & (dt <= a.max_dt_s) & (tg[:-1] >= tg[0] + 5_000_000_000)
      & (tg[:-1] > ti[0] + 20_000_000) & (tg[1:] < ti[-1] - 20_000_000)]
vg = log(np.einsum('nji,njk->nik', Rg[j], Rg[j+1])); dtp = dt[j]
third = np.minimum(3 * np.arange(len(j)) // len(j), 2)
ds = np.arange(-2.0, 10.0001, 0.25)


def resid(mode, d):
    Ra, Rb = orient(mode, tg[j] + int(round(d*1e6))), orient(mode, tg[j+1] + int(round(d*1e6)))
    vi = log(np.einsum('nji,njk->nik', Ra, Rb)); b = np.zeros(3)
    for _ in range(3):  # vg ~ X (vi - b dt)
        X = kabsch(vi - b*dtp[:, None], vg)
        b = ((vi - vg @ X) * dtp[:, None]).sum(0) / (dtp**2).sum()
    return np.linalg.norm(vg - (vi - b*dtp[:, None]) @ X.T, axis=1)**2


def minimum(ms):  # parabola through the lowest scan point and its neighbours
    k = min(max(int(np.argmin(ms)), 1), len(ds) - 2)
    c = np.polyfit(ds[k-1:k+2], ms[k-1:k+2], 2)
    return -c[1] / (2*c[0]) if c[0] > 0 else ds[k]


out = {}
for mode in ("mid", "hold"):
    R = np.array([resid(mode, d) for d in ds])  # scan x pairs
    whole = R.mean(1); thirds = [minimum(R[:, third == t].mean(1)) for t in range(3)]
    dmin = minimum(whole); out[mode] = dmin
    print(f"{mode:4s} d={dmin:+.2f} ms (thirds {thirds[0]:.2f} / {thirds[1]:.2f} / {thirds[2]:.2f})  "
          f"rms {np.sqrt(whole.min())*1e3:.2f} mrad at {ds[np.argmin(whole)]:+.2f} ms, {np.sqrt(whole[np.argmin(np.abs(ds))])*1e3:.2f} at 0  "
          f"(pairs {len(j)}, dt p50 {np.median(dtp)*1e3:.0f} ms, apply {a.apply_ns} ns)")
line = f"d_mid {out['mid']:+.2f} ms  d_hold {out['hold']:+.2f} ms  imu_time_offset_ns = {a.apply_ns - int(round(out['hold']*1e6))}"
if a.factory:
    imu = [i for i in json.load(open(a.factory))["ImuCalibrations"] if i["Label"] == "imu-right"][0]
    fg = imu["Gyroscope"]["TimeOffsetSec_Device_Gyro"]*1e3; fa = imu["Accelerometer"]["TimeOffsetSec_Device_Accel"]*1e3
    line += f"  factory gyro {fg:.3f} accel {fa:.3f} ms  d_mid - factory gyro {out['mid'] + a.apply_ns/1e6 - fg:+.2f} ms"
print(line)
