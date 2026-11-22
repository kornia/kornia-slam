"""imuedges.py STDERR IMU_CSV CAM0_CSV [--imu-offset-ns N] [--skip-frames 300] [--tail]: interior IMU edges.

From the `[vi_ba] FINAL edge` lines of a debug log, each attached to the `[vi_ba] WINDOW kf=` solve that
follows it: interior (boundary=false) edges of solves at frame >= skip-frames. Prints n, chi2 p50/p90,
the share with Huber weight < 1 and the weight p10, |r_R| p90, and the slope (least squares through
the origin) of |r_R| on |w(t1) - w(t0)|, w the 10 ms mean gyro at the edge ends. A camera-IMU clock
error delta gives r_R ~ (w(t1) - w(t0)) x delta, so the slope reads the residual offset in ms.
Edge ends: the newest edge of a solve ends at the solve frame's cam0 stamp, the older ones chain back
by their dt. --imu-offset-ns is the run's imu_time_offset_ns (added to the IMU stamps, as the app).
--tail splits chi2 > 100 edges from the rest by row, in sigma (LaMAria calibrations: gyro noise
1.745e-4, accel noise 7.845e-4), dt, |w(t1) - w(t0)| and |w|.
"""
import argparse, re, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("log"); ap.add_argument("imu"); ap.add_argument("cam0")
ap.add_argument("--imu-offset-ns", type=int, default=0); ap.add_argument("--skip-frames", type=int, default=300)
ap.add_argument("--tail", action="store_true")
a = ap.parse_args()

num = r"([-+0-9.eE]+)"
fin = re.compile(rf"\[vi_ba\] FINAL edge (\d+)→(\d+)\s+dt={num}s\s+\|r_R\|={num}\s+\|r_v\|={num}\s+\|r_p\|={num}.*chi2={num} hw={num} boundary=(true|false)")
win = re.compile(r"\[vi_ba\] WINDOW kf=(\d+) ")
rows, pend = [], []  # kf0, kf1, dt, rR, rv, rp, chi2, hw, boundary, frame
for line in open(a.log, errors="replace"):
    m = fin.search(line)
    if m: g = m.groups(); pend.append([int(g[0]), int(g[1])] + [float(x) for x in g[2:8]] + [g[8] == "true"]); continue
    m = win.search(line)
    if m: rows += [p + [int(m[1])] for p in pend]; pend = []
F = np.array(rows, dtype=float)
cam = np.loadtxt(a.cam0, delimiter=",", comments="#", usecols=0).astype(np.int64)

t_end = np.zeros(len(F)); i = 0
while i < len(F):
    j = i
    while j + 1 < len(F) and F[j+1, 9] == F[i, 9] and F[j+1, 0] == F[j, 1]: j += 1
    te, back = cam[int(F[i, 9])] / 1e9, 0.0
    for k in range(j, i - 1, -1): t_end[k] = te - back; back += F[k, 2]
    i = j + 1
t_st = t_end - F[:, 2]
m = (F[:, 9] >= a.skip_frames) & (F[:, 8] == 0)
x, ts, te = F[m], t_st[m], t_end[m]

I = np.loadtxt(a.imu, delimiter=",", comments="#")
ti = (I[:, 0].astype(np.int64) + a.imu_offset_ns) / 1e9; gy = I[:, 1:4]


def w_at(t):
    lo, hi = np.searchsorted(ti, t - 0.005), np.searchsorted(ti, t + 0.005)
    return gy[lo:hi].mean(0) if hi > lo else np.full(3, np.nan)


w0 = np.array([w_at(t) for t in ts]); w1 = np.array([w_at(t) for t in te])
dw = np.linalg.norm(w1 - w0, axis=1); wn = 0.5*(np.linalg.norm(w0, axis=1) + np.linalg.norm(w1, axis=1))
rR, chi2, hw = x[:, 3], x[:, 6], x[:, 7]; ok = ~np.isnan(dw)
slope = (dw[ok] @ rR[ok]) / (dw[ok] @ dw[ok])
print(f"interior edges frame>={a.skip_frames}: n {len(x)}  chi2 p50 {np.percentile(chi2, 50):.1f} p90 {np.percentile(chi2, 90):.0f}  "
      f"share hw<1 {np.mean(hw < 1):.3f}  hw p10 {np.percentile(hw, 10):.2f}  |r_R| p90 {np.percentile(rR, 90)*1e3:.2f} mrad  "
      f"slope |r_R|/|dw| {slope*1e3:.2f} ms  (offset {a.imu_offset_ns} ns)")
if a.tail:
    dt = x[:, 2]; sq = np.sqrt(3)
    sR = rR / (1.745e-4*np.sqrt(dt)*sq); sV = x[:, 4] / (7.845e-4*np.sqrt(dt)*sq); sP = x[:, 5] / (7.845e-4*dt**1.5*sq)
    tail = chi2 > 100
    print(f"  chi2>100: {tail.mean()*100:.1f} % of {len(x)}")
    for lab, v in (("r_R sigma", sR), ("r_v sigma", sV), ("r_p sigma", sP), ("dt s", dt), ("|dw| rad/s", dw), ("|w| rad/s", wn), ("hw", hw)):
        print(f"  {lab:11s} tail median {np.nanmedian(v[tail]):9.3f}   rest median {np.nanmedian(v[~tail]):9.3f}")
    print(f"  corr(|r_R|, |dw|) {np.corrcoef(rR[ok], dw[ok])[0, 1]:.2f}  corr(|r_R|, |w|) {np.corrcoef(rR[ok], wn[ok])[0, 1]:.2f}  "
          f"corr(log chi2, dt) {np.corrcoef(np.log(chi2 + 1e-9), dt)[0, 1]:.2f}")
