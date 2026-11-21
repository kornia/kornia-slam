"""biaswalk.py STDERR CALIB_JSON [SKIP]: bias random-walk residuals and bias wander from a debug log.

From `[vi_ba] FINAL` lines: p50/p90 of z = |r| / (sigma * sqrt(dt) * sqrt(3)) for r_bg and r_ba,
boundary and interior edges apart. sigma is the calibration's bias random walk, so z ~ 1 means
the edge sits at its sensor-model strength (card S: the 1e-2 boundary scale lets it reach 8-15).
From `[vi_ba] STATE` lines (one per solve, the newest keyframe) after SKIP solves (default 100;
use 60 on 800 frames): per-axis mean, SD and step RMS of ba and bg.
"""
import json, re, sys, numpy as np

log, calib = sys.argv[1], sys.argv[2]
skip = int(sys.argv[3]) if len(sys.argv) > 3 else 100
imu = json.load(open(calib))["imu0"]
s_g, s_a = imu["gyro_bias_random_walk_sigma"], imu["acc_bias_random_walk_sigma"]

num = r"([-+0-9.eE]+)"
fin = re.compile(rf"\[vi_ba\] FINAL edge .*dt={num}s.*\|r_bg\|={num}\s+\|r_ba\|={num}.*boundary=(true|false)")
vec = lambda k: rf"{k}=\({num},{num},{num}\)"
st = re.compile(rf"\[vi_ba\] STATE .*{vec('ba')} {vec('bg')}")

z = {(b, k): [] for b in (True, False) for k in ("g", "a")}
states = []
for line in open(log, errors="replace"):
    m = fin.search(line)
    if m:
        dt, rg, ra, b = float(m[1]), float(m[2]), float(m[3]), m[4] == "true"
        if dt > 0:
            z[(b, "g")].append(rg / (s_g * np.sqrt(dt) * np.sqrt(3)))
            z[(b, "a")].append(ra / (s_a * np.sqrt(dt) * np.sqrt(3)))
        continue
    m = st.search(line)
    if m:
        states.append([float(x) for x in m.groups()])

for b, name in ((True, "boundary"), (False, "interior")):
    out = []
    for k, lab in (("g", "gyro"), ("a", "accel")):
        v = np.array(z[(b, k)])
        out.append(f"{lab} p50 {np.percentile(v, 50):.2f} p90 {np.percentile(v, 90):.2f}" if len(v) else f"{lab} -")
    print(f"{name:8s} n={len(z[(b, 'g')]):5d}  " + "  ".join(out))

s = np.array(states)
print(f"solves {len(s)}, skip {skip}")
s = s[skip:]
if len(s) > 1:
    for name, sl in (("ba", slice(0, 3)), ("bg", slice(3, 6))):
        v = s[:, sl]
        step = np.sqrt((np.diff(v, axis=0) ** 2).mean(axis=0))
        fmt = lambda a: "(" + ", ".join(f"{x:.3e}" for x in a) + ")"
        sd = v.std(axis=0)
        rms = lambda a: np.sqrt((a**2).mean())
        print(f"{name} mean {fmt(v.mean(axis=0))}  SD {fmt(sd)} rms {rms(sd):.3e}  step RMS {fmt(step)} rms {rms(step):.3e}")
