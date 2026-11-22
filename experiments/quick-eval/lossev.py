"""lossev.py RUN_DIR SEQ_DIR [SESSION]: what each tracking loss of one run cost (card AC, G39 G40 G46 G47).

RUN_DIR holds stderr.txt and traj.txt (a run with --trajectory-include-coasted); SESSION (default
RUN_DIR/session.cbor) gives the segment starts. SEQ_DIR is the LaMAria sequence directory (`asl/<seq>/aria/cam0`,
`aria/imu0`, `ground_truth/pseudo_dense/<seq>.txt`). Works on a frame window (start_frame): stderr frame indices are
absolute. Ported from the 2026-11-22 diagnosis scripts (diagV/track/{episodes,events,classify,yaw,coast2,kfstate}.py).

Frame classes from the `[ i] Status kf= pts=` lines: T tracked, C coasted (Skipped with a keyframe), B bootstrap stall.
A give-up is the start of every session segment after the first. Its previous segment's age at the loss classes it:
startup (no segment has lived 3 s yet), young (< 3 s), primary.

Prints:
  episodes:  frames per class, give-ups per class, young deaths and their share of give-ups.
  link:      per give-up, the gap (s), the gyro rotation over it, GT and SLAM rotation from the last tracked frame to
             the first tracked frame after it, the rotation error between them (GT taken into the trajectory's body
             frame by the hand-eye rotation C fitted on tracked 1 s pairs; printed as "body"), and the position error of that
             restart against GT after a local rotation fit on the 10 s before the loss (needs GT; medians per class).
  coast:     coasted frames of IMU-confident episodes (segment >= 3.5 s old), error against GT by age since the
             last tracked frame, the hold-still error (GT displacement), and the SLAM / GT coast speed.
  vel:       share of coasted frames whose `[vel]` line (DEBUG=1) says src=prop.
  keyframes: session keyframes over 4 m/s (initialised ones), velocity median and max.
"""
import os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from worldrot import load_session, segment_id, q2R  # noqa: E402

STATUS = re.compile(r"^\[ *(\d+)\] (\w+)\s+kf=(\d+)\s+pts=(\d+)")
VEL = re.compile(r"\[vel\] frame=(\d+) .* src=(\w+)")


def angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def log(R):
    th = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v * (th / (2 * np.sin(th)) if th > 1e-9 else 0.5)


def classes(stderr):
    """{frame: class} with T/C/B; a lone B between C's is a lost keyframe insert, so C."""
    st = {}
    for line in open(stderr, errors="replace"):
        m = STATUS.match(line)
        if m:
            st[int(m[1])] = (m[2], int(m[3]))
    cls = {}
    for i, (s, kf) in st.items():
        cls[i] = "T" if s != "Skipped" else ("B" if kf == i else "C")
    for i in sorted(cls):
        if cls[i] == "B" and cls.get(i - 1) == "C" and cls.get(i + 1) == "C":
            cls[i] = "C"
    return cls


def src_share(stderr, coast_frames):
    src = {}
    for line in open(stderr, errors="replace"):
        m = VEL.search(line)
        if m: src[int(m[1])] = m[2]
    seen = [src[f] for f in coast_frames if f in src]
    return (sum(s == "prop" for s in seen) / len(seen) if seen else float("nan")), len(seen)


class Gt:
    def __init__(self, path):
        self.ok = os.path.exists(path) and os.path.getsize(path) > 0
        if self.ok:
            g = np.loadtxt(path); self.t = g[:, 0] / 1e9; self.p = g[:, 1:4]; self.q = g[:, 4:8]

    def pos(self, t):
        if not self.ok: return None
        k = np.searchsorted(self.t, t)
        if k <= 0 or k >= len(self.t) or self.t[k] - self.t[k - 1] > 1.0: return None
        a = (t - self.t[k - 1]) / (self.t[k] - self.t[k - 1])
        return self.p[k - 1] * (1 - a) + self.p[k] * a

    def rot(self, t):
        if not self.ok: return None
        k = int(np.argmin(np.abs(self.t - t)))
        return q2R(self.q[k]) if abs(self.t[k] - t) <= 0.15 else None


def kabsch(X, Y):
    mx, my = X.mean(0), Y.mean(0)
    U, _, Vt = np.linalg.svd((Y - my).T @ (X - mx)); S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0: S[2, 2] = -1
    return U @ S @ Vt


class Run:
    def __init__(self, run, seq_dir, session):
        name = os.path.basename(os.path.normpath(seq_dir))
        aria = f"{seq_dir}/asl/{name}/aria"
        self.ts = np.loadtxt(f"{aria}/cam0/data.csv", delimiter=",", comments="#", usecols=0, dtype=np.int64)
        self.tsec = self.ts / 1e9
        imu = np.loadtxt(f"{aria}/imu0/data.csv", delimiter=",", comments="#")
        self.it, self.gyro = imu[:, 0] / 1e9, imu[:, 1:4]
        self.gt = Gt(f"{seq_dir}/ground_truth/pseudo_dense/{name}.txt")
        self.cls = classes(f"{run}/stderr.txt")
        self.lo, self.hi = min(self.cls), max(self.cls) + 1
        self.traj = {}
        for line in open(f"{run}/traj.txt"):
            v = line.split()
            if len(v) >= 8 and not line.startswith("#"):
                self.traj[int(v[0])] = (np.array(list(map(float, v[1:4]))), q2R(list(map(float, v[4:8]))))
        self.kfs, _ = load_session(session) if os.path.exists(session) else ([], {})
        starts = {}
        for kf in self.kfs:
            starts.setdefault(segment_id(kf), kf["timestamp_sec"])
        self.seg_t = np.array(sorted(starts.values()))
        self.C = self.body_alignment()

    def body_alignment(self):
        """Rotation C with GT relative rotations = C (SLAM relative rotations) C^T, fitted on the rotation vectors of
        tracked 1 s pairs (hand-eye): the GT body frame is not the trajectory's on every sequence (2_11: tracked 1 s
        pairs read 34-47 deg apart with equal magnitudes; R_01: 0.2 deg)."""
        T = [f for f in sorted(self.cls) if self.cls[f] == "T" and self.ts[f] in self.traj]
        X, Y = [], []
        for a, b in zip(T[::20], T[20::20]):
            ga, gb = self.gt.rot(self.tsec[a]), self.gt.rot(self.tsec[b])
            if self.tsec[b] - self.tsec[a] > 1.2 or ga is None or gb is None: continue
            ws, wg = log(self.traj[self.ts[a]][1].T @ self.traj[self.ts[b]][1]), log(ga.T @ gb)
            if np.linalg.norm(wg) > np.radians(10): X.append(ws); Y.append(wg)
        if len(X) < 5: return np.eye(3)
        X, Y = np.array(X), np.array(Y)
        U, _, Vt = np.linalg.svd(Y.T @ X); S = np.eye(3)
        if np.linalg.det(U @ Vt) < 0: S[2, 2] = -1
        return U @ S @ Vt

    def frame_at(self, t):
        return int(np.argmin(np.abs(self.tsec - t)))

    def last_tracked_before(self, f):
        f -= 1
        while f > self.lo and self.cls.get(f) != "T": f -= 1
        return f

    def first_tracked_from(self, f):
        while f < self.hi and self.cls.get(f) != "T": f += 1
        return f

    def gyro_angle(self, t0, t1):
        m = (self.it >= t0) & (self.it <= t1); w = self.gyro[m]
        dt = np.diff(self.it[m], append=self.it[m][-1] if m.any() else 0)
        R = np.eye(3)
        for wi, d in zip(w, dt):
            th = np.linalg.norm(wi) * d
            if th > 0:
                k = wi / np.linalg.norm(wi); K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
                R = R @ (np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K)
        return angle(R)

    def local_rotation(self, f_last, seg_start_t):
        """SLAM-to-GT rotation fitted on the tracked frames of the 10 s before f_last, same segment."""
        X, Y = [], []
        for f in range(max(self.lo, f_last - 200), f_last + 1):
            if self.cls.get(f) != "T" or self.tsec[f] < seg_start_t or self.ts[f] not in self.traj: continue
            g = self.gt.pos(self.tsec[f])
            if g is not None: X.append(self.traj[self.ts[f]][0]); Y.append(g)
        if len(X) < 20: return None
        X, Y = np.array(X), np.array(Y)
        return kabsch(X, Y) if np.ptp(Y, axis=0).max() >= 1.0 else None

    def give_ups(self):
        rows, startup = [], True
        for n in range(1, len(self.seg_t)):
            seed = self.frame_at(self.seg_t[n])
            a = self.last_tracked_before(seed)
            age = self.tsec[a] - self.seg_t[n - 1]
            c = "primary" if age >= 3 else ("startup" if startup else "young")
            startup = startup and age < 3
            rows.append(self.link(c, a, self.first_tracked_from(a + 1), age, n))
        return rows

    def link(self, c, a, s, age, n):
        r = dict(cls=c, t=self.tsec[a] - self.tsec[0], age=age, gap=self.tsec[min(s, len(self.ts) - 1)] - self.tsec[a],
                 gyro=self.gyro_angle(self.tsec[a], self.tsec[min(s, len(self.ts) - 1)]), rot_err=np.nan, pos_err=np.nan,
                 gt_rot=np.nan, slam_rot=np.nan)
        if s >= self.hi or self.ts[a] not in self.traj or self.ts[s] not in self.traj: return r
        (pa, Ra), (pb, Rb) = self.traj[self.ts[a]], self.traj[self.ts[s]]
        r["slam_rot"] = angle(Ra.T @ Rb)
        ga, gb = self.gt.rot(self.tsec[a]), self.gt.rot(self.tsec[s])
        if ga is not None and gb is not None:
            g = self.C.T @ (ga.T @ gb) @ self.C
            r["gt_rot"] = angle(g); r["rot_err"] = angle(g.T @ (Ra.T @ Rb))
        Rm = self.local_rotation(a, self.seg_t[n - 1])
        p0, p1 = self.gt.pos(self.tsec[a]), self.gt.pos(self.tsec[s])
        if Rm is not None and p0 is not None and p1 is not None:
            r["pos_err"] = float(np.linalg.norm(Rm @ (pb - pa) - (p1 - p0)))
        return r

    def episodes(self):
        out, f = [], self.first_tracked_from(self.lo)
        while f < self.hi:
            if self.cls.get(f) == "T": f += 1; continue
            j = self.first_tracked_from(f); out.append((f, j)); f = j
        return out

    def coast_rows(self):
        rows = []
        for i, j in self.episodes():
            k = np.searchsorted(self.seg_t, self.tsec[i]) - 1
            if k < 0 or self.tsec[i] - self.seg_t[k] < 3.5: continue
            last = i - 1
            if self.ts[last] not in self.traj: continue
            Rm = self.local_rotation(last, self.seg_t[k]); g0 = self.gt.pos(self.tsec[last])
            if Rm is None or g0 is None: continue
            p0 = self.traj[self.ts[last]][0]
            for f in range(i, j):
                g = self.gt.pos(self.tsec[f])
                if self.ts[f] not in self.traj or g is None: continue
                ds, dg = Rm @ (self.traj[self.ts[f]][0] - p0), g - g0
                rows.append((self.tsec[f] - self.tsec[last], float(np.linalg.norm(ds - dg)), float(np.linalg.norm(dg)),
                             float(np.linalg.norm(ds))))
        return np.array(rows).reshape(-1, 4)


def med(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return (f"{np.median(x):.2f}" if len(x) else "nan"), len(x)


def report(r):
    c = "".join(r.cls[f] for f in sorted(r.cls))
    gu = r.give_ups()
    print(f"body: GT-to-trajectory body rotation C {angle(r.C):.1f} deg (hand-eye on tracked 1 s pairs)")
    n = {k: sum(g["cls"] == k for g in gu) for k in ("startup", "young", "primary")}
    print(f"episodes: frames {len(c)} T {c.count('T')} C {c.count('C')} B {c.count('B')}; give_ups {len(gu)} {n}; "
          f"young deaths {n['young']} ({100 * n['young'] / max(len(gu), 1):.0f} %)")
    print("link: cls t_s age_s gap_s gyro_deg gt_rot_deg slam_rot_deg rot_err_deg restart_pos_err_m")
    for g in gu:
        if g["cls"] == "primary":
            print(f"  {g['cls']} {g['t']:.1f} {g['age']:.1f} {g['gap']:.2f} {g['gyro']:.1f} {g['gt_rot']:.1f} "
                  f"{g['slam_rot']:.1f} {g['rot_err']:.1f} {g['pos_err']:.2f}")
    for k in ("all", "primary", "young", "startup"):
        sel = [g for g in gu if k == "all" or g["cls"] == k]
        if sel:
            (re_, nr), (pe, npe) = med([g["rot_err"] for g in sel]), med([g["pos_err"] for g in sel])
            print(f"  {k}: n={len(sel)} rot_err median {re_} deg (n={nr}); restart pos_err median {pe} m (n={npe})")
    rows = r.coast_rows()
    print(f"coast: IMU-confident coasted frames with GT {len(rows)}")
    for a, b in zip([0, 0.5, 1, 2, 3], [0.5, 1, 2, 3, 5.1]):
        m = rows[(rows[:, 0] >= a) & (rows[:, 0] < b)]
        if len(m):
            sp, gsp = m[:, 3] / np.maximum(m[:, 0], 1e-3), m[:, 2] / np.maximum(m[:, 0], 1e-3)
            print(f"  {a}-{b}s n={len(m)} coast_err med {np.median(m[:, 1]):.2f} p90 {np.percentile(m[:, 1], 90):.2f} | "
                  f"hold err med {np.median(m[:, 2]):.2f} | speed slam {np.median(sp):.2f} gt {np.median(gsp):.2f} "
                  f"err {np.median(np.abs(sp - gsp)):.2f} m/s")
    share, nv = src_share(f"{RUN}/stderr.txt", [f for f in sorted(r.cls) if r.cls[f] == "C"])
    print(f"vel: coasted frames with a [vel] line {nv}, src=prop {100 * share:.0f} %")
    v = np.array([k["velocity_world"] for k in r.kfs]).reshape(-1, 3); sp = np.linalg.norm(v, axis=1)
    init = sp > 0
    if init.any():
        print(f"keyframes: {len(r.kfs)}, initialised {init.sum()}, over 4 m/s {(sp > 4).sum()}, "
              f"speed median {np.median(sp[init]):.2f} max {sp.max():.1f} m/s")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr); sys.exit(2)
    RUN = sys.argv[1]
    report(Run(RUN, sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else f"{RUN}/session.cbor"))
