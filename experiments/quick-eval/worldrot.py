"""worldrot.py: rotations of the whole world about the origin (card AC, G46).

  worldrot.py jumps TRAJ [MIN_JUMP_M]
      Single-frame jumps (dt <= 0.06 s) over MIN_JUMP_M (default 1 m) in a `t x y z qx qy qz qw` file, t in ns.
      A jump that keeps the distance from the origin to 0.1 m is an origin rotation: the later segment's inertial
      alignment (`Map::apply_inertial_alignment`) moved the tracking pose with the whole map. Prints time (s from
      the first pose), jump, distance from the origin before and after, the angle between the two positions about
      the origin, the orientation change across the jump, and the axis. The position angle is only a lower bound
      of the world rotation (equal when the axis is normal to the position); the orientation change is the
      rotation itself, the one to compare with an `[imu_init]` gravity angle.
  worldrot.py segrot TRAJ SESSION.cbor
      Per session segment: the rotation about the origin that takes the online trajectory's positions to the
      session dump's keyframe centres (-R^T t of `pose_cw`), its angle, the tilt of world Y under it, and the fit
      RMS. A level, unrotated segment shows 0 deg and 0 deg.
  worldrot.py gravity STDERR
      Angle between each solved gravity on an `[imu_init] accepted` line (DEBUG=1 runs) and (0, 9.81, 0).

The CBOR walk is diagV/drift/sess.py's: it decodes only the keyframe headers and skips the observations.
"""
import struct, sys, re
import numpy as np


def load_traj(path, rotations=False):
    a = np.loadtxt(path, comments="#", ndmin=2)
    if rotations:
        return a[:, 0].astype(np.int64), a[:, 1:4], a[:, 4:8]
    return a[:, 0].astype(np.int64), a[:, 1:4]


def rotation_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def angle_deg(a, b):
    c = np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def jumps(t, p, min_jump=1.0):
    """Rows (i, jump_m, r0, r1, angle_deg, axis, origin_preserving) for single-frame jumps over min_jump."""
    out = []
    dt = np.diff(t) / 1e9
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    r = np.linalg.norm(p, axis=1)
    for i in np.nonzero((dt <= 0.06) & (d > min_jump))[0]:
        a, b = p[i], p[i + 1]
        ax = np.cross(a, b)
        ax = ax / max(np.linalg.norm(ax), 1e-12)
        out.append((i + 1, float(d[i]), float(r[i]), float(r[i + 1]), angle_deg(a, b), ax, abs(r[i + 1] - r[i]) <= 0.1))
    return out


class Cbor:
    """Minimal CBOR reader for a SessionDump: keyframe headers only."""

    def __init__(self, path):
        self.b = open(path, "rb").read()

    def head(self, i):
        b = self.b[i]; mt = b >> 5; ai = b & 31; i += 1
        if ai < 24: return mt, ai, i
        n = {24: 1, 25: 2, 26: 4, 27: 8}[ai]
        return mt, int.from_bytes(self.b[i:i + n], "big"), i + n

    def dec(self, i):
        b = self.b[i]; mt = b >> 5; ai = b & 31
        if mt == 7:
            if ai == 25:
                return float(np.frombuffer(self.b[i + 1:i + 3], dtype=">f2")[0]), i + 3
            if ai == 26: return struct.unpack(">f", self.b[i + 1:i + 5])[0], i + 5
            if ai == 27: return struct.unpack(">d", self.b[i + 1:i + 9])[0], i + 9
            return {20: False, 21: True, 22: None, 23: None}[ai], i + 1
        mt, n, i = self.head(i)
        if mt == 0: return n, i
        if mt == 1: return -1 - n, i
        if mt in (2, 3):
            s = self.b[i:i + n]; return (s.decode() if mt == 3 else s), i + n
        if mt == 4:
            out = []
            for _ in range(n):
                v, i = self.dec(i); out.append(v)
            return out, i
        if mt == 5:
            out = {}
            for _ in range(n):
                k, i = self.dec(i); v, i = self.dec(i); out[k] = v
            return out, i
        return self.dec(i)  # tag

    def skip(self, i):
        b = self.b[i]; mt = b >> 5; ai = b & 31
        if mt == 7:
            return i + {25: 3, 26: 5, 27: 9}.get(ai, 1)
        mt, n, i = self.head(i)
        if mt in (0, 1): return i
        if mt in (2, 3): return i + n
        if mt in (4, 5):
            for _ in range(n * (2 if mt == 5 else 1)): i = self.skip(i)
            return i
        return self.skip(i)


def load_session(path):
    """Keyframes as dicts (observations replaced by their count) and the top-level scalars."""
    c = Cbor(path)
    mt, n, i = c.head(0)
    kfs, meta = [], {}
    for _ in range(n):
        k, i = c.dec(i)
        if k == "keyframes":
            _, nk, i = c.head(i)
            for _ in range(nk):
                _, nf, i = c.head(i); kf = {}
                for _ in range(nf):
                    kk, i = c.dec(i)
                    if kk == "observations":
                        _, no, _ = c.head(i); kf["observations_n"] = no; i = c.skip(i)
                    else:
                        v, i = c.dec(i); kf[kk] = v
                kfs.append(kf)
        elif k in ("landmarks", "imu", "frames", "rig"):
            i = c.skip(i)
        else:
            v, i = c.dec(i); meta[k] = v
    return kfs, meta


def segment_id(kf):
    s = kf["segment"]
    return list(s.values())[0] if isinstance(s, dict) else s


def q2R(q):
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def kf_centre(kf):
    R = q2R(kf["pose_cw"]["rotation_xyzw"]); t = np.asarray(kf["pose_cw"]["translation"], float)
    return -R.T @ t


def origin_rotation(P, Q):
    """R minimising |R P - Q| (rows are points), rotation about the origin only."""
    U, _, Vt = np.linalg.svd(Q.T @ P); S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0: S[2, 2] = -1
    return U @ S @ Vt


def segment_rotations(t, p, kfs):
    """Rows (segment, t0_s, t1_s, n, rot_deg, tilt_deg, rms_m) for every segment with >= 3 matched keyframes."""
    by_seg = {}
    for kf in kfs:
        by_seg.setdefault(segment_id(kf), []).append(kf)
    rows = []
    for seg, ks in sorted(by_seg.items(), key=lambda kv: kv[1][0]["timestamp_sec"]):
        tk = np.array([round(k["timestamp_sec"] * 1e9) for k in ks], dtype=np.int64)
        idx = np.searchsorted(t, tk).clip(1, len(t) - 1)
        idx = np.where(np.abs(t[idx - 1] - tk) < np.abs(t[idx] - tk), idx - 1, idx)
        ok = np.abs(t[idx] - tk) < 1_000_000
        if ok.sum() < 3: continue
        P = p[idx[ok]]; Q = np.array([kf_centre(k) for k, o in zip(ks, ok) if o])
        R = origin_rotation(P, Q)
        rot = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
        tilt = angle_deg(R @ np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0]))
        rms = float(np.sqrt(np.mean(np.sum((P @ R.T - Q) ** 2, axis=1))))
        rows.append((seg, ks[0]["timestamp_sec"], ks[-1]["timestamp_sec"], int(ok.sum()), rot, tilt, rms))
    return rows


GRAVITY = re.compile(r"\[imu_init\] accepted .*gravity=\(([-\d.]+),([-\d.]+),([-\d.]+)\)")


def gravity_angles(stderr):
    out = []
    for line in open(stderr, errors="replace"):
        m = GRAVITY.search(line)
        if m:
            g = np.array([float(m[k]) for k in (1, 2, 3)])
            out.append((g, angle_deg(g, np.array([0.0, 9.81, 0.0]))))
    return out


def main(argv):
    if len(argv) >= 2 and argv[0] == "jumps":
        t, p, q = load_traj(argv[1], rotations=True); mj = float(argv[2]) if len(argv) > 2 else 1.0
        rows = jumps(t, p, mj)
        print(f"jumps over {mj} m: {len(rows)}, origin-preserving: {sum(r[6] for r in rows)}")
        print("  t_s     jump_m  r_before  r_after  angle_deg  orient_deg  origin  axis")
        for i, d, r0, r1, ang, ax, keep in rows:
            orient = rotation_deg(q2R(q[i]) @ q2R(q[i - 1]).T)
            print(f"  {(t[i] - t[0]) / 1e9:8.1f} {d:7.1f} {r0:8.1f} {r1:8.1f} {ang:8.1f} {orient:9.1f}     "
                  f"{'yes' if keep else 'no ':3s}   ({ax[0]:.2f}, {ax[1]:.2f}, {ax[2]:.2f})")
        return 0
    if len(argv) == 3 and argv[0] == "segrot":
        t, p = load_traj(argv[1]); kfs, meta = load_session(argv[2])
        print(f"keyframes {len(kfs)}, gravity_world {meta.get('gravity_world')}")
        print("  segment  t0_s      t1_s      n_kf  rot_deg  tilt_deg  rms_m")
        t0 = t[0] / 1e9
        for seg, a, b, n, rot, tilt, rms in segment_rotations(t, p, kfs):
            print(f"  {seg:7d} {a - t0:8.1f} {b - t0:8.1f} {n:6d} {rot:8.1f} {tilt:8.1f} {rms:7.2f}")
        return 0
    if len(argv) == 2 and argv[0] == "gravity":
        for g, ang in gravity_angles(argv[1]):
            print(f"  gravity ({g[0]:.3f}, {g[1]:.3f}, {g[2]:.3f}) angle to +Y {ang:.1f} deg")
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
