"""biasevent.py RUN_DIR|TRAJ SEQ_DIR [--body FILE] [--window LO:HI] [--cache DIR]: does the bias carry the events (card AH, G49)?

RUN_DIR holds traj.txt and optionally session.cbor and stderr.txt (BASE gate runs: no session, so M2 only); a file
argument is scored for M2 alone (ORB-SLAM3 estimate.txt, a `synth` trajectory). Times are s from the first cam0 frame.
  b_true:  gyro bias per 60 s piece from the GT orientation. Pairs are non-overlapping GT samples about 1 s apart (GT
           rotation noise over dt sets the rate residual). The GT body frame is taken to the IMU by clockscan.py's
           constant-bias fit with its rotation, refitted on the gyro; the body file of `lossev.py body` (`--body`, else
           gateH0/body, cardAC/body) is printed as the check. Model: Log(dR_gyro^T X^T dR_gt X) = J b, J the bias
           Jacobian of the gyro chain (sum of the step rotations to the pair's end); not clockscan's `vi - b dt`, which at
           160-380 deg/s misses by more than the bias. The camera-IMU clock d is scanned (-2..10 ms) on the same fit.
           Prints the residual sd per pair (rad/s) and the standard error of each piece's b_true (s_fit). Fitted once per
           sequence and cached (--cache, default .pr180-fix/cardAH/s0/btrue): window runs read it.
           Without GT (5_11, 5_12) b_true is the run median of the estimate's 30 s bias medians, and s_fit their spread.
  walk:    consecutive keyframes, same segment, 0 < dt < 2 s, nonzero biases: z = |db| / (sigma_w sqrt(3 dt)), sigma_w the
           bias random walk of imu0 in the sequence calibration (as biaswalk.py). p50 / p90 gyro and accel.
  ba30:    |ba| median per 30 s grid piece (the planner preview's number), and the trailing 30 s vector median ba30 on a
           5 s grid against the segment reference (median of the segment's ba30). Bound 3 sigma_a sqrt(T), T the span of
           the segment's nonzero-bias keyframes inside the window: how far a random walk may go in that time.
  bg-b_true: 5 s median of the keyframe gyro bias minus b_true.
  M2:      per 5 s window, the yaw part (about up) of Log(dR_gyro(b_true)^T dR_est) in the IMU frame at the window end,
           dR_est the estimate's relative rotation taken to the IMU by a hand-eye rotation fitted on the run's 1 s pairs
           (identity-like for ours, C for GT). Up from the mean specific force over the window. Flagged when |yaw| >
           3 sigma_psi, sigma_psi^2 = sigma_g^2 tau + sigma_w^2 tau^3 / 3 + (s_fit tau)^2, sigma_g the gyro noise density,
           s_fit the standard error of the piece's b_true.
  events:  driftbudget.py's 60 s piece events. Per event: class (restart if a segment start lies in [lo - 30 s, t_ev]),
           M2 sum over its windows (signed, all windows), flagged windows, the bias-carried heading -int (bg - b_true).up dt
           over [lo, hi] (same sign as M2: a too-large bias turns the estimate the other way), its share of the yaw
           step, and the lead of the accel-bias excursion: t_ev minus the start of the ba30 run over the bound in
           progress at t_ev (runs merge over gaps up to 10 s; t_ev the refined step time, lo for scale-only events).
The last lines are `SUMMARY key=value ...` and one `EVENT ...` per event, for collection.

biasevent.py synth SEQ_DIR OUT [--ramp LO:HI:DEG]: GT poses as a `t x y z qx qy qz qw` estimate, optionally with a yaw
ramp about GT up from LO to HI s (positions re-integrated through the turned heading): the M2 self-tests.
"""
import hashlib, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from worldrot import load_session, segment_id  # noqa: E402
from clockscan import read_gyro, chain, orient, exp, log, kabsch, q2R, bias_fit  # noqa: E402
import driftbudget as db  # noqa: E402

P = "/home/edgar/dev/split/.pr180-fix"
CACHE = f"{P}/cardAH/s0/btrue"
PIECE_S, WIN_S, BA_S = 60.0, 5.0, 30.0
PAIR_S = 1.0
OFFSETS_MS = np.arange(-2.0, 10.0001, 0.5)


def arg(argv, k, default=None):
    if k in argv:
        i = argv.index(k); v = argv[i + 1]; del argv[i:i + 2]; return v
    return default


def angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def R2q(R):  # xyzw
    w = np.sqrt(max(1e-12, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    q = np.array([(R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w), w])
    if w < 1e-3:  # near 180 deg: let a quaternion via the eigenvector take over
        K = np.array([[R[0, 0] - R[1, 1] - R[2, 2], R[1, 0] + R[0, 1], R[2, 0] + R[0, 2], R[1, 2] - R[2, 1]],
                      [R[1, 0] + R[0, 1], R[1, 1] - R[0, 0] - R[2, 2], R[2, 1] + R[1, 2], R[2, 0] - R[0, 2]],
                      [R[2, 0] + R[0, 2], R[2, 1] + R[1, 2], R[2, 2] - R[0, 0] - R[1, 1], R[0, 1] - R[1, 0]],
                      [R[1, 2] - R[2, 1], R[2, 0] - R[0, 2], R[0, 1] - R[1, 0], R[0, 0] + R[1, 1] + R[2, 2]]]) / 3
        v = np.linalg.eigh(K)[1][:, -1]; q = np.array([v[0], v[1], v[2], v[3]])
    return q / np.linalg.norm(q)


class Seq:
    def __init__(self, seq_dir, body=None):
        self.dir = seq_dir.rstrip("/"); self.name = os.path.basename(self.dir)
        asl = f"{self.dir}/asl/{self.name}"
        self.cam = np.loadtxt(f"{asl}/aria/cam0/data.csv", delimiter=",", comments="#", usecols=0, dtype=np.int64)
        self.t0 = int(self.cam[0])
        cal = __import__("json").load(open(f"{self.dir}/pinhole_calibrations/{self.name}.json"))["imu0"]
        self.sw_g, self.sw_a = cal["gyro_bias_random_walk_sigma"], cal["acc_bias_random_walk_sigma"]
        self.sn_g = cal["gyro_noise_density"]
        self.ti, self.gy, self.dti = read_gyro(asl)
        self.acc = np.loadtxt(f"{asl}/aria/imu0/data.csv", delimiter=",", comments="#", usecols=(4, 5, 6))
        gtp = f"{self.dir}/ground_truth/pseudo_dense/{self.name}.txt"
        self.has_gt = os.path.exists(gtp) and os.path.getsize(gtp) > 0
        self.body_file = None
        if self.has_gt:
            g = np.loadtxt(gtp, comments="#", ndmin=2)
            self.tg = g[:, 0].astype(np.int64); self.pg = g[:, 1:4]; self.Rg = q2R(g[:, 4:8])
            self.body_file = body or next((f for f in (f"{P}/gateH0/body/{self.name}.txt", f"{P}/cardAC/body/{self.name}.txt")
                                           if os.path.exists(f)), None)
            if self.body_file is None: raise SystemExit(f"biasevent: no body rotation for {self.name}; pass --body")
            self.C = np.loadtxt(self.body_file)
        self.Ch = chain(self.gy, self.dti, "hold")
        self.d_ns = 0

    def rel(self, t_ns):
        return (np.asarray(t_ns) - self.t0) / 1e9

    def abs_ns(self, t_s):
        return (np.asarray(t_s) * 1e9).astype(np.int64) + self.t0

    # ---- b_true ----
    def pairs(self):
        """Non-overlapping GT pairs of about PAIR_S: the GT rotation noise (about 1.5 mrad per sample on 1_19) over dt
        sets the rate residual, so 0.2 s neighbours would bury the bias."""
        tg = self.tg; ja, jb = [], []; k = int(np.searchsorted(tg, self.ti[0] + 20_000_000))
        while k < len(tg) - 1:
            e = int(np.searchsorted(tg, tg[k] + int(PAIR_S * 1e9)))
            if e >= len(tg) or tg[e] > self.ti[-1] - 20_000_000: break
            if (tg[e] - tg[k]) / 1e9 <= 2 * PAIR_S and np.diff(tg[k:e + 1]).max() <= 1e9: ja.append(k); jb.append(e)
            k = e
        return np.array(ja), np.array(jb)

    def pair_model(self, ja, jb, d_ns, X, S):
        """rv (IMU frame at the pair end) and J for GT pairs at clock offset d, GT taken to the IMU by X."""
        ta, tb = self.tg[ja] + d_ns, self.tg[jb] + d_ns
        Ra = orient(self.ti, self.gy, self.dti, self.Ch, "hold", ta); Rb = orient(self.ti, self.gy, self.dti, self.Ch, "hold", tb)
        dRi = np.einsum("nji,njk->nik", Ra, Rb)
        Gi = X.T @ np.einsum("nji,njk->nik", self.Rg[ja], self.Rg[jb]) @ X
        rv = log(np.einsum("nji,njk->nik", dRi, Gi))
        ia = np.searchsorted(self.ti, ta).clip(0, len(S) - 1); ib = np.searchsorted(self.ti, tb).clip(0, len(S) - 1)
        J = -np.einsum("nji,njk->nik", Rb, S[ib] - S[ia])
        return rv, J, (self.tg[jb] - self.tg[ja]) / 1e9, log(dRi)

    def hand_eye_gt(self, ja, jb, d_ns):
        """clockscan.py's constant-bias fit with its rotation: X with GT rotation vectors = X (gyro - b dt)."""
        wi = log(np.einsum("nji,njk->nik", *[orient(self.ti, self.gy, self.dti, self.Ch, "hold", t + d_ns)
                                              for t in (self.tg[ja], self.tg[jb])]))
        wg = log(np.einsum("nji,njk->nik", self.Rg[ja], self.Rg[jb]))
        big = np.linalg.norm(wg, axis=1) > np.radians(10)
        return bias_fit(wi[big], wg[big], ((self.tg[jb] - self.tg[ja]) / 1e9)[big])[1]

    def bias_only_sse(self, rv, J, tmid):  # clock scan objective: per-piece bias, no scale
        tot = 0.0
        for lo in np.arange(0.0, tmid.max() + PIECE_S, PIECE_S):
            m = (tmid >= lo) & (tmid < lo + PIECE_S)
            if m.sum() < 15: continue
            b = np.linalg.lstsq(J[m].reshape(-1, 3), rv[m].reshape(-1), rcond=None)[0]
            tot += float(np.sum((rv[m] - J[m] @ b) ** 2))
        return tot

    def scale_jacobian(self, ia, ib):
        """K (n x 3 x 9): the pair rotation's change for a gyro correction w -> (I + dM) w, C_b^T sum C_{k+1} dM w_k dt,
        linear in vec(dM) (column-major). Prefix sums in chunks, kept at the needed indices only (3x9 per sample
        would be 200 MB on a 900 s sequence)."""
        need = np.unique(np.concatenate([ia, ib])); at = {}; tot = np.zeros((3, 9)); step = 50_000
        for c in range(0, len(self.dti), step):
            g = self.gy[c + 1:c + 1 + step]; Cn = self.Ch[c + 1:c + 1 + step]; dt = self.dti[c:c + step]
            blk = np.einsum("ni,njk->njik", g * dt[:, None], Cn).reshape(len(dt), 3, 9)  # kron(w^T, C)
            cs = np.cumsum(blk, axis=0)
            for n in need[(need > c) & (need <= c + len(dt))]: at[n] = tot + cs[n - c - 1]
            for n in need[need == c]: at[n] = tot.copy()
            tot = tot + cs[-1]
        for n in need[need > len(self.dti)]: at[n] = tot
        Pa = np.array([at[n] for n in ia]); Pb = np.array([at[n] for n in ib])
        return np.einsum("nji,njk->nik", self.Ch[ib], Pb - Pa)

    def fit_joint(self, rv, J, K, dtp, tmid):
        """Per-piece gyro bias and one global dM; refit without residuals over 5 MAD (pseudo-dense spikes)."""
        P_ = np.floor(tmid / PIECE_S).astype(int); npc = P_.max() + 1
        cnt = np.bincount(P_, minlength=npc); use = cnt[P_] >= 15; keep = use.copy()
        for _ in range(2):
            idx = np.nonzero(keep)[0]; A = np.zeros((3 * len(idx), 3 * npc + 9))
            for r, k in enumerate(idx):
                A[3 * r:3 * r + 3, 3 * P_[k]:3 * P_[k] + 3] = J[k]; A[3 * r:3 * r + 3, 3 * npc:] = K[k]
            x = np.linalg.lstsq(A, rv[idx].reshape(-1), rcond=None)[0]
            b = x[:3 * npc].reshape(npc, 3); dM = x[3 * npc:].reshape(3, 3, order="F")
            res = np.linalg.norm(rv - np.einsum("nij,nj->ni", J, b[P_]) - np.einsum("nij,j->ni", K, x[3 * npc:]), axis=1) / dtp
            keep = use & (res <= 5 * 1.4826 * np.median(res[use]))
        rows = []
        for k in range(npc):
            m = keep & (P_ == k); ok = (P_ == k).sum() >= 15
            rows.append((k * PIECE_S, b[k] if ok else None, float(np.sqrt(np.mean(res[m] ** 2))) if ok else np.nan,
                         int(m.sum()), res[m] if ok else None))
        return rows, dM

    def btrue_fit(self):
        ja, jb = self.pairs()
        S = np.concatenate([np.zeros((1, 3, 3)), np.cumsum(self.Ch[1:] * self.dti[:, None, None], axis=0)])
        tmid = self.rel((self.tg[ja] + self.tg[jb]) // 2)
        # The body file C comes from SLAM-trajectory pairs; 0.7 deg off on 1_19, it doubles the residual at turns,
        # so the rotation is refitted on the gyro and C kept as the check.
        X = self.hand_eye_gt(ja, jb, 0); sse = []
        for d in OFFSETS_MS:
            rv, J, dtp, _ = self.pair_model(ja, jb, int(round(d * 1e6)), X, S)
            sse.append(self.bias_only_sse(rv, J, tmid))
        sse = np.array(sse); k = min(max(int(np.argmin(sse)), 1), len(sse) - 2)
        c = np.polyfit(OFFSETS_MS[k - 1:k + 2], sse[k - 1:k + 2], 2)
        d_ms = float(-c[1] / (2 * c[0])) if c[0] > 0 else float(OFFSETS_MS[k])
        X = self.hand_eye_gt(ja, jb, int(round(d_ms * 1e6)))
        rv, J, dtp, wi = self.pair_model(ja, jb, int(round(d_ms * 1e6)), X, S)
        ia = np.searchsorted(self.ti, self.tg[ja] + int(round(d_ms * 1e6))).clip(0, len(S) - 1)
        ib = np.searchsorted(self.ti, self.tg[jb] + int(round(d_ms * 1e6))).clip(0, len(S) - 1)
        rows, dM = self.fit_joint(rv, J, self.scale_jacobian(ia, ib), dtp, tmid)
        allr = np.concatenate([r for _, b, _, _, r in rows if b is not None])
        sd = np.array([r[2] for r in rows]); n = np.array([r[3] for r in rows])
        return dict(lo=np.array([r[0] for r in rows]), b=np.array([r[1] if r[1] is not None else [np.nan] * 3 for r in rows]),
                    sd=sd, se=sd / np.sqrt(np.maximum(n, 1)), n=n, d_ms=d_ms, X=X, dM=dM, version=2,
                    sd_all=float(np.sqrt(np.mean(allr ** 2))), se_p50=float(np.nanmedian(sd / np.sqrt(np.maximum(n, 1)))),
                    c_vs_gyro_deg=angle_deg(X.T @ self.C), pairs=len(ja), dt_p50=float(np.median(dtp)),
                    body=self.body_file, body_md5=hashlib.md5(open(self.body_file, "rb").read()).hexdigest())

    def btrue(self, cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        p = f"{cache_dir}/{self.name}.npz"
        md5 = hashlib.md5(open(self.body_file, "rb").read()).hexdigest()
        if os.path.exists(p):
            z = dict(np.load(p))
            if str(z["body_md5"]) == md5 and int(z.get("version", 0)) == 2:
                bt = {k: (v.item() if v.shape == () else v) for k, v in z.items()}; bt["cached"] = True; return bt
        bt = self.btrue_fit()
        self.d_ns = int(round(bt["d_ms"] * 1e6)); self.set_btrue(bt["lo"], bt["b"], bt["se"], bt["dM"])
        bt["sig_gt"], bt["sig_gt_n"] = self.gt_window_noise(bt["X"])
        np.savez(p, **bt); bt["cached"] = False; return bt

    def gt_window_noise(self, X):
        """GT endpoint noise of a 5 s window: 1.4826 MAD of the yaw part of Log(dR_gyro(b_true)^T X^T dR_gt X) over GT
        samples 5 s apart, after the predicted scale yaw is removed (it has its own term)."""
        ys = []; tr = self.rel(self.tg)
        for a in np.arange(tr[0], tr[-1] - WIN_S, WIN_S):
            i = int(np.argmin(np.abs(tr - a))); j = int(np.argmin(np.abs(tr - a - WIN_S)))
            if abs(tr[i] - a) > 0.25 or abs(tr[j] - a - WIN_S) > 0.25 or j <= i: continue
            Ri, Rj = self.Rc(np.array([self.tg[i], self.tg[j]]))
            v = log((Rj.T @ Ri @ X.T @ self.Rg[i].T @ self.Rg[j] @ X)[None])[0]
            u = self.up(np.array([self.tg[j]]))[0]
            ys.append(float(v @ u - self.scale_pred(self.tg[i], self.tg[j]) @ u))
        ys = np.array(ys)
        return float(1.4826 * np.median(np.abs(ys - np.median(ys)))), len(ys)

    def set_btrue(self, lo, b, sd, dM=None):
        self.dM = np.zeros((3, 3)) if dM is None else dM
        v = np.einsum("nij,nj->ni", self.Ch[1:], self.gy[1:] @ self.dM.T) * self.dti[:, None]
        self.Q = np.concatenate([np.zeros((1, 3)), np.cumsum(v, axis=0)])
        ok = np.isfinite(b).all(1)
        self.bt_c = lo[ok] + PIECE_S / 2; self.bt_b = b[ok]; self.bt_lo = lo; self.bt_sd = sd
        bti = np.stack([np.interp(self.rel(self.ti), self.bt_c, self.bt_b[:, k]) for k in range(3)], 1)
        self.gyc = self.gy - bti
        self.Cc = chain(self.gyc, self.dti, "hold")

    def b_true(self, t_s):
        return np.stack([np.interp(t_s, self.bt_c, self.bt_b[:, k]) for k in range(3)], -1)

    def s_fit(self, t_s):
        k = np.clip(((np.asarray(t_s) - self.bt_lo[0]) // PIECE_S).astype(int), 0, len(self.bt_sd) - 1)
        sd = self.bt_sd[k]; fallback = np.nanmedian(self.bt_sd)
        return np.where(np.isfinite(sd), sd, fallback)

    def scale_pred(self, ta, tb):
        """Rotation (IMU frame at tb) the fitted gyro scale dM adds between ta and tb: C_b^T sum C_{k+1} dM w_k dt."""
        ia, ib = np.searchsorted(self.ti, [ta + self.d_ns, tb + self.d_ns]).clip(0, len(self.Q) - 1)
        return self.Ch[ib].T @ (self.Q[ib] - self.Q[ia])

    def Rc(self, t_ns):  # bias-corrected gyro orientation, clock applied
        return orient(self.ti, self.gyc, self.dti, self.Cc, "hold", np.asarray(t_ns) + self.d_ns)

    def up(self, t_ns, half_s=WIN_S / 2):
        """Up in the IMU frame at each t: mean specific force over t +- half_s, rotated to the frame at t."""
        out = []
        for t in np.atleast_1d(t_ns):
            a, b = np.searchsorted(self.ti, [t - int(half_s * 1e9), t + int(half_s * 1e9)])
            k = np.arange(a, max(b, a + 1), 10).clip(0, len(self.ti) - 1)
            Rt = self.Ch[min(np.searchsorted(self.ti, t), len(self.ti) - 1)]
            f = np.einsum("ji,njk,nk->i", Rt, self.Ch[k], self.acc[k])
            out.append(f / max(np.linalg.norm(f), 1e-9))
        return np.array(out)


class Est:
    def __init__(self, path, seq):
        a = np.loadtxt(path, comments="#", ndmin=2)
        o = np.argsort(a[:, 0], kind="stable"); a = a[o]
        self.t = a[:, 0].astype(np.int64); self.p = a[:, 1:4]; self.R = q2R(a[:, 4:8]); self.ts = seq.rel(self.t)

    def near(self, t_s, tol):
        i = np.searchsorted(self.ts, t_s).clip(1, len(self.ts) - 1)
        i = np.where(np.abs(self.ts[i - 1] - t_s) < np.abs(self.ts[i] - t_s), i - 1, i)
        return i, np.abs(self.ts[i] - t_s) <= tol


def hand_eye(est, seq, segs):
    """X with w_est = X w_imu on the run's 1 s pairs (same segment, both turns over 10 deg), refit on inliers."""
    a = np.arange(est.ts[0], est.ts[-1] - 1.0, 0.5)
    ia, oka = est.near(a, 0.06); ib, okb = est.near(a + 1.0, 0.06)
    ok = oka & okb & (segs(est.ts[ia]) == segs(est.ts[ib])) & (ib > ia)
    ia, ib = ia[ok], ib[ok]
    we = log(np.einsum("nji,njk->nik", est.R[ia], est.R[ib]))
    wi = log(np.einsum("nji,njk->nik", seq.Rc(est.t[ia]), seq.Rc(est.t[ib])))
    big = (np.linalg.norm(we, axis=1) > np.radians(10)) & (np.linalg.norm(wi, axis=1) > np.radians(10))
    X = kabsch(wi[big], we[big])
    r = np.linalg.norm(wi[big] @ X.T - we[big], axis=1); keep = r < max(3 * np.median(r), np.radians(2))
    X = kabsch(wi[big][keep], we[big][keep])
    r = np.linalg.norm(wi[big] @ X.T - we[big], axis=1)
    # 3D residual norm over two endpoints; per-axis sigma from its median (Maxwell: median = 1.538 sigma).
    return X, int(big.sum()), float(np.degrees(np.median(r))), float(np.median(r) / 1.538)


def m2_windows(est, seq, X, segs, lo, hi, tol, sig_run):
    rows = []
    for a in np.arange(np.floor(lo / WIN_S) * WIN_S, hi - WIN_S + 1e-9, WIN_S):
        (i, oi), (j, oj) = est.near(np.array([a]), tol), est.near(np.array([a + WIN_S]), tol)
        i, j = int(i[0]), int(j[0])
        if not (oi[0] and oj[0]) or j <= i or segs(est.ts[i]) != segs(est.ts[j]) or est.ts[i] < lo or est.ts[j] > hi + tol:
            continue
        dRe = X.T @ est.R[i].T @ est.R[j] @ X
        Ri, Rj = seq.Rc(np.array([est.t[i], est.t[j]]))
        v = log((Rj.T @ Ri @ dRe)[None])[0]  # dR_gyro^T dR_est, IMU frame at the window end
        u = seq.up(np.array([est.t[j]]))[0]
        tau = (est.t[j] - est.t[i]) / 1e9
        sf = float(seq.s_fit(np.array([a + WIN_S / 2]))[0])
        sc = float(seq.scale_pred(est.t[i], est.t[j]) @ u)
        sig = np.sqrt(seq.sn_g ** 2 * tau + seq.sw_g ** 2 * tau ** 3 / 3 + (sf * tau) ** 2 + seq.sig_gt ** 2 + sig_run ** 2 + sc ** 2)
        yaw = float(v @ u); tilt = float(np.linalg.norm(v - yaw * u))
        rows.append(dict(lo=a, hi=a + WIN_S, yaw=np.degrees(yaw), tilt=np.degrees(tilt), sig=np.degrees(sig), sc=abs(np.degrees(sc)),
                         flag1=abs(yaw) > 3 * sig, seg=segs(est.ts[i])))
    # A flag counts only in a run of >= 2 adjacent windows over 3 sigma with the same sign (owner, 2026-11-23): one
    # jittered pose shared by two windows flags both with opposite signs.
    for r in rows: r["flag"] = False
    k = 0
    while k < len(rows):
        j = k
        while (rows[k]["flag1"] and j + 1 < len(rows) and rows[j + 1]["flag1"] and abs(rows[j + 1]["lo"] - rows[j]["hi"]) < 1e-6
               and np.sign(rows[j + 1]["yaw"]) == np.sign(rows[k]["yaw"])):
            j += 1
        if j > k:
            for r in rows[k:j + 1]: r["flag"] = True
        k = j + 1
    return rows


def flag_runs(rows):
    out, cur = [], None
    for r in rows:
        if r["flag"] and cur is not None and abs(r["lo"] - cur[1]) < 1e-6: cur[1] = r["hi"]
        elif r["flag"]: cur = [r["lo"], r["hi"]]; out.append(cur)
        else: cur = None
    return out


def summary_stderr(run):
    p = f"{run}/stderr.txt"; out = {}
    if not run or not os.path.exists(p): return out
    for line in open(p, errors="replace"):
        if line.startswith("[summary]"):
            for k, v in re.findall(r"(\w+)=([-\d.]+)", line): out[k] = v
    return out


def main(argv):
    if len(argv) >= 3 and argv[0] == "synth":
        return synth(argv[1:])
    body = arg(argv, "--body"); win = arg(argv, "--window"); cache = arg(argv, "--cache", CACHE)
    if len(argv) != 2:
        print(__doc__, file=sys.stderr); return 2
    src, seq_dir = argv
    run = src if os.path.isdir(src) else None
    traj = f"{run}/traj.txt" if run else src
    seq = Seq(seq_dir, body)
    est = Est(traj, seq)
    lo, hi = (float(x) for x in win.split(":")) if win else (max(0.0, est.ts[0]), est.ts[-1])
    print(f"== {traj} {seq.name}: poses {len(est.t)}, t {est.ts[0]:.1f}-{est.ts[-1]:.1f} s, window {lo:.0f}-{hi:.0f} s")

    kfs = load_session(f"{run}/session.cbor")[0] if run and os.path.exists(f"{run}/session.cbor") else []
    if kfs:
        kfs.sort(key=lambda k: k["timestamp_sec"])
        st = {}
        for k in kfs: st.setdefault(segment_id(k), k["timestamp_sec"])
        starts = sorted(seq.rel(np.array(list(st.values())) * 1e9))
        src_st = "session"
    elif run and os.path.exists(f"{run}/stderr.txt") and seq.has_gt:
        starts = db.segment_starts_stderr(db.frame_classes(f"{run}/stderr.txt"), type("G", (), dict(cam=seq.cam, t0=seq.t0)))
        src_st = "stderr"
    else:
        starts, src_st = [], "none"
    starts = np.array(sorted(starts)) if len(starts) else np.array([])
    segs = lambda t: np.searchsorted(starts, t, side="right") if len(starts) else np.zeros_like(np.asarray(t), int)
    smry = summary_stderr(run)
    print(f"   segments {len(starts) if len(starts) else 1} ({src_st}): " + ", ".join(f"{x:.1f}" for x in starts)
          + f"; later_alignments {smry.get('later_alignments', '-')}")

    # keyframe biases
    if kfs:
        kt = np.array([seq.rel(round(k["timestamp_sec"] * 1e9)) for k in kfs]); kseg = segs(kt)
        bg = np.array([k["bias_gyro"] for k in kfs], float); ba = np.array([k["bias_accel"] for k in kfs], float)
        nz = (np.abs(bg).sum(1) > 0) & (np.abs(ba).sum(1) > 0)
        inw = (kt >= lo) & (kt <= hi)

    # b_true
    if seq.has_gt:
        bt = seq.btrue(cache)
        seq.d_ns = int(round(bt["d_ms"] * 1e6))
        seq.set_btrue(bt["lo"], bt["b"], bt["se"], bt["dM"]); seq.sig_gt = bt["sig_gt"]
        print(f"   b_true from GT ({'cache' if bt['cached'] else 'fitted'}): {bt['pairs']} pairs, dt p50 {bt['dt_p50']:.2f} s, "
              f"clock d {bt['d_ms']:+.2f} ms, body {os.path.relpath(bt['body'], P)} ({bt['c_vs_gyro_deg']:.2f} deg from the "
              f"gyro fit X used), residual sd {bt['sd_all']:.2e} rad/s per pair, standard error of b_true per piece p50 {bt['se_p50']:.2e} (s_fit)")
        ev_ = np.linalg.eigvalsh(0.5 * (bt["dM"] + bt["dM"].T))
        print(f"   gyro scale fit dM (symmetric part eigenvalues {ev_[0]:+.4f} {ev_[1]:+.4f} {ev_[2]:+.4f}); GT 5 s endpoint noise "
              f"sigma_gt {np.degrees(bt['sig_gt']):.3f} deg ({bt['sig_gt_n']} GT windows)")
        print("     lo_s   n   b_true (rad/s)                      sd        se")
        for l, b, s, e, n in zip(bt["lo"], bt["b"], bt["sd"], bt["se"], bt["n"]):
            if lo - PIECE_S < l <= hi:
                print(f"     {l:5.0f} {n:4d}  ({b[0]:+.2e}, {b[1]:+.2e}, {b[2]:+.2e})  {s:.2e}  {e:.2e}")
        bt_src, sd_all = "gt", bt["sd_all"]
    elif kfs:
        g = [np.median(bg[nz & (kt >= a) & (kt < a + BA_S)], axis=0) for a in np.arange(0, kt.max(), BA_S)
             if (nz & (kt >= a) & (kt < a + BA_S)).sum() >= 10]
        g = np.array(g); b0 = np.median(g, axis=0)
        sfit = float(1.4826 * np.sqrt((np.median(np.abs(g - b0), axis=0) ** 2).sum()))
        seq.set_btrue(np.array([0.0]), b0[None], np.array([sfit])); seq.sig_gt = 0.0
        print(f"   b_true WITHOUT GT: run median of {len(g)} 30 s medians of the estimate's bias, ({b0[0]:+.2e}, {b0[1]:+.2e}, "
              f"{b0[2]:+.2e}) rad/s, s_fit (spread) {sfit:.2e}; clock, gyro scale and GT endpoint noise not available")
        bt_src, sd_all = "run-median", sfit
    else:
        print("   no GT and no session: nothing to fit b_true against"); return 1

    S = dict(run=run or traj, seq=seq.name, window=f"{lo:.0f}:{hi:.0f}", btrue=bt_src, fit_sd=f"{sd_all:.2e}", fit_se=f"{np.nanmedian(seq.bt_sd):.2e}",
             segments=len(starts) if len(starts) else 1, later_alignments=smry.get("later_alignments", "-"))

    if kfs:
        # walk z
        k = np.arange(len(kt) - 1); dt = np.diff(kt)
        ok = (kseg[1:] == kseg[:-1]) & (dt > 1e-3) & (dt < 2.0) & nz[1:] & nz[:-1] & inw[1:] & inw[:-1]
        zg = np.linalg.norm(np.diff(bg, axis=0), axis=1)[ok] / (seq.sw_g * np.sqrt(3 * dt[ok]))
        za = np.linalg.norm(np.diff(ba, axis=0), axis=1)[ok] / (seq.sw_a * np.sqrt(3 * dt[ok]))
        print(f"   walk z ({ok.sum()} pairs): gyro p50 {np.median(zg):.2f} p90 {np.percentile(zg, 90):.2f}; "
              f"accel p50 {np.median(za):.2f} p90 {np.percentile(za, 90):.2f}")
        S.update(walk_g_p50=f"{np.median(zg):.2f}", walk_g_p90=f"{np.percentile(zg, 90):.2f}",
                 walk_a_p50=f"{np.median(za):.2f}", walk_a_p90=f"{np.percentile(za, 90):.2f}")
        # 30 s grid |ba| medians
        grid = [(a, float(np.median(np.linalg.norm(ba[(kt >= a) & (kt < a + BA_S) & nz], axis=1))))
                for a in np.arange(np.floor(lo / BA_S) * BA_S, hi, BA_S) if ((kt >= a) & (kt < a + BA_S) & nz).sum() >= 5]
        print("   |ba| 30 s medians (m/s^2): " + " ".join(f"{a:.0f}:{m:.2f}" for a, m in grid))
        g300 = [m for a, m in grid if 300 <= a < 480]
        if g300: S["ba30_max_300_480"] = f"{max(g300):.2f}"
        # trailing ba30 against the segment reference
        tq = np.arange(np.ceil(lo / WIN_S) * WIN_S, hi + 1e-9, WIN_S); exc = []
        ba30 = np.full((len(tq), 3), np.nan); qseg = segs(tq)
        for n, t in enumerate(tq):
            m = nz & (kt > t - BA_S) & (kt <= t) & (kseg == qseg[n]) & inw
            if m.sum() >= 10: ba30[n] = np.median(ba[m], axis=0)
        dev = np.full(len(tq), np.nan); env = np.full(len(tq), np.nan)
        for s_ in np.unique(qseg):
            m = (qseg == s_) & np.isfinite(ba30[:, 0]); mk = nz & inw & (kseg == s_)
            if m.sum() < 3 or mk.sum() < 2: continue
            ref = np.median(ba30[m], axis=0); T = kt[mk].max() - kt[mk].min()
            dev[m] = np.linalg.norm(ba30[m] - ref, axis=1); env[m] = 3 * seq.sw_a * np.sqrt(max(T, BA_S))
        over = np.isfinite(dev) & (dev > env)
        for n in np.nonzero(over)[0]:
            if exc and tq[n] - exc[-1][1] <= 10 + 1e-9 and qseg[n] == exc[-1][2]: exc[-1][1] = tq[n]; exc[-1][3] = max(exc[-1][3], dev[n])
            else: exc.append([tq[n], tq[n], qseg[n], dev[n]])
        fin = np.isfinite(dev)
        print(f"   ba30 - ref: max {np.nanmax(dev):.2f} m/s^2, bound {np.nanmin(env):.2f}-{np.nanmax(env):.2f}, over the bound "
              f"{100 * over[fin].mean():.0f} % of 5 s points; runs: " + ", ".join(f"{a:.0f}-{b:.0f} (seg {s_}, max {d:.2f})" for a, b, s_, d in exc))
        S.update(ba30_dev_max=f"{np.nanmax(dev):.2f}", ba30_bound=f"{np.nanmax(env):.2f}", ba30_over_pct=f"{100 * over[fin].mean():.0f}")
        # bg - b_true, 5 s medians
        dg = []
        for a in np.arange(np.floor(lo / WIN_S) * WIN_S, hi, WIN_S):
            m = nz & inw & (kt >= a) & (kt < a + WIN_S)
            if m.sum() >= 3: dg.append((a, float(np.linalg.norm(np.median(bg[m] - seq.b_true(kt[m]), axis=0)))))
        dv = np.array([d for _, d in dg])
        top = sorted(dg, key=lambda x: -x[1])[:5]
        print(f"   |bg - b_true| 5 s medians (rad/s): p50 {np.median(dv):.2e} p90 {np.percentile(dv, 90):.2e} max {dv.max():.2e}; "
              "largest at " + ", ".join(f"{a:.0f} s {d:.2e}" for a, d in top))
        S.update(dbg_p50=f"{np.median(dv):.2e}", dbg_max=f"{dv.max():.2e}")

    # M2
    tol = 0.06 if np.median(np.diff(est.ts)) < 0.1 else 0.25
    X, npair, hres, sig_run = hand_eye(est, seq, segs)
    print(f"   hand-eye estimate-to-IMU {angle_deg(X):.2f} deg ({npair} 1 s pairs, residual median {hres:.2f} deg)"
          + (f", {angle_deg(X.T @ seq.C):.2f} deg from the GT body C" if seq.has_gt else ""))
    m2 = m2_windows(est, seq, X, segs, lo, hi, tol, sig_run)
    nf = sum(r["flag"] for r in m2); nf1 = sum(r["flag1"] for r in m2)
    print(f"   sigma_psi terms: s_fit p50 {np.nanmedian(seq.bt_sd):.2e} rad/s, GT endpoint {np.degrees(seq.sig_gt):.3f} deg, "
          f"run 1 s hand-eye {np.degrees(sig_run):.3f} deg, scale |yaw| p50 {np.median([r['sc'] for r in m2]) if m2 else 0:.3f} deg")
    runs5 = flag_runs(m2)
    print(f"   M2: {len(m2)} windows of {WIN_S:.0f} s, 3 sigma_psi p50 {3 * np.median([r['sig'] for r in m2]):.2f} deg, "
          f"single windows over 3 sigma {nf1}, flagged (same-sign runs >= 2) {nf} ({100 * nf / max(len(m2), 1):.1f} %); |yaw| p50 {np.median([abs(r['yaw']) for r in m2]):.2f} deg; "
          f"flag runs: " + ", ".join(f"{a:.0f}-{b:.0f}" for a, b in runs5))
    for r in m2:
        if r["flag1"]: print(("   *" if r["flag"] else "    ") + f" {r['lo']:5.0f}-{r['hi']:<5.0f} yaw {r['yaw']:+7.2f} tilt {r['tilt']:6.2f} 3sig {3 * r['sig']:.2f} deg")
    S.update(m2_windows=len(m2), m2_single=nf1, m2_flagged=nf, sig_run_deg=f"{np.degrees(sig_run):.3f}", m2_flag_pct=f"{100 * nf / max(len(m2), 1):.1f}",
             m2_runs=";".join(f"{a:.0f}-{b:.0f}" for a, b in runs5) or "-")

    # events
    evs = []
    if seq.has_gt:
        gt = type("G", (), dict(t=seq.tg, p=seq.pg, cam=seq.cam, t0=seq.t0))
        m = (est.ts >= lo) & (est.ts <= hi)
        tr = db.Track(est.t[m], est.p[m], gt)
        rows = db.pieces(tr, 60, [x for x in starts if lo <= x <= hi]) if len(tr.t) > 20 else []
        evs, sig, thr = db.events(rows, tr) if rows else ([], 0, 0)
        print(f"   events (60 s pieces, threshold {thr:.2f} deg): {len(evs)}")
        S["events"] = len(evs)
        in_ev = lambda t: any(e["lo"] <= t < e["hi"] for e in evs)
        nonev = [r for r in rows if not any(e["lo"] <= 0.5 * (r["lo"] + r["hi"]) < e["hi"] for e in evs)]
        nonev_fl = [r for r in nonev if any(w["flag"] and r["lo"] <= w["lo"] + WIN_S / 2 < r["hi"] for w in m2)]
        print(f"   non-event 60 s pieces with an M2 flag: {len(nonev_fl)}/{len(nonev)}")
        S.update(nonev_pieces=len(nonev), nonev_flagged=len(nonev_fl))
        print("     lo-hi_s     kind       step_deg t_ev  class    M2sum  flagged  carried_deg  carried/step  ba_lead_s")
        for e in evs:
            t_ev = e["t"] if e["t"] is not None else e["lo"]
            cls = "restart" if any(e["lo"] - 30 <= s_ <= t_ev for s_ in starts) else "within"
            ws = [w for w in m2 if e["lo"] <= w["lo"] + WIN_S / 2 < e["hi"]]
            msum = sum(w["yaw"] for w in ws); mfl = sum(w["flag"] for w in ws)
            carried = np.nan; lead = None
            if kfs:
                es = segs(t_ev)
                mk = nz & (kt >= e["lo"]) & (kt < e["hi"]) & (kseg == es)
                if mk.sum() >= 3:
                    tk = kt[mk]; u = seq.up(seq.abs_ns(tk))
                    rate = -np.einsum("ni,ni->n", bg[mk] - seq.b_true(tk), u)
                    dtk = np.diff(tk); okd = dtk < 2.0
                    carried = float(np.degrees(np.sum((0.5 * (rate[1:] + rate[:-1]) * dtk)[okd])))
                for a, b, s_, d in exc:
                    if s_ == es and a <= t_ev and b >= t_ev - 30: lead = t_ev - a
            step = e["step"]
            share = 100 * carried * np.sign(msum) / abs(step) if np.isfinite(step) and step != 0 and np.isfinite(carried) else np.nan
            ls = f"{lead:.0f}" if lead is not None else "-"
            print(f"     {e['lo']:5.0f}-{e['hi']:<5.0f} {e['kind']:10s} {step:+7.1f} {t_ev:5.0f}  {cls:8s} {msum:+7.1f} {mfl:3d}/{len(ws):<3d}"
                  f"  {carried:+9.1f}  {share:9.0f} %  {ls:>6s}")
            evs_line = dict(lo=f"{e['lo']:.0f}", hi=f"{e['hi']:.0f}", kind=e["kind"], step=f"{step:+.1f}", t=f"{t_ev:.0f}", cls=cls,
                            m2sum=f"{msum:+.1f}", m2flag=f"{mfl}/{len(ws)}", carried=f"{carried:+.1f}", share=f"{share:.0f}", lead=ls)
            e["line"] = evs_line
    print("SUMMARY " + " ".join(f"{k}={v}" for k, v in S.items()))
    for e in evs:
        print("EVENT " + " ".join(f"{k}={v}" for k, v in e["line"].items()))
    return 0


def synth(argv):
    ramp = arg(argv, "--ramp")
    seq_dir, out = argv
    name = os.path.basename(seq_dir.rstrip("/"))
    g = np.loadtxt(f"{seq_dir}/ground_truth/pseudo_dense/{name}.txt", comments="#", ndmin=2)
    t0 = int(np.loadtxt(f"{seq_dir}/asl/{name}/aria/cam0/data.csv", delimiter=",", comments="#", usecols=0, dtype=np.int64, max_rows=1))
    ts = (g[:, 0] - t0) / 1e9; R = q2R(g[:, 4:8]); p = g[:, 1:4].copy()
    if ramp:
        a, b, deg = (float(x) for x in ramp.split(":"))
        th = np.radians(deg) * np.clip((ts - a) / (b - a), 0, 1)
        Rz = lambda x: np.array([[np.cos(x), -np.sin(x), 0], [np.sin(x), np.cos(x), 0], [0, 0, 1]])
        q = p.copy()
        for k in range(1, len(p)): q[k] = q[k - 1] + Rz(th[k]) @ (p[k] - p[k - 1])
        p = q; R = np.array([Rz(x) @ r for x, r in zip(th, R)])
    with open(out, "w") as f:
        for t, x, r in zip(g[:, 0].astype(np.int64), p, R):
            f.write(f"{t} {x[0]:.6f} {x[1]:.6f} {x[2]:.6f} " + " ".join(f"{v:.9f}" for v in R2q(r)) + "\n")
    print(f"wrote {len(p)} poses to {out}" + (f" with a {ramp} yaw ramp" if ramp else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
