"""driftbudget.py RUN_DIR|TRAJ SEQ [--emit DIR] [--grid T0]: where a run's trajectory error sits (card AG).

RUN_DIR holds traj.txt, and optionally stderr.txt and session.cbor (H0 gate runs keep it; BASE runs do not). A file
argument is scored alone (ORB-SLAM3 estimate.txt, a synthetic test trajectory). Prints:
  global:   one Sim3 (Umeyama) of all poses to the pseudo-dense GT, as segsim3.py.
  pieces:   Sim3 per 120/60/30/10 s piece (grid from the first cam0 frame, cut at session segment starts; pieces
            whose matched poses span under half the piece are dropped): RMS,
            scale, path ratio ours/GT, and the heading error (yaw about GT up) and tilt against the segment's first
            piece. Heading error is yaw(R_0 R_k^T), with R_k the piece's estimate-to-GT rotation: positive when our
            path turned left of GT.
  events:   60 s piece steps with |d yaw| > 3 sqrt(2) sigma, or piece scale outside 0.8-1.2. sigma is the sequence's
            piece-fit noise, the median over its 60 s pieces of atan(RMS / gyration radius of the GT piece), so the
            threshold comes from the fit and not from a constant; the median keeps the event pieces from inflating
            it. Adjacent flagged steps merge into one event, whose step runs from the clean piece before to the clean
            piece after. Its time is refined to 1 s by the split that minimises the two-piece Sim3 residual.
  keyframes: the same pieces and events on the session's final keyframe centres (the map, not the causal output).
  windows:  per 60 s piece with a session: |bg| max, |ba| median, speed ratio, and the M1 bias step z-score
            max |b(t + 5 s) - b(t)| / (sigma_walk sqrt(dt)), sigma_walk from imu0 in pinhole_calibrations/<seq>.json.
            Pairs cross no segment start and skip a still-zero bias (before inertial init).
  coasts:   runs of coasted frames (`[idx] Skipped` with kf != idx, lossev.py's class C) from stderr.txt.
--emit DIR writes the split lists for oracle.py (O1.txt segment starts, O3.txt plus event times, O4-60.txt, O4-30.txt)
and coasted.txt (coasted pose timestamps, ns). Splits are s from the first cam0 frame, comma-separated.
--grid T0 offsets the piece grid by T0 s (to check an event is not a grid artefact).
"""
import json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from worldrot import kf_centre, load_session, segment_id  # noqa: E402

D = "/home/edgar/dev/.flux-connect-scratch/bench/lamaria/data"
WIDTHS = (120, 60, 30, 10)
# Pseudo-dense GT is cm-level; without this floor a GT-as-estimate run has zero fit noise and flags round-off.
RMS_FLOOR_M = 0.02
SCALE_BAND = (0.8, 1.2)
BIAS_DT_S = 5.0
STATUS = re.compile(r"^\[\s*(\d+)\] (\w+)\s+kf=(\d+)")


def q2R(q):
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def assoc(te, tg):
    idx = np.searchsorted(tg, te).clip(1, len(tg) - 1)
    idx = np.where(np.abs(tg[idx - 1] - te) < np.abs(tg[idx] - te), idx - 1, idx)
    return idx, np.abs(tg[idx] - te) < 10_000_000


def umeyama(A, B):
    ma, mb = A.mean(0), B.mean(0); X, Y = A - ma, B - mb
    U, Dg, Vt = np.linalg.svd(Y.T @ X / len(A)); S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0: S[2, 2] = -1
    R = U @ S @ Vt; s = (Dg * np.diag(S)).sum() / X.var(0).sum()
    res = s * A @ R.T + (mb - s * R @ ma) - B
    return s, R, float(np.sqrt(np.mean(np.sum(res ** 2, 1)))), float(np.sum(res ** 2))


def yaw_tilt(M):
    x = M @ np.array([1.0, 0, 0]); z = M @ np.array([0, 0, 1.0])
    return float(np.degrees(np.arctan2(x[1], x[0]))), float(np.degrees(np.arccos(np.clip(z[2], -1, 1))))


def path(P):
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) if len(P) > 1 else 0.0


class Gt:
    def __init__(self, seq):
        g = np.loadtxt(f"{D}/{seq}/ground_truth/pseudo_dense/{seq}.txt")
        self.t = g[:, 0].astype(np.int64); self.p = g[:, 1:4]
        cam = np.loadtxt(f"{D}/{seq}/asl/{seq}/aria/cam0/data.csv", delimiter=",", comments="#", usecols=0,
                         dtype=np.int64)
        self.cam = cam; self.t0 = cam[0]
        imu = json.load(open(f"{D}/{seq}/pinhole_calibrations/{seq}.json"))["imu0"]
        self.sg, self.sa = imu["gyro_bias_random_walk_sigma"], imu["acc_bias_random_walk_sigma"]


class Track:
    """Estimate positions matched to GT samples, times in s from the first cam0 frame."""

    def __init__(self, te, pe, gt):
        idx, ok = assoc(te, gt.t)
        self.t = ((te - gt.t0) / 1e9)[ok]; self.p = pe[ok]; self.g = gt.p[idx[ok]]


def pieces(tr, width, starts, offset=0.0):
    """Rows per piece: (lo, hi, n, s, R, rms, gyr, path_ratio, seg). Grid cut at segment starts."""
    edges = sorted(set(list(np.arange(offset, tr.t[-1] + width, width)) + [x for x in starts if x > tr.t[0]]))
    seg_of = lambda x: int(np.searchsorted(starts, x, side="right")) if len(starts) else 0
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (tr.t >= lo) & (tr.t < hi)
        # A stub piece (sequence end, segment cut) fits rotation badly and fakes a yaw step.
        if m.sum() < 10 or np.ptp(tr.t[m]) < 0.5 * min(width, hi - lo): continue
        A, B = tr.p[m], tr.g[m]
        if np.ptp(B, axis=0).max() < 1.0: continue
        s, R, rms, _ = umeyama(A, B)
        gyr = float(np.sqrt(np.mean(np.sum((B - B.mean(0)) ** 2, 1))))
        rows.append(dict(lo=lo, hi=hi, n=int(m.sum()), s=s, R=R, rms=rms, gyr=gyr,
                         ratio=path(A) / max(path(B), 1e-9), seg=seg_of(0.5 * (lo + hi))))
    first = {}
    for r in rows:
        R0 = first.setdefault(r["seg"], r["R"])
        r["yaw"], r["tilt"] = yaw_tilt(R0 @ r["R"].T)
    return rows


def sigma_deg(r):
    return float(np.degrees(np.arctan(max(r["rms"], RMS_FLOOR_M) / max(r["gyr"], 1e-6))))


def refine(tr, lo, hi):
    """Split time in [lo, hi] (1 s grid) minimising the summed residual of the two Sim3 fits on either side."""
    best = (np.inf, 0.5 * (lo + hi))
    a, b = lo, hi
    for tau in np.arange(np.ceil(lo + 1), hi - 1 + 1e-9, 1.0):
        m1 = (tr.t >= a - 30) & (tr.t < tau); m2 = (tr.t >= tau) & (tr.t < b + 30)
        if m1.sum() < 10 or m2.sum() < 10: continue
        sse = umeyama(tr.p[m1], tr.g[m1])[3] + umeyama(tr.p[m2], tr.g[m2])[3]
        if sse < best[0]: best = (sse, float(tau))
    return best[1]


def events(rows, tr):
    """Event dicts from 60 s piece rows: span, kind, yaw step (clean piece to clean piece), refined time."""
    if not rows: return [], 0.0, 0.0
    sig = float(np.median([sigma_deg(r) for r in rows])); thr = 3 * np.sqrt(2) * sig
    flag_step = [False] + [rows[k]["seg"] == rows[k - 1]["seg"] and abs(rows[k]["yaw"] - rows[k - 1]["yaw"]) > thr
                           for k in range(1, len(rows))]
    flag_scale = [not (SCALE_BAND[0] <= r["s"] <= SCALE_BAND[1]) for r in rows]
    out, k = [], 0
    while k < len(rows):
        if not (flag_step[k] or flag_scale[k]):
            k += 1; continue
        j = k
        while j + 1 < len(rows) and rows[j + 1]["seg"] == rows[k]["seg"] and (flag_step[j + 1] or flag_scale[j + 1]):
            j += 1
        # A yaw group [k, j] of flagged steps: before = k - 1, after = j. A scale piece is itself the event.
        first_bad = k if flag_scale[k] and not flag_step[k] else k - 1
        before = first_bad - 1 if flag_scale[k] and not flag_step[k] else k - 1
        after = j + 1 if flag_scale[j] and j + 1 < len(rows) and rows[j + 1]["seg"] == rows[k]["seg"] else j
        ok_b = before >= 0 and rows[before]["seg"] == rows[k]["seg"]
        step = rows[after]["yaw"] - rows[before]["yaw"] if ok_b else float("nan")
        kinds = (["yaw"] if any(flag_step[k:j + 1]) else []) + (["scale"] if any(flag_scale[k:j + 1]) else [])
        lo = rows[max(first_bad, 0)]["lo"]; hi = rows[j]["hi"]
        ev = dict(lo=lo, hi=hi, kind="+".join(kinds), step=step, smin=min(r["s"] for r in rows[k:j + 1]),
                  smax=max(r["s"] for r in rows[k:j + 1]))
        ev["t"] = refine(tr, lo, hi) if "yaw" in kinds else None
        # M1 windows: scale pieces, pieces between merged steps, and the piece holding the step time.
        wins = {rows[i]["lo"] for i in range(k, j + 1) if flag_scale[i]} | {rows[i]["lo"] for i in range(k, j)}
        wins |= {r["lo"] for r in rows if ev["t"] is not None and r["lo"] <= ev["t"] < r["hi"]}
        ev["wins"] = wins
        out.append(ev); k = j + 1
    return out, sig, thr


def segment_starts_session(kfs, t0):
    st = {}
    for k in kfs: st.setdefault(segment_id(k), k["timestamp_sec"])
    return sorted((v * 1e9 - t0) / 1e9 for v in st.values())


def frame_classes(stderr):
    cls = {}
    for line in open(stderr, errors="replace"):
        m = STATUS.match(line)
        if m:
            i, s, kf = int(m[1]), m[2], int(m[3])
            cls[i] = "T" if s != "Skipped" else ("B" if kf == i else "C")
    for i in sorted(cls):
        if cls[i] == "B" and cls.get(i - 1) == "C" and cls.get(i + 1) == "C": cls[i] = "C"
    return cls


def segment_starts_stderr(cls, gt, min_run=3):
    """First tracked frame after each bootstrap run: the segment starts when no session is kept.

    A single B frame is a keyframe inserted while coasting, not a restart (1_19: restarts stall 9-15 frames).
    """
    out, run = [], 0
    for i in sorted(cls):
        if cls[i] == "B": run += 1; continue
        if cls[i] == "T" and run >= min_run: out.append((gt.cam[i] - gt.t0) / 1e9)
        run = 0
    return out


def runs(frames):
    out = []
    for f in frames:
        if out and f == out[-1][1] + 1: out[-1][1] = f
        else: out.append([f, f])
    return out


def print_pieces(label, tr, starts, offset):
    print(f"-- {label}")
    s, R, rms, _ = umeyama(tr.p, tr.g)
    print(f"   global Sim3: n {len(tr.t)}, RMS {rms:.2f} m, scale {s:.3f}, GT path {path(tr.g):.0f} m")
    by_w = {}
    for w in WIDTHS:
        rows = pieces(tr, w, starts, offset); by_w[w] = rows
        print(f"   {w} s pieces: lo-hi_s  n  RMS_m  scale  ours/GT  yaw_deg  tilt_deg  sigma_deg  seg")
        if w == 10:
            y = np.array([r["yaw"] for r in rows]); rr = np.array([r["rms"] for r in rows])
            print(f"     ({len(rows)} pieces) RMS median {np.median(rr):.2f} p90 {np.percentile(rr, 90):.2f} m; "
                  f"yaw min {y.min():.1f} max {y.max():.1f} deg")
            continue
        for r in rows:
            print(f"     {r['lo']:5.0f}-{r['hi']:<5.0f} {r['n']:5d} {r['rms']:6.2f} {r['s']:6.3f} {r['ratio']:6.2f} "
                  f"{r['yaw']:8.1f} {r['tilt']:7.1f} {sigma_deg(r):8.2f} {r['seg']:4d}")
    ev, sig, thr = events(by_w[60], tr)
    print(f"   events (60 s): sigma {sig:.2f} deg, yaw threshold {thr:.2f} deg, scale band {SCALE_BAND}: {len(ev)}")
    for e in ev:
        ts = f"{e['t']:.0f}" if e["t"] is not None else "-"
        print(f"     {e['lo']:5.0f}-{e['hi']:<5.0f} {e['kind']:10s} yaw step {e['step']:+6.1f} deg  t {ts} s  "
              f"scale {e['smin']:.2f}-{e['smax']:.2f}")
    return by_w, ev


def bias_windows(kfs, gt, rows60, tr, starts, ev):
    t = np.array([(round(k["timestamp_sec"] * 1e9) - gt.t0) / 1e9 for k in kfs])
    o = np.argsort(t); t = t[o]; ks = [kfs[i] for i in o]
    bg = np.array([k["bias_gyro"] for k in ks], float); ba = np.array([k["bias_accel"] for k in ks], float)
    seg = np.searchsorted(starts, t, side="right")
    j = np.searchsorted(t, t + BIAS_DT_S).clip(0, len(t) - 1)
    dt = t[j] - t
    ok = (j > np.arange(len(t))) & (seg[j] == seg) & (np.abs(bg).sum(1) > 0) & (dt < 2 * BIAS_DT_S)
    zg = np.where(ok, np.linalg.norm(bg[j] - bg, axis=1) / (gt.sg * np.sqrt(np.maximum(dt, 1e-3))), 0)
    za = np.where(ok, np.linalg.norm(ba[j] - ba, axis=1) / (gt.sa * np.sqrt(np.maximum(dt, 1e-3))), 0)
    in_ev = lambda lo, hi: any(lo in e["wins"] for e in ev)
    print(f"   windows (60 s), M1 z over {BIAS_DT_S:.0f} s, sigma_walk gyro {gt.sg:.3g} accel {gt.sa:.3g}:")
    print("     lo-hi_s  event  |bg|max  |ba|med  speed  z_gyro  z_accel  obs_min")
    out = []
    for r in rows60:
        m = (t >= r["lo"]) & (t < r["hi"])
        if not m.any(): continue
        obs = min(k.get("observations_n", 0) for k, f in zip(ks, m) if f)
        e = in_ev(r["lo"], r["hi"])
        w = dict(lo=r["lo"], hi=r["hi"], ev=e, bg=float(np.linalg.norm(bg[m], axis=1).max()),
                 ba=float(np.median(np.linalg.norm(ba[m], axis=1))), zg=float(zg[m].max()), za=float(za[m].max()))
        out.append(w)
        print(f"     {r['lo']:5.0f}-{r['hi']:<5.0f} {'EVENT' if e else '     '} {w['bg']:7.3f} {w['ba']:7.2f} "
              f"{r['ratio']:6.2f} {w['zg']:7.1f} {w['za']:7.1f} {obs:6d}")
    evw = [w for w in out if w["ev"]]; nev = [w for w in out if not w["ev"]]
    hit = lambda w: max(w["zg"], w["za"]) > 10
    print(f"   M1: event windows with z > 10: {sum(map(hit, evw))}/{len(evw)}; "
          f"non-event windows with z > 10: {sum(map(hit, nev))}/{len(nev)}")
    return out


def main(argv):
    if len(argv) < 2:
        print(__doc__, file=sys.stderr); return 2
    src, seq = argv[0], argv[1]
    emit = argv[argv.index("--emit") + 1] if "--emit" in argv else None
    offset = float(argv[argv.index("--grid") + 1]) if "--grid" in argv else 0.0
    run = src if os.path.isdir(src) else None
    traj = f"{run}/traj.txt" if run else src
    gt = Gt(seq)
    a = np.loadtxt(traj, comments="#", ndmin=2); te = a[:, 0].astype(np.int64)
    tr = Track(te, a[:, 1:4], gt)
    kfs = []
    if run and os.path.exists(f"{run}/session.cbor"): kfs, _ = load_session(f"{run}/session.cbor")
    cls = frame_classes(f"{run}/stderr.txt") if run and os.path.exists(f"{run}/stderr.txt") else {}
    st_err = segment_starts_stderr(cls, gt) if cls else []
    starts = segment_starts_session(kfs, gt.t0) if kfs else st_err
    print(f"== {traj} {seq}: poses {len(te)}, matched to GT {len(tr.t)}, t {tr.t[0]:.0f}-{tr.t[-1]:.0f} s")
    print(f"   segment starts ({'session' if kfs else 'stderr' if cls else 'none'}): "
          + ", ".join(f"{x:.1f}" for x in starts))
    if kfs and cls: print("   segment starts from stderr (check): " + ", ".join(f"{x:.1f}" for x in st_err))
    by_w, ev = print_pieces("trajectory", tr, starts, offset)
    if kfs:
        tk = np.array([round(k["timestamp_sec"] * 1e9) for k in kfs], dtype=np.int64)
        pk = np.array([kf_centre(k) for k in kfs]); o = np.argsort(tk)
        print_pieces(f"session final keyframes ({len(kfs)})", Track(tk[o], pk[o], gt), starts, offset)
        bias_windows(kfs, gt, by_w[60], tr, starts, ev)
    coasted = sorted(i for i, c in cls.items() if c == "C")
    if cls:
        rr = runs(coasted); long = [r for r in rr if r[1] - r[0] + 1 >= 10]
        print(f"   coasts: {len(coasted)} coasted frames in {len(rr)} runs, longest {max((r[1] - r[0] + 1 for r in rr), default=0)}"
              f" frames; runs >= 10 frames: " + ", ".join(f"{(gt.cam[r[0]] - gt.t0) / 1e9:.1f} s x{r[1] - r[0] + 1}"
                                                          for r in long))
    if emit:
        os.makedirs(emit, exist_ok=True)
        ev_t = [e["t"] for e in ev if e["t"] is not None] + [x for e in ev if "scale" in e["kind"] for x in (e["lo"], e["hi"])]
        lists = {"O1": starts, "O3": sorted(set(starts + ev_t)),
                 "O4-60": sorted(set(starts + list(np.arange(60.0, tr.t[-1], 60.0)))),
                 "O4-30": sorted(set(starts + list(np.arange(30.0, tr.t[-1], 30.0))))}
        for k, v in lists.items():
            open(f"{emit}/{k}.txt", "w").write(",".join(f"{x:.3f}" for x in v) + "\n")
        open(f"{emit}/coasted.txt", "w").write("".join(f"{gt.cam[i]}\n" for i in coasted))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
