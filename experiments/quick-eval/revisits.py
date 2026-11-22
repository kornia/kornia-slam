"""revisits.py SEQ|all [--session SESSION.cbor]: places the pseudo-dense GT path revisits (card AG, G10).

A revisit is a GT pose i and an earlier pose j within X m, with cam0 optical axes within 30 deg, after at least
L m of GT path between them. Per X in {1, 2, 3, 5, 10} m and L in {20, 50, 100} m it prints: revisit share (poses
with a partner), episodes (revisit poses merged across gaps up to 5 s), longest loop (largest path between a pair),
enclosed path share (union of the [j, i] path intervals over the total path) and the share of control-point
observations (`ground_truth/sparse/<seq>.json`, `images[*].timestamp`) and of control points with any observation
inside an enclosed interval. A loop closer can only correct the path between its two keyframes, so the enclosed
share bounds what loops can recover on a sequence.

Why X: verification is a cam0 PnP of the candidate's landmarks (`loop_closure/verification.rs:101-103`, at least
30 inliers in 6 of 16 cells, `:28-36`). ORB matches hold to about 30 deg of view change, so the largest useful
baseline is X = d_med * tan 30 deg, with d_med the median landmark depth. `--session` reads d_med from a session dump
and prints that X. The table spans X to 10 m to show the answer does not hang on it.
Why L: a loop is useful when the drift across it exceeds 3 loop sigmas, L_min = 3 * 0.10 m / drift rate
(`closer.rs:100`): 67 m at our 0.45 %, 107 m at ORB-SLAM3's 0.28 %. Covisible neighbours are excluded anyway
(`closer.rs:301-309`).

Conventions (checked on 1_19, cam0 axis vs GT velocity cos 0.85): GT rows are `t x y z qx qy qz qw`, world from
body; calibration `T_b_s.qvec` is w x y z. Times print as s from the first cam0 frame.
"""
import json, sys
import numpy as np

D = "/home/edgar/dev/.flux-connect-scratch/bench/lamaria/data"
GT_SEQS = ["R_11_5cp", "sequence_1_19", "sequence_1_20", "sequence_2_11", "sequence_2_12", "sequence_3_17",
           "sequence_3_18", "sequence_4_10", "sequence_4_11"]
XS, LS, VIEW_DEG, EPISODE_GAP_S = (1, 2, 3, 5, 10), (20, 50, 100), 30.0, 5.0


def q2R(q):
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def load(seq):
    g = np.loadtxt(f"{D}/{seq}/ground_truth/pseudo_dense/{seq}.txt")
    c = json.load(open(f"{D}/{seq}/pinhole_calibrations/{seq}.json"))["cam0"]["T_b_s"]["qvec"]
    axis_b = q2R([c[1], c[2], c[3], c[0]])[:, 2]
    view = np.array([q2R(q) @ axis_b for q in g[:, 4:8]])
    cam = np.loadtxt(f"{D}/{seq}/asl/{seq}/aria/cam0/data.csv", delimiter=",", comments="#", usecols=0, dtype=np.int64)
    t = (g[:, 0].astype(np.int64) - cam[0]) / 1e9
    p = g[:, 1:4]
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    sp = json.load(open(f"{D}/{seq}/ground_truth/sparse/{seq}.json"))
    obs = [((v["timestamp"] - cam[0]) / 1e9, v["control_point"]) for v in sp["images"].values()]
    return t, p, view, s, obs


def pairs(p, view, s, X, any_view=False, chunk=400):
    """Per pose i: the earliest partner j (largest path gap) within X m and the view cone; -1 if none."""
    n = len(p); far = np.full(n, -1); cosv = np.cos(np.radians(VIEW_DEG))
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        d = np.linalg.norm(p[a:b, None, :] - p[None, :b, :], axis=2) <= X
        if not any_view: d &= (view[a:b] @ view[:b].T) >= cosv
        d &= np.arange(b)[None, :] < np.arange(a, b)[:, None]
        has = d.any(1); j = d.argmax(1)
        far[a:b] = np.where(has, j, -1)
    return far


def pairs_L(p, view, s, X, L, chunk=400):
    """Per pose i: the earliest partner j with path gap >= L (the earliest gives the widest interval)."""
    n = len(p); far = np.full(n, -1); cosv = np.cos(np.radians(VIEW_DEG))
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        d = np.linalg.norm(p[a:b, None, :] - p[None, :b, :], axis=2) <= X
        d &= (view[a:b] @ view[:b].T) >= cosv
        d &= (s[a:b, None] - s[None, :b]) >= L
        has = d.any(1); j = d.argmax(1)
        far[a:b] = np.where(has, j, -1)
    return far


def union(iv):
    out = []
    for lo, hi in sorted(iv):
        if out and lo <= out[-1][1]: out[-1][1] = max(out[-1][1], hi)
        else: out.append([lo, hi])
    return out


def stats(t, s, obs, far):
    rev = np.nonzero(far >= 0)[0]
    if len(rev) == 0:
        return dict(share=0.0, episodes=0, longest=0.0, lt=None, enclosed=0.0, cp_obs=0.0, cp_any=0.0, iv=[])
    ep = 1 + int((np.diff(t[rev]) > EPISODE_GAP_S).sum())
    gaps = s[rev] - s[far[rev]]; k = rev[gaps.argmax()]
    iv = union([(far[i], i) for i in rev])
    enc = sum(s[hi] - s[lo] for lo, hi in iv) / s[-1]
    tiv = [(t[lo], t[hi]) for lo, hi in iv]
    inside = [any(lo <= to <= hi for lo, hi in tiv) for to, _ in obs]
    cps = {c for _, c in obs}; cin = {c for (to, c), f in zip(obs, inside) if f}
    return dict(share=len(rev) / len(t), episodes=ep, longest=float(gaps.max()), lt=(t[far[k]], t[k]), enclosed=enc,
                cp_obs=float(np.mean(inside)) if obs else 0.0, cp_any=len(cin) / max(len(cps), 1), iv=tiv)


def d_med_from_session(path, every=20):
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from worldrot import Cbor
    c = Cbor(path); _, n, i = c.head(0); lm = {}; kfs = []
    for _ in range(n):
        k, i = c.dec(i)
        if k == "landmarks":
            v, i = c.dec(i); lm = {x["idx"]: np.asarray(x["position_world"], float) for x in v}
        elif k == "keyframes":
            _, nk, i = c.head(i)
            for q in range(nk):
                if q % every:
                    i = c.skip(i); continue
                kf, i = c.dec(i); kfs.append(kf)
        else:
            i = c.skip(i)
    dep = []
    for kf in kfs:
        R = q2R(kf["pose_cw"]["rotation_xyzw"]); tr = np.asarray(kf["pose_cw"]["translation"], float)
        for o in kf["observations"]:
            if o["camera"] == 0 and o["landmark"] in lm: dep.append((R @ lm[o["landmark"]] + tr)[2])
    dep = np.array(dep); dep = dep[dep > 0]
    return float(np.median(dep)), len(dep)


def one(seq):
    t, p, view, s, obs = load(seq)
    print(f"== {seq}: GT poses {len(t)}, path {s[-1]:.0f} m, start-end {np.linalg.norm(p[-1] - p[0]):.0f} m, "
          f"{len(obs)} CP observations of {len({c for _, c in obs})} CPs")
    print("   X_m L_m  revisit%  episodes  longest_m  longest_t_s      enclosed%  cp_obs_in%  cp_in%")
    res = {}
    for X in XS:
        for L in LS:
            r = stats(t, s, obs, pairs_L(p, view, s, X, L)); res[(X, L)] = r
            lt = f"{r['lt'][0]:6.0f}-{r['lt'][1]:<6.0f}" if r["lt"] else "      -      "
            print(f"   {X:3d} {L:3d}  {100 * r['share']:7.1f}  {r['episodes']:8d}  {r['longest']:9.0f}  {lt}  "
                  f"{100 * r['enclosed']:9.1f}  {100 * r['cp_obs']:10.1f}  {100 * r['cp_any']:6.1f}")
    # Longest loop with no path floor except leaving the trivial neighbourhood (L 20), and with any view.
    av = pairs(p, view, s, 5, any_view=True); rv = np.nonzero(av >= 0)[0]
    any_long = float((s[rv] - s[av[rv]]).max()) if len(rv) else 0.0
    r35 = res[(3, 50)]
    print("   enclosed intervals X 3 L 50: " + (", ".join(f"{a:.0f}-{b:.0f} s" for a, b in r35["iv"]) or "none"))
    return dict(seq=seq, path=s[-1], se=np.linalg.norm(p[-1] - p[0]), long5=res[(5, 20)]["longest"],
                lt5=res[(5, 20)]["lt"], any5=any_long, enc=r35["enclosed"], cp=r35["cp_obs"], cpa=r35["cp_any"])


def main(argv):
    if not argv:
        print(__doc__, file=sys.stderr); return 2
    seqs = GT_SEQS if argv[0] == "all" else [argv[0]]
    if "--session" in argv:
        dm, n = d_med_from_session(argv[argv.index("--session") + 1])
        print(f"d_med {dm:.2f} m over {n} cam0 observations; X = d_med * tan 30 deg = {dm * np.tan(np.radians(30)):.2f} m")
    rows = [one(q) for q in seqs]
    print("\nsummary (section 0 table): longest loop at X 5 m, L 20 m; enclosed at X 3 m, L 50 m")
    print("| seq | path m | start-end m | longest loop, X 5 m | longest, X 5 m any view | loop-enclosed path, X 3 m, L 50 m | CP obs inside | CPs inside |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        lt = f" (t {r['lt5'][0]:.0f} to {r['lt5'][1]:.0f} s)" if r["lt5"] else ""
        print(f"| {r['seq'].replace('sequence_', '')} | {r['path']:.0f} | {r['se']:.0f} | {r['long5']:.0f} m{lt} | "
              f"{r['any5']:.0f} m | {100 * r['enc']:.1f} % | {100 * r['cp']:.1f} % | {100 * r['cpa']:.1f} % |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
