"""coasts.py RUN_DIR GT NFRAMES [COASTED]: price the coasted poses of one run (stage-0 card S0-C2).

RUN_DIR/traj.txt is the scored file; COASTED (default RUN_DIR/coasted.txt, the `--trajectory-coasted` file)
holds the frames carried by the IMU prediction. Both are `t x y z qx qy qz qw` rows, t in ns. The scored poses are
the union of the two files, in time order, so the script works whether or not the run used
`--trajectory-include-coasted`. Prints:
  runs:  how many coast runs (consecutive coasted rows in that order), their length in frames (min / p50 / max),
         and the share of the scored poses that are coasted;
  ate:   SE3 ATE (m) of all scored poses, and of the scored poses without the coasted ones, both against GT
         with the same 10 ms association and Umeyama alignment as ate.py, plus the relative difference.
Card S0-C2 builds a knob if the ATE with coasts is over 10 % worse than without on any sequence.
"""
import os, sys, numpy as np


def load(p):
    a = np.loadtxt(p, comments="#", ndmin=2)
    return a[:, 0].astype(np.int64), a[:, 1:4]


def ate_se3(te, pe, tg, pg):
    idx = np.searchsorted(tg, te).clip(1, len(tg) - 1)
    idx = np.where(np.abs(tg[idx - 1] - te) < np.abs(tg[idx] - te), idx - 1, idx)
    ok = np.abs(tg[idx] - te) < 10_000_000
    if ok.sum() < 3:
        return float("nan"), int(ok.sum())
    A, B = pe[ok], pg[idx[ok]]
    ma, mb = A.mean(0), B.mean(0)
    X, Y = A - ma, B - mb
    U, _, Vt = np.linalg.svd(Y.T @ X / len(A))
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    return float(np.sqrt((((R @ X.T).T + mb - B) ** 2).sum(1).mean())), int(ok.sum())


def main(argv):
    if len(argv) < 4:
        sys.exit(__doc__)
    run, gt_path, n = argv[1], argv[2], int(argv[3])
    coasted_path = argv[4] if len(argv) > 4 else os.path.join(run, "coasted.txt")
    for p in (os.path.join(run, "traj.txt"), coasted_path, gt_path):
        if not os.path.isfile(p):
            sys.exit(f"coasts.py: missing {p}")
    ts, ps = load(os.path.join(run, "traj.txt"))
    tc, pc = load(coasted_path)
    tg, pg = load(gt_path)

    coast_set = set(tc.tolist())
    t_all = np.concatenate([ts, tc[~np.isin(tc, ts)]])
    p_all = np.concatenate([ps, pc[~np.isin(tc, ts)]])
    order = np.argsort(t_all, kind="stable")
    t_all, p_all = t_all[order], p_all[order]
    is_coast = np.array([t in coast_set for t in t_all.tolist()])

    lens, cur = [], 0
    for c in is_coast:
        if c:
            cur += 1
        elif cur:
            lens.append(cur)
            cur = 0
    if cur:
        lens.append(cur)
    name = os.path.basename(os.path.normpath(run))
    share = is_coast.mean() if len(is_coast) else float("nan")
    print(f"== {name}  scored={len(t_all)}/{n} coasted={int(is_coast.sum())} ({share:.1%})")
    if lens:
        print(f"   runs={len(lens)} length min={min(lens)} p50={int(np.median(lens))} max={max(lens)} frames")
    else:
        print("   runs=0")
    with_c, n_with = ate_se3(t_all, p_all, tg, pg)
    without, n_without = ate_se3(t_all[~is_coast], p_all[~is_coast], tg, pg)
    rel = (with_c - without) / without if without > 0 else float("nan")
    print(f"   ate_se3 with coasts={with_c:.3f}m (n={n_with}) without={without:.3f}m (n={n_without}) "
          f"with/without={rel:+.1%}" + ("  EXCEEDS 10%" if rel > 0.10 else ""))


main(sys.argv)
