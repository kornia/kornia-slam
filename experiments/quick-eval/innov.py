"""innov.py STDERR [--fps 20]: the pose innovation statistic of one DEBUG=1 run (card AC-2b, G48).

Reads the `[nis] frame= nis6= nis_rot= nis_trans= c= inl= lost= verdict=` lines of `pose_gate: Innovation`, the
`[ i] Status kf= pts=` lines (lossev.py's frame classes) and the `[summary]` lines.

Healthy frames: 100+ inliers and not lost. Event frames: judged frames within 3 s before the onset of a primary loss,
a coasted episode of at least 0.5 s that starts at least 3 s of frames after the last bootstrap frame (jumps.py's
primary age, read from frame counts at --fps). NIS/c needs the scale, so warm-up frames (c=nan) are left out of it.

Prints:
  m0:     judged against judged + unavailable, from the [nis] lines and from the [summary] loss counters, and against
          tracked frames.
  nis:    quantiles of NIS/c (6-dof) and of the raw 3-dof parts for healthy and event frames; K5m's ratio
          (healthy p99.9 / event p10) for the 6-dof and each part (K7 reads the parts).
  share:  would-reject share (verdict over or reject) of judged frames, all and healthy.
  losses: per primary loss, onset frame, coast frames, judged and would-reject frames in the 3 s before it.
"""
import os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lossev import classes  # noqa: E402

NIS = re.compile(r"\[nis\] frame=(\d+) nis6=(\S+) nis_rot=(\S+) nis_trans=(\S+) c=(\S+) inl=(\d+) lost=(\d) "
                 r"verdict=(\w+)")
KV = re.compile(r"(\w+)=(\S+)")
# chi2 at 99.9 %: 6 dof (the gate), 3 dof (each logged part).
CHI2_6, CHI2_3 = 22.458, 16.266
HEALTHY_INLIERS = 100
BEFORE_SEC, MIN_AGE_SEC, MIN_COAST_SEC = 3.0, 3.0, 0.5


def q(x, ps=(10, 50, 90, 99, 99.9)):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if not len(x): return "n=0"
    return f"n={len(x)} " + " ".join(f"p{p:g} {np.percentile(x, p):.2f}" for p in ps)


def primary_losses(cls, fps):
    out, frames = [], sorted(cls)
    last_b, f, hi = frames[0] - 1, frames[0], frames[-1] + 1
    while f < hi:
        c = cls.get(f)
        if c == "B": last_b = f
        if c != "C": f += 1; continue
        j = f
        while j < hi and cls.get(j) == "C": j += 1
        if (f - last_b) >= MIN_AGE_SEC * fps and (j - f) >= MIN_COAST_SEC * fps: out.append((f, j))
        f = j
    return out


def main(path, fps):
    rows, summary = {}, {}
    for line in open(path, errors="replace"):
        m = NIS.search(line)
        if m:
            rows[int(m[1])] = (float(m[2]), float(m[3]), float(m[4]), float(m[5]), int(m[6]), int(m[7]), m[8])
        elif line.startswith("[summary]"):
            summary.update(dict(KV.findall(line)))
    cls = classes(path)
    judged = {f: r for f, r in rows.items() if r[6] in ("ok", "over", "reject")}
    na = sum(r[6] in ("na", "warm") for r in rows.values())
    print(f"m0: [nis] judged {len(judged)} unavailable {na} "
          f"({100 * len(judged) / max(len(judged) + na, 1):.1f} %); summary judged "
          f"{summary.get('pose_innovation_judged')} unavailable {summary.get('pose_innovation_unavailable')} "
          f"over {summary.get('pose_innovation_over')} rejects {summary.get('pose_innovation_rejects')} "
          f"scale_max {summary.get('pose_innovation_scale_max')}; tracked {summary.get('tracked')}")
    losses = primary_losses(cls, fps) if cls else []
    before = int(round(BEFORE_SEC * fps))
    event = set()
    for i, _ in losses: event.update(range(i - before, i))
    healthy = [r for r in judged.values() if r[4] >= HEALTHY_INLIERS and r[5] == 0]
    ev = [judged[f] for f in sorted(event) if f in judged]
    scaled = lambda rs: [r[0] / r[3] for r in rs]
    print(f"nis: healthy (inl >= {HEALTHY_INLIERS}, not lost) and event (3 s before {len(losses)} primary losses)")
    for name, k, f in (("nis6/c", None, scaled), ("nis_rot", 1, None), ("nis_trans", 2, None)):
        get = f if f else (lambda rs, k=k: [r[k] for r in rs])
        h, e = np.asarray(get(healthy), float), np.asarray(get(ev), float)
        h, e = h[np.isfinite(h)], e[np.isfinite(e)]
        print(f"  {name} healthy {q(h)}")
        print(f"  {name} event   {q(e)}")
        if len(h) and len(e):
            hp, ep = np.percentile(h, 99.9), np.percentile(e, 10)
            print(f"  {name} K5m: healthy p99.9 {hp:.2f} / event p10 {ep:.2f} = {hp / ep:.3f} "
                  f"({'passes' if hp < ep / 10 else 'fails'}: under 1/10 needed)")
    over = lambda rs: sum(r[6] in ("over", "reject") for r in rs)
    print(f"share: would-reject {over(judged.values())} of {len(judged)} judged "
          f"({100 * over(judged.values()) / max(len(judged), 1):.2f} %); healthy {over(healthy)} of {len(healthy)}; "
          f"3-dof parts over {CHI2_3} raw: rot {sum(r[1] > CHI2_3 for r in judged.values())} "
          f"trans {sum(r[2] > CHI2_3 for r in judged.values())}")
    print("losses: onset coast_frames judged_3s over_3s max_nis6/c_3s")
    for i, j in losses:
        rs = [judged[f] for f in range(i - before, i) if f in judged]
        mx = max((r[0] / r[3] for r in rs if np.isfinite(r[0] / r[3])), default=float("nan"))
        print(f"  {i} {j - i} {len(rs)} {over(rs)} {mx:.1f}")


if __name__ == "__main__":
    args = sys.argv[1:]
    fps = 20.0
    if "--fps" in args:
        i = args.index("--fps"); fps = float(args[i + 1]); del args[i:i + 2]
    if len(args) != 1:
        print(__doc__, file=sys.stderr); sys.exit(2)
    main(args[0], fps)
