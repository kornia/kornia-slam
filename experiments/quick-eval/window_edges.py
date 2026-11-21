"""window_edges.py STDERR [STDERR ...]: first-iteration IMU edges of each local VI-BA, by position (card R).

Reads the `[kf] frame=`, `[vi_ba] edge` and `[vi_ba] WINDOW` debug lines. The edge lines before a
WINDOW line belong to that solve. n_new is the number of keyframes inserted after the previous
solve's newest keyframe, up to this solve's newest. Edges are classed by their end slot:
  old       both ends at or before the previous newest keyframe;
  crossing  previous newest -> first new keyframe;
  new       both ends new.
A keyframe absorbed during a slow solve (n_new >= 2) starts the next window from its seed, so
crossing and new edges show how far that seed is from the IMU chain.
"""
import re, sys, numpy as np

KF = re.compile(r"^\[kf\] frame=(\d+) grown=")
EDGE = re.compile(r"^\[vi_ba\] edge (\d+)\D+(\d+)\s.*\|r_v\|=(\S+)\s+\|r_p\|=(\S+).*hw=(\S+)")
WIN = re.compile(r"^\[vi_ba\] WINDOW kf=(\d+) .*?\bshift=(\S+)")


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def solves(path):
    # A synchronous solve logs before its keyframe's `[kf]` line, so count over the whole file.
    lines = open(path, errors="replace").read().splitlines()
    kf_frames = sorted({int(m[1]) for line in lines if (m := KF.match(line))})
    edges, prev, out = [], None, []
    skipped = {"first": 0, "no_edges": 0, "not_newer": 0, "edge_lines_unparsed": 0}
    for line in lines:
        if m := EDGE.match(line):
            edges.append((int(m[1]), int(m[2]), float(m[3]), float(m[4]), float(m[5])))
        elif line.startswith("[vi_ba] edge "):
            skipped["edge_lines_unparsed"] += 1
        elif m := WIN.match(line):
            newest, shift = int(m[1]), float(m[2])
            if prev is None:
                skipped["first"] += 1
            elif not edges:
                skipped["no_edges"] += 1
            elif newest <= prev:
                # A re-seed or a restart moved the newest keyframe back: n_new would be 0 and every edge
                # would land in the wrong class, so leave the solve out and say so.
                skipped["not_newer"] += 1
            else:
                n_new = sum(prev < f <= newest for f in kf_frames)
                out.append((n_new, shift, edges))
            prev, edges = newest, []
    return out, skipped


def classify(n_new, edges):
    last = max(j for _, j, *_ in edges)
    first_new = last - n_new + 1
    old, cross, new = [], [], []
    for i, j, rv, rp, hw in edges:
        (new if i >= first_new else cross if j == first_new else old).append((rv, rp, hw))
    return old, cross, new


def report(path):
    s, skipped = solves(path)
    print(f"== {path}: left out " + " ".join(f"{k}={v}" for k, v in skipped.items()))
    if not s:
        print(f"== {path}: no solves")
        return
    old, cross, new, cross_multi = [], [], [], []
    for n_new, _, edges in s:
        o, c, n = classify(n_new, edges)
        old += o
        cross += c
        new += n
        if n_new >= 2:
            cross_multi += c
    nn = np.array([n for n, _, _ in s])
    sh = np.array([x for _, x, _ in s])
    col = lambda rows, k: np.array([r[k] for r in rows])
    print(
        f"   solves={len(s)} n_new mean={nn.mean():.2f} p50={pct(nn, 50):.0f} max={nn.max()} "
        f"large(>30cm)={(sh > 0.3).sum()} shift p50={pct(sh, 50) * 100:.1f}cm"
    )
    print(f"   old      n={len(old)} r_v p50={pct(col(old, 0), 50):.4f}")
    print(
        f"   crossing n={len(cross)} r_v p50={pct(col(cross, 0), 50):.4f} r_p p50={pct(col(cross, 1), 50):.4f} "
        f"p90={pct(col(cross, 1), 90):.4f} hw p50={pct(col(cross, 2), 50):.4f}"
        f" | n_new>=2: n={len(cross_multi)} r_v p50={pct(col(cross_multi, 0), 50):.4f}"
    )
    print(
        f"   new      n={len(new)} r_v p50={pct(col(new, 0), 50):.4f} r_p p50={pct(col(new, 1), 50):.4f} "
        f"hw p50={pct(col(new, 2), 50):.4f}"
    )
    bins = [(1, 2), (3, 4), (5, 6), (7, 10**9)]
    parts = []
    for lo, hi in bins:
        b = sh[(nn >= lo) & (nn <= hi)]
        parts.append(f"{lo}-{hi if hi < 10**9 else ''}: n={len(b)} p50={pct(b, 50) * 100:.1f}cm large={(b > 0.3).sum()}")
    print("   shift by n_new  " + "; ".join(parts))


for p in sys.argv[1:]:
    report(p)
