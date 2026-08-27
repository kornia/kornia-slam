#!/usr/bin/env python3
"""Run, score and compare the EuRoC stereo-inertial benchmark.

    bench.py run      [--seqs ...] [--runs N] [--configs ...] [--orb3] --out DIR
    bench.py score    DIR [--baseline FILE]
    bench.py compare  DIR --baseline FILE
    bench.py baseline DIR --save FILE

`run` is resumable: a run directory that already holds a trajectory is skipped,
so an interrupted sweep continues where it stopped.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from traj_eval import EUROC, SEQS, SHORT, evaluate, require_euroc  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
ORB3 = Path(os.environ.get("KORNIA_BENCH_ORB3", REPO/"extra"/"ORB_SLAM3"))
VOCAB = Path(os.environ.get("KORNIA_BENCH_VOCAB", ORB3/"Vocabulary"/"ORBvoc.txt"))
BIN = Path(os.environ.get("KORNIA_BENCH_BIN", REPO/"target"/"release"/"kornia-slam"))

# Sequence dir -> the stem ORB-SLAM3's EuRoC timestamp files use.
TS_STEM = {s: s.split("_")[0] + s.split("_")[1] for s in SEQS}

# kornia-slam configurations. `si_loop` is the default the report quotes; the
# others exist to attribute error to a subsystem rather than guess at it.
CONFIGS = {
    "si_loop":   dict(flags=["--stereo", "--imu"], vocab=True,
                      desc="stereo-inertial + loop closure"),
    "si_noloop": dict(flags=["--stereo", "--imu"], vocab=False,
                      desc="stereo-inertial, loop closure off"),
    "s_only":    dict(flags=["--stereo"], vocab=False,
                      desc="stereo only, no IMU"),
    "mi_loop":   dict(flags=["--imu"], vocab=True,
                      desc="mono-inertial + loop closure"),
}
# Run-to-run spread measured over 3 runs of both systems on all 11 sequences.
# Differences smaller than this are not evidence of anything.
NOISE_FLOOR = 0.19


# ── running ──────────────────────────────────────────────────────────────

def run_kornia(cfg, seq, out, extra=()):
    spec = CONFIGS[cfg]
    cmd = [str(BIN), "--no-tui"]
    if spec["vocab"]:
        cmd += ["--vocab", str(VOCAB), "--apply-pgo"]
    cmd += ["euroc", "--data", str(EUROC/seq), *spec["flags"],
            "--evaluate", "--eval-out", str(out), *extra]
    t0 = time.time()
    with open(out/"stdout.log", "w") as so, open(out/"stderr.log", "w") as se:
        rc = subprocess.call(cmd, stdout=so, stderr=se)
    tum = out/"kornia_slam_traj_tum.txt"
    if tum.exists() and tum.stat().st_size:
        (out/"traj.txt").write_bytes(tum.read_bytes())
    # Keep the loop-closure lines; the raw log is large and mostly VI-BA noise.
    events = [l for l in (out/"stderr.log").read_text(errors="ignore").splitlines()
              if "loop-closure" in l or "[pgo]" in l or "panicked" in l]
    (out/"events.log").write_text("\n".join(events))
    subprocess.call(["gzip", "-f", str(out/"stderr.log")])
    return rc, time.time()-t0


def run_orb3(seq, out):
    exe = ORB3/"Examples"/"Stereo-Inertial"/"stereo_inertial_euroc"
    times = ORB3/"Examples"/"Stereo-Inertial"/"EuRoC_TimeStamps"/f"{TS_STEM[seq]}.txt"
    if not exe.exists():
        return 127, 0.0
    cmd = [str(exe), str(VOCAB),
           str(ORB3/"Examples"/"Stereo-Inertial"/"EuRoC.yaml"),
           str(EUROC/seq), str(times), "run"]
    t0 = time.time()
    with open(out/"stdout.log", "w") as so, open(out/"stderr.log", "w") as se:
        rc = subprocess.call(cmd, cwd=out, stdout=so, stderr=se)
    f = out/"f_run.txt"
    if f.exists() and f.stat().st_size:
        (out/"traj.txt").write_bytes(f.read_bytes())
    return rc, time.time()-t0


def cmd_run(a):
    require_euroc()
    if not BIN.exists():
        raise SystemExit(f"kornia-slam binary not found at {BIN}\n"
                         "Build it with: cargo build --release -p kornia-slam-app")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out/"progress.log"
    jobs = []
    for cfg in a.configs:
        for seq in a.seqs:
            for r in range(1, a.runs+1):
                jobs.append(("kornia-"+cfg, cfg, seq, r))
    if a.orb3:
        for seq in a.seqs:
            for r in range(1, a.runs+1):
                jobs.append(("orb3", None, seq, r))

    done = 0
    for name, cfg, seq, r in jobs:
        d = out/name/seq/f"run{r}"
        if (d/"traj.txt").exists() and (d/"traj.txt").stat().st_size:
            done += 1
            continue
        d.mkdir(parents=True, exist_ok=True)
        rc, secs = (run_orb3(seq, d) if cfg is None else run_kornia(cfg, seq, d))
        line = f"{name} {seq} run{r} rc={rc} {secs:.0f}s"
        print(line, flush=True)
        with open(log, "a") as f:
            f.write(line+"\n")
        done += 1
    print(f"\n{done}/{len(jobs)} runs present in {out}")
    return 0


# ── scoring ──────────────────────────────────────────────────────────────

METRICS = ("ate_rmse", "ate_rot_rmse", "rpe_trans_rmse", "rpe_rot_rmse",
           "path_ratio", "scale", "coverage")


def score(out):
    """Median metrics per (config, sequence) across however many runs exist."""
    out = Path(out)
    runs = []
    for cfgdir in sorted(p for p in out.iterdir() if p.is_dir()):
        frame = "body" if cfgdir.name == "orb3" else "camera"
        for seq in SEQS:
            for d in sorted((cfgdir/seq).glob("run*")) if (cfgdir/seq).exists() else []:
                p = d/"traj.txt"
                if not p.exists() or not p.stat().st_size:
                    continue
                try:
                    r = evaluate(p, seq, frame=frame)
                except Exception as e:                       # noqa: BLE001
                    print(f"  ! {cfgdir.name}/{seq}/{d.name}: {e}", file=sys.stderr)
                    continue
                if r:
                    runs.append(dict(config=cfgdir.name, seq=seq, run=d.name,
                                     **{m: r[m] for m in METRICS}))
    import numpy as np
    med = {}
    for cfg in {r["config"] for r in runs}:
        for seq in SEQS:
            rs = [r for r in runs if r["config"] == cfg and r["seq"] == seq]
            if not rs:
                continue
            e = dict(config=cfg, seq=seq, n_runs=len(rs))
            for m in METRICS:
                e[m] = float(np.median([r[m] for r in rs]))
            ate = [r["ate_rmse"] for r in rs]
            e["ate_spread"] = float(max(ate)-min(ate))
            med[f"{cfg}|{seq}"] = e
    return runs, med


def cmd_score(a):
    runs, med = score(a.dir)
    Path(a.dir, "runs.json").write_text(json.dumps(runs, indent=1))
    Path(a.dir, "medians.json").write_text(json.dumps(med, indent=1))
    for cfg in sorted({v["config"] for v in med.values()}):
        rows = [v for v in med.values() if v["config"] == cfg]
        print(f"\n── {cfg}  ({CONFIGS.get(cfg.replace('kornia-',''), {}).get('desc', '')})")
        for e in sorted(rows, key=lambda r: SEQS.index(r["seq"])):
            print(f"   {SHORT[e['seq']]:6s} n={e['n_runs']} "
                  f"ATE={e['ate_rmse']:7.4f}m ±{e['ate_spread']:.4f}  "
                  f"rot={e['ate_rot_rmse']:6.2f}°  "
                  f"RPEr={e['rpe_rot_rmse']:6.2f}°/s  "
                  f"path={(e['path_ratio']-1)*100:+5.1f}%")
    print(f"\n{len(runs)} runs scored → {a.dir}/medians.json")
    if a.baseline:
        print()
        return compare(med, json.loads(Path(a.baseline).read_text()))
    return 0


# ── comparing ────────────────────────────────────────────────────────────

def compare(cur, base, cfg="kornia-si_loop"):
    """Current against a stored baseline, with the noise floor applied."""
    rows, better, worse = [], 0, 0
    for seq in SEQS:
        k = f"{cfg}|{seq}"
        if k not in cur or k not in base:
            continue
        c, b = cur[k]["ate_rmse"], base[k]["ate_rmse"]
        delta = (c-b)/b
        verdict = "same"
        if delta < -NOISE_FLOOR:
            verdict, better = "BETTER", better+1
        elif delta > NOISE_FLOOR:
            verdict, worse = "WORSE", worse+1
        rows.append((SHORT[seq], b, c, delta, verdict))
    if not rows:
        print("no overlapping sequences between run and baseline")
        return 1
    print(f"{'seq':7s} {'baseline':>9s} {'current':>9s} {'change':>8s}   verdict")
    for name, b, c, d, v in rows:
        print(f"{name:7s} {b:9.4f} {c:9.4f} {d*100:+7.1f}%   {v}")
    import numpy as np
    mb = np.median([r[1] for r in rows])
    mc = np.median([r[2] for r in rows])
    print(f"\nmedian ATE {mb:.4f} → {mc:.4f} m ({(mc-mb)/mb*100:+.1f}%)")
    print(f"{better} sequence(s) better, {worse} worse, "
          f"{len(rows)-better-worse} within the ±{NOISE_FLOOR*100:.0f}% noise floor")
    return 1 if worse > better else 0


def cmd_compare(a):
    _, med = score(a.dir)
    return compare(med, json.loads(Path(a.baseline).read_text()))


def cmd_baseline(a):
    _, med = score(a.dir)
    Path(a.save).parent.mkdir(parents=True, exist_ok=True)
    Path(a.save).write_text(json.dumps(med, indent=1))
    print(f"baseline written to {a.save} ({len(med)} cells)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="execute the sweep")
    r.add_argument("--out", required=True)
    r.add_argument("--seqs", nargs="+", default=SEQS)
    r.add_argument("--configs", nargs="+", default=["si_loop"], choices=list(CONFIGS))
    r.add_argument("--runs", type=int, default=1,
                   help="repeats per cell; 3 to see past run-to-run noise")
    r.add_argument("--orb3", action="store_true", help="also run the ORB-SLAM3 baseline")
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("score", help="score a sweep directory")
    s.add_argument("dir")
    s.add_argument("--baseline", help="also compare against this baseline file")
    s.set_defaults(fn=cmd_score)

    c = sub.add_parser("compare", help="compare a sweep against a baseline")
    c.add_argument("dir")
    c.add_argument("--baseline", required=True)
    c.set_defaults(fn=cmd_compare)

    b = sub.add_parser("baseline", help="save a sweep as the reference baseline")
    b.add_argument("dir")
    b.add_argument("--save", required=True)
    b.set_defaults(fn=cmd_baseline)

    a = ap.parse_args()
    sys.exit(a.fn(a))


if __name__ == "__main__":
    main()
