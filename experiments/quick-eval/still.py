#!/usr/bin/env python3
"""Keyframes and still frames inside still spans (card AB).

  still.py STDERR CAM_CSV [--window A-B] [--quiet S0-S1,S2-S3] [--gt GT_CSV]

STDERR is a run's debug log (run.sh DEBUG=1). CAM_CSV is the camera's data.csv (EuRoC
mav0/cam0/data.csv: frame k is row k, so a source with start_frame 0). --window is a frame
range (inclusive), --quiet the IMU-quiet spans in seconds from the first frame, --gt the EuRoC
state_groundtruth_estimate0/data.csv for the speed check (K4).

Prints key=value lines:
  keyframes_in_window, keyframes_per_min_in_longest_still_run, still_frames_in_window,
  still_frames_in_quiet, quiet_frames, still_frames_outside_window, frames_outside_window,
  withheld_in_window, heartbeat_in_window, quality_in_window, lost_withheld_in_window,
  landmarks_in_window (close_points + grown of the keyframes in the window),
  withheld_above_5cm (withheld or lost-withheld frames with GT speed over 5 cm/s),
  runs_closed_<voter> counts in the window.
"""
import argparse
import bisect
import re

KF = re.compile(r"\[kf\] frame=(\d+) grown=(\d+)")
CLOSE = re.compile(r"\[kf_stereo\] frame=(\d+) close_points=(\d+)")
STILL = re.compile(r"\[still\] frame=(\d+) t=\S+ still=([01])")
ASK = re.compile(r"\[still\] frame=(\d+) ask=(\w+)")
CLOSED = re.compile(r"\[still\] run kf=(\d+) closed by=(\w+) at=([0-9.]+)")
RUN_START = re.compile(r"\[still\] run kf=(\d+) t=")


def span(text):
    a, b = text.split("-")
    return float(a), float(b)


def read_frames(cam_csv):
    stamps = []
    with open(cam_csv) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            stamps.append(int(line.split(",")[0]) * 1e-9)
    return stamps


def read_gt_speed(gt_csv):
    times, speeds = [], []
    with open(gt_csv) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            v = line.split(",")
            times.append(int(v[0]) * 1e-9)
            speeds.append(sum(float(x) ** 2 for x in v[8:11]) ** 0.5)
    return times, speeds


def speed_at(gt, t):
    times, speeds = gt
    i = bisect.bisect_left(times, t)
    if i <= 0 or i >= len(times):
        return None
    return speeds[i]


def parse(stderr):
    log = {"kf": {}, "close": {}, "still": {}, "ask": {}, "closed": [], "runs": []}
    with open(stderr, errors="replace") as f:
        for line in f:
            if m := KF.search(line):
                log["kf"][int(m[1])] = int(m[2])
            elif m := CLOSE.search(line):
                log["close"][int(m[1])] = int(m[2])
            elif m := STILL.search(line):
                log["still"].setdefault(int(m[1]), m[2] == "1")
            elif m := ASK.search(line):
                log["ask"][int(m[1])] = m[2]
            elif m := CLOSED.search(line):
                log["closed"].append((int(m[1]), m[2]))
            elif m := RUN_START.search(line):
                log["runs"].append(int(m[1]))
    return log


def longest_still_run(still, keyframes, frames):
    """Keyframes per minute inside the longest stretch of consecutive still frames."""
    best = (0, 0)
    start = None
    for k in sorted(still) + [None]:
        if k is not None and still[k] and (start is None or k == prev + 1):
            start = k if start is None else start
            prev = k
            continue
        if start is not None and prev - start > best[1] - best[0]:
            best = (start, prev)
        start, prev = (k, k) if k is not None and still[k] else (None, None)
    a, b = best
    if b <= a:
        return None, 0.0
    secs = frames[b] - frames[a]
    n = sum(1 for k in keyframes if a <= k <= b)
    return n / secs * 60.0 if secs > 0 else None, secs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stderr")
    p.add_argument("cam_csv")
    p.add_argument("--window", type=span, default=None)
    p.add_argument("--quiet", default="")
    p.add_argument("--gt", default=None)
    a = p.parse_args()

    frames = read_frames(a.cam_csv)
    t0 = frames[0]
    log = parse(a.stderr)
    lo, hi = a.window if a.window else (0, len(frames) - 1)
    in_w = lambda k: lo <= k <= hi
    quiet = [span(s) for s in a.quiet.split(",") if s]
    in_quiet = lambda k: k < len(frames) and any(s <= frames[k] - t0 <= e for s, e in quiet)

    kfs = sorted(log["kf"])
    print(f"keyframes_in_window={sum(1 for k in kfs if in_w(k))}")
    landmarks = sum(log["kf"][k] + log["close"].get(k, 0) for k in kfs if in_w(k))
    print(f"landmarks_in_window={landmarks}")
    still = log["still"]
    print(f"still_frames_in_window={sum(1 for k, s in still.items() if s and in_w(k))}")
    quiet_frames = [k for k in range(len(frames)) if in_quiet(k) and in_w(k)]
    print(f"quiet_frames={len(quiet_frames)}")
    print(f"still_frames_in_quiet={sum(1 for k in quiet_frames if still.get(k))}")
    judged_outside = [k for k in still if not in_w(k)]
    print(f"frames_outside_window={len(judged_outside)}")
    print(f"still_frames_outside_window={sum(1 for k in judged_outside if still[k])}")
    for kind in ("withheld", "heartbeat", "quality", "lost_withheld"):
        n = sum(1 for k, v in log["ask"].items() if v == kind and in_w(k))
        print(f"{kind}_in_window={n}")
    withheld_outside_still = sum(
        1 for k, v in log["ask"].items() if v in ("withheld", "lost_withheld") and not still.get(k)
    )
    print(f"withheld_on_non_still_frames={withheld_outside_still}")
    closers = {}
    for kf, voter in log["closed"]:
        if in_w(kf):
            closers[voter] = closers.get(voter, 0) + 1
    print("runs_closed_in_window " + " ".join(f"{k}={v}" for k, v in sorted(closers.items())))
    per_min, secs = longest_still_run({k: s for k, s in still.items() if in_w(k)}, kfs, frames)
    print(f"longest_still_run_sec={secs:.2f} keyframes_per_min_in_it={per_min}")
    if a.gt:
        gt = read_gt_speed(a.gt)
        fast = [
            k
            for k, v in log["ask"].items()
            if v in ("withheld", "lost_withheld")
            and k < len(frames)
            and (speed_at(gt, frames[k]) or 0.0) > 0.05
        ]
        print(f"withheld_above_5cm={len(fast)} frames={sorted(fast)[:20]}")


if __name__ == "__main__":
    main()
