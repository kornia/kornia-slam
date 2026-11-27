#!/usr/bin/env python3
"""evaluate.py SEQ TRAJ [--fmt tum|kornia] [options]: score one trajectory against ground truth.

SEQ is a sequence in the EuRoC layout. Ground truth is the first of
mav0/state_groundtruth_estimate0/data.csv and mav0/mocap0/data.csv that exists (the app's order).
evo does the association, the Umeyama alignment and the APE.

Output: one JSON line with
  n_est, n_assoc   : estimated poses, and those associated with a ground-truth pose
  total_frames     : cam0 frames with a cam1 partner when cam1/data.csv exists, else all cam0 frames
  tracked_pct      : n_assoc / total_frames
  ate_se3_*        : ATE RMSE / mean (m) after SE(3) alignment (scale fixed): the primary number,
                     the ORB-SLAM3 stereo-inertial convention
  ate_sim3_*       : the same after Sim(3) alignment, and its sim3_scale
  jumps            : consecutive associated pairs whose relative motion differs from ground truth's
                     by more than --jump-m after the SE(3) alignment
  pose             : "body" (tum) or "camera_centre" (kornia); never compare the two in one table
  rpe<d>           : with --rpe, relative pose error over pairs d seconds apart (within 25 ms):
                     translation RMSE (m) in frame i's axes, and yaw RMSE (deg) of the world-frame
                     relative rotation about ground truth's z (gravity) axis. Needs orientations (tum).

Formats:
  tum    : rows "t x y z qx qy qz qw" (space or comma; t in s or ns, autodetected; # lines skipped).
  kornia : the app's kornia_slam_raw.csv "idx,x,y,z" (camera centres, position only). Needs --log, the
           app's stderr: row i takes the cam0 timestamp of the frame named by the i-th status line.
"""
import argparse
import csv
import json
import os
import re
import sys

import numpy as np
from evo.core import metrics, sync
from evo.core.trajectory import PoseTrajectory3D

GT_FILES = ('mav0/state_groundtruth_estimate0/data.csv', 'mav0/mocap0/data.csv')
STATUS_LINE = re.compile(r'^\[\s*(\d+)\] \w')


def fail(msg):
    print(f'evaluate.py: {msg}', file=sys.stderr)
    raise SystemExit(2)


def lines(path):
    with open(path, errors='replace') as f:
        return f.read().splitlines()


def csv_rows(path):
    return [r for r in csv.reader(lines(path)) if r]


def read_ts(path):
    return [int(r[0]) for r in csv_rows(path) if not r[0].startswith('#')]


def load_gt(seq):
    paths = [os.path.join(seq, p) for p in GT_FILES]
    path = next((p for p in paths if os.path.exists(p)), None)
    if path is None:
        fail(f'no ground truth: neither {paths[0]} nor {paths[1]}')
    rows = [r for r in csv_rows(path) if not r[0].startswith('#')]
    a = np.array([[float(x) for x in r[:8]] for r in rows])
    # EuRoC ground-truth quaternion order is w x y z.
    return PoseTrajectory3D(positions_xyz=a[:, 1:4], orientations_quat_wxyz=a[:, 4:8], timestamps=a[:, 0] * 1e-9)


def frame_count(seq):
    cam0 = read_ts(os.path.join(seq, 'mav0/cam0/data.csv'))
    cam1_csv = os.path.join(seq, 'mav0/cam1/data.csv')
    if not os.path.exists(cam1_csv):
        return len(cam0)
    right = set(read_ts(cam1_csv))
    return sum(t in right for t in cam0)


def status_frame_indices(log):
    return [int(m.group(1)) for m in map(STATUS_LINE.match, lines(log)) if m]


def load_kornia(seq, path, log):
    rows = [r for r in csv_rows(path) if r[0] != 'idx']
    frames = status_frame_indices(log)
    if len(rows) != len(frames):
        fail(f'{path} has {len(rows)} rows but {log} has {len(frames)} status lines')
    cam0 = read_ts(os.path.join(seq, 'mav0/cam0/data.csv'))
    if frames and max(frames) >= len(cam0):
        fail(f'{log} names frame {max(frames)} but cam0 has {len(cam0)} frames')
    xyz = np.array([[float(x) for x in r[1:4]] for r in rows]).reshape(-1, 3)
    ts = np.array([cam0[i] for i in frames], dtype=float) * 1e-9
    q = np.tile([1.0, 0, 0, 0], (len(rows), 1))
    return PoseTrajectory3D(positions_xyz=xyz, orientations_quat_wxyz=q, timestamps=ts)


def load_tum(path):
    rows = []
    for line in lines(path):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        rows.append([float(x) for x in line.replace(',', ' ').split()[:8]])
    a = np.array(rows)
    a = a[np.argsort(a[:, 0])]
    t = a[:, 0] * (1e-9 if a[0, 0] > 1e12 else 1.0)
    # TUM order qx qy qz qw; evo wants w x y z.
    q = a[:, [7, 4, 5, 6]]
    return PoseTrajectory3D(positions_xyz=a[:, 1:4], orientations_quat_wxyz=q, timestamps=t)


def copy_traj(t):
    return PoseTrajectory3D(t.positions_xyz.copy(), t.orientations_quat_wxyz.copy(), t.timestamps.copy())


def ate(est, gt, scale):
    e = copy_traj(est)
    scale_f = e.align(gt, correct_scale=scale)
    m = metrics.APE(metrics.PoseRelation.translation_part)
    m.process_data((gt, e))
    return m.get_statistic(metrics.StatisticsType.rmse), m.get_statistic(metrics.StatisticsType.mean), scale_f, e


def count_jumps(est, gt, jump_m):
    de = np.diff(est.positions_xyz, axis=0)
    dg = np.diff(gt.positions_xyz, axis=0)
    return int(np.sum(np.linalg.norm(de - dg, axis=1) > jump_m))


def yaw_about_z(r):
    return np.degrees(np.arctan2(r[1, 0] - r[0, 1], r[0, 0] + r[1, 1]))


def rpe(est, gt, delta, tol=0.025):
    """est, gt: associated PoseTrajectory3D, est already SE(3)-aligned to gt."""
    t = est.timestamps
    pe, pg = est.poses_se3, gt.poses_se3
    trans, yaw = [], []
    for i in range(len(t)):
        j = np.searchsorted(t, t[i] + delta)
        best = None
        for k in (j - 1, j):
            if 0 <= k < len(t) and k != i and abs(t[k] - t[i] - delta) < tol:
                if best is None or abs(t[k] - t[i] - delta) < abs(t[best] - t[i] - delta):
                    best = k
        if best is None:
            continue
        ei, ej, gi, gj = pe[i], pe[best], pg[i], pg[best]
        de = ei[:3, :3].T @ (ej[:3, 3] - ei[:3, 3])
        dg = gi[:3, :3].T @ (gj[:3, 3] - gi[:3, 3])
        trans.append(np.linalg.norm(de - dg))
        rw_e = ej[:3, :3] @ ei[:3, :3].T
        rw_g = gj[:3, :3] @ gi[:3, :3].T
        yaw.append(yaw_about_z(rw_g.T @ rw_e))
    if not trans:
        return {'n': 0}
    return {'n': len(trans), 'trans_rmse': round(float(np.sqrt(np.mean(np.square(trans)))), 4),
            'yaw_rmse_deg': round(float(np.sqrt(np.mean(np.square(yaw)))), 3)}


def to_cam0_body(seq, traj):
    """Rotates a cam0 trajectory into the body frame by R_BS of cam0/sensor.yaml (lever arm not modelled)."""
    txt = '\n'.join(lines(os.path.join(seq, 'mav0/cam0/sensor.yaml')))
    vals = [float(x) for x in re.search(r'T_BS:.*?data:\s*\[([^\]]*)\]', txt, re.S).group(1).replace('\n', ' ').split(',')]
    r_bs = np.array(vals).reshape(4, 4)[:3, :3]
    poses = [np.block([[p[:3, :3] @ r_bs.T, p[:3, 3:4]], [np.zeros((1, 3)), np.ones((1, 1))]]) for p in traj.poses_se3]
    return PoseTrajectory3D(poses_se3=poses, timestamps=traj.timestamps.copy())


def keep_near(traj, stamps_ns, tol=0.005):
    """Mask of the poses of traj within tol seconds of one of stamps_ns."""
    ref = np.sort(np.asarray(stamps_ns, dtype=float) * 1e-9)
    t = traj.timestamps
    if len(ref) == 0:
        return np.zeros(len(t), dtype=bool)
    i = np.searchsorted(ref, t)
    hi, lo = np.clip(i, 0, len(ref) - 1), np.clip(i - 1, 0, len(ref) - 1)
    return np.minimum(np.abs(ref[hi] - t), np.abs(ref[lo] - t)) <= tol


def associate(gt, est, max_diff):
    """Associated (gt, est), or None when nothing associates."""
    if est is None:
        return None
    try:
        return sync.associate_trajectories(gt, est, max_diff=max_diff)
    except sync.SyncException:
        return None


def evaluate(seq, traj, fmt='tum', log=None, max_diff=0.005, nearest_gt=False, total=None,
             rpe_deltas=None, cam0_body=False, jump_m=0.3, keep=None):
    """Scores TRAJ against SEQ's ground truth; returns the output dict. Bad input raises SystemExit(2).

    keep: optional cam0 stamps (ns); only poses within 5 ms of one are scored.
    """
    if fmt == 'kornia':
        if log is None:
            fail('--fmt kornia needs --log (the app stderr with the status lines)')
        est = load_kornia(seq, traj, log)
    else:
        est = load_tum(traj)
    gt = load_gt(seq)
    if nearest_gt:
        max_diff = float('inf')
    n_est = len(est.timestamps)
    if keep is not None:
        mask = keep_near(est, keep)
        n_est = int(mask.sum())
        est = PoseTrajectory3D(est.positions_xyz[mask], est.orientations_quat_wxyz[mask],
                               est.timestamps[mask]) if n_est else None
    pair = associate(gt, est, max_diff)
    n_assoc = len(pair[1].timestamps) if pair else 0
    total = total or frame_count(seq)
    out = {'pose': 'camera_centre' if fmt == 'kornia' else 'body',
           'n_est': n_est, 'n_assoc': n_assoc, 'total_frames': total,
           'tracked_pct': round(100.0 * n_assoc / total, 2)}
    if n_assoc >= 3:
        g2, e2 = pair
        r, m, _, aligned = ate(e2, g2, False)
        out['ate_se3_rmse'] = round(r, 4)
        out['ate_se3_mean'] = round(m, 4)
        out['jumps'] = count_jumps(aligned, g2, jump_m)
        r, m, s, _ = ate(e2, g2, True)
        out['ate_sim3_rmse'] = round(r, 4)
        out['ate_sim3_mean'] = round(m, 4)
        out['sim3_scale'] = round(float(s[2]) if isinstance(s, tuple) else float(s), 5)
        if rpe_deltas:
            if cam0_body:
                aligned = to_cam0_body(seq, aligned)
            for d in rpe_deltas.split(','):
                out[f'rpe{d}'] = rpe(aligned, g2, float(d))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('seq')
    ap.add_argument('traj')
    ap.add_argument('--fmt', default='tum', choices=['tum', 'kornia'])
    ap.add_argument('--log', default=None, help='app stderr with the per-frame status lines (--fmt kornia)')
    ap.add_argument('--max-diff', type=float, default=0.005, help='association limit (s)')
    ap.add_argument('--nearest-gt', action='store_true',
                    help='nearest ground truth with no time limit (reproduces the app --evaluate printout)')
    ap.add_argument('--total', type=int, default=None, help='override total frames for tracked_pct')
    ap.add_argument('--rpe', default=None, help='comma-separated deltas in seconds, e.g. 1,10')
    ap.add_argument('--cam0-body', action='store_true',
                    help='before RPE, rotate a cam0 trajectory into the body frame by cam0 R_BS')
    ap.add_argument('--jump-m', type=float, default=0.3, help='relative-motion error (m) counted as a jump')
    a = ap.parse_args()
    out = evaluate(a.seq, a.traj, fmt=a.fmt, log=a.log, max_diff=a.max_diff, nearest_gt=a.nearest_gt,
                   total=a.total, rpe_deltas=a.rpe, cam0_body=a.cam0_body, jump_m=a.jump_m)
    print(json.dumps(out))


if __name__ == '__main__':
    main()
