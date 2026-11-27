"""Synthetic EuRoC-layout sequences and run directories for the eval script tests."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

T0_NS = 1_403_636_579_763_555_840
DT_NS = 50_000_000


def stamps(n):
    return [T0_NS + i * DT_NS for i in range(n)]


def rot(axis, angle):
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def quat_wxyz(r):
    w = np.sqrt(max(1.0 + np.trace(r), 1e-12)) / 2
    return np.array([w, (r[2, 1] - r[1, 2]) / (4 * w), (r[0, 2] - r[2, 0]) / (4 * w), (r[1, 0] - r[0, 1]) / (4 * w)])


def poses(n):
    """Ground-truth body poses (R, p) along a smooth 3D curve."""
    out = []
    for i in range(n):
        s = i * DT_NS * 1e-9
        p = np.array([2 * np.cos(0.5 * s), 2 * np.sin(0.7 * s), 0.3 * np.sin(s)])
        r = rot([0, 0, 1], 0.3 * s) @ rot([1, 0, 0], 0.1 * np.sin(s))
        out.append((r, p))
    return out


def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(header + '\n')
        for r in rows:
            f.write(','.join(str(x) for x in r) + '\n')


def make_sequence(root, n, gt_dir='state_groundtruth_estimate0', cam1=True):
    ts = stamps(n)
    write_csv(os.path.join(root, 'mav0/cam0/data.csv'), '#timestamp [ns],filename', [[t, f'{t}.png'] for t in ts])
    if cam1:
        write_csv(os.path.join(root, 'mav0/cam1/data.csv'), '#timestamp [ns],filename', [[t, f'{t}.png'] for t in ts])
    rows = []
    for t, (r, p) in zip(ts, poses(n)):
        rows.append([t, *p, *quat_wxyz(r), *([0.0] * 9)])
    write_csv(os.path.join(root, 'mav0', gt_dir, 'data.csv'), '#timestamp,p_x,p_y,p_z,q_w,q_x,q_y,q_z', rows)
    return ts


def write_tum(path, ts, ps):
    with open(path, 'w') as f:
        for t, (r, p) in zip(ts, ps):
            w, x, y, z = quat_wxyz(r)
            f.write(f'{t} {p[0]:.9f} {p[1]:.9f} {p[2]:.9f} {x:.9f} {y:.9f} {z:.9f} {w:.9f}\n')


def write_kornia(path, ps):
    with open(path, 'w') as f:
        f.write('idx,x,y,z\n')
        for i, (_, p) in enumerate(ps):
            f.write(f'{i},{p[0]:.9f},{p[1]:.9f},{p[2]:.9f}\n')


def write_status_log(path, frame_indices):
    with open(path, 'w') as f:
        f.write('Dataset: synthetic\n')
        for i in frame_indices:
            f.write(f'[{i:>5}] Tracked  kf=0    pts=100     1.0ms fe=0.5ms\n')


def moved(ps, r0, t0, scale=1.0):
    return [(r0 @ r, scale * (r0 @ p + t0)) for r, p in ps]
