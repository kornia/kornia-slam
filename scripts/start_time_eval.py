#!/usr/bin/env python3
"""Start-time sensitivity evaluation harness for kornia-slam.

Evaluates how initialisation success, time-to-init, scale error, and
trajectory accuracy (Sim(3) ATE RMSE) vary as the sequence start offset
changes, mirroring the evaluation protocol of AnchorFlow-VIO (arXiv:2610.26210).

Usage:
    # Run synthetic smoke test (no dataset required):
    python scripts/start_time_eval.py --smoke-test --output-dir eval_output/smoke_test

    # Run on an EuRoC sequence:
    python scripts/start_time_eval.py \
        --dataset-path /path/to/MH_05_difficult \
        --sequence-name MH_05_difficult \
        --config configs/euroc.ron \
        --start-offsets 0,100,200,300,400,600 \
        --output-dir eval_output/mh05
"""

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ==============================================================================
# Mathematical Helpers: 3D Vectors, Matrices & Umeyama Sim(3) Alignment
# Matches apps/kornia-slam-app/src/evaluation.rs
# ==============================================================================

@dataclass
class Vec3:
    x: float
    y: float
    z: float

    def __add__(self, o: "Vec3") -> "Vec3":
        return Vec3(self.x + o.x, self.y + o.y, self.z + o.z)

    def __sub__(self, o: "Vec3") -> "Vec3":
        return Vec3(self.x - o.x, self.y - o.y, self.z - o.z)

    def __mul__(self, s: float) -> "Vec3":
        return Vec3(self.x * s, self.y * s, self.z * s)

    def __truediv__(self, s: float) -> "Vec3":
        return Vec3(self.x / s, self.y / s, self.z / s)

    def dot(self, o: "Vec3") -> float:
        return self.x * o.x + self.y * o.y + self.z * o.z

    def norm_sq(self) -> float:
        return self.dot(self)

    def norm(self) -> float:
        return math.sqrt(self.norm_sq())


def mat3_mul_vec(m: List[List[float]], v: Vec3) -> Vec3:
    return Vec3(
        m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
        m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
        m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z,
    )


def mat3_mul_mat3(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    res = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            res[i][j] = sum(a[i][k] * b[k][j] for k in range(3))
    return res


def mat3_transpose(m: List[List[float]]) -> List[List[float]]:
    return [[m[j][i] for j in range(3)] for i in range(3)]


def mat3_det(m: List[List[float]]) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def svd3(a: List[List[float]]) -> Tuple[List[List[float]], List[float], List[List[float]]]:
    """Robust 3x3 SVD via one-sided Jacobi orthogonalization (A = U * S * V^T)."""
    # W columns initialized to A's columns
    w = [[a[r][c] for r in range(3)] for c in range(3)]  # w[col][row]
    v = [[1.0 if r == c else 0.0 for r in range(3)] for c in range(3)]  # v[col][row]

    for _ in range(30):
        max_c = 0.0
        for j in range(3):
            for k in range(j + 1, 3):
                # Column dot products
                dot_jj = sum(w[j][r] * w[j][r] for r in range(3))
                dot_kk = sum(w[k][r] * w[k][r] for r in range(3))
                dot_jk = sum(w[j][r] * w[k][r] for r in range(3))

                max_c = max(max_c, abs(dot_jk))
                if abs(dot_jk) < 1e-14 * math.sqrt(max(dot_jj * dot_kk, 1e-14)):
                    continue

                zeta = (dot_kk - dot_jj) / (2.0 * dot_jk)
                t = (1.0 if zeta >= 0 else -1.0) / (abs(zeta) + math.sqrt(1.0 + zeta * zeta))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c

                for r in range(3):
                    w_jr = c * w[j][r] - s * w[k][r]
                    w_kr = s * w[j][r] + c * w[k][r]
                    w[j][r] = w_jr
                    w[k][r] = w_kr

                    v_jr = c * v[j][r] - s * v[k][r]
                    v_kr = s * v[j][r] + c * v[k][r]
                    v[j][r] = v_jr
                    v[k][r] = v_kr

        if max_c < 1e-12:
            break

    singular_values = [math.sqrt(sum(w[c][r] * w[c][r] for r in range(3))) for c in range(3)]
    u_cols = []
    for c in range(3):
        sigma = singular_values[c]
        if sigma > 1e-12:
            u_cols.append([w[c][r] / sigma for r in range(3)])
        else:
            u_cols.append([1.0 if r == c else 0.0 for r in range(3)])

    # Order by descending singular value
    order = sorted(range(3), key=lambda i: singular_values[i], reverse=True)
    s_sorted = [singular_values[i] for i in order]
    u = [[u_cols[order[c]][r] for c in range(3)] for r in range(3)]
    v_mat = [[v[order[c]][r] for c in range(3)] for r in range(3)]

    return u, s_sorted, v_mat


@dataclass
class Sim3Alignment:
    scale: float
    rotation: List[List[float]]
    translation: Vec3


def align_sim3(est: List[Vec3], gt: List[Vec3]) -> Sim3Alignment:
    """Umeyama Sim3 fit mapping est onto gt.

    Replicates apps/kornia-slam-app/src/evaluation.rs `align_sim3`.
    """
    n = float(len(est))
    if n < 2:
        return Sim3Alignment(scale=1.0, rotation=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], translation=Vec3(0, 0, 0))

    # Centroids
    mu_est = Vec3(0, 0, 0)
    mu_gt = Vec3(0, 0, 0)
    for p_e, p_g in zip(est, gt):
        mu_est = mu_est + p_e
        mu_gt = mu_gt + p_g
    mu_est = mu_est / n
    mu_gt = mu_gt / n

    # Cross-covariance and variance
    sigma = [[0.0] * 3 for _ in range(3)]
    var_est = 0.0
    for p_e, p_g in zip(est, gt):
        pe = p_e - mu_est
        pg = p_g - mu_gt
        sigma[0][0] += pg.x * pe.x
        sigma[0][1] += pg.x * pe.y
        sigma[0][2] += pg.x * pe.z
        sigma[1][0] += pg.y * pe.x
        sigma[1][1] += pg.y * pe.y
        sigma[1][2] += pg.y * pe.z
        sigma[2][0] += pg.z * pe.x
        sigma[2][1] += pg.z * pe.y
        sigma[2][2] += pg.z * pe.z
        var_est += pe.norm_sq()

    for i in range(3):
        for j in range(3):
            sigma[i][j] /= n
    var_est /= n

    u, s, v = svd3(sigma)
    vt = mat3_transpose(v)
    uvt = mat3_mul_mat3(u, vt)

    diag_s = [1.0, 1.0, 1.0]
    if mat3_det(uvt) < 0.0:
        diag_s[2] = -1.0

    diag_mat = [[diag_s[0], 0, 0], [0, diag_s[1], 0], [0, 0, diag_s[2]]]
    r = mat3_mul_mat3(mat3_mul_mat3(u, diag_mat), vt)

    trace = sum(s[i] * diag_s[i] for i in range(3))
    scale = trace / var_est if var_est > 1e-12 else 1.0

    r_mu = mat3_mul_vec(r, mu_est)
    translation = mu_gt - (r_mu * scale)

    return Sim3Alignment(scale=scale, rotation=r, translation=translation)


def compute_ate_rmse(aligned_est: List[Vec3], gt: List[Vec3]) -> float:
    """Absolute Trajectory Error (ATE) RMSE in metres."""
    if not aligned_est or len(aligned_est) != len(gt):
        return float("nan")
    sq_errs = [(p - g).norm_sq() for p, g in zip(aligned_est, gt)]
    return math.sqrt(sum(sq_errs) / len(sq_errs))


def apply_sim3(alignment: Sim3Alignment, points: List[Vec3]) -> List[Vec3]:
    """Applies Sim(3) transform: p_aligned = translation + s * (R * p)."""
    return [
        alignment.translation + (mat3_mul_vec(alignment.rotation, p) * alignment.scale)
        for p in points
    ]


# ==============================================================================
# Metric Structures
# ==============================================================================

@dataclass
class StartTimeMetrics:
    offset_frames: int
    offset_sec: float
    init_success: bool
    time_to_init_sec: float
    frames_evaluated: int
    scale: float
    scale_error: float
    ate_rmse_m: float
    tracking_loss_events: int
    wall_time_sec: float
    status: str  # "ok", "diverged", "init_failed", "skipped"


# ==============================================================================
# Synthetic Smoke Test
# ==============================================================================

def run_synthetic_smoke_test(
    start_offsets: List[int], fps: float = 20.0
) -> List[StartTimeMetrics]:
    """Simulates a deterministic synthetic trajectory across start offsets.

    Validates that the harness correctly executes Sim(3) alignment, computes
    ATE RMSE, tracks time-to-init, measures scale errors, and reports statistics.
    """
    total_frames = 1200
    # Ground truth: a 3D helix
    gt_trajectory: List[Vec3] = []
    for i in range(total_frames):
        t = i / fps
        x = 5.0 * math.cos(0.4 * t)
        y = 5.0 * math.sin(0.4 * t)
        z = 0.5 * t
        gt_trajectory.append(Vec3(x, y, z))

    results: List[StartTimeMetrics] = []

    for offset in start_offsets:
        offset_sec = offset / fps
        if offset >= total_frames - 50:
            results.append(
                StartTimeMetrics(
                    offset_frames=offset,
                    offset_sec=offset_sec,
                    init_success=False,
                    time_to_init_sec=float("nan"),
                    frames_evaluated=0,
                    scale=float("nan"),
                    scale_error=float("nan"),
                    ate_rmse_m=float("nan"),
                    tracking_loss_events=0,
                    wall_time_sec=0.001,
                    status="init_failed",
                )
            )
            continue

        # Simulating start-time sensitivity effect:
        # Later starts in certain phases have slight initialization latency and drift
        # Deterministic variation function:
        phase = (offset / fps) * 0.4
        synthetic_scale = 0.96 + 0.05 * math.sin(phase)  # scale oscillates slightly
        init_delay_frames = 20 + int(8 * abs(math.sin(phase * 1.5)))
        time_to_init = init_delay_frames / fps

        # Suffix evaluated
        eval_gt = gt_trajectory[offset:]
        eval_est: List[Vec3] = []

        # Synthetic tracking drift
        for idx, g in enumerate(eval_gt):
            # Noise drift increases over time
            drift_x = 0.02 * math.sin(idx * 0.05) + 0.0005 * idx
            drift_y = 0.02 * math.cos(idx * 0.05) - 0.0003 * idx
            drift_z = 0.01 * math.sin(idx * 0.02)
            # Unaligned monocular estimate at synthetic scale
            p = Vec3(
                (g.x + drift_x) * synthetic_scale,
                (g.y + drift_y) * synthetic_scale,
                (g.z + drift_z) * synthetic_scale,
            )
            eval_est.append(p)

        # Umeyama Sim3 alignment
        alignment = align_sim3(eval_est, eval_gt)
        aligned_est = apply_sim3(alignment, eval_est)
        ate = compute_ate_rmse(aligned_est, eval_gt)

        # Tracking loss events simulated
        loss_events = 1 if abs(synthetic_scale - 1.0) > 0.04 else 0

        results.append(
            StartTimeMetrics(
                offset_frames=offset,
                offset_sec=offset_sec,
                init_success=True,
                time_to_init_sec=time_to_init,
                frames_evaluated=len(aligned_est),
                scale=alignment.scale,
                scale_error=abs(alignment.scale - 1.0),
                ate_rmse_m=ate,
                tracking_loss_events=loss_events,
                wall_time_sec=round(0.01 + len(eval_gt) * 0.0001, 3),
                status="ok",
            )
        )

    return results


# ==============================================================================
# Real Dataset Execution Harness
# ==============================================================================

def generate_override_ron(
    base_ron_path: Path, dataset_path: Path, start_frame: int, out_ron_path: Path
) -> None:
    """Creates a temporary run RON config with the specified start_frame and synchronous mapping."""
    with open(base_ron_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Normalize Windows backslashes for RON string paths
    data_str = str(dataset_path.resolve()).replace("\\", "/")

    # Replace or insert data path
    content = re.sub(r'data:\s*"[^"]*"', f'data: "{data_str}"', content)
    # Replace or insert start_frame
    if "start_frame:" in content:
        content = re.sub(r"start_frame:\s*\d+", f"start_frame: {start_frame}", content)
    else:
        content = re.sub(
            r"(source:\s*Euroc\(\(\s*)",
            rf"\1start_frame: {start_frame},\n        ",
            content,
        )

    # Force synchronous mapping for deterministic evaluation replay
    if "execution: Asynchronous" in content:
        content = content.replace("execution: Asynchronous", "execution: Synchronous")

    with open(out_ron_path, "w", encoding="utf-8") as f:
        f.write(content)


def run_offset_real(
    kornia_slam_bin: str,
    base_ron_path: Path,
    dataset_path: Path,
    start_frame: int,
    offset_out_dir: Path,
    fps: float = 20.0,
) -> StartTimeMetrics:
    """Executes kornia-slam on a real sequence from a specific start_frame offset."""
    offset_out_dir.mkdir(parents=True, exist_ok=True)
    temp_ron = offset_out_dir / f"run_offset_{start_frame}.ron"
    generate_override_ron(base_ron_path, dataset_path, start_frame, temp_ron)

    cmd = [
        kornia_slam_bin,
        "--config",
        str(temp_ron),
        "--no-tui",
        "--evaluate",
        "--eval-out",
        str(offset_out_dir),
    ]

    start_wall = time.time()
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=1800,  # 30 min cap
        )
        wall_time = time.time() - start_wall
        output = proc.stderr + "\n" + proc.stdout
    except subprocess.TimeoutExpired:
        return StartTimeMetrics(
            offset_frames=start_frame,
            offset_sec=start_frame / fps,
            init_success=False,
            time_to_init_sec=float("nan"),
            frames_evaluated=0,
            scale=float("nan"),
            scale_error=float("nan"),
            ate_rmse_m=float("nan"),
            tracking_loss_events=0,
            wall_time_sec=time.time() - start_wall,
            status="timeout",
        )
    except Exception as e:
        return StartTimeMetrics(
            offset_frames=start_frame,
            offset_sec=start_frame / fps,
            init_success=False,
            time_to_init_sec=float("nan"),
            frames_evaluated=0,
            scale=float("nan"),
            scale_error=float("nan"),
            ate_rmse_m=float("nan"),
            tracking_loss_events=0,
            wall_time_sec=time.time() - start_wall,
            status=f"error: {e}",
        )

    # Parse stdout/stderr for metrics
    # Look for [bootstrap] or [imu_init] or tracking loss
    init_success = False
    init_frame_idx = None
    tracking_loss_count = 0

    for line in output.splitlines():
        if "[bootstrap]" in line or "bootstrap accepted" in line or "[imu_init] applying" in line:
            init_success = True
            m = re.search(r"\[(\d+)\]", line)
            if m and init_frame_idx is None:
                init_frame_idx = int(m.group(1))
        if "Skipped" in line or "tracking lost" in line.lower():
            tracking_loss_count += 1

    time_to_init = (
        (init_frame_idx - start_frame) / fps
        if (init_frame_idx is not None and init_frame_idx >= start_frame)
        else (20.0 / fps if init_success else float("nan"))
    )

    # Read evaluation CSV if generated
    aligned_csv = offset_out_dir / "kornia_slam_aligned.csv"
    scale = float("nan")
    scale_error = float("nan")
    ate_rmse = float("nan")
    frames_evaluated = 0

    if aligned_csv.exists():
        est_pts: List[Vec3] = []
        gt_pts: List[Vec3] = []
        with open(aligned_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                est_pts.append(
                    Vec3(float(row["x"]), float(row["y"]), float(row["z"]))
                )
                gt_pts.append(
                    Vec3(float(row["gt_x"]), float(row["gt_y"]), float(row["gt_z"]))
                )

        frames_evaluated = len(est_pts)
        if frames_evaluated >= 2:
            ate_rmse = compute_ate_rmse(est_pts, gt_pts)

        # Parse scale from output text if present
        m_scale = re.search(r"Scale\s*:\s*([0-9.]+)", output)
        if m_scale:
            scale = float(m_scale.group(1))
            scale_error = abs(scale - 1.0)

    status = "ok" if (init_success and not math.isnan(ate_rmse)) else "init_failed"

    return StartTimeMetrics(
        offset_frames=start_frame,
        offset_sec=start_frame / fps,
        init_success=init_success,
        time_to_init_sec=time_to_init,
        frames_evaluated=frames_evaluated,
        scale=scale,
        scale_error=scale_error,
        ate_rmse_m=ate_rmse,
        tracking_loss_events=tracking_loss_count,
        wall_time_sec=round(wall_time, 2),
        status=status,
    )


# ==============================================================================
# Summary Statistics & Reporting
# ==============================================================================

def compute_summary(metrics: List[StartTimeMetrics]) -> Dict[str, Dict[str, float]]:
    valid_ate = [m.ate_rmse_m for m in metrics if not math.isnan(m.ate_rmse_m)]
    valid_scale = [m.scale for m in metrics if not math.isnan(m.scale)]
    valid_time_to_init = [
        m.time_to_init_sec for m in metrics if not math.isnan(m.time_to_init_sec)
    ]
    valid_wall = [m.wall_time_sec for m in metrics]

    def stats(vals: List[float]) -> Dict[str, float]:
        if not vals:
            return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
        mean = sum(vals) / len(vals)
        variance = sum((x - mean) ** 2 for x in vals) / (len(vals) if len(vals) > 1 else 1)
        return {
            "mean": round(mean, 4),
            "std": round(math.sqrt(variance), 4),
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
        }

    failures = sum(1 for m in metrics if not m.init_success or m.status != "ok")
    failure_rate = (failures / len(metrics)) * 100.0 if metrics else 0.0

    return {
        "ate_rmse_m": stats(valid_ate),
        "scale": stats(valid_scale),
        "time_to_init_sec": stats(valid_time_to_init),
        "wall_time_sec": stats(valid_wall),
        "summary": {
            "total_runs": len(metrics),
            "successful_runs": len(metrics) - failures,
            "failure_rate_pct": round(failure_rate, 2),
        },
    }


def print_table(metrics: List[StartTimeMetrics], summary: Dict[str, Dict[str, float]]) -> None:
    print("\n" + "=" * 90)
    print("                 KORNIA-SLAM START-TIME SENSITIVITY EVALUATION")
    print("=" * 90)
    print(
        f"{'Offset (f)':<11} {'Offset (s)':<11} {'Status':<12} {'Init (s)':<10} "
        f"{'Scale':<10} {'Scale Err':<11} {'ATE RMSE (m)':<14} {'Losses':<8} {'Wall (s)':<9}"
    )
    print("-" * 90)
    for m in metrics:
        scale_str = f"{m.scale:.4f}" if not math.isnan(m.scale) else "N/A"
        scale_err_str = f"{m.scale_error:.4f}" if not math.isnan(m.scale_error) else "N/A"
        ate_str = f"{m.ate_rmse_m:.4f}" if not math.isnan(m.ate_rmse_m) else "N/A"
        init_str = f"{m.time_to_init_sec:.2f}" if not math.isnan(m.time_to_init_sec) else "N/A"
        print(
            f"{m.offset_frames:<11} {m.offset_sec:<11.2f} {m.status:<12} {init_str:<10} "
            f"{scale_str:<10} {scale_err_str:<11} {ate_str:<14} {m.tracking_loss_events:<8} {m.wall_time_sec:<9.2f}"
        )
    print("=" * 90)
    print("SUMMARY STATISTICS across start offsets:")
    ate_st = summary["ate_rmse_m"]
    scale_st = summary["scale"]
    init_st = summary["time_to_init_sec"]
    wall_st = summary["wall_time_sec"]
    summ = summary["summary"]
    print(
        f"  ATE RMSE (m)     : Mean={ate_st['mean']:.4f}  Std={ate_st['std']:.4f}  "
        f"Min={ate_st['min']:.4f}  Max={ate_st['max']:.4f}"
    )
    print(
        f"  Sim(3) Scale     : Mean={scale_st['mean']:.4f}  Std={scale_st['std']:.4f}  "
        f"Min={scale_st['min']:.4f}  Max={scale_st['max']:.4f}"
    )
    print(
        f"  Time-to-Init (s) : Mean={init_st['mean']:.2f}  Std={init_st['std']:.2f}  "
        f"Min={init_st['min']:.2f}  Max={init_st['max']:.2f}"
    )
    print(
        f"  Wall Time (s)    : Mean={wall_st['mean']:.2f}  Std={wall_st['std']:.2f}  "
        f"Min={wall_st['min']:.2f}  Max={wall_st['max']:.2f}"
    )
    print(
        f"  Total Runs       : {int(summ['total_runs'])} | Succeeded: {int(summ['successful_runs'])} | "
        f"Failure Rate: {summ['failure_rate_pct']:.1f}%"
    )
    print("=" * 90 + "\n")


# ==============================================================================
# Main Entry Point
# ==============================================================================

def parse_offsets(offsets_arg: str, fps: float) -> List[int]:
    parts = [p.strip() for p in offsets_arg.split(",") if p.strip()]
    offsets = []
    for p in parts:
        if p.endswith("s"):
            sec = float(p[:-1])
            offsets.append(int(round(sec * fps)))
        else:
            offsets.append(int(p))
    return sorted(list(set(offsets)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="kornia-slam start-time sensitivity evaluation harness."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Path to sequence dataset directory (e.g. MH_05_difficult).",
    )
    parser.add_argument(
        "--sequence-name",
        type=str,
        default="sequence",
        help="Identifier name for the sequence.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/euroc.ron"),
        help="Base RON configuration file.",
    )
    parser.add_argument(
        "--start-offsets",
        type=str,
        default="0,100,200,300,400,600",
        help="Comma-separated start offsets in frames or seconds (e.g. '0,100,200' or '0s,5s,10s').",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="Camera capture frame rate (default: 20.0 Hz).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_output"),
        help="Directory where CSV and JSON reports are saved.",
    )
    parser.add_argument(
        "--bin-path",
        type=str,
        default="target/release/kornia-slam",
        help="Path to compiled kornia-slam binary (default: target/release/kornia-slam).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run self-contained synthetic smoke test without external datasets.",
    )

    args = parser.parse_args()
    start_offsets = parse_offsets(args.start_offsets, args.fps)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke_test:
        print(f"[*] Running synthetic smoke test over offsets: {start_offsets}")
        metrics = run_synthetic_smoke_test(start_offsets, fps=args.fps)
    else:
        if not args.dataset_path or not args.dataset_path.exists():
            print(
                f"[ERROR] Dataset path '{args.dataset_path}' does not exist.\n"
                "To test the harness without local data, run with '--smoke-test'.",
                file=sys.stderr,
            )
            sys.exit(1)

        print(
            f"[*] Running start-time evaluation on '{args.sequence_name}' "
            f"({len(start_offsets)} start offsets)..."
        )
        metrics = []
        for offset in start_offsets:
            print(f"  -> Testing start offset: {offset} frames ({offset / args.fps:.2f} s)...")
            res = run_offset_real(
                kornia_slam_bin=args.bin_path,
                base_ron_path=args.config,
                dataset_path=args.dataset_path,
                start_frame=offset,
                offset_out_dir=args.output_dir / f"offset_{offset}",
                fps=args.fps,
            )
            metrics.append(res)

    summary = compute_summary(metrics)
    print_table(metrics, summary)

    # Save CSV
    csv_path = args.output_dir / "start_time_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(metrics[0]).keys()))
        writer.writeheader()
        for m in metrics:
            writer.writerow(asdict(m))

    # Save JSON
    json_path = args.output_dir / "start_time_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "sequence_name": args.sequence_name,
                "is_smoke_test": args.smoke_test,
                "fps": args.fps,
                "metrics": [asdict(m) for m in metrics],
                "summary": summary,
            },
            f,
            indent=2,
        )

    print(f"[+] Saved metrics CSV to: {csv_path}")
    print(f"[+] Saved summary JSON to: {json_path}")


if __name__ == "__main__":
    main()
