"""Convert a LaMAria training sequence to the EuRoC MAV layout read by kornia-slam-app.

LaMAria (https://github.com/cvg/lamaria) ships an ASL export that is EuRoC-like but
differs in ways the EuRoC loader rejects: the root is `aria/` not `mav0/`, there is no
`sensor.yaml`, PNGs are 3-channel gray, and the pseudo ground truth is a separate
`timestamp tx ty tz qx qy qz qw` file holding world_from_imu poses.

Usage (then run configs/lamaria-imu.ron; evaluate with `--trajectory-out` and
LaMAria's `evaluate_wrt_mps.py`):
    python3 -I scripts/lamaria_to_euroc.py <lamaria>/training/R_01_easy data/lamaria/R_01_easy
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

CAMERA_RATE_HZ = 20


def quat_xyzw_to_matrix(q: list[float]) -> np.ndarray:
    x, y, z, w = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat_wxyz(r: np.ndarray) -> np.ndarray:
    w = np.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    x = np.sqrt(max(0.0, 1 + r[0, 0] - r[1, 1] - r[2, 2])) / 2
    y = np.sqrt(max(0.0, 1 - r[0, 0] + r[1, 1] - r[2, 2])) / 2
    z = np.sqrt(max(0.0, 1 - r[0, 0] - r[1, 1] + r[2, 2])) / 2
    x = np.copysign(x, r[2, 1] - r[1, 2])
    y = np.copysign(y, r[0, 2] - r[2, 0])
    z = np.copysign(z, r[1, 0] - r[0, 1])
    return np.array([w, x, y, z])


def pose_matrix(rotation: np.ndarray, translation) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rotation
    t[:3, 3] = translation
    return t


def body_from_sensor(calib: dict) -> np.ndarray:
    # LaMAria stores qvec as x, y, z, w (the IMU's identity is [0, 0, 0, 1]).
    return pose_matrix(quat_xyzw_to_matrix(calib["T_b_s"]["qvec"]), calib["T_b_s"]["tvec"])


def sensor_yaml(calib: dict) -> str:
    fx, fy, cx, cy = calib["params"]
    res = calib["resolution"]
    t_bs = ", ".join(f"{v:.17g}" for v in body_from_sensor(calib).ravel())
    return (
        "sensor_type: camera\n"
        f"rate_hz: {CAMERA_RATE_HZ}\n"
        f"resolution: [{res['width']}, {res['height']}]\n"
        "camera_model: pinhole\n"
        f"intrinsics: [{fx!r}, {fy!r}, {cx!r}, {cy!r}]\n"
        "distortion_model: radial-tangential\n"
        "distortion_coefficients: [0.0, 0.0, 0.0, 0.0]\n"
        "T_BS:\n"
        "  cols: 4\n"
        "  rows: 4\n"
        f"  data: [{t_bs}]\n"
    )


def to_gray(paths: tuple[Path, Path]) -> None:
    src, dst = paths
    if not dst.exists():
        Image.open(src).convert("L").save(dst)


def convert_camera(src_cam: Path, dst_cam: Path, calib: dict, workers: int) -> int:
    (dst_cam / "data").mkdir(parents=True, exist_ok=True)
    rows = [line.split(",") for line in src_cam.joinpath("data.csv").read_text().split()]
    with open(dst_cam / "data.csv", "w") as f:
        f.write("#timestamp [ns],filename\n")
        for ts, name in rows:
            f.write(f"{ts},{name}\n")
    (dst_cam / "sensor.yaml").write_text(sensor_yaml(calib))
    jobs = [(src_cam / "data" / name, dst_cam / "data" / name) for _, name in rows]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(to_gray, jobs, chunksize=32))
    return len(rows)


def convert_imu(src: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    with open(src) as fin, open(dst_dir / "data.csv", "w") as fout:
        fout.write(
            "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
            "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n"
        )
        for line in fin:
            if line.strip():
                fout.write(line if line.endswith("\n") else line + "\n")


def convert_ground_truth(src: Path, dst_dir: Path, imu_from_frame: np.ndarray) -> int:
    """Writes EuRoC GT (`ts, p, q_wxyz`) as world_from_frame = world_from_imu · imu_from_frame."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(src) as fin, open(dst_dir / "data.csv", "w") as fout:
        fout.write("#timestamp,p_x,p_y,p_z,q_w,q_x,q_y,q_z\n")
        for line in fin:
            cols = line.split()
            if len(cols) < 8 or line.startswith("#"):
                continue
            ts, vals = cols[0], [float(v) for v in cols[1:8]]
            world_from_imu = pose_matrix(quat_xyzw_to_matrix(vals[3:7]), vals[0:3])
            pose = world_from_imu @ imu_from_frame
            p, q = pose[:3, 3], matrix_to_quat_wxyz(pose[:3, :3])
            fout.write(f"{ts}," + ",".join(f"{v:.9f}" for v in (*p, *q)) + "\n")
            n += 1
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("sequence_dir", type=Path, help="LaMAria <split>/<seq> download folder")
    parser.add_argument("output_dir", type=Path, help="EuRoC-layout output folder")
    parser.add_argument(
        "--gt-frame",
        choices=["cam0", "imu"],
        default="cam0",
        help="Frame the GT positions refer to; kornia-slam evaluates cam0 centres",
    )
    parser.add_argument("--no-stereo", action="store_true", help="Skip cam1")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    seq = args.sequence_dir.name
    aria = args.sequence_dir / "asl_folder" / seq / "aria"
    calib = json.loads((args.sequence_dir / "pinhole_calibrations" / f"{seq}.json").read_text())
    mav0 = args.output_dir / "mav0"

    cams = ["cam0"] if args.no_stereo else ["cam0", "cam1"]
    for cam in cams:
        n = convert_camera(aria / cam, mav0 / cam, calib[cam], args.workers)
        print(f"[{cam}] {n} frames")
    convert_imu(aria / "imu0" / "data.csv", mav0 / "imu0")
    print("[imu0] done")

    gt = args.sequence_dir / "ground_truth" / "pGT" / f"{seq}.txt"
    if gt.exists():
        imu_from_frame = body_from_sensor(calib["cam0"]) if args.gt_frame == "cam0" else np.eye(4)
        n = convert_ground_truth(gt, mav0 / "state_groundtruth_estimate0", imu_from_frame)
        print(f"[gt] {n} poses ({args.gt_frame} frame)")
    else:
        print("[gt] no pGT for this sequence (test split?); --evaluate unavailable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
