#!/usr/bin/env python3
"""Export CLIDD features for every image of a EuRoC sequence, for
kornia-slam's `Precomputed` frontend.

    export_clidd_features.py ~/datasets/euroc/MH_01_easy -o MH_01_easy.clidd_l64.kfeat
    export_clidd_features.py ~/datasets/euroc/MH_01_easy --view left -o MH_01_easy.clidd_l64.left.kfeat
    export_clidd_features.py ~/datasets/euroc/MH_01_easy --view right -o MH_01_easy.clidd_l64.right.kfeat

Needs torch, kornia (with `kornia.feature.CLIDD`), opencv-python and numpy.
Keypoints are in the pixels of the image the system is given: the raw
(distorted) cam0 image for monocular runs (`--view raw`), and for stereo the
rectified cam0 and cam1 images (`--view left`, `--view right`), rectified as
the EuRoC stereo source does it. CLIDD detects on the image resized to a
multiple of 32 (736 wide for EuRoC's 752), so x is an integer scaled by
752/736 and y, at 480 rows, an integer.

Little-endian layout, read by `crates/kornia-slam/src/frontend/feature_file.rs`:

    magic         8 bytes   b"KSLMFEAT"
    version       u32       1
    dim           u32       floats per descriptor
    n_frames      u32
    n_frames times, by strictly increasing timestamp:
      timestamp   i64       image timestamp, ns
      n           u32       keypoints
      keypoints   f32[n][2] (x, y)
      scores      f32[n]
      descriptors f32[n][dim], L2-normalised
"""

import argparse
import re
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

MAGIC = b"KSLMFEAT"
VERSION = 1


def camera_frames(root: Path, camera: str) -> list[tuple[int, Path]]:
    """Every (timestamp ns, image path) in mav0/<camera>/data.csv, in time order."""
    folder = root / "mav0" / camera
    frames = []
    for line in (folder / "data.csv").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        stamp, name = line.split(",")[:2]
        frames.append((int(stamp), folder / "data" / name.strip()))
    return sorted(frames)


def calibration(root: Path, camera: str) -> dict:
    """K, distortion (k1, k2, p1, p2), size and T_BS from mav0/<camera>/sensor.yaml."""
    text = (root / "mav0" / camera / "sensor.yaml").read_text()
    text = re.sub(r"#[^\n]*", "", text)

    def numbers(key: str) -> np.ndarray:
        found = re.search(rf"^\s*{key}:\s*\[([^\]]*)\]", text, re.MULTILINE)
        if found is None:
            raise SystemExit(f"{camera}/sensor.yaml has no {key}")
        return np.array([float(v) for v in found.group(1).split(",")])

    fu, fv, cu, cv_ = numbers("intrinsics")
    width, height = (int(v) for v in numbers("resolution"))
    return {
        "K": np.array([[fu, 0, cu], [0, fv, cv_], [0, 0, 1]]),
        "D": numbers("distortion_coefficients"),
        "size": (width, height),
        "T_BS": numbers("data").reshape(4, 4),
    }


def rodrigues(v: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(v, dtype=np.float64).reshape(3, 1))[0]


def rectification_maps(root: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-view undistort + rectify remap tables, as kornia-3d's
    `StereoRectifier::from_calib` builds them from EuRoC's calibration:
    Bouguet's rotations, a shared focal length (the mean of both cameras' fx
    and fy) and the principal point at the image centre."""
    left, right = calibration(root, "cam0"), calibration(root, "cam1")
    if left["size"] != right["size"]:
        raise SystemExit("cam0 and cam1 resolutions differ")
    r_l, t_l = left["T_BS"][:3, :3], left["T_BS"][:3, 3]
    r_r, t_r = right["T_BS"][:3, :3], right["T_BS"][:3, 3]
    r_rel = r_r.T @ r_l
    t_rel = r_r.T @ (t_l - t_r)

    half = rodrigues(-0.5 * cv2.Rodrigues(r_rel)[0].ravel())
    t = half @ t_rel
    axis = 0 if abs(t[0]) > abs(t[1]) else 1
    target = np.zeros(3)
    target[axis] = 1.0 if t[axis] > 0 else -1.0
    w = np.cross(t, target)
    norm = np.linalg.norm(w)
    if norm > 0:
        w *= np.arccos(np.clip(abs(t[axis]) / np.linalg.norm(t), -1.0, 1.0)) / norm
    align = rodrigues(w)

    width, height = left["size"]
    f = (left["K"][0, 0] + left["K"][1, 1] + right["K"][0, 0] + right["K"][1, 1]) / 4.0
    rectified = np.array([[f, 0, (width - 1) / 2], [0, f, (height - 1) / 2], [0, 0, 1]])
    maps = {}
    for view, calib, rotation in (("left", left, align @ half.T), ("right", right, align @ half)):
        maps[view] = cv2.initUndistortRectifyMap(
            calib["K"], calib["D"], rotation, rectified, (width, height), cv2.CV_32FC1
        )
    return maps


def load_model(variant: str, top_k: int, device: str):
    import kornia.feature as KF

    return KF.CLIDD.from_pretrained(variant, top_k=top_k).to(device).eval()


@torch.inference_mode()
def extract(model, gray: np.ndarray, device: str):
    """Keypoints (N, 2), scores (N,) and unit descriptors (N, D), float32."""
    x = torch.from_numpy(gray).to(device).float()[None, None] / 255.0
    out = model(x)[0]
    desc = torch.nn.functional.normalize(out["descriptors"].float(), dim=1)
    return (
        out["keypoints"].float().cpu().numpy(),
        out["scores"].float().cpu().numpy(),
        desc.cpu().numpy(),
    )


def inside(keypoints: np.ndarray, width: int, height: int) -> np.ndarray:
    """Keypoints clipped into the image, which the reader requires."""
    upper = np.array([width, height], dtype=np.float32) - 1e-3
    return np.clip(keypoints, 0.0, upper)


def frame_record(stamp: int, keypoints, scores, desc) -> bytes:
    le = np.dtype("<f4")
    return b"".join(
        [
            struct.pack("<qI", stamp, len(keypoints)),
            np.ascontiguousarray(keypoints, dtype=le).tobytes(),
            np.ascontiguousarray(scores, dtype=le).tobytes(),
            np.ascontiguousarray(desc, dtype=le).tobytes(),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sequence", type=Path, help="EuRoC sequence root (contains mav0/)")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument(
        "--view",
        choices=["raw", "left", "right"],
        default="raw",
        help="raw cam0 images (monocular), or the rectified cam0 / cam1 images (stereo)",
    )
    parser.add_argument("--variant", default="L64", help="CLIDD variant, 64-D only")
    parser.add_argument("--top-k", type=int, default=1000, help="keypoints per frame")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    frames = camera_frames(args.sequence, "cam1" if args.view == "right" else "cam0")
    remap = None if args.view == "raw" else rectification_maps(args.sequence)[args.view]
    model = load_model(args.variant, args.top_k, args.device)
    clipped = 0
    with open(args.output, "wb") as out:
        out.write(MAGIC)
        dim_offset = out.tell()
        out.write(struct.pack("<III", VERSION, 0, len(frames)))
        dim = None
        for i, (stamp, path) in enumerate(frames):
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise SystemExit(f"cannot read {path}")
            if remap is not None:
                gray = cv2.remap(gray, *remap, cv2.INTER_LINEAR)
            keypoints, scores, desc = extract(model, gray, args.device)
            if dim is None:
                dim = desc.shape[1]
            elif desc.shape[1] != dim:
                raise SystemExit(f"{path}: descriptor size changed to {desc.shape[1]}")
            height, width = gray.shape
            kept = inside(keypoints, width, height)
            clipped += int(np.any(kept != keypoints))
            out.write(frame_record(stamp, kept, scores, desc))
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(frames)} frames", file=sys.stderr)
        out.seek(dim_offset)
        out.write(struct.pack("<III", VERSION, dim or 0, len(frames)))
    print(
        f"wrote {len(frames)} frames to {args.output} ({clipped} with keypoints clipped into the image)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
