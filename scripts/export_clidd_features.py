#!/usr/bin/env python3
"""Export CLIDD features for every cam0 image of a EuRoC sequence, for
kornia-slam's `Precomputed` frontend.

    export_clidd_features.py ~/datasets/euroc/MH_01_easy -o MH_01_easy.clidd_l64.kfeat

Needs torch, kornia (with `kornia.feature.CLIDD`), opencv-python and numpy.
Keypoints are in pixels of the raw (distorted) cam0 image, as the monocular
EuRoC source hands images to the system. CLIDD detects on the image resized to a
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
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

MAGIC = b"KSLMFEAT"
VERSION = 1


def cam0_frames(root: Path) -> list[tuple[int, Path]]:
    """Every (timestamp ns, image path) in mav0/cam0/data.csv, in time order."""
    cam0 = root / "mav0" / "cam0"
    frames = []
    for line in (cam0 / "data.csv").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        stamp, name = line.split(",")[:2]
        frames.append((int(stamp), cam0 / "data" / name.strip()))
    return sorted(frames)


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
    parser.add_argument("--variant", default="L64", help="CLIDD variant, 64-D only")
    parser.add_argument("--top-k", type=int, default=1000, help="keypoints per frame")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    frames = cam0_frames(args.sequence)
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
