"""Write OpenCV reference outputs for the parity tests of the
`upstream/kornia_imgproc` kernels (`lk_window`, `good_features`,
`corner_subpix`) and the LK break cases.

Usage:
  python3 opencv_frontend_fixtures.py <frame1.png> <frame2.png>
  python3 opencv_frontend_fixtures.py <frame1.png> <frame2.png> --out F
      [--levels L] [--win W] [--crop X,Y,W,H] [--init-flow DX,DY]

Use two consecutive grayscale frames of one size, for example kornia-rs's
`tests/data/mh01_frame1.png` and `mh01_frame2.png` (752x480). One thread.

Without options it writes version 1 to
`tests/fixtures/upstream/frontend_opencv.bin`; the parity tests stay
`#[ignore]` until that file exists. Any option writes version 2 to `--out`
(required then), for example a break case's 390x292 crop of two LaMAria
frames with the production pyramid.

- goodFeaturesToTrack(frame1, 200, 0.01, 23, mask = 9 px border, blockSize 3)
- cornerSubPix(frame1, those corners, winSize (5, 5), zeroZone (-1, -1),
  COUNT|EPS 30 / 0.01)
- calcOpticalFlowPyrLK(frame1, frame2, the refined corners, winSize (W, W),
  maxLevel L, COUNT|EPS 8 / 0.01, minEigThreshold 1e-4); version 1 has
  W 11 and L 0. With `--init-flow` every point starts at corner + (DX, DY)
  under OPTFLOW_USE_INITIAL_FLOW.

Version 1 layout, little endian:
  u32 w, u32 h
  u8[h][w]   frame1
  u8[h][w]   frame2
  u32 n
  f32[n][2]  goodFeaturesToTrack corners
  f32[n][2]  cornerSubPix corners
  f32[n][2]  calcOpticalFlowPyrLK next points
  u8[n]      calcOpticalFlowPyrLK status

Version 2 layout, little endian: a header, then version 1 on the cropped frames.
  u8[4]      b"KOF2"
  u32 levels, u32 win
  u32 crop_x, u32 crop_y   (top-left of the crop in the source frames)
  f32 flow_dx, f32 flow_dy (0, 0 without --init-flow)
  ...        version 1, w and h the crop's
"""

import argparse
import struct
from pathlib import Path

import cv2
import numpy as np

MAX_CORNERS, QUALITY, MIN_DISTANCE, BORDER = 200, 0.01, 23, 9
SUBPIX_HALF_WIN, SUBPIX_CRIT = 5, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.01)
LK_WIN, LK_CRIT = 11, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 8, 0.01)
MAGIC_V2 = b"KOF2"


def pair(text: str, kind):
    return [kind(v) for v in text.split(",")]


def reference(f1, f2, levels: int, win: int, flow) -> bytes:
    h, w = f1.shape
    mask = np.zeros_like(f1)
    mask[BORDER : h - BORDER, BORDER : w - BORDER] = 255
    corners = cv2.goodFeaturesToTrack(f1, MAX_CORNERS, QUALITY, MIN_DISTANCE, mask=mask, blockSize=3)
    corners = corners.reshape(-1, 2).astype(np.float32)
    refined = corners.copy().reshape(-1, 1, 2)
    cv2.cornerSubPix(f1, refined, (SUBPIX_HALF_WIN, SUBPIX_HALF_WIN), (-1, -1), SUBPIX_CRIT)
    refined = refined.reshape(-1, 2)
    start, flags = None, 0
    if flow != (0.0, 0.0):
        start = (refined + np.float32(flow)).reshape(-1, 1, 2).copy()
        flags = cv2.OPTFLOW_USE_INITIAL_FLOW
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(
        f1,
        f2,
        refined.reshape(-1, 1, 2),
        start,
        winSize=(win, win),
        maxLevel=levels,
        criteria=LK_CRIT,
        flags=flags,
        minEigThreshold=1e-4,
    )
    blob = struct.pack("<II", w, h) + f1.tobytes() + f2.tobytes()
    blob += struct.pack("<I", len(corners)) + corners.tobytes() + refined.astype(np.float32).tobytes()
    blob += nxt.reshape(-1, 2).astype(np.float32).tobytes() + st.reshape(-1).astype(np.uint8).tobytes()
    print(f"{w}x{h}, {len(corners)} corners, {int(st.sum())} tracked, cv2 {cv2.__version__}")
    return blob


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("frame1", type=Path)
    ap.add_argument("frame2", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--levels", type=int)
    ap.add_argument("--win", type=int)
    ap.add_argument("--crop", type=lambda s: pair(s, int))
    ap.add_argument("--init-flow", type=lambda s: pair(s, float))
    a = ap.parse_args()
    cv2.setNumThreads(1)
    f1 = cv2.imread(str(a.frame1), cv2.IMREAD_GRAYSCALE)
    f2 = cv2.imread(str(a.frame2), cv2.IMREAD_GRAYSCALE)
    assert f1 is not None and f2 is not None and f1.shape == f2.shape
    v2 = any(v is not None for v in (a.out, a.levels, a.win, a.crop, a.init_flow))
    if not v2:
        blob = reference(f1, f2, 0, LK_WIN, (0.0, 0.0))
        out = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "upstream" / "frontend_opencv.bin"
    else:
        assert a.out is not None, "version 2 needs --out"
        x, y, w, h = a.crop if a.crop else (0, 0, f1.shape[1], f1.shape[0])
        assert len(a.crop or [0] * 4) == 4 and x + w <= f1.shape[1] and y + h <= f1.shape[0], "crop outside the frame"
        levels = 0 if a.levels is None else a.levels
        win = LK_WIN if a.win is None else a.win
        flow = tuple(a.init_flow) if a.init_flow else (0.0, 0.0)
        assert len(flow) == 2, "--init-flow is DX,DY"
        crop1 = np.ascontiguousarray(f1[y : y + h, x : x + w])
        crop2 = np.ascontiguousarray(f2[y : y + h, x : x + w])
        header = MAGIC_V2 + struct.pack("<IIIIff", levels, win, x, y, *flow)
        blob = header + reference(crop1, crop2, levels, win, flow)
        out = a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(blob)
    print(f"{out}: {len(blob)} bytes")


if __name__ == "__main__":
    main()
