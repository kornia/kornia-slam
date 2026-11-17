"""Write `tests/fixtures/upstream/frontend_opencv.bin`: OpenCV reference
outputs for the parity tests of the `upstream/kornia_imgproc` kernels
(`lk_window`, `good_features`, `corner_subpix`). Those tests stay
`#[ignore]` until this file exists.

Usage: python3 opencv_frontend_fixtures.py <frame1.png> <frame2.png>

Use two consecutive grayscale frames of one size, for example kornia-rs's
`tests/data/mh01_frame1.png` and `mh01_frame2.png` (752x480). One thread.

- goodFeaturesToTrack(frame1, 200, 0.01, 23, mask = 9 px border, blockSize 3)
- cornerSubPix(frame1, those corners, winSize (5, 5), zeroZone (-1, -1),
  COUNT|EPS 30 / 0.01)
- calcOpticalFlowPyrLK(frame1, frame2, the refined corners, winSize (11, 11),
  maxLevel 0, COUNT|EPS 8 / 0.01, minEigThreshold 1e-4)

Layout, little endian:
  u32 w, u32 h
  u8[h][w]   frame1
  u8[h][w]   frame2
  u32 n
  f32[n][2]  goodFeaturesToTrack corners
  f32[n][2]  cornerSubPix corners
  f32[n][2]  calcOpticalFlowPyrLK next points
  u8[n]      calcOpticalFlowPyrLK status
"""

import struct
import sys
from pathlib import Path

import cv2
import numpy as np

MAX_CORNERS, QUALITY, MIN_DISTANCE, BORDER = 200, 0.01, 23, 9
SUBPIX_HALF_WIN, SUBPIX_CRIT = 5, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.01)
LK_WIN, LK_CRIT = 11, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 8, 0.01)


def main(frame1: Path, frame2: Path) -> None:
    cv2.setNumThreads(1)
    f1 = cv2.imread(str(frame1), cv2.IMREAD_GRAYSCALE)
    f2 = cv2.imread(str(frame2), cv2.IMREAD_GRAYSCALE)
    assert f1 is not None and f2 is not None and f1.shape == f2.shape
    h, w = f1.shape
    mask = np.zeros_like(f1)
    mask[BORDER : h - BORDER, BORDER : w - BORDER] = 255
    corners = cv2.goodFeaturesToTrack(f1, MAX_CORNERS, QUALITY, MIN_DISTANCE, mask=mask, blockSize=3)
    corners = corners.reshape(-1, 2).astype(np.float32)
    refined = corners.copy().reshape(-1, 1, 2)
    cv2.cornerSubPix(f1, refined, (SUBPIX_HALF_WIN, SUBPIX_HALF_WIN), (-1, -1), SUBPIX_CRIT)
    refined = refined.reshape(-1, 2)
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(
        f1,
        f2,
        refined.reshape(-1, 1, 2),
        None,
        winSize=(LK_WIN, LK_WIN),
        maxLevel=0,
        criteria=LK_CRIT,
        minEigThreshold=1e-4,
    )
    blob = struct.pack("<II", w, h) + f1.tobytes() + f2.tobytes()
    blob += struct.pack("<I", len(corners)) + corners.tobytes() + refined.astype(np.float32).tobytes()
    blob += nxt.reshape(-1, 2).astype(np.float32).tobytes() + st.reshape(-1).astype(np.uint8).tobytes()
    out = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "upstream" / "frontend_opencv.bin"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(blob)
    print(f"{w}x{h}, {len(corners)} corners, {int(st.sum())} tracked, {len(blob)} bytes, cv2 {cv2.__version__}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
