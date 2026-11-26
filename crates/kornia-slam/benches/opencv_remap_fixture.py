"""Write `tests/fixtures/upstream/remap_opencv.bin`: OpenCV's fixed-point
remap of one EuRoC frame, for the `remap_u8` `OpenCvQ5` parity test. The
test stays `#[ignore]` until this file exists.

Usage: python3 opencv_remap_fixture.py <cam0 frame.png>

Use a 752x480 EuRoC `cam0` frame, for example MH_01
`mav0/cam0/data/1403636579763555584.png`. One thread.

- maps: initUndistortRectifyMap of the EuRoC cam0 calibration (radtan,
  k1 -0.283), rotation from stereoRectify with cam1, CV_32FC1
- convertMaps(map_x, map_y, CV_16SC2): 1/32 px coordinates
- remap(frame, fixed maps, INTER_LINEAR, BORDER_CONSTANT 0)

Layout, little endian:
  u32 w, u32 h
  u8[h][w]   frame
  f32[h][w]  map_x
  f32[h][w]  map_y
  u8[h][w]   remap output
"""

import struct
import sys
from pathlib import Path

import cv2
import numpy as np

K0 = np.array([[458.654, 0, 367.215], [0, 457.296, 248.375], [0, 0, 1]])
D0 = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05])
K1 = np.array([[457.587, 0, 379.999], [0, 456.134, 255.238], [0, 0, 1]])
D1 = np.array([-0.28368365, 0.07451284, -0.00010473, -3.55590700e-05])
T_BS0 = np.array(
    [
        [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
        [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
        [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
        [0, 0, 0, 1],
    ]
)
T_BS1 = np.array(
    [
        [0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556],
        [0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024],
        [-0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038],
        [0, 0, 0, 1],
    ]
)


def main(frame: Path) -> None:
    cv2.setNumThreads(1)
    img = cv2.imread(str(frame), cv2.IMREAD_GRAYSCALE)
    assert img is not None
    h, w = img.shape
    t_10 = np.linalg.inv(T_BS1) @ T_BS0
    r0, _, p0, _, _, _, _ = cv2.stereoRectify(K0, D0, K1, D1, (w, h), t_10[:3, :3], t_10[:3, 3:4], alpha=0)
    map_x, map_y = cv2.initUndistortRectifyMap(K0, D0, r0, p0, (w, h), cv2.CV_32FC1)
    fixed_xy, fixed_frac = cv2.convertMaps(map_x, map_y, cv2.CV_16SC2)
    out = cv2.remap(img, fixed_xy, fixed_frac, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    dst = Path(__file__).resolve().parent.parent / "tests/fixtures/upstream/remap_opencv.bin"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as f:
        f.write(struct.pack("<II", w, h))
        f.write(img.astype(np.uint8).tobytes())
        f.write(map_x.astype("<f4").tobytes())
        f.write(map_y.astype("<f4").tobytes())
        f.write(out.astype(np.uint8).tobytes())
    print(f"wrote {dst} ({w}x{h}, OpenCV {cv2.__version__})")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
