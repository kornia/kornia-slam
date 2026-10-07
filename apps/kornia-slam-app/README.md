# kornia-slam-app

This package is the composition root for the `kornia-slam` executable. It wires the
`kornia_slam::SlamSystem` runtime to four interchangeable frame sources — offline EuRoC MAV image sequences, offline MCAP recordings (e.g. bubbaloop captures), a live OAK-D camera, and any UVC-class camera (laptop webcams, USB cams, CSI-to-UVC adapters on a Pi…). All feed the same `SlamSystem::process` loop, and the TUI / Rerun visualizers work for any of them. Sources supply calibrated, synchronized images and IMU samples (rectified for stereo); feature extraction, stereo matching and fisheye keypoint mapping happen in the library. EuRoC, MCAP, and OAK-D additionally support a **stereo mode** (see below) that yields metric depth; UVC is monocular only.

## Pipeline configuration

What the pipeline runs is set by a RON file passed with `--config`: which sensors it uses, the ORB settings, keyframe policy, local-mapping execution and loop closing. Lower-level algorithm thresholds are not part of the file; library users set them in Rust. Without `--config` it runs monocular ORB with loop closing disabled. [`configs/`](../../configs) has ready-made files:

| File | Pipeline |
| --- | --- |
| `mono.ron` | Monocular; lists every setting with its default |
| `mono-imu.ron` | Monocular + IMU |
| `stereo.ron` | Rectified stereo |
| `stereo-imu.ron` | Stereo + IMU |
| `stereo-imu-loop.ron` | Stereo + IMU with loop detection and pose-graph correction |
| `hilti.ron` | Monocular with 3000 keypoints for the Hilti fisheye frames |

Omitted fields keep their defaults, so a file only lists what it changes. Unknown fields and invalid values are rejected before any data is read, and a configuration that selects a sensor the source cannot provide fails with an explicit error. Loop closing needs an ORB vocabulary (DBoW2 `ORBvoc.txt`, or a `.bin` from `convert_orbvoc`); relative vocabulary paths resolve against the config file's directory.

Sensors each source can provide:

| Source | Stereo | IMU |
| --- | --- | --- |
| `euroc` | yes | yes |
| `mcap` | yes, with `--calib` | no |
| `oakd` | yes, with `--calib` | no |
| `hilti` | no | no |
| `uvc` | no | no |

## Frame sources

Selectable via subcommand, after the global options (`--config`, visualizer flags):

```text
kornia-slam [--config FILE.ron] euroc --data /path/to/V1_01_easy [--start-frame N] [--max-frames N] [--evaluate] [--eval-out DIR]
kornia-slam [--config FILE.ron] mcap  --path FILE.mcap [--channel mono_left] [--max-frames N] [--calib calib.yaml --right-channel mono_right]
kornia-slam [--config FILE.ron] oakd  [--width 640 --height 400 --fps 30] [--max-frames N] [--calib calib.yaml]
kornia-slam [--config FILE.ron] uvc   --fx F --fy F --cx C --cy C [--index 0] [--width 640 --height 480] [--max-frames N]
```

`oakd` requires `--features oakd`; `uvc` requires `--features uvc`. The default build needs no extra system dependencies.

## EuRoC dataset

Download the EuRoC MAV dataset from the OpenVINS dataset guide:
<https://docs.openvins.com/gs-datasets.html#gs-data-euroc>

Standard directory layout:

```text
V1_01_easy/
└── mav0/
    ├── cam0/
    │   ├── data.csv
    │   ├── sensor.yaml
    │   └── data/
    │       ├── 1403636579763555584.png
    │       └── ...
    └── state_groundtruth_estimate0/
        └── data.csv
```

`mav0/cam0/{data.csv,sensor.yaml}` and the PNGs under `data/` are required. Ground truth is optional and only parsed by the dataset reader.

[Machine Hall sequences](https://www.research-collection.ethz.ch/entities/researchdata/bcaf173e-5dac-484b-bc37-faf97a594f1f) (MH_01–MH_05) are recommended for initial testing.

```bash
cargo run --release -p kornia-slam-app -- euroc --data /path/to/MH_01_easy
```

## OAK-D camera

Build prerequisites (`depthai-sys` builds [depthai-core](https://github.com/luxonis/depthai-core) v3 from source on first compile — ~5–10 min wall, several GB of `target/`):

- `cmake` (3.20+) and a C/C++ toolchain (`gcc`/`g++` or `clang`)
- `pkg-config`
- udev rules for non-root device access — see `/etc/udev/rules.d/80-movidius.rules` in the [depthai docs](https://docs.luxonis.com/projects/api/en/latest/install/)
- **libclang 14** (or older) so `autocxx`/bindgen can parse depthai-core headers. Clang 19+ rejects a libnop template construct used in vcpkg-installed deps; libclang is pinned at the workspace level via `.cargo/config.toml`:

  ```toml
  [env]
  LIBCLANG_PATH = "/usr/lib/llvm-14/lib"
  ```

  Adjust for your system, or set the env var when invoking cargo.

The Rerun feature (`viz`, default-on) collides with `depthai-sys`'s vendored lz4 at link time. Until either upstream stops vendoring lz4, pass:

```text
RUSTFLAGS="-C link-arg=-Wl,--allow-multiple-definition"
```

at cargo invocation time when building with both `viz` and `oakd`.

```bash
# Live, with Rerun visualization:
RUSTFLAGS="-C link-arg=-Wl,--allow-multiple-definition" \
  cargo run --release -p kornia-slam-app --features oakd -- --rerun-stream oakd

# Live, TUI only (no Rerun, no lz4 clash):
cargo run --release -p kornia-slam-app --no-default-features --features oakd -- oakd
```

In **mono** mode intrinsics are placeholder (rough scale of the OAK-D Pro factory fx/fy at 1280×800); reading the on-device factory calibration is a TODO. In **stereo** mode (a stereo config plus `--calib …`) the intrinsics come from the calibration YAML and online rectification produces metric pairs — see [Stereo mode](#stereo-mode).

## Stereo mode

Selecting `cameras: Stereo` in the configuration opens a left/right pair instead of a single image. Each rectified pair is matched along its rows (`compute_stereo_matches`) to recover per-keypoint disparity, and `depth = bf / disparity` (with `bf = fx · baseline`) gives **metric** depth. This makes initialization metric (no scale ambiguity) and feeds depth into bundle adjustment.

| Source  | How rectification is obtained                                            | Extra flags                          |
| ------- | ------------------------------------------------------------------------ | ------------------------------------ |
| `euroc` | From `cam0`/`cam1` `sensor.yaml` (intrinsics + `T_BS`), computed in-proc | none                                 |
| `mcap`  | From a calibration YAML; left/right channels paired by timestamp         | `--calib c.yaml [--right-channel …]` |
| `oakd`  | From a calibration YAML; CamB+CamC streamed and rectified online         | `--calib c.yaml`                     |

EuRoC is already rectifiable from its `sensor.yaml`, so it needs no `--calib`. MCAP and OAK-D record **raw** (unrectified) frames, so they need a calibration YAML.

### Calibration YAML

The YAML holds per-camera pinhole intrinsics plus 8-coefficient OpenCV distortion `[k1, k2, p1, p2, k3, k4, k5, k6]`, and the left→right extrinsic (row-major 3×3 rotation, translation in metres). Both views are assumed calibrated at the same `width`×`height`:

```yaml
width: 640
height: 400
left:
  fx: 452.1
  fy: 452.1
  cx: 320.5
  cy: 200.2
  distortion: [-0.045, 0.012, 0.0001, -0.0002, 0.0, 0.0, 0.0, 0.0]
right:
  fx: 451.8
  fy: 451.8
  cx: 318.9
  cy: 199.7
  distortion: [-0.043, 0.010, 0.0001, -0.0001, 0.0, 0.0, 0.0, 0.0]
r_left_to_right: [1, 0, 0, 0, 1, 0, 0, 0, 1]   # row-major 3x3
t_left_to_right_m: [-0.075, 0, 0]              # baseline in metres
```

For an OAK-D this is the device's factory calibration (readable once via the depthai Python API's `readCalibration`) dumped to this schema.

### Examples

```bash
# EuRoC stereo (metric), first 500 frames, with evaluation CSVs:
cargo run --release -p kornia-slam-app -- --config configs/stereo.ron \
    euroc --data /path/to/MH_01_easy --max-frames 500 --evaluate

# Offline MCAP stereo (raw OAK-D recording + calibration):
cargo run --release -p kornia-slam-app -- --config configs/stereo.ron \
    mcap --path recording.mcap --calib calib.yaml \
    --channel mono_left --right-channel mono_right

# Live OAK-D stereo (free the device first if a daemon holds it, e.g.
# `bubbaloop node stop oak-camera`):
cargo run --release -p kornia-slam-app --no-default-features --features oakd -- \
    --config configs/stereo.ron oakd --calib calib.yaml --fps 30
```

For the live OAK-D case, stereo uses the `width`/`height` from the YAML; `--width`/`--height` apply to mono only. CamB/CamC are hardware-synced, so consecutive items from each queue are paired directly.

## UVC camera

Any UVC-class device works (built-in laptop webcam, USB camera, CSI-to-UVC adapter on a Raspberry Pi). Unlike EuRoC and OAK-D, there's no on-device calibration, so you must pass intrinsics on the command line — they have to match the resolution the device actually streams at (nokhwa picks the closest supported mode if the exact one is missing).

```bash
# /dev/video0 at 640x480, rough pinhole calibration:
cargo run --release -p kornia-slam-app --features uvc -- \
    uvc --index 0 --fx 600 --fy 600 --cx 320 --cy 240
```

## Visualizers

The TUI is the default — just run the app. Override with one of:

- `--rerun-stream` — spawn a Rerun viewer and stream image / keypoints / trajectory / camera / map points (requires `--features viz`, default on). Disables the TUI.
- `--no-tui` — fall back to plain stderr status lines (no TUI, no Rerun).
- `--debug` — show the debug panel inside the TUI (or extra diagnostic lines on stderr in `--no-tui` mode). Toggle live with the `d` key while the TUI is running.

## Local checks

```bash
cargo fmt -p kornia-slam-app -- --check
cargo clippy -p kornia-slam-app --all-targets -- -D warnings
cargo run -p kornia-slam-app -- --help
```
