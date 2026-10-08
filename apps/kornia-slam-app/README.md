# kornia-slam-app

This package is the composition root for the `kornia-slam` executable. It wires the
`kornia_slam::SlamSystem` runtime to four interchangeable frame sources — offline EuRoC MAV image sequences, offline MCAP recordings (e.g. bubbaloop captures), a live OAK-D camera, and any UVC-class camera (laptop webcams, USB cams, CSI-to-UVC adapters on a Pi…). All feed the same `SlamSystem::process` loop, and the TUI / Rerun visualizers work for any of them. Sources supply calibrated, synchronized images and IMU samples (rectified for stereo); feature extraction, stereo matching and fisheye keypoint mapping happen in the library. EuRoC, MCAP, and OAK-D additionally support a **stereo mode** (see below) that yields metric depth; UVC is monocular only.

## Run files

A run is described by one RON file passed with `--config`: the `source` to read and the `system` to run on it.

```ron
(
    source: Euroc((data: "../data/euroc/MH_01_easy", max_frames: 500)),
    system: (version: 1, sensors: (cameras: Stereo, imu: true)),
)
```

```bash
cargo run --release -p kornia-slam-app -- --config run.ron
```

- **`source`** is one of `Euroc`, `Hilti`, `Mcap`, `Oakd` or `Uvc`, with the dataset path or device settings, frame range (`start_frame`, `max_frames`; 0 = all) and, where needed, a calibration file. Its options are documented in [`src/config.rs`](src/config.rs).
- **`system`** selects the sensors, ORB settings, keyframe policy, local-mapping execution and loop closing. Omitted, the default monocular pipeline runs. Lower-level algorithm thresholds are not part of the file; library users set them in Rust.
- Omitted fields keep their defaults. Unknown fields and invalid values are rejected before any data is read, and selecting a sensor the source cannot provide fails with an explicit error.
- Relative paths (datasets, recordings, calibration, vocabulary) resolve against the run file's directory.
- Loop closing needs an ORB vocabulary (DBoW2 `ORBvoc.txt`, or a `.bin` from `convert_orbvoc`) and stereo or IMU input.

[`configs/`](../../configs) has one example per source; the data paths are placeholders:

| File | Run |
| --- | --- |
| `euroc.ron` | EuRoC, monocular; lists every system setting with its default |
| `euroc-stereo-imu-loop.ron` | EuRoC, stereo + IMU with loop closing |
| `hilti.ron` | Hilti fisheye sequence, 3000 keypoints |
| `mcap-stereo.ron` | MCAP recording, rectified stereo |
| `oakd-stereo.ron` | Live OAK-D stereo (`--features oakd`) |
| `uvc.ron` | Live UVC camera (`--features uvc`) |

Sensors each source can provide:

| Source | Stereo | IMU |
| --- | --- | --- |
| `Euroc` | yes | yes |
| `Mcap` | yes, with `calib` | no |
| `Oakd` | yes, with `calib` | no |
| `Hilti` | no | no |
| `Uvc` | no | no |

`Oakd` requires `--features oakd` and `Uvc` requires `--features uvc`; a run file naming them in a build without the feature fails with a message saying which. The default build needs no extra system dependencies.

Command-line options are only about evaluation and display: `--evaluate` and `--eval-out DIR` (sources with ground truth: EuRoC, Hilti), and the visualizer flags below.

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
cargo run --release -p kornia-slam-app -- --config configs/euroc.ron
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
  cargo run --release -p kornia-slam-app --features oakd -- \
  --config configs/oakd-stereo.ron --rerun-stream

# Live, TUI only (no Rerun, no lz4 clash):
cargo run --release -p kornia-slam-app --no-default-features --features oakd -- \
  --config configs/oakd-stereo.ron
```

In **mono** mode intrinsics are placeholder (rough scale of the OAK-D Pro factory fx/fy at 1280×800); reading the on-device factory calibration is a TODO. In **stereo** mode (stereo cameras plus `calib`) the intrinsics come from the calibration YAML and online rectification produces metric pairs — see [Stereo mode](#stereo-mode).

## Stereo mode

Selecting `cameras: Stereo` in the configuration opens a left/right pair instead of a single image. Each rectified pair is matched along its rows (`compute_stereo_matches`) to recover per-keypoint disparity, and `depth = bf / disparity` (with `bf = fx · baseline`) gives **metric** depth. This makes initialization metric (no scale ambiguity) and feeds depth into bundle adjustment.

| Source  | How rectification is obtained                                            | Run-file settings                    |
| ------- | ------------------------------------------------------------------------ | ------------------------------------ |
| `Euroc` | From `cam0`/`cam1` `sensor.yaml` (intrinsics + `T_BS`), computed in-proc | none                                 |
| `Mcap`  | From a calibration YAML; left/right channels paired by timestamp         | `calib`, optionally `right_channel`  |
| `Oakd`  | From a calibration YAML; CamB+CamC streamed and rectified online         | `calib`                              |

EuRoC is already rectifiable from its `sensor.yaml`, so it needs no `calib`. MCAP and OAK-D record **raw** (unrectified) frames, so they need a calibration YAML.

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
# EuRoC stereo + IMU with loop closing, and evaluation CSVs:
cargo run --release -p kornia-slam-app -- \
    --config configs/euroc-stereo-imu-loop.ron --evaluate

# Offline MCAP stereo (raw OAK-D recording + calibration):
cargo run --release -p kornia-slam-app -- --config configs/mcap-stereo.ron

# Live OAK-D stereo (free the device first if a daemon holds it, e.g.
# `bubbaloop node stop oak-camera`):
cargo run --release -p kornia-slam-app --no-default-features --features oakd -- \
    --config configs/oakd-stereo.ron
```

For the live OAK-D case, stereo uses the `width`/`height` from the YAML; the run file's `width`/`height` apply to mono only. CamB/CamC are hardware-synced, so consecutive items from each queue are paired directly.

## UVC camera

Any UVC-class device works (built-in laptop webcam, USB camera, CSI-to-UVC adapter on a Raspberry Pi). Unlike EuRoC and OAK-D, there's no on-device calibration, so the run file gives the intrinsics (`fx`, `fy`, `cx`, `cy` and optional distortion) — they have to match the resolution the device actually streams at (nokhwa picks the closest supported mode if the exact one is missing).

```bash
# /dev/video0 at 640x480, rough pinhole calibration (see configs/uvc.ron):
cargo run --release -p kornia-slam-app --features uvc -- --config configs/uvc.ron
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
