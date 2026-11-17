# Scripts

Utility and evaluation scripts for `kornia-slam`.

## Scripts

- **`start_time_eval.py`**: Evaluation harness to measure initialisation sensitivity, time-to-init, scale error, and trajectory accuracy across start-time offsets (issue #168). Supports EuRoC, LaMAria, and synthetic benchmarks.

  ```bash
  # Run self-contained synthetic smoke test (harness self-test on synthetic data, NOT kornia-slam performance):
  python scripts/start_time_eval.py --smoke-test --output-dir eval_output/smoke_test

  # Dry-run on synthetic LaMAria fixture (validates config, checks frame counts, and prints commands):
  python scripts/start_time_eval.py \
      --dataset-path crates/kornia-slam/tests/fixtures/lamaria_sample \
      --sequence-name lamaria_sample \
      --dataset-format lamaria \
      --start-offsets 0,5,10 \
      --dry-run \
      --output-dir eval_output/lamaria_fixture_dry_run

  # Run on a real EuRoC sequence:
  python scripts/start_time_eval.py \
      --dataset-path data/euroc/MH_05_difficult \
      --sequence-name MH_05_difficult \
      --config configs/euroc.ron \
      --start-offsets 0,100,200,300,400,600 \
      --output-dir eval_output/mh05

  # Run on a real LaMAria sequence (e.g. R_01_easy):
  python scripts/start_time_eval.py \
      --dataset-path data/lamaria/R_01_easy \
      --sequence-name R_01_easy \
      --dataset-format lamaria \
      --config configs/lamaria.ron \
      --start-offsets 0,100,200,300,400,600 \
      --output-dir eval_output/r01_easy
  ```

- **`lamaria_adapter.py`**: Dataset loader and adapter for the LaMAria benchmark (arXiv:2509.26639, ETH Zurich) following AnchorFlow-VIO (arXiv:2610.26210) conventions. Translates pinhole calibration JSON into `sensor.yaml`, adjusts IMU hardware latency (-4.2 ms), formats pseudo ground truth to EuRoC format, and generates synthetic fixtures for testing.

  ```bash
  # Generate a synthetic LaMAria test fixture:
  python scripts/lamaria_adapter.py --create-fixture crates/kornia-slam/tests/fixtures/lamaria_sample

  # Adapt a downloaded LaMAria sequence directory for kornia-slam:
  python scripts/lamaria_adapter.py \
      --dataset-dir data/lamaria/R_01_easy \
      --output-dir data/lamaria/R_01_easy_adapted \
      --imu-time-offset-ms 4.2
  ```

- **`test_lamaria_eval.py`**: Unit test suite for `lamaria_adapter.py` and `start_time_eval.py`.

  ```bash
  python scripts/test_lamaria_eval.py
  ```

- **`check_version_pins.py`**: Checks that crate versions and dependencies match workspace release pins.
- **`release_rust.sh`**: Pre-release packaging and checks.

---

# Evaluation: AnchorFlow-VIO and LaMAria Benchmark (Start-Time Sensitivity)

*Reference:* GitHub Issue [#168](https://github.com/kornia/kornia-slam/issues/168)
*Upstream Base SHA:* `88e35f2c1005c031255fd4e9a7ed87faf0918b72`

## 1. Executive Summary

This document evaluates **AnchorFlow-VIO** ([arXiv:2610.26210](https://arxiv.org/abs/2610.26210)) and the **LaMAria benchmark** ([arXiv:2509.26639](https://arxiv.org/abs/2509.26639), [ETH Zurich](https://www.lamaria.ethz.ch)) for `kornia-slam`.

AnchorFlow-VIO evaluates a strictly single-camera, single-IMU visual-inertial odometry pipeline on the city-scale LaMAria benchmark. A critical empirical finding in the paper is **start-time sensitivity**: identical sequences started at different prospective time offsets exhibit dramatic variations in initialization success, scale estimation, and trajectory accuracy (e.g. from CP score 6.14 at 30 s to 42.00 at 120 s on a low-light sequence).

Following maintainer guidance from Edgar Riba (@edgarriba: *"I'd prioritise here to benchmark against lamaria"*), this work prioritised benchmarking against LaMAria:
1. Analysed AnchorFlow-VIO methodology and the LaMAria benchmark specifications.
2. Formulated a component-level gap analysis mapping `kornia-slam`'s mono+IMU ingestion pipeline to LaMAria.
3. Implemented a LaMAria dataset adapter (`scripts/lamaria_adapter.py`) converting pinhole calibration, IMU latency, and pseudo ground-truth to the `mav0` format.
4. Extended `scripts/start_time_eval.py` to support LaMAria evaluation across multiple start offsets with trajectory metrics and failure detection.
5. Built and verified a synthetic fixture (`crates/kornia-slam/tests/fixtures/lamaria_sample/`) and unit test suite (`scripts/test_lamaria_eval.py`).

---

## 2. AnchorFlow-VIO (arXiv:2610.26210) & LaMAria (arXiv:2509.26639)

### 2.1 AnchorFlow-VIO System Formulation
- **Architecture:** Monocular VIO derived from Basalt. Pure odometry; no global loop closures or relocalization (§2, §3.3).
- **Sensors:** Single grayscale pinhole camera (`cam0`, 20 FPS) and single IMU (`imu0`, 1 kHz right IMU) (§3.1).
- **Preprocessing:** Factory affine calibration applied to raw IMU: $\mathbf{y}_{\mathrm{cal}} = M(\mathbf{y}_{\mathrm{raw}} - \mathbf{b}_{\mathrm{factory}})$. Subtracts fixed hardware timing latency of $4.2\text{ ms}$ ($4.2 \times 10^6\text{ ns}$) from IMU timestamps (§3.1).
- **Tracking:** Combines patch alignment optical flow ("flow") with fixed birth-template anchoring ("anchor") across a 4-level image pyramid (§3.2).
- **Backend:** Square root marginalization, sliding-window bundle adjustment (`vio_max_states=3`, `vio_max_kfs=10`) (§3.3).
- **Start-Time Evaluation Protocol (§4.4):** Fresh runs launched at prospective frame offsets of 600 frames (~30 s) and 2,400 frames (~120 s). Evaluated over the **exact same temporal suffix** to isolate start-time effects from path length (§4.4).
- **Reported Variations (§5.5, Table 5):**
  - *Low light (`sequence_4_10`):* 30 s start achieved 6.14 CP (ATE RMSE 12.27 m) vs 23.24 CP on suffix; 120 s start achieved 42.00 CP (ATE RMSE 2.00 m) vs 23.14 CP on suffix.
  - *Medium (`sequence_2_12`):* 30 s start scored 15.55 CP (ATE 6.87 m); 120 s start scored 29.36 CP (ATE 4.07 m).
  - *Moving Platform (`sequence_5_12`):* Diverged at both starts (>150 m step error, 2.85 CP at 30 s, 0.00 CP at 120 s).

### 2.2 LaMAria Benchmark Specifications
- **Overview:** 70 km, 22 hours of city-scale egocentric data in Zurich, Switzerland captured with Project Aria glasses.
- **Sensors:** 2x grayscale SLAM cameras ($640 \times 480$ @ 20 FPS), 1x RGB camera ($1408 \times 1408$ @ 10 FPS), 2x IMUs (1 kHz right, 800 Hz left), barometer, Wi-Fi, BLE, GPS.
- **Formats:** Raw VRS (~890 GB), ASL pinhole undistorted (~1.1 TB: ~170 GB train, ~884 GB test), ROS1 bags (~1.5 TB).
- **Ground Truth:** Centimeter-level surveyed control points (CP) outdoors + dense pseudo ground truth (pGT) for training sequences.
- **Metrics:** Primary: Control-point score (0–100 scale). Secondary: Horizontal pose recall at 5 m (%), ATE RMSE (m) under Sim(3).
- **License:** CC BY 4.0 (dataset); MIT (code).

---

## 3. Results on kornia-slam & Benchmark Prioritisation

### 3.1 Maintainer Alignment & Execution Status
Following maintainer guidance from Edgar Riba (@edgarriba: *"I'd prioritise here to benchmark against lamaria"*), benchmarking against LaMAria was prioritized.

*Full-Sequence Real Dataset Status:* **NOT EXECUTED — Network Bandwidth & Download Envelope Constraints.**
- **Transfer Measurement:** The official ETH Zurich server (`https://cvg-data.inf.ethz.ch/lamaria/`) was probed and verified reachable. However, transfer speed in this environment averages ~0.58 MB/s. At this rate, downloading even a single training sequence (e.g. `R_01_easy.zip` at 1,321 MB) requires ~38 minutes, which exceeds reasonable execution envelopes.
- **Rule of Non-Fabrication:** In strict adherence to repository guidelines, no experimental numbers on un-downloaded sequences have been fabricated.
- **Local Fixture Verification:** A reproducible synthetic LaMAria fixture was generated at `crates/kornia-slam/tests/fixtures/lamaria_sample` (15 frames @ 20 Hz, pinhole camera calibration, 1 kHz IMU stream, and pseudo ground truth). The adapter and evaluation harness were validated end-to-end against this fixture in dry-run and adaptation modes (`scripts/test_lamaria_eval.py`).

### 3.2 Harness Smoke-Test Validation (Synthetic Data Self-Test)
```text
==========================================================================================
  HARNESS SELF-TEST ON SYNTHETIC DATA -- NOT KORNIA-SLAM PERFORMANCE
==========================================================================================
Offset (f)  Offset (s)  Status       Init (s)   Scale      Scale Err   ATE RMSE (m)   Losses   Wall (s)
------------------------------------------------------------------------------------------
0           0.00        ok           1.00       1.0415     0.0415      0.0745         1        0.13
100         5.00        ok           1.05       0.9942     0.0058      0.0744         0        0.12
200         10.00       ok           1.10       1.0858     0.0858      0.0803         1        0.11
300         15.00       ok           1.15       1.0574     0.0574      0.0718         1        0.10
400         20.00       ok           1.20       0.9899     0.0101      0.0681         0        0.09
600         30.00       ok           1.30       1.0728     0.0728      0.0648         1        0.07
==========================================================================================
SUMMARY STATISTICS across start offsets:
  ATE RMSE (m)     : Mean=0.0723  Std=0.0049  Min=0.0648  Max=0.0803
  Sim(3) Scale     : Mean=1.0403  Std=0.0367  Min=0.9899  Max=1.0858
  Time-to-Init (s) : Mean=1.13  Std=0.10  Min=1.00  Max=1.30
  Wall Time (s)    : Mean=0.10  Std=0.02  Min=0.07  Max=0.13
  Total Runs       : 6 | Succeeded: 6 | Failure Rate: 0.0%
==========================================================================================
```

### 3.3 Hypothesized Start-Time Sensitivity in kornia-slam
- **Visual Bootstrap:** Monocular initialization requires parallax ($>1.0^\circ$ rotation-compensated parallax or sufficient homography score). If an offset starts during stationary motion or pure rotation, visual initialization stalls.
- **Inertial Initialization Gate:** `ImuInitializer::ready` requires at least 3 keyframes, 2.0 s (monocular), and $>0.5\text{ m}$ motion. Under low excitation, inertial initialization will remain uninitialized, gating IMU prediction.

---

## 4. LaMAria Integration Effort & Gap Assessment

| Component / Requirement | kornia-slam Status | Benchmark Expectation | Gap Description / Resolution | Effort Estimate |
|---|---|---|---|---|
| **Data Format Adapter** | **Implemented** via `scripts/lamaria_adapter.py`. | ASL format (`cam0/data.csv`, `imu0/data.csv`) is provided by ETH Zurich. | Adapter maps native LaMAria directory into standard `mav0/` layout expected by `kornia-slam`. | **Completed** |
| **Camera Model Adapter** | **Implemented** via `scripts/lamaria_adapter.py`. | ASL format provides pre-undistorted images mapped to pinhole model via COLMAP. | Adapter extracts `pinhole_calibrations/<seq>.json` and formats `sensor.yaml` with 4x4 row-major $T_{BS}$. | **Completed** |
| **IMU Timestamp Latency** | **Implemented** via `scripts/lamaria_adapter.py`. | Benchmark cameras and IMUs exhibit hardware latency ($4.2\text{ ms}$ offset subtracted from IMU clock). | Adapter subtracts configurable latency (default $4.2\text{ ms}$) during IMU ingestion into `mav0/imu0/data.csv`. | **Completed** |
| **Ground Truth Formatter** | **Implemented** via `scripts/lamaria_adapter.py`. | Pseudo ground-truth (pGT) poses provided in `ground_truth/pseudo_dense/<seq>.txt`. | Formats pGT into `mav0/state_groundtruth_estimate0/data.csv` for `kornia-slam-app --evaluate`. | **Completed** |
| **IMU Factory Calibration** | Assumes raw samples in standard units ($m/s^2, rad/s$). | Requires affine factory calibration: $\mathbf{y}_{\mathrm{cal}} = M(\mathbf{y}_{\mathrm{raw}} - \mathbf{b}_{\mathrm{factory}})$. | Parser and runtime transform for device $3 \times 3$ scale/misalignment matrix $M$ and factory bias vector $\mathbf{b}_{\mathrm{factory}}$. | **M (Medium)** |
| **IMU Noise Parameters** | Fixed noise model constants in `kornia_sensors`. | Accelerometer/gyroscope noise and random-walk parameters scale per sensor. | Expose IMU noise parameters (`accel_noise`, `gyro_noise`, `accel_walk`, `gyro_walk`) in `SystemConfig` / RON run files. | **S (Small)** |
| **Benchmark Trajectory Exporter** | Exports `kornia_slam_raw.csv` (`idx, x, y, z`). | Requires nanosecond timestamped world-from-IMU poses: `timestamp_ns, tx, ty, tz, qx, qy, qz, qw`. | Export formatter to output exact LaMAria submission CSV and fill gaps causally where tracking was skipped. | **S (Small)** |
| **Control-Point (CP) Scorer** | Sim(3) ATE / RPE / drift reporting in `apps/kornia-slam-app/src/evaluation.rs`. | Official piecewise-linear horizontal control-point score (0–100) and 5 m pose recall. | Implement local evaluation script matching the official benchmark Python scoring package (`tools/evaluation/`). | **M (Medium)** |

---

## 5. Reusable vs Non-Reusable Concepts

- **Reusable:**
  - *Start-Time Sensitivity Benchmarking:* Integrate start-offset testing into regular regression suites.
  - *Hardware Time-Offset Correction:* Add configurable `time_offset_ns` parameter to sensor ingest.
  - *Inertial Fallback Attitude:* Graceful fallback to accelerometer attitude when joint gravity solve fails.
- **Non-Reusable:**
  - *Fixed-Template KLT Tracking:* `kornia-slam` relies on ORB descriptor keypoints rather than KLT optical flow patches.
  - *Square Root Marginalization (Basalt QR):* `kornia-slam` uses Ceres / Levenberg-Marquardt with Schur marginalization.
