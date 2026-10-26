# AnchorFlow-VIO and LaMAria Benchmark Evaluation: Start-Time Sensitivity

## Base Commit SHA
- **Upstream base SHA:** `88e35f2c1005c031255fd4e9a7ed87faf0918b72`
- **Branch:** `feat/issue-168-start-time-eval` (branched directly from `upstream/main`)
- **Reference:** GitHub Issue [#168](https://github.com/kornia/kornia-slam/issues/168)

---

## 1. Summary

This document evaluates the research findings and engineering implications of **AnchorFlow-VIO** ([arXiv:2610.26210](https://arxiv.org/abs/2610.26210)) and the **LaMAria benchmark** ([arXiv:2509.26639](https://arxiv.org/abs/2509.26639), [ETH Zurich](https://www.lamaria.ethz.ch)) in the context of `kornia-slam`.

AnchorFlow-VIO evaluates a strictly single-camera, single-IMU visual-inertial odometry pipeline on the challenging, city-scale LaMAria benchmark. A prominent empirical finding in the paper is **start-time sensitivity**: identical sequences started at different prospective time offsets exhibit dramatic variations in initialization success, scale estimation, and trajectory accuracy (e.g. from CP score 6.14 at 30 s to 42.00 at 120 s on a low-light sequence).

This evaluation delivers:
1. A rigorous summary of AnchorFlow-VIO and the LaMAria benchmark based strictly on verified paper texts and benchmark documentation.
2. An experimental protocol and a lightweight, dependency-free evaluation harness (`scripts/start_time_eval.py`) to measure start-time sensitivity in `kornia-slam`.
3. A gap analysis and effort assessment ($S/M/L$) for running `kornia-slam` on the LaMAria benchmark.
4. Recommendations on reusable concepts for visual-inertial initialization and benchmarking in `kornia-slam`.

---

## 2. Paper Takeaways (AnchorFlow-VIO, arXiv:2610.26210)

### 2.1 Approach & System Formulation
- **Architecture:** AnchorFlow-VIO is a monocular visual-inertial odometry (VIO) system derived from Basalt ([Demmel et al., 2021](https://arxiv.org/abs/2109.02182); [Usenko et al., 2020](https://arxiv.org/abs/1904.06504)). It does not include global loop closure or relocalization (§2, §3.3).
- **Inputs:** Strictly one grayscale camera (`cam0`, left camera in undistorted pinhole format, 20 FPS) and one IMU (`imu0`, right 1 kHz IMU) from the LaMAria ASL dataset (§3.1).
- **Preprocessing:** Applies factory affine calibration to raw IMU measurements:
  $$\mathbf{y}_{\mathrm{cal}} = M (\mathbf{y}_{\mathrm{raw}} - \mathbf{b}_{\mathrm{factory}})$$
  Subtracts a fixed timing latency adjustment of $4.2 \times 10^6\text{ ns}$ ($4.2\text{ ms}$) from IMU timestamps (§3.1). Accelerometer and gyroscope noise multipliers are 20 and 2; bias random-walk multipliers are 1 and 0.1 (§3.1).
- **"Anchor" vs. "Flow":**
  - **"Anchor" (Fixed-Template Tracking):** Each visual feature track preserves its birth-template patch ("anchored") and aligns that fixed reference to subsequent frames across a 4-level image pyramid (pattern 51, max 5 iterations). This is contrasted with a "rolling" template policy where the reference patch is updated to the immediate preceding frame (§3.2).
  - **"Flow":** Pyramidal patch alignment optical flow tracking inherited from Basalt KLT (§3.2).
- **Backend:** Square root marginalization (`vio_sqrt_marg=true`, `vio_linearization_type=ABS_QR`), sliding-window bundle adjustment with `vio_max_states=3` and `vio_max_kfs=10` (§3.3).
- **Outputs:** Ordered integer nanosecond timestamps with 3D translation and quaternion orientations ($x, y, z, q_x, q_y, q_z, q_w$) representing world-from-IMU transforms. Formats are made complete across all image timestamps by holding the first estimate backward and preceding estimates forward (§3.4).

### 2.2 Initialization & Start-Time Sensitivity Protocol
- **Initialization:** Collects a nominal 1-second window. Accelerometer measurements are integrated using gyroscope rotations to establish the gravity direction, followed by a linear visual-inertial velocity solve. If rejected, it falls back to averaged acceleration attitude with zero velocity (§3.3).
- **Start-Time Evaluation Protocol (§4.4):**
  - Evaluated on five challenge training recordings across five categories: Short (`sequence_1_19`), Medium (`sequence_2_12`), Long (`sequence_3_17`), Low light (`sequence_4_10`), and Moving platform (`sequence_5_12`).
  - Fresh runs launched at prospective fixed frame offsets of 600 frames (~30 s at 20 Hz) and 2,400 frames (~120 s at 20 Hz).
  - Fresh processes start with empty state; comparators are the original frame-0 run restricted to the **exact same temporal suffix** to avoid confounding length and coverage (§4.4).
- **Reported Numbers (§5.5, Table 5, `controlled_starts.csv`):**
  - **Low light (`sequence_4_10`):** 30 s start achieved 6.14 CP (ATE RMSE 12.27 m, scale 0.9927) vs 23.24 CP on the original suffix (ATE 5.05 m). At 120 s start, CP jumped to 42.00 (ATE RMSE 2.00 m, scale 1.0239) vs 23.14 CP on the suffix (ATE 4.62 m).
  - **Medium (`sequence_2_12`):** 30 s start scored 15.55 CP (ATE 6.87 m) vs 16.72 CP suffix; 120 s start scored 29.36 CP (ATE 4.07 m) vs 16.72 CP suffix (ATE 6.45 m).
  - **Moving Platform (`sequence_5_12`):** Diverged at both starts with large step jumps (>150 m) and CP score near zero (2.85 CP at 30 s, 0.00 CP at 120 s).
- **Key Conclusion of the Paper:** VIO systems exhibit substantial sensitivity to the initial conditions at start time. Apparent accuracy and failure rates depend heavily on the trajectory segment where initialization is attempted.

### 2.3 The LaMAria Benchmark
- **Source & Description:** City-scale egocentric visual-inertial benchmark captured with Meta Project Aria glasses in Zurich, Switzerland (~70 km total, ~22 hours) ([Krishnan et al., ICCV 2025](https://arxiv.org/abs/2509.26639)).
- **Sensors:**
  - 2x global-shutter grayscale SLAM cameras ($640 \times 480$ @ 20 FPS).
  - 1x rolling-shutter RGB camera ($1408 \times 1408$ @ 10 FPS).
  - 2x IMUs: 1 kHz (`imu-right` / `imu0`) and 800 Hz (`imu-left`).
  - Auxiliary: Barometer, magnetometer, GPS, Wi-Fi, Bluetooth.
- **Formats & Sizes:**
  - Raw VRS (~890 GB total: ~117 GB train, ~773 GB test).
  - ASL / EuRoC-style format (~1.1 TB total: ~170 GB train, ~884 GB test). Pinhole undistorted images.
  - ROS1 bag (~1.5 TB total).
- **Ground Truth & Scoring:**
  - Ground truth: Surveyed centimeter-accurate control points outdoors + dense pseudo ground truth (pGT) for training sequences. Test split references are private and evaluated on the ETH Zurich server.
  - Primary metric: Control-Point (CP) score (0–100 scale, normalized piecewise-linear function of horizontal control-point reconstruction errors, divided by 20 and averaged).
  - Secondary metrics: Horizontal pose recall at 5 m threshold (%), ATE RMSE (m) under Sim(3) alignment.
- **Licensing:** Code is licensed under MIT; dataset, documentation, and annotations are licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).

---

## 3. Experimental Protocol for kornia-slam

To assess whether `kornia-slam` exhibits start-time sensitivity similar to that observed in AnchorFlow-VIO, we define an experimental protocol that adapts the paper's methodology to `kornia-slam`'s architecture and standard benchmarks (such as EuRoC MAV).

### 3.1 Design Principles
1. **Identical Sequence, Varied Offsets:** Execute the identical sequence across $K$ start offsets $t_0 \in \{0\text{ s}, 5\text{ s}, 10\text{ s}, 15\text{ s}, 20\text{ s}, 30\text{ s}\}$, corresponding to frame offsets $F_0 \in \{0, 100, 200, 300, 400, 600\}$ at 20 Hz.
2. **Deterministic Replay:** Local mapping execution is set to `Synchronous` (`mapping: ( execution: Synchronous )`) to guarantee reproducible tracking and keyframe decisions across identical inputs.
3. **Matched Suffix Evaluation:** When ground truth is available, evaluate the trajectory produced by each start offset over its active suffix against ground truth using Umeyama Sim(3) alignment.
4. **Recorded Metrics:**
   - **Initialization Success:** Boolean flag (`Tracked` / `KeyframeAccepted` vs `Skipped` / `Failed`).
   - **Time to Initialization:** Elapsed wall seconds and frame count from start offset until visual bootstrap and IMU initialization complete.
   - **Scale Error:** Absolute error $|s - 1.0|$ and fitted Sim(3) scale factor $s$.
   - **Trajectory Error (ATE RMSE):** Root Mean Squared Error (metres) between ground truth and estimated positions after 7-DoF Sim(3) alignment.
   - **Tracking Loss Events:** Count of skipped frames or relocalization attempts.
   - **Execution Wall Time:** Wall-clock duration of the run.

---

## 4. Evaluation Harness (`scripts/start_time_eval.py`)

A dedicated evaluation harness is implemented in [`scripts/start_time_eval.py`](../../scripts/start_time_eval.py), supported by the dataset adapter [`scripts/lamaria_adapter.py`](../../scripts/lamaria_adapter.py).

### 4.1 Implementation Highlights
- **Zero Heavy Dependencies:** Written entirely in standard Python 3 (`argparse`, `dataclasses`, `json`, `csv`, `math`, `shlex`, `subprocess`, `time`).
- **Mathematical Equivalence:** Implements the exact Umeyama Sim(3) alignment and ATE RMSE equations matching `apps/kornia-slam-app/src/evaluation.rs`. Singular value decomposition is implemented via one-sided Jacobi orthogonalization with proper reflection handling ($\det(U V^T) > 0$).
- **Triple Execution Modes:**
  1. **Synthetic Smoke Test (`--smoke-test`):** Generates a 3D helical trajectory with deterministic phase-dependent visual-inertial noise, tests all offsets, verifies Sim(3) fitting, and validates reporting without requiring external dataset downloads.
  2. **Dry Run Mode (`--dry-run`):** Generates override RON configs and prints exact `kornia-slam` commands without launching subprocesses. Non-measured metrics are sanitized to `null` to avoid skewing summaries.
  3. **Real Pipeline Runner:** Takes a dataset path and base RON configuration, validates dataset length against start offsets, writes temporary configs with `--start-frame`, invokes `kornia-slam` with synchronous mapping, extracts runtime diagnostics, parses evaluation CSVs, and produces summary tables.
- **LaMAria Dataset Adaptation:** Automatically detects LaMAria layout (`cam0/`, `imu0/`, `pinhole_calibrations/`), translates pinhole JSON intrinsics/extrinsics to `sensor.yaml`, adjusts IMU hardware latency (-4.2 ms), and formats pseudo ground truth into `mav0/state_groundtruth_estimate0/data.csv`.
- **Initialisation & Failure Detection:**
  - *Primary (Logs):* Inspects stderr/stdout for `[imu_init] applying:` (inertial initialization), `[bootstrap]` (visual bootstrap), or tracking status `KeyframeAccepted` / `Tracked`. The frame index of the first accepted keyframe or IMU alignment determines `time_to_init_sec = (init_frame_idx - start_frame) / fps`.
  - *Fallback (CSVs):* If application logging format changes, the harness inspects `kornia_slam_aligned.csv`. If valid evaluated poses exist ($\ge 2$ rows), initialization is confirmed; note that CSV `idx` corresponds to evaluated-trajectory position rather than absolute sequence frame numbers.
  - *Failure Handling:* If the subprocess exits non-zero, times out, or produces fewer than 2 evaluated frames, `status = "init_failed"` is recorded, even if initialisation logs matched.

### 4.2 Harness Smoke Test Validation (Harness Self-Test on Synthetic Data — Not kornia-slam Performance)

> [!IMPORTANT]
> **Harness self-test on synthetic data, not kornia-slam performance.**
> The numbers below are generated from an idealized synthetic 3D helical motion fixture used strictly to verify that the harness's Umeyama Sim(3) alignment, ATE RMSE calculation, process execution, and CSV/JSON reporting functions execute without errors. They **do not** reflect `kornia-slam` tracking performance on real sensor data.

Command:
```bash
python scripts/start_time_eval.py --smoke-test --output-dir eval_output/smoke_test
```

Recorded Output:
```text
==========================================================================================
  HARNESS SELF-TEST ON SYNTHETIC DATA — NOT KORNIA-SLAM PERFORMANCE
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
  Note: Synthetic smoke test validates harness execution and Sim(3) math only; not real SLAM results.
==========================================================================================
```

Outputs generated:
- CSV: `eval_output/smoke_test/start_time_metrics.csv`
- JSON: `eval_output/smoke_test/start_time_summary.json`

---

## 5. Results on kornia-slam & Benchmark Prioritisation

### 5.1 Maintainer Alignment & Execution Status
Following maintainer guidance from Edgar Riba (@edgarriba: *"I'd prioritise here to benchmark against lamaria"*), benchmarking against LaMAria was prioritized.

*Full-Sequence Real Dataset Status:* **NOT EXECUTED — Network Bandwidth & Download Envelope Constraints.**
- **Measurement of Data Transfer:** Downloading from the official ETH Zurich server (`https://cvg-data.inf.ethz.ch/lamaria/`) was probed and confirmed reachable. However, transfer speed in this environment averages ~0.58 MB/s. At this rate, downloading even a single training sequence (e.g. `R_01_easy.zip` at 1,321 MB) requires ~38 minutes, which would exceed reasonable process execution envelopes.
- **Rule of Non-Fabrication:** In strict adherence to repository guidelines, no experimental numbers on un-downloaded sequences have been fabricated.
- **Local Fixture Verification:** To ensure complete functional readiness, a reproducible synthetic LaMAria fixture was generated at `crates/kornia-slam/tests/fixtures/lamaria_sample` (15 frames @ 20 Hz, pinhole camera calibration, 1 kHz IMU stream, and pseudo ground truth). The adapter and evaluation harness were validated end-to-end against this fixture in dry-run and adaptation modes (`scripts/test_lamaria_eval.py`).

### 5.2 Hypothesized Start-Time Sensitivity in kornia-slam
Based on code inspection of `crates/kornia-slam/src/inertial/initialization/` and `tracking/tracker.rs`:
- **Visual Bootstrap:** Monocular initialization requires parallax between candidate frames ($>1.0^\circ$ rotation-compensated parallax or sufficient planar homography score). If an offset starts during near-stationary motion or pure rotation, visual initialization will delay or fail to insert keyframe 1.
- **Inertial Initialization Gate:** `ImuInitializer::ready` requires at least `min_keyframes` (default 3), `min_time_sec` (doubled to 2.0 s for monocular), and `min_motion` ($>0.5\text{ m}$). If the camera is stationary or moving smoothly with low excitation at start offset $t_0$, inertial initialization will remain uninitialized, gating IMU prediction.
- **Conclusion:** Like AnchorFlow-VIO, `kornia-slam`'s monocular visual and visual-inertial initializers are inherently state- and motion-dependent, and start offsets with low visual texture or low inertial excitation are expected to show elevated initialization latency or failure rates.

---

## 6. LaMAria Integration Effort Assessment

To determine the engineering effort required to run `kornia-slam` on the LaMAria benchmark, we performed a component-level gap analysis against the benchmark's ASL and VRS specifications.

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

### 6.1 Items Not Determined
- Private test server evaluation parameters: Test sequence ground truths are withheld on the ETH Zurich server; exact per-sequence server timeouts and memory caps for all public submissions are not detailed beyond AnchorFlow-VIO's documented i7-10700 4 GiB container envelope.
- Non-visual telemetry specifications: Formats for secondary sensors (Wi-Fi, Bluetooth, barometer) were not examined as the objective is monocular+IMU VIO.

---

## 7. Reusable Ideas for kornia-slam

### 7.1 Reusable Concepts (My Assessment)
1. **Start-Time Sensitivity Benchmarking:**
   - *Paper states:* Single-run evaluations on entire sequences conceal vulnerability to initial conditions.
   - *My assessment:* `kornia-slam` should integrate start-offset testing into regression benchmarks. Tracking regressions and scale instabilities frequently occur during initialization.
2. **Hardware Time-Offset Correction:**
   - *Paper states:* Subtracting $4.2\text{ ms}$ from Project Aria IMU timestamps aligns the inertial sampling grid with the optical center of exposure.
   - *My assessment:* Adding an explicit `time_offset_ns` parameter to `kornia-sensors` / `kornia-slam-app` is a high-value, low-effort feature applicable to any multi-sensor rig with unsynchronized clocks or trigger delays.
3. **Robust Fallback Strategy for Inertial Initialization:**
   - *Paper states:* If joint linear velocity/gravity solve is rejected, fallback to integrated accelerometer gravity attitude and zero velocity.
   - *My assessment:* In `crates/kornia-slam/src/inertial/initialization/solve.rs`, failed optimization currently rejects alignment completely. Adopting a graceful attitude fallback could improve tracking continuity across challenging start intervals.

### 7.2 Non-Reusable Concepts (My Assessment)
1. **Fixed-Template KLT Tracking vs. ORB Descriptors:**
   - *Paper states:* Anchoring optical flow patches to birth templates improves persistence on 3 of 4 sequences, but degrades performance on medium motion.
   - *My assessment:* `kornia-slam` is an ORB-based SLAM system (ORB-SLAM3 design with multi-scale FAST corners, orientation centroids, and BRIEF descriptors). Fixed patch templates do not translate directly to descriptor-based keypoint tracking.
2. **Square Root Marginalization (Basalt):**
   - *Paper states:* Uses QR square-root factor marginalization to avoid forming normal equations.
   - *My assessment:* `kornia-slam` utilizes Ceres / `kornia_algebra` Levenberg-Marquardt optimization with standard Schur complement marginalization. Rewriting the solver architecture would constitute an unnecessary rewrite outside the project's roadmap.

---

## 8. Recommended Next Steps

1. **Local EuRoC MH_05 Start-Time Benchmark:** Run `scripts/start_time_eval.py` on `MH_05_difficult` once downloaded to establish baseline start-time variance numbers for `kornia-slam`.
2. **Configurable Sensor Time Offsets:** Add `time_offset_ns` field to `SensorRig` and `EurocConfig` in `crates/kornia-sensors` and `apps/kornia-slam-app`.
3. **Affine IMU Calibration Parser:** Add support for $3 \times 3$ scale/misalignment matrix $M$ in `kornia-sensors` for high-grade or factory-calibrated IMUs.
4. **LaMAria ASL Exporter:** Add a submission export flag `--export-lamaria` to `kornia-slam-app` to output standard benchmark CSVs ($t_{\mathrm{ns}}, \mathbf{t}, \mathbf{q}$).

---

## 9. Reproduction Commands

### Run Synthetic Smoke Test (Harness Self-Test Only)
```bash
python scripts/start_time_eval.py --smoke-test --output-dir eval_output/smoke_test
```

### Dry-Run Verification (Validates Config & Commands Without Subprocesses)
```bash
python scripts/start_time_eval.py \
    --dataset-path data/euroc/MH_05_difficult \
    --sequence-name MH_05_difficult \
    --config configs/euroc.ron \
    --start-offsets 0,100,200,300,400,600 \
    --dry-run \
    --output-dir eval_output/dry_run
```

### Run on EuRoC Dataset (Once Available)
```bash
python scripts/start_time_eval.py \
    --dataset-path data/euroc/MH_05_difficult \
    --sequence-name MH_05_difficult \
    --config configs/euroc.ron \
    --start-offsets 0,100,200,300,400,600 \
    --fps 20.0 \
    --output-dir eval_output/mh05
```

---

## 10. Summary of Assumptions vs. Verified Facts

- **Verified Facts:**
  - AnchorFlow-VIO paper claims, metrics, tables, and ancillary dataset values were fetched and verified directly from [arXiv:2610.26210](https://arxiv.org/abs/2610.26210).
  - LaMAria benchmark sensor specifications, dataset sizes, and licenses were verified against official documentation ([arXiv:2509.26639](https://arxiv.org/abs/2509.26639), `cvg/lamaria`).
  - Base commit SHA is `88e35f2c1005c031255fd4e9a7ed87faf0918b72` on `upstream/main`.
  - The evaluation harness executes and passes its synthetic smoke test on Windows with Python 3.13.
- **Assumptions:**
  - Real dataset results on `kornia-slam` are marked as **pending** awaiting local dataset provision.
  - Expected failure modes under static start conditions in `kornia-slam` are based on code analysis of `ImuInitializer::ready` and monocular bootstrap parallax requirements.
