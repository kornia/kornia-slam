# Roadmap

This roadmap will keep evolving. It sets a direction, not a commitment.

## Deployment realism

Themes derived from ongoing studies of industrial and egocentric deployments. These guide our backlog and evaluation, not parity with MicroAGI microSLAM or any external system, and we do not claim SOTA.

- Dynamic‑object‑aware tracking — https://github.com/kornia/kornia-slam/issues/139
- Long‑sequence consistency — https://github.com/kornia/kornia-slam/issues/140
- Dense/TSDF consistency from fixed poses — https://github.com/kornia/kornia-slam/issues/141
- Monocular scale (already covered) — https://github.com/kornia/kornia-slam/issues/71
- Place recognition / loop features — https://github.com/kornia/kornia-slam/issues/48
- Appearance‑robust relocalization — https://github.com/kornia/kornia-slam/issues/68
- LaMAria / egocentric evaluation — https://github.com/kornia/kornia-slam/issues/78
- Multi‑camera + IMU — https://github.com/kornia/kornia-slam/issues/138

## Now — egocentric accuracy on LaMAria

We are working toward top results on the [LaMAria](https://github.com/cvg/lamaria) egocentric benchmark ([#78](https://github.com/kornia/kornia-slam/issues/78)), using the Aria rig and IMU and no GPS. The work is sequenced by where the benchmark score is lost: each change states its theory and predictions first, ships behind a config knob that defaults to the old behaviour, and is compared window by window against ORB-SLAM3 in the same harness.

Losses and recovery:
- [x] Keep every pose rigid at the source; rotation scale crept in through repeated pose corrections
- [ ] Lost-mode keyframes as fixed anchors (built, knob `keyframes_while_lost`, not yet validated; no gain on its own)
- [ ] Keep gravity, bias and velocity through a tracking loss, so a re-seeded map shares the old world frame (built, knob `tracking_loss.reseed.keep_inertial`, not yet validated; no gain on its own)
- [ ] Reject tracked poses that disagree with the gyro prediction (built, knob `tracking_loss.pose_gate`, not yet validated; no gain on the proxy)
- [ ] Global PnP relocalisation while lost; a re-seeded map with its own local window (built, knobs `tracking_loss.relocalization.while_lost` and `.solver`, `tracking_loss.reseed.own_window_keyframes` and `.relocalize_when_lost`, not yet validated)

Scale and drift while tracking:
- [ ] Longer visual-inertial BA window, and keyframes no longer dragged away from the map by BA (built, knobs `mapping.local_ba.large_window`, `mapping.local_ba.imu_huber`, `tracking.snapback_fix`, not yet validated; `mapping.local_ba.active_keyframes` already existed)
- [ ] Per-frame velocity state and a pose-inertial solve per frame (built, knob `tracking.pose_inertial`, not yet validated)
- [ ] ORB-SLAM3's keyframe insertion, local map and growth rules (built, knobs `keyframes.rules`, `tracking.local_map`, `mapping.growth_neighbours`, not yet validated)
- [ ] IMU bias and noise model, compared across ORB-SLAM3, Basalt and other references (built in part, knob `inertial.sample_interpolation`, not yet validated)
- [ ] IMU covariance trust: trust the inertial prediction by its propagated covariance (built, knob `tracking.imu_trust`, not yet validated)
- [ ] Offline global visual-inertial BA (built, run file `offline_ba` and the `offline-ba` subcommand over a `--session-out` dump, not yet validated)
- [ ] Calibration states in the offline BA: camera rotation and focal drift, gyro intrinsics, clock offset (built, knobs `offline_ba.calibration.*`, not yet validated)

Field of view and speed:
- [ ] Raw Aria fisheye cameras instead of the pinhole crop: deferred to its own round (card F)
- [ ] ORB only on keyframes with KLT in between, on every rig camera (built, knob `klt_rig`, not yet validated; `klt_first` already existed)
- [ ] Optical flow on every rig camera, forward-backward check, refill, gyro guess and flow-only pose (built, knobs `klt_rig`, `forward_backward`, `track_refill`, `gyro_guess` and `flow_pose`, not yet validated)
- [ ] Window-only fixed-point LK at an 11x11 window (built, knobs `klt_kernel: WindowedU8` and `klt_window`, not yet validated)
- [ ] Flow workers across 2-3 cores (built, knob `flow_threads`, not yet validated)
- [ ] Keyframe ORB on the mapping thread (`keyframe_orb: Mapping`): deferred, its design is open (lane T step 12)

## Next — complete the SLAM stack

- [ ] GPU acceleration of the per-frame hot paths (feature extraction, matching, KLT
      tracking), starting with the KLT frontend from Rerun's
      [slam-rs](https://github.com/rerun-io/examples-monorepo/tree/main/packages/slam-rs)
- [ ] Relocalization on tracking loss, robust to appearance change (day/night, lighting) —
      measure the degradation first ([#68](https://github.com/kornia/kornia-slam/issues/68))
- [ ] Metric scale for monocular maps: Sim3 loop closure, and AprilTag anchoring
      ([#71](https://github.com/kornia/kornia-slam/issues/71), [#72](https://github.com/kornia/kornia-slam/pull/72))
- [ ] Match ORB-SLAM3 on trajectory accuracy and tracking robustness
- [ ] Upstream general-purpose code (solvers, camera models, image ops) to
      [kornia-rs](https://github.com/kornia/kornia-rs). Staged in `crates/kornia-slam/src/upstream/`, one module per target (built, not yet upstreamed):
  - [ ] `kornia_3d::pgo`: PGO manifolds and edges
  - [ ] `kornia_3d::pnp::refine`: analytic 2x6 pose refine
  - [ ] `kornia_3d::pnp::gpnp`: generalised PnP and AP3P on rig rays
  - [ ] `kornia_3d::pose`: `camera_centre` and the `Pose3d`/`SE3` conversions
  - [ ] `kornia_algebra`: f32/f64 casts

## Library structure

- [ ] Pluggable feature frontend for non-ORB and learned features, e.g. XFeat ([#49](https://github.com/kornia/kornia-slam/issues/49))
- [ ] Pluggable place recognition, including learned global descriptors such as DINOv3
      ([#48](https://github.com/kornia/kornia-slam/issues/48))
- [ ] Multi-camera rigs (2+ cameras), ported from Rerun's
      [slam-rs](https://github.com/rerun-io/examples-monorepo/tree/main/packages/slam-rs);
      generic rig calibration file, a ray per rig pixel and a rig-aware AP3P (built, knobs `rig_calibration`
      and `tracking_loss.relocalization.solver: RigAp3p`, not yet validated)
- [ ] Explore [Atlas](https://github.com/kornia/kornia-slam/issues/59), a manifold-native estimation and factor-graph foundation, as
      the base for BA, VI-BA and pose-graph optimization
- [ ] Split `SlamSystem` into tracking, mapping, inertial, and loop-closing subsystems

## Tooling and integrations

- [ ] Pixi environment and tasks for reproducible cross-platform builds
      ([#77](https://github.com/kornia/kornia-slam/issues/77), [#79](https://github.com/kornia/kornia-slam/pull/79))
- [ ] ROS 2 integration via [ros2_rust](https://github.com/ros2-rust/ros2_rust)
      ([#75](https://github.com/kornia/kornia-slam/issues/75))

## Experimental — sensors, maps, agents

- [ ] RGB-D, LiDAR and GNSS estimators, estimator fusion in odometry
- [ ] Map representations beyond sparse landmarks — dense, TSDF, voxel, Gaussian splats ([#141](https://github.com/kornia/kornia-slam/issues/141))
- [ ] Embedded compute targets alongside desktop/server
- [ ] Map server exposing pose and map queries over MCP
- [ ] Agentic SLAM — agents monitoring subsystems at runtime, switching strategies and tuning parameters
