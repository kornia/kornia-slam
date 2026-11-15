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
- [x] Lost-mode keyframes as fixed anchors (knob)
- [x] Keep gravity, bias and velocity through a tracking loss, so a re-seeded map shares the old world frame (knob)
- [x] Reject tracked poses that disagree with the gyro prediction (knob)
- [ ] Global PnP relocalisation while lost; a re-seeded map with its own local window

Scale and drift while tracking:
- [ ] Longer visual-inertial BA window, and keyframes no longer dragged away from the map by BA
- [ ] Per-frame velocity state and a pose-inertial solve per frame
- [ ] IMU bias and noise model, compared across ORB-SLAM3, Basalt and other references
- [ ] Offline global visual-inertial BA

Field of view and speed:
- [ ] Raw Aria fisheye cameras instead of the pinhole crop
- [ ] ORB only on keyframes with KLT in between, on every rig camera

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
      [kornia-rs](https://github.com/kornia/kornia-rs)

## Library structure

- [ ] Pluggable feature frontend for non-ORB and learned features, e.g. XFeat ([#49](https://github.com/kornia/kornia-slam/issues/49))
- [ ] Pluggable place recognition, including learned global descriptors such as DINOv3
      ([#48](https://github.com/kornia/kornia-slam/issues/48))
- [ ] Multi-camera rigs (2+ cameras), ported from Rerun's
      [slam-rs](https://github.com/rerun-io/examples-monorepo/tree/main/packages/slam-rs)
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
