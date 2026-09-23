# Roadmap

This roadmap will keep evolving. It sets a direction, not a commitment.

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
- [ ] Map representations beyond sparse landmarks — dense, TSDF, voxel, Gaussian splats
- [ ] Embedded compute targets alongside desktop/server
- [ ] Map server exposing pose and map queries over MCP
- [ ] Agentic SLAM — agents monitoring subsystems at runtime, switching strategies and tuning parameters
