# Roadmap

This roadmap will keep evolving. It sets a direction, not a commitment.

## Next — complete the SLAM stack

- [ ] Relocalization on tracking loss
- [ ] Metric scale for monocular maps: Sim3 loop closure, and AprilTag anchoring
      ([#72](https://github.com/kornia/kornia-slam/pull/72))
- [ ] Match ORB-SLAM3 on trajectory accuracy and tracking robustness
- [ ] Upstream general-purpose code (solvers, camera models, image ops) to
      [kornia-rs](https://github.com/kornia/kornia-rs)

## Library structure

- [ ] Pluggable feature frontend for non-ORB and learned features
- [ ] GPU KLT frontend (CubeCL) and multi-camera rigs (2+ cameras), ported from Rerun's
      [slam-rs](https://github.com/rerun-io/examples-monorepo/tree/main/packages/slam-rs)
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
