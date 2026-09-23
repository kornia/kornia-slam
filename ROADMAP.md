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
- [ ] Split `SlamSystem` into tracking, mapping, inertial, and loop-closing subsystems

## Experimental — sensors, maps, agents

- [ ] RGB-D, LiDAR and GNSS estimators, estimator fusion in odometry
- [ ] Map representations beyond sparse landmarks — dense, TSDF, voxel, Gaussian splats
- [ ] Embedded compute targets alongside desktop/server
- [ ] Map server exposing pose and map queries over MCP
- [ ] Agentic SLAM — agents monitoring subsystems at runtime, switching strategies and tuning parameters
