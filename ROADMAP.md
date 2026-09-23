# Roadmap

Direction, not commitment. Items move as the project evolves.

## Next — complete the SLAM stack

- [ ] Relocalization on tracking loss
- [ ] Sim3 loop closure and scale correction for non-metric monocular maps
- [ ] Redundant keyframe culling
- [ ] Robust visual-inertial initialization and scale stability

## Structure — turn the pipeline into a library API

- [ ] Pluggable feature frontend: a `FeatureFrontend`/`Descriptor` seam so non-ORB and learned
      descriptors (and their matchers) drop in, with an async, device-capable variant
- [ ] Split `SlamSystem` into tracking, mapping, inertial, and loop-closing subsystems
- [ ] Telemetry contract: one canonical per-frame outcome, a stable diagnostic vocabulary, and
      versioned run artifacts that tooling and agents can read
- [ ] Crate split — `kornia-slam-telemetry`, `kornia-slam-eval`, and an isolated crate for
      GPU/TensorRT frontends so the default build stays CPU-only
- [ ] Upstream anything not SLAM-specific (camera models, solvers, image ops) to
      [kornia-rs](https://github.com/kornia/kornia-rs)

## Robustness and evaluation

- [ ] Profiles — one app binary with `--profile`, where a profile is earned by a
      composition-root recipe, a CI-gated dataset metric, and a stated compute budget
- [ ] Match a strong ORB-SLAM baseline on trajectory quality and tracking robustness
- [ ] Evaluation across datasets (EuRoC, TUM-VI, Hilti) and challenging scenarios

## Later — sensors, maps, agents

- [ ] RGB-D, LiDAR and GNSS estimators, estimator fusion in odometry
- [ ] Map representations beyond sparse landmarks — dense, TSDF, voxel, Gaussian splats
- [ ] Embedded compute targets alongside desktop/server
- [ ] Map server exposing pose and map queries over MCP
- [ ] Agentic SLAM — agents monitoring subsystems at runtime, switching strategies and tuning parameters
