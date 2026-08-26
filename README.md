# 🚧 kornia-slam 📷🧭🗺️📍🤖

Spatial runtime for real-time pose estimation, mapping, and agent interaction.

> **Early stage, active development.** Today this is an ORB-based visual (and visual-inertial)
> SLAM pipeline that runs end-to-end on EuRoC, Hilti, MCAP recordings, and live cameras.
> System orchestration now lives in the library as `SlamPipeline`, but the API and module
> layout are still moving. Expect breaking changes, and treat the roadmap below as a
> direction that is subject to change rather than a commitment.
> Contributions and feedback welcome.

## What works today

- **ORB front end** — extraction, matching, two-view bootstrap (essential matrix + triangulation)
- **Tracking** — map-projection PnP with RANSAC against a local map built from the covisibility graph
- **Mapping** — keyframe insertion, map-point triangulation and fusion into neighbors, map-point culling
- **Optimization** — initial and local bundle adjustment via Schur complement (`kornia-3d`), plus a
  visual-inertial local BA (`vi_ba_schur`)
- **Stereo** — rectification, row-wise stereo matching, metric depth into initialization and BA
- **IMU** — preintegration and ORB-SLAM3-style inertial initialization (gyro bias, gravity, and
  scale in monocular mode)
- **Frame sources** — EuRoC, Hilti-Trimble (fisheye), MCAP recordings, OAK-D, UVC webcams,
  behind a common `FrameSource` trait
- **Place recognition** — DBoW2 bag-of-words vocabulary and keyframe database for candidate
  retrieval (`--vocab`)
- **Loop closure** — geometric verification of retrieved candidates, loop fusion, and metric
  pose-graph optimization applied to the live map (`--apply-pgo`); gravity-preserving
  four-degree-of-freedom optimization once the IMU is initialized
- **Tooling** — default terminal UI with live BEV and debug panel, optional Rerun streaming,
  ATE/RPE evaluation against ground truth

Not yet: relocalization, Sim3 loop closure for non-metric monocular maps, map serving over MCP.

## Quick start

```bash
# EuRoC, monocular (no extra deps)
cargo run --release -p kornia-slam-app -- euroc --data /path/to/MH_01_easy

# EuRoC, stereo + IMU, with trajectory evaluation
cargo run --release -p kornia-slam-app -- euroc --data /path/to/MH_01_easy --stereo --imu --evaluate

# Live UVC camera (laptop webcam, USB cam, …)
cargo run --release -p kornia-slam-app --features uvc -- uvc --fx 600 --fy 600 --cx 320 --cy 240
```

Press `d` in the TUI to toggle the debug panel. Use `--rerun-stream` for a Rerun viewer,
`--no-tui` for plain stderr.

Full source, calibration, and stereo docs: [apps/kornia-slam-app/README.md](apps/kornia-slam-app/README.md).

## Layout

```text
crates/kornia-slam     library: the SlamPipeline runtime plus frame, map, estimation (two-view,
                       PnP, map projection, IMU init), stereo, visual-inertial BA, place
                       recognition, and loop closure
crates/kornia-sensors  sensor types (IMU)
apps/kornia-slam-app   composition root for the `kornia-slam` executable: CLI, frame sources,
                       ORB extraction, TUI, Rerun, evaluation
```

The library owns the algorithms and the temporal orchestration; the app wires a concrete
product around them.

## Local checks

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --workspace
```

## Branching

`develop` is the working default branch and takes all PRs for now. It is temporary: when
v0.1.0 is tagged it folds into `main`, which then becomes the only long-lived branch, with
short-lived `feat/*` and `fix/*` branches on top and tags for releases.

## Roadmap

Everything below is a roadmap entry, not a shipped capability.

**Next — complete the SLAM stack**
- [ ] Relocalization on tracking loss
- [x] Place recognition and loop closure on metric maps (verification, fusion, pose-graph
      optimization)
- [ ] Sim3 loop closure and scale correction for non-metric monocular maps
- [ ] Redundant keyframe culling
- [ ] Robust visual-inertial initialization and scale stability

**Structure — turn the pipeline into a library API**
- [ ] Pluggable feature frontend: a `FeatureFrontend`/`Descriptor` seam so non-ORB and learned
      descriptors (and their matchers) drop in, with an async, device-capable variant
- [x] Move temporal orchestration (state machine, keyframe policy, map-update ordering) into the
      library; `apps/` holds composition roots rather than a second pipeline
- [ ] Split `SlamPipeline` into tracking, mapping, inertial, and loop-closing subsystems
- [ ] Telemetry contract: one canonical per-frame outcome, a stable diagnostic vocabulary, and
      versioned run artifacts that tooling and agents can read
- [ ] Crate split — `kornia-slam-telemetry`, `kornia-slam-eval`, and an isolated crate for
      GPU/TensorRT frontends so the default build stays CPU-only
- [ ] Upstream anything not SLAM-specific (camera models, solvers, image ops) to
      [kornia-rs](https://github.com/kornia/kornia-rs)

**Robustness and evaluation**
- [ ] Profiles — one app binary with `--profile`, where a profile is earned by a
      composition-root recipe, a CI-gated dataset metric, and a stated compute budget
- [ ] Match a strong ORB-SLAM baseline on trajectory quality and tracking robustness
- [ ] Evaluation across datasets (EuRoC, TUM-VI, Hilti) and challenging scenarios

**Later — sensors, maps, agents**
- [ ] RGB-D, LiDAR and GNSS estimators, estimator fusion in odometry
- [ ] Map representations beyond sparse landmarks — dense, TSDF, voxel, Gaussian splats
- [ ] Embedded compute targets alongside desktop/server
- [ ] Map server exposing pose and map queries over MCP
- [ ] Agentic SLAM — agents monitoring subsystems at runtime, switching strategies and tuning parameters

## License

Apache-2.0
