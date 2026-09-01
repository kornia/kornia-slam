# SLAM System Module Design

## Goal

Replace the layer-oriented `pipeline`, `estimation`, and state-only `system.rs`
layout with feature-oriented `system`, `tracking`, and `initialization` modules
without changing runtime behavior.

## Module ownership

`system` is the public runtime facade. It owns `SlamSystem`, configuration,
frame dispatch, cross-subsystem coordination, and externally visible system
events.

`tracking` owns frame-to-map pose tracking and the state used by that process.
It contains map-projection pose estimation, PnP, optical-flow track
bookkeeping, keyframe policy, tracking-loss policy, and per-frame results.

`initialization` owns algorithms that establish a usable visual or inertial
state. It contains two-view initialization, IMU initialization, and the
inertial initialization factor.

The remaining modules retain their existing ownership: `map` stores and
optimizes map data, while `loop_closure` verifies loop constraints and computes
global corrections.

## Public API

The preferred public names become:

- `SlamSystem` instead of `SlamPipeline`.
- `SlamConfig` instead of `PipelineConfig`.
- `LoopClosingConfig` instead of `PgoPipelineConfig`.

Deprecated root-level type aliases preserve source compatibility for the old
names during the transition. The old public `pipeline` and `estimation` module
paths are not retained as duplicate module trees; consumers should migrate to
`system`, `tracking`, and `initialization`.

## State scope

This change moves the existing state and policy types into `tracking` but does
not alter their transition semantics. Separating inertial state from tracking,
adding explicit recently-lost/relocalizing states, and redesigning reset
semantics require behavioral tests and belong in a follow-up.

## File layout

```text
crates/kornia-slam/src/
├── system/
│   ├── mod.rs
│   └── config.rs
├── tracking/
│   ├── mod.rs
│   ├── state.rs
│   ├── policy.rs
│   ├── optical_flow.rs
│   └── pose_estimation/
│       ├── mod.rs
│       ├── pnp.rs
│       └── map_projection/
└── initialization/
    ├── mod.rs
    ├── two_view.rs
    ├── imu.rs
    └── inertial_factor.rs
```

## Verification

The refactor must pass formatting, the complete `kornia-slam` library test
suite, the public API integration test, and a workspace-level check. A public
API test will exercise both the preferred names and the deprecated aliases.
