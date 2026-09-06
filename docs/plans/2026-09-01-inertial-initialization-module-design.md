# Inertial Initialization Module Design

## Goal

Group inertial initialization and its private optimizer factor under a dedicated
`initialization/inertial/` module without changing the public initialization API
or runtime behavior.

## Structure

```text
initialization/
├── mod.rs
├── two_view.rs
└── inertial/
    ├── mod.rs
    └── factor.rs
```

`initialization/mod.rs` declares private `inertial` and `two_view` modules and
continues to re-export the supported types and functions. Existing imports such
as `kornia_slam::initialization::ImuInitializer` therefore remain unchanged.

`inertial/mod.rs` owns the initializer, configuration, typed results, rejection
reasons, numeric helpers, and tests. Its private `factor` child contains the
optimizer factors. No code outside the inertial module accesses those factors.

## Behavior and errors

This is a file-layout-only refactor. Initialization still computes typed results,
and `SlamSystem` remains solely responsible for applying those results to the map
and tracking state. Error types, validation, debug reporting, and data flow are
unchanged.

## Verification

Formatting, Clippy with warnings denied, workspace tests, workspace checking, and
the public API compatibility test must continue to pass.
