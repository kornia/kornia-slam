# Pipeline duplication note

`src/pipeline.rs` and `src/config.rs` in this crate are byte-for-byte copies of
the example's orchestration:

- Source of truth: `examples/orb_slam/src/pipeline.rs`
- Source of truth: `examples/orb_slam/src/config.rs`

Forked at commit: `cd853b8` (2026-05-22).

This duplication was an intentional design decision (see
`docs/superpowers/specs/2026-05-22-pyo3-orb-slam-bindings-design.md`): the
example crate stays untouched, and this crate carries its own copy that can
evolve independently.

If you change the example's pipeline, decide whether the binding's copy
should follow. There is no automated diff/CI check; that would need its own
divergence policy.
