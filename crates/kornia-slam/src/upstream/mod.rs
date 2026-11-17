//! Upstream target: kornia-rs, one module per crate. Kept here until upstreamed; no kornia-slam types.

pub(crate) mod kornia_3d;
pub(crate) mod kornia_algebra;
// Public so the kernel benches reach it under `bench_internals`.
// Skeleton for the 1 ms frontend: lane K fills these, lane T calls them.
#[allow(dead_code)]
pub mod kornia_imgproc;
