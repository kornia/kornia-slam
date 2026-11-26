//! One window step of the sliding-window odometry on the synthetic stereo
//! head walk: predict, observations, keyframe and landmarks, the LM solve
//! and the marginalization with its nullspace twin (design card P5).
//!
//! `RAYON_NUM_THREADS=1 taskset -c 8 cargo bench -p kornia-slam --features bench_internals --bench bench_odometry_window`

use criterion::{Criterion, criterion_group, criterion_main};
use kornia_slam::odometry_bench::WindowBench;

fn window_step(c: &mut Criterion) {
    let mut bench = WindowBench::new();
    c.bench_function("odometry_window_step", |b| b.iter(|| bench.step()));
}

criterion_group!(benches, window_step);
criterion_main!(benches);
