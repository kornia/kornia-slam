//! Upstream target: kornia_imgproc (test data). Kept here until upstreamed; no kornia-slam types.
//!
//! Reader of `tests/fixtures/upstream/frontend_opencv.bin`, the OpenCV
//! outputs `benches/opencv_frontend_fixtures.py` writes; its docstring has
//! the layout and the OpenCV parameters the parity tests repeat.

use kornia_image::{Image, ImageSize};

pub(super) const MAX_CORNERS: usize = 200;
pub(super) const QUALITY: f32 = 0.01;
pub(super) const MIN_DISTANCE: f32 = 23.0;
pub(super) const BORDER: usize = 9;
pub(super) const SUBPIX_HALF_WIN: usize = 5;
pub(super) const SUBPIX_MAX_ITER: usize = 30;
pub(super) const LK_WIN: usize = 11;
pub(super) const LK_MAX_ITER: usize = 8;
pub(super) const EPSILON: f32 = 0.01;

pub(super) const IGNORE: &str =
    "needs tests/fixtures/upstream/frontend_opencv.bin from benches/opencv_frontend_fixtures.py";

pub(super) struct OpencvFixture {
    pub frame1: Image<u8, 1>,
    pub frame2: Image<u8, 1>,
    pub corners: Vec<[f32; 2]>,
    pub refined: Vec<[f32; 2]>,
    pub tracked: Vec<[f32; 2]>,
    pub status: Vec<u8>,
}

/// Panics with the regeneration command when the file is missing.
pub(super) fn load() -> OpencvFixture {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/upstream/frontend_opencv.bin"
    );
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}; {IGNORE}"));
    let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as usize;
    let f32_at = |o: usize| f32::from_le_bytes(b[o..o + 4].try_into().unwrap());
    let (w, h) = (u32_at(0), u32_at(4));
    let size = ImageSize {
        width: w,
        height: h,
    };
    let frame = |o: usize| Image::new(size, b[o..o + w * h].to_vec()).unwrap();
    let (f1, f2, n_at) = (8, 8 + w * h, 8 + 2 * w * h);
    let n = u32_at(n_at);
    let points = |o: usize| -> Vec<[f32; 2]> {
        (0..n)
            .map(|i| [f32_at(o + 8 * i), f32_at(o + 8 * i + 4)])
            .collect()
    };
    let corners_at = n_at + 4;
    let status_at = corners_at + 24 * n;
    OpencvFixture {
        frame1: frame(f1),
        frame2: frame(f2),
        corners: points(corners_at),
        refined: points(corners_at + 8 * n),
        tracked: points(corners_at + 16 * n),
        status: b[status_at..status_at + n].to_vec(),
    }
}
