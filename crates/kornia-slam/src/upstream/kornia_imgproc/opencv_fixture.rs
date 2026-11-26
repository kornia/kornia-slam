//! Upstream target: kornia_imgproc (test data). Kept here until upstreamed; no kornia-slam types.
//!
//! Reader of `tests/fixtures/upstream/frontend_opencv.bin` (version 1) and of
//! the version 2 files the break cases use, the OpenCV outputs
//! `benches/opencv_frontend_fixtures.py` writes; its docstring has both
//! layouts and the OpenCV parameters the parity tests repeat.

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

const MAGIC_V2: &[u8; 4] = b"KOF2";
const HEADER_V2: usize = 4 + 6 * 4;

/// The `calcOpticalFlowPyrLK` settings behind a fixture; version 1 files
/// carry no header and hold the defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct LkSettings {
    pub levels: usize,
    pub win: usize,
    /// Top-left of the crop in the source frames (px).
    pub crop_origin: [usize; 2],
    /// Start offset of every point under `OPTFLOW_USE_INITIAL_FLOW` (px).
    pub initial_flow: [f32; 2],
}

impl Default for LkSettings {
    fn default() -> Self {
        Self {
            levels: 0,
            win: LK_WIN,
            crop_origin: [0, 0],
            initial_flow: [0.0, 0.0],
        }
    }
}

pub(super) struct OpencvFixture {
    pub settings: LkSettings,
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
    parse(&b)
}

/// Parses either layout; panics on a truncated file, as a test should.
pub(super) fn parse(file: &[u8]) -> OpencvFixture {
    let u32_in = |b: &[u8], o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as usize;
    let f32_in = |b: &[u8], o: usize| f32::from_le_bytes(b[o..o + 4].try_into().unwrap());
    let (settings, b) = if file.starts_with(MAGIC_V2) {
        let settings = LkSettings {
            levels: u32_in(file, 4),
            win: u32_in(file, 8),
            crop_origin: [u32_in(file, 12), u32_in(file, 16)],
            initial_flow: [f32_in(file, 20), f32_in(file, 24)],
        };
        (settings, &file[HEADER_V2..])
    } else {
        (LkSettings::default(), file)
    };
    let u32_at = |o: usize| u32_in(b, o);
    let f32_at = |o: usize| f32_in(b, o);
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
        settings,
        frame1: frame(f1),
        frame2: frame(f2),
        corners: points(corners_at),
        refined: points(corners_at + 8 * n),
        tracked: points(corners_at + 16 * n),
        status: b[status_at..status_at + n].to_vec(),
    }
}

#[test]
fn reads_a_version_2_header_and_the_version_1_body_after_it() {
    let (w, h) = (3u32, 2u32);
    let mut body = Vec::new();
    for v in [w, h] {
        body.extend(v.to_le_bytes());
    }
    body.extend((0..12u8).collect::<Vec<_>>());
    body.extend(1u32.to_le_bytes());
    for v in [1.0f32, 1.0, 1.25, 0.75, 2.5, 1.5] {
        body.extend(v.to_le_bytes());
    }
    body.push(1);

    let v1 = parse(&body);
    assert_eq!(v1.settings, LkSettings::default());
    assert_eq!(v1.frame2.as_slice(), &[6, 7, 8, 9, 10, 11]);
    assert_eq!(
        (v1.corners[0], v1.refined[0], v1.tracked[0], v1.status[0]),
        ([1.0, 1.0], [1.25, 0.75], [2.5, 1.5], 1)
    );

    let mut v2 = MAGIC_V2.to_vec();
    for v in [3u32, 21, 195, 146] {
        v2.extend(v.to_le_bytes());
    }
    for v in [4.5f32, -2.0] {
        v2.extend(v.to_le_bytes());
    }
    v2.extend(&body);
    let fx = parse(&v2);
    assert_eq!(
        fx.settings,
        LkSettings {
            levels: 3,
            win: 21,
            crop_origin: [195, 146],
            initial_flow: [4.5, -2.0],
        }
    );
    assert_eq!(fx.tracked, v1.tracked);
}
