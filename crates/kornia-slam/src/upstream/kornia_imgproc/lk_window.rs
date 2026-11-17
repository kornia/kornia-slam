//! Upstream target: kornia_imgproc::optical_flow_pyr_lk. Kept here until upstreamed; no kornia-slam types.
//!
//! Window-only fixed-point LK: OpenCV's scheme (Q14 weights, i16 Scharr, the
//! same status rules) on `wide` vectors, with the Scharr gradients computed on
//! each template's patch only, so no full-image pyramid or gradient pass runs.
//! Single-threaded and allocation-free once a frame and a template exist.

use kornia_image::Image;

/// A frame [`track_template`] reads: a padded level-0 plane, with coarse
/// levels built only on request.
pub struct LkFrame {
    /// Level 0, then the coarse levels built so far.
    levels: Vec<Image<u8, 1>>,
    /// Border replicated around each level, so a window never reads outside.
    pad: usize,
    frame_id: u64,
}

impl LkFrame {
    /// Copies `image` into a level-0 plane padded for windows of `win`.
    pub fn new(_image: &Image<u8, 1>, _win: usize, _frame_id: u64) -> Result<Self, LkWindowError> {
        todo!("lane K: padded level-0 plane")
    }

    /// Re-fills this frame from `image` in place; drops the coarse levels.
    pub fn rebuild(&mut self, _image: &Image<u8, 1>, _frame_id: u64) -> Result<(), LkWindowError> {
        todo!("lane K: padded level-0 plane")
    }

    /// Builds levels `1..=max_level` if they are not built yet.
    pub fn build_coarse(&mut self, _max_level: usize) -> Result<(), LkWindowError> {
        todo!("lane K: coarse levels on request")
    }

    /// The caller's id for the image this frame holds.
    pub fn frame_id(&self) -> u64 {
        self.frame_id
    }

    /// Levels built so far, level 0 included.
    pub fn levels(&self) -> usize {
        todo!("lane K: level count")
    }

    /// The level-0 plane, unpadded.
    pub fn plane(&self) -> &Image<u8, 1> {
        todo!("lane K: level-0 plane")
    }
}

/// One track's window: `win × win` i16 values and their Scharr gradients, Q14.
#[derive(Debug, Clone, Default)]
pub struct LkTemplate {
    values: Vec<i16>,
    grad_x: Vec<i16>,
    grad_y: Vec<i16>,
    win: usize,
    level: usize,
    pixel: [f32; 2],
}

/// The result of one [`track_template`] solve, in level-0 pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LkTrack {
    pub pixel: [f32; 2],
    pub status: bool,
    /// Mean absolute intensity difference over the window, as OpenCV's.
    pub error: f32,
    pub iterations: u32,
}

/// Extracts the template of `pixel` (level-0 pixels) at `level` of `frame`
/// into `out`; `false` when the window leaves the padded plane.
pub fn extract_template(
    _frame: &LkFrame,
    _pixel: [f32; 2],
    _level: usize,
    _win: usize,
    _out: &mut LkTemplate,
) -> bool {
    todo!("lane K: window-only template")
}

/// Tracks `template` into `next` from `start` (level-0 pixels). A template
/// cut at `level > 0` tracks on `next`'s level of the same index, and the
/// result is scaled back to level 0.
pub fn track_template(
    _template: &LkTemplate,
    _next: &LkFrame,
    _start: [f32; 2],
    _max_iter: usize,
    _epsilon: f32,
) -> LkTrack {
    todo!("lane K: fixed-point LK on one window")
}

/// An image or window that [`LkFrame`] cannot hold.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum LkWindowError {
    /// Windows must be odd and at least 3.
    InvalidWindow(usize),
    /// The image is smaller than one window.
    ImageTooSmall { width: usize, height: usize },
}

impl std::fmt::Display for LkWindowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidWindow(win) => write!(f, "LK window {win} must be odd and at least 3"),
            Self::ImageTooSmall { width, height } => {
                write!(f, "a {width}x{height} image is smaller than one LK window")
            }
        }
    }
}

impl std::error::Error for LkWindowError {}
