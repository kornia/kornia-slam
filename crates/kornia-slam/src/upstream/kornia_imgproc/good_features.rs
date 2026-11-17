//! Upstream target: kornia_imgproc::features::good_features. Kept here until upstreamed; no kornia-slam types.
//!
//! Shi-Tomasi corners (the minimum-eigenvalue response) on `wide` vectors,
//! detected only in grid cells no existing point occupies: a quality
//! threshold, a 3×3 maximum, a minimum spacing and a border mask.

use kornia_image::Image;

/// What [`good_features_to_track`] keeps.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GoodFeaturesParams {
    pub max_corners: usize,
    /// Fraction of the strongest response a corner must reach.
    pub quality: f32,
    /// No corner closer than this to another, or to an occupied point (pixels).
    pub min_spacing_px: f32,
    /// Corners within this many pixels of the image edge are dropped.
    pub border_px: usize,
    /// Respond on a 2× downsampled image; corners are returned in full-size pixels.
    pub half_resolution: bool,
}

/// Buffers reused across calls.
#[derive(Debug, Clone, Default)]
pub struct GoodFeaturesScratch {
    response: Vec<f32>,
    occupied_cells: Vec<bool>,
    candidates: Vec<(f32, [f32; 2])>,
}

/// Appends to `out` the strongest corners of `image` away from `occupied`
/// (pixels), strongest first.
pub fn good_features_to_track(
    _image: &Image<u8, 1>,
    _occupied: &[[f32; 2]],
    _params: &GoodFeaturesParams,
    _scratch: &mut GoodFeaturesScratch,
    _out: &mut Vec<[f32; 2]>,
) {
    todo!("lane K: Shi-Tomasi in empty cells")
}
