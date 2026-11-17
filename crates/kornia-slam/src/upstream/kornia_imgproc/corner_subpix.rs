//! Upstream target: kornia_imgproc::features::corner_subpix. Kept here until upstreamed; no kornia-slam types.
//!
//! OpenCV's `cornerSubPix`: each corner moves to where the image gradients in
//! its window are orthogonal to the offsets from it.

use kornia_image::Image;

/// Refines `corners` (pixels) in place over a `(2 half_win + 1)²` window,
/// for at most `max_iter` steps or until a step is below `epsilon` pixels.
pub fn corner_sub_pix(
    _image: &Image<u8, 1>,
    _corners: &mut [[f32; 2]],
    _half_win: usize,
    _max_iter: usize,
    _epsilon: f32,
) {
    todo!("lane K: sub-pixel corner refinement")
}
