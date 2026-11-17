//! Upstream target: kornia_imgproc::optical_flow_pyr_lk. Kept here until upstreamed; no kornia-slam types.
//!
//! Window-only fixed-point LK: OpenCV's scheme (Q14 weights, i16 Scharr, the
//! same status rules) on `wide` vectors, with the Scharr gradients computed on
//! each template's patch only, so no full-image pyramid or gradient pass runs.
//! Single-threaded and allocation-free once a frame and a template exist.
//!
//! One level of kornia-rs's `calc_optical_flow_pyr_lk_u8`, bit for bit. That
//! kernel pads every level with reflect-101 and its derivative planes with
//! zeros. Here the planes stay unpadded, so [`LkFrame::plane`] is the frame
//! itself and the per-frame copy is the only one: interior windows read the
//! plane directly, and a window that reaches the border takes a scalar path
//! that reads the same reflect-101 pixels and zero derivatives.

use kornia_image::{Image, ImageError, ImageSize};
use wide::{i16x8, i32x4, i32x8};

use super::lanes::{LANES, load8_i16};
use super::pyrdown_u8::{pyrdown_u8, reflect_101};

/// Fractional bits of the bilinear weights.
const W_BITS: i32 = 14;
const W_ONE: i32 = 1 << W_BITS;
/// Window samples keep 5 fractional bits (`value * 32`).
const SAMPLE_SHIFT: i32 = W_BITS - 5;
/// Scale from the integer tensor and mismatch sums to OpenCV's float ones.
const FLT_SCALE: f32 = 1.0 / (1u32 << 20) as f32;
/// `|J - I|` is in units of 1/32 pixel value.
const SAMPLE_UNIT: f32 = 32.0;
/// Two consecutive updates that cancel to within this (pixels) mean the
/// iteration oscillates; OpenCV then takes the half step and stops.
const OSCILLATION_TOL: f32 = 0.01;
/// kornia-rs's `PyrLKParams::default().min_eigen_threshold`.
const MIN_EIGEN_THRESHOLD: f32 = 1e-4;
/// Bound of `|J - I|` in window units, and of an unscaled Scharr derivative.
const MAX_DIFF: i64 = 255 * 32;
const MAX_DERIV: i64 = 16 * 255;
/// Chunks summed into i32 lanes before they spill to i64: each chunk adds
/// two products per lane.
const TENSOR_FLUSH: usize = (i32::MAX as i64 / (2 * MAX_DERIV * MAX_DERIV)) as usize;
const MISMATCH_FLUSH: usize = (i32::MAX as i64 / (2 * MAX_DIFF * MAX_DERIV)) as usize;

/// A frame [`track_template`] reads: the level-0 plane, with coarse levels
/// built only on request.
pub struct LkFrame {
    /// Level 0, then coarse levels; the buffers outlive [`LkFrame::rebuild`].
    levels: Vec<Image<u8, 1>>,
    /// Levels valid for the current image.
    built: usize,
    /// Coarse levels stop where this window no longer fits, as OpenCV's.
    win: usize,
    frame_id: u64,
    pyrdown_rows: Vec<u16>,
}

impl LkFrame {
    /// Copies `image` into a level-0 plane for windows of `win`.
    pub fn new(image: &Image<u8, 1>, win: usize, frame_id: u64) -> Result<Self, LkWindowError> {
        check_window(win)?;
        check_size(image.size(), win)?;
        Ok(Self {
            levels: vec![copy_image(image)?],
            built: 1,
            win,
            frame_id,
            pyrdown_rows: Vec::new(),
        })
    }

    /// Re-fills this frame from `image` in place; drops the coarse levels.
    pub fn rebuild(&mut self, image: &Image<u8, 1>, frame_id: u64) -> Result<(), LkWindowError> {
        check_size(image.size(), self.win)?;
        if self.levels[0].size() == image.size() {
            self.levels[0]
                .as_slice_mut()
                .copy_from_slice(image.as_slice());
        } else {
            let plane = copy_image(image)?;
            self.levels.clear();
            self.levels.push(plane);
        }
        self.built = 1;
        self.frame_id = frame_id;
        Ok(())
    }

    /// Builds levels `1..=max_level` if they are not built yet. Like
    /// `cv::buildOpticalFlowPyramid`, it stops before a level whose size is
    /// at most the window in either axis; [`LkFrame::levels`] says where.
    pub fn build_coarse(&mut self, max_level: usize) -> Result<(), LkWindowError> {
        while self.built <= max_level {
            let k = self.built;
            let below = self.levels[k - 1].size();
            let size = ImageSize {
                width: below.width.div_ceil(2),
                height: below.height.div_ceil(2),
            };
            if size.width <= self.win || size.height <= self.win {
                break;
            }
            if self.levels.len() == k {
                self.levels.push(blank_image(size)?);
            } else if self.levels[k].size() != size {
                self.levels[k] = blank_image(size)?;
            }
            let (finer, coarser) = self.levels.split_at_mut(k);
            pyrdown_u8(&finer[k - 1], &mut coarser[0], &mut self.pyrdown_rows)
                .map_err(image_error)?;
            self.built += 1;
        }
        Ok(())
    }

    /// The caller's id for the image this frame holds.
    pub fn frame_id(&self) -> u64 {
        self.frame_id
    }

    /// Levels built so far, level 0 included.
    pub fn levels(&self) -> usize {
        self.built
    }

    /// The level-0 plane, unpadded.
    pub fn plane(&self) -> &Image<u8, 1> {
        &self.levels[0]
    }

    fn level(&self, level: usize) -> Option<&Image<u8, 1>> {
        self.levels[..self.built].get(level)
    }
}

/// One track's window: `win × win` i16 values and their Scharr gradients, Q14.
#[derive(Debug, Clone, Default)]
pub struct LkTemplate {
    /// `I * 32`, `win` rows of `cols` lanes; lanes `x >= win` are zero, as
    /// are those of the gradients.
    values: Vec<i16>,
    grad_x: Vec<i16>,
    grad_y: Vec<i16>,
    /// Scharr of the plane under the window, `win + 1` rows of `cols + LANES`.
    deriv_x: Vec<i16>,
    deriv_y: Vec<i16>,
    win: usize,
    cols: usize,
    level: usize,
    /// `(A11, A12, A22)` in OpenCV's float scale, and `1 / det`.
    tensor: [f32; 3],
    inv_det: f32,
    /// The window is in the plane and its tensor is well conditioned.
    trackable: bool,
}

impl LkTemplate {
    fn reshape(&mut self, win: usize) {
        let cols = win.next_multiple_of(LANES);
        let (len, deriv_len) = (win * cols, (win + 1) * (cols + LANES));
        self.values.resize(len, 0);
        self.grad_x.resize(len, 0);
        self.grad_y.resize(len, 0);
        self.deriv_x.resize(deriv_len, 0);
        self.deriv_y.resize(deriv_len, 0);
        self.win = win;
        self.cols = cols;
    }

    fn deriv_stride(&self) -> usize {
        self.cols + LANES
    }
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
/// into `out`; `false` when the window leaves the image by more than its
/// size (OpenCV's rule), the frame lacks `level`, or `win` is not odd and at
/// least 3. A template whose structure tensor is ill-conditioned is kept, and
/// [`track_template`] reports it with status `false`.
pub fn extract_template(
    frame: &LkFrame,
    pixel: [f32; 2],
    level: usize,
    win: usize,
    out: &mut LkTemplate,
) -> bool {
    out.trackable = false;
    let Some(plane) = frame.level(level) else {
        return false;
    };
    if check_window(win).is_err() {
        return false;
    }
    let scale = 1.0 / (1u32 << level) as f32;
    let half_win = (win as f32 - 1.0) * 0.5;
    let tl = [pixel[0] * scale - half_win, pixel[1] * scale - half_win];
    let (lw, lh) = (plane.width(), plane.height());
    let Some([x0, y0]) = corner(tl, win, lw, lh) else {
        return false;
    };
    let w = q14_weights(tl[0] - x0 as f32, tl[1] - y0 as f32);
    out.reshape(win);
    out.level = level;
    let data = plane.as_slice();
    let interior =
        x0 >= 1 && y0 >= 1 && x0 as usize + out.cols + LANES < lw && y0 as usize + win + 1 < lh;
    if interior {
        let (x0, y0) = (x0 as usize, y0 as usize);
        deriv_interior(data, lw, x0, y0, out);
        values_interior(data, lw, x0, y0, w, out);
    } else {
        deriv_border(data, lw, lh, x0, y0, out);
        values_border(data, lw, lh, x0, y0, w, out);
    }
    let (ia11, ia12, ia22) = gradients(w, out);
    let (a11, a12, a22) = (
        ia11 as f32 * FLT_SCALE,
        ia12 as f32 * FLT_SCALE,
        ia22 as f32 * FLT_SCALE,
    );
    let area = (win * win) as f32;
    let det = a11 * a22 - a12 * a12;
    let min_eig = (a22 + a11 - ((a11 - a22) * (a11 - a22) + 4.0 * a12 * a12).sqrt()) / (2.0 * area);
    out.tensor = [a11, a12, a22];
    out.inv_det = 1.0 / det;
    out.trackable = min_eig >= MIN_EIGEN_THRESHOLD && det >= f32::EPSILON;
    true
}

/// Tracks `template` into `next` from `start` (level-0 pixels). A template
/// cut at `level > 0` tracks on `next`'s level of the same index, and the
/// result is scaled back to level 0; that level takes OpenCV's level-0
/// status rules. Iterations stop after `max_iter`, once a step is at most
/// `epsilon` pixels (at that level), or on an oscillation half step. A
/// failed track keeps its last estimate, and `start` when it never moved.
pub fn track_template(
    template: &LkTemplate,
    next: &LkFrame,
    start: [f32; 2],
    max_iter: usize,
    epsilon: f32,
) -> LkTrack {
    let mut out = LkTrack {
        pixel: start,
        status: false,
        error: 0.0,
        iterations: 0,
    };
    let Some(plane) = next.level(template.level) else {
        return out;
    };
    if !template.trackable {
        return out;
    }
    let win = template.win;
    let level_scale = (1u32 << template.level) as f32;
    let half_win = (win as f32 - 1.0) * 0.5;
    let (lw, lh) = (plane.width(), plane.height());
    let eps2 = f64::from(epsilon) * f64::from(epsilon);
    let [a11, a12, a22] = template.tensor;
    let inv_det = template.inv_det;
    let mut pt = [start[0] / level_scale, start[1] / level_scale];
    let mut np = [pt[0] - half_win, pt[1] - half_win];
    let mut prev_delta = [0.0f32; 2];
    let mut status = true;
    for it in 0..max_iter {
        let Some([jx, jy]) = corner(np, win, lw, lh) else {
            status = false;
            break;
        };
        let w = q14_weights(np[0] - jx as f32, np[1] - jy as f32);
        let (ib1, ib2) = mismatch(plane, jx, jy, w, template);
        out.iterations += 1;
        let (b1, b2) = (ib1 as f32 * FLT_SCALE, ib2 as f32 * FLT_SCALE);
        let delta = [
            (a12 * b2 - a22 * b1) * inv_det,
            (a12 * b1 - a11 * b2) * inv_det,
        ];
        np = [np[0] + delta[0], np[1] + delta[1]];
        pt = [np[0] + half_win, np[1] + half_win];
        let d2 =
            f64::from(delta[0]) * f64::from(delta[0]) + f64::from(delta[1]) * f64::from(delta[1]);
        if d2 <= eps2 {
            break;
        }
        if it > 0
            && (delta[0] + prev_delta[0]).abs() < OSCILLATION_TOL
            && (delta[1] + prev_delta[1]).abs() < OSCILLATION_TOL
        {
            pt = [pt[0] - delta[0] * 0.5, pt[1] - delta[1] * 0.5];
            break;
        }
        prev_delta = delta;
    }
    if status {
        let tl = [pt[0] - half_win, pt[1] - half_win];
        match corner(tl, win, lw, lh) {
            Some([jx, jy]) => {
                let w = q14_weights(tl[0] - jx as f32, tl[1] - jy as f32);
                let sum = abs_error(plane, jx, jy, w, template);
                out.error = sum as f32 / (SAMPLE_UNIT * (win * win) as f32);
            }
            None => status = false,
        }
    }
    out.status = status;
    out.pixel = [pt[0] * level_scale, pt[1] * level_scale];
    out
}

/// Unscaled 3×3 Scharr of the rows around `y0..=y0 + win`, every stored lane;
/// the caller checked that all reads are inside the plane.
fn deriv_interior(data: &[u8], lw: usize, x0: usize, y0: usize, t: &mut LkTemplate) {
    let ds = t.deriv_stride();
    let at = |o: usize| load8_i16(&data[o..]);
    for v in 0..=t.win {
        let center = (y0 + v) * lw + x0;
        for u in (0..ds).step_by(LANES) {
            let (a, c, b) = (center + u - lw, center + u, center + u + lw);
            let (am, a0, ap) = (at(a - 1), at(a), at(a + 1));
            let (cm, cp) = (at(c - 1), at(c + 1));
            let (bm, b0, bp) = (at(b - 1), at(b), at(b + 1));
            let gx = (ap + bp) * 3i16 + cp * 10i16 - ((am + bm) * 3i16 + cm * 10i16);
            let gy = ((bm - am) + (bp - ap)) * 3i16 + (b0 - a0) * 10i16;
            let o = v * ds + u;
            t.deriv_x[o..o + LANES].copy_from_slice(gx.as_array());
            t.deriv_y[o..o + LANES].copy_from_slice(gy.as_array());
        }
    }
}

/// [`deriv_interior`] for a window that reaches the border: reflect-101
/// reads, zero outside the image (kornia-rs's zero-padded derivative planes).
fn deriv_border(data: &[u8], lw: usize, lh: usize, x0: isize, y0: isize, t: &mut LkTemplate) {
    let ds = t.deriv_stride();
    let p = |x: isize, y: isize| pixel_101(data, lw, lh, x, y);
    for v in 0..=t.win {
        for u in 0..ds {
            let (x, y) = (x0 + u as isize, y0 + v as isize);
            let inside =
                u <= t.win && (0..lw as isize).contains(&x) && (0..lh as isize).contains(&y);
            let (gx, gy) = if inside {
                let s = |x: isize| (p(x, y - 1) + p(x, y + 1)) * 3 + p(x, y) * 10;
                let d = |x: isize| p(x, y + 1) - p(x, y - 1);
                (s(x + 1) - s(x - 1), (d(x - 1) + d(x + 1)) * 3 + d(x) * 10)
            } else {
                (0, 0)
            };
            t.deriv_x[v * ds + u] = gx as i16;
            t.deriv_y[v * ds + u] = gy as i16;
        }
    }
}

/// Window samples `I * 32` at Q14 weights `w`, from inside the plane.
fn values_interior(data: &[u8], lw: usize, x0: usize, y0: usize, w: [i32; 4], t: &mut LkTemplate) {
    let wv = weight_lanes(w);
    let mask = tail_mask(t.win, t.cols);
    for y in 0..t.win {
        let r0 = (y0 + y) * lw + x0;
        for x in (0..t.cols).step_by(LANES) {
            let iv = bilinear_u8(data, r0 + x, lw, wv) & lane_mask(x, t.cols, mask);
            let o = y * t.cols + x;
            t.values[o..o + LANES].copy_from_slice(iv.as_array());
        }
    }
}

fn values_border(
    data: &[u8],
    lw: usize,
    lh: usize,
    x0: isize,
    y0: isize,
    w: [i32; 4],
    t: &mut LkTemplate,
) {
    for y in 0..t.win {
        for x in 0..t.cols {
            t.values[y * t.cols + x] = if x < t.win {
                let (px, py) = (x0 + x as isize, y0 + y as isize);
                descale(bilinear_101(data, lw, lh, px, py, w), SAMPLE_SHIFT) as i16
            } else {
                0
            };
        }
    }
}

/// Bilinear Q14 gradients from the derivative rows, and the exact integer
/// structure tensor `(A11, A12, A22)`.
fn gradients(w: [i32; 4], t: &mut LkTemplate) -> (i64, i64, i64) {
    let (ds, cols) = (t.deriv_stride(), t.cols);
    let wv = weight_lanes(w);
    let mask = tail_mask(t.win, cols);
    let zero = i32x4::splat(0);
    let (mut q11, mut q12, mut q22) = (zero, zero, zero);
    let (mut a11, mut a12, mut a22) = (0i64, 0i64, 0i64);
    let mut pending = 0;
    for y in 0..t.win {
        for x in (0..cols).step_by(LANES) {
            let o = y * ds + x;
            let lanes = lane_mask(x, cols, mask);
            let gx = bilinear_i16(&t.deriv_x, o, ds, wv) & lanes;
            let gy = bilinear_i16(&t.deriv_y, o, ds, wv) & lanes;
            let g = y * cols + x;
            t.grad_x[g..g + LANES].copy_from_slice(gx.as_array());
            t.grad_y[g..g + LANES].copy_from_slice(gy.as_array());
            q11 += gx.dot(gx);
            q12 += gx.dot(gy);
            q22 += gy.dot(gy);
            pending += 1;
            if pending == TENSOR_FLUSH {
                (a11, a12, a22) = (a11 + hsum(q11), a12 + hsum(q12), a22 + hsum(q22));
                (q11, q12, q22) = (zero, zero, zero);
                pending = 0;
            }
        }
    }
    (a11 + hsum(q11), a12 + hsum(q12), a22 + hsum(q22))
}

/// `(sum (J - I) Ix, sum (J - I) Iy)` of the window at integer corner
/// `(jx, jy)` of `plane` with weights `w`; exact, so both paths agree.
fn mismatch(plane: &Image<u8, 1>, jx: isize, jy: isize, w: [i32; 4], t: &LkTemplate) -> (i64, i64) {
    let (lw, lh) = (plane.width(), plane.height());
    let data = plane.as_slice();
    if !window_inside(jx, jy, t, lw, lh) {
        let (mut b1, mut b2) = (0i64, 0i64);
        for y in 0..t.win {
            for x in 0..t.win {
                let o = y * t.cols + x;
                let (px, py) = (jx + x as isize, jy + y as isize);
                let jv = descale(bilinear_101(data, lw, lh, px, py, w), SAMPLE_SHIFT);
                let diff = jv - i32::from(t.values[o]);
                b1 += i64::from(diff * i32::from(t.grad_x[o]));
                b2 += i64::from(diff * i32::from(t.grad_y[o]));
            }
        }
        return (b1, b2);
    }
    let (jx, jy) = (jx as usize, jy as usize);
    let wv = weight_lanes(w);
    let zero = i32x4::splat(0);
    let (mut q1, mut q2) = (zero, zero);
    let (mut b1, mut b2) = (0i64, 0i64);
    let mut pending = 0;
    for y in 0..t.win {
        let r0 = (jy + y) * lw + jx;
        for x in (0..t.cols).step_by(LANES) {
            let o = y * t.cols + x;
            let diff =
                bilinear_u8(data, r0 + x, lw, wv) - i16x8::from_slice_unaligned(&t.values[o..]);
            q1 += diff.dot(i16x8::from_slice_unaligned(&t.grad_x[o..]));
            q2 += diff.dot(i16x8::from_slice_unaligned(&t.grad_y[o..]));
            pending += 1;
            if pending == MISMATCH_FLUSH {
                (b1, b2) = (b1 + hsum(q1), b2 + hsum(q2));
                (q1, q2) = (zero, zero);
                pending = 0;
            }
        }
    }
    (b1 + hsum(q1), b2 + hsum(q2))
}

/// `sum |J - I|` over the real window, in window units.
fn abs_error(plane: &Image<u8, 1>, jx: isize, jy: isize, w: [i32; 4], t: &LkTemplate) -> i64 {
    let (lw, lh) = (plane.width(), plane.height());
    let data = plane.as_slice();
    let mut sum = 0i64;
    if !window_inside(jx, jy, t, lw, lh) {
        for y in 0..t.win {
            for x in 0..t.win {
                let (px, py) = (jx + x as isize, jy + y as isize);
                let jv = descale(bilinear_101(data, lw, lh, px, py, w), SAMPLE_SHIFT);
                sum += i64::from((jv - i32::from(t.values[y * t.cols + x])).abs());
            }
        }
        return sum;
    }
    let (jx, jy) = (jx as usize, jy as usize);
    let wv = weight_lanes(w);
    let mask = tail_mask(t.win, t.cols);
    for y in 0..t.win {
        let r0 = (jy + y) * lw + jx;
        for x in (0..t.cols).step_by(LANES) {
            let o = y * t.cols + x;
            let diff =
                bilinear_u8(data, r0 + x, lw, wv) - i16x8::from_slice_unaligned(&t.values[o..]);
            let diff = (diff & lane_mask(x, t.cols, mask)).abs();
            sum += i64::from(i32x8::from_i16x8(diff).reduce_add());
        }
    }
    sum
}

/// Every vector read of a `t`-sized window at `(jx, jy)` and its bilinear
/// neighbour lies in the plane.
fn window_inside(jx: isize, jy: isize, t: &LkTemplate, lw: usize, lh: usize) -> bool {
    jx >= 0 && jy >= 0 && jx as usize + t.cols < lw && jy as usize + t.win < lh
}

/// `descale(sum w_k p_k, shift)` of 8 consecutive u8 samples at `o` (row `o`
/// and `o + 1`, the next row at `o + stride`), as i16 lanes.
#[inline(always)]
fn bilinear_u8(data: &[u8], o: usize, stride: usize, w: [i16x8; 4]) -> i16x8 {
    let p = [o, o + 1, o + stride, o + stride + 1].map(|k| load8_i16(&data[k..]));
    combine(p, w, SAMPLE_SHIFT)
}

#[inline(always)]
fn bilinear_i16(data: &[i16], o: usize, stride: usize, w: [i16x8; 4]) -> i16x8 {
    let p = [o, o + 1, o + stride, o + stride + 1].map(|k| i16x8::from_slice_unaligned(&data[k..]));
    combine(p, w, W_BITS)
}

/// `(sum w_k p_k + 2^(shift - 1)) >> shift`; every sum fits i32 and every
/// result fits i16, so truncation equals OpenCV's saturating pack.
#[inline(always)]
fn combine(p: [i16x8; 4], w: [i16x8; 4], shift: i32) -> i16x8 {
    let sum =
        p[0].mul_widen(w[0]) + p[1].mul_widen(w[1]) + p[2].mul_widen(w[2]) + p[3].mul_widen(w[3]);
    i16x8::from_i32x8_truncate((sum + (1i32 << (shift - 1))) >> shift)
}

fn weight_lanes(w: [i32; 4]) -> [i16x8; 4] {
    // Each Q14 weight is in [-1, 2^14], so it fits i16.
    w.map(|k| i16x8::splat(k as i16))
}

/// Lanes of the last chunk that lie inside the window.
fn tail_mask(win: usize, cols: usize) -> i16x8 {
    i16x8::new(std::array::from_fn(|k| {
        if cols - LANES + k < win { -1 } else { 0 }
    }))
}

#[inline(always)]
fn lane_mask(x: usize, cols: usize, tail: i16x8) -> i16x8 {
    if x + LANES == cols {
        tail
    } else {
        i16x8::splat(-1)
    }
}

fn hsum(v: i32x4) -> i64 {
    v.to_array().iter().map(|&l| i64::from(l)).sum()
}

#[inline]
fn pixel_101(data: &[u8], lw: usize, lh: usize, x: isize, y: isize) -> i32 {
    let (x, y) = (reflect_101(x, lw as isize), reflect_101(y, lh as isize));
    i32::from(data[y as usize * lw + x as usize])
}

/// Undescaled Q14 bilinear sample of the reflect-101 plane at corner `(x, y)`.
#[inline]
fn bilinear_101(data: &[u8], lw: usize, lh: usize, x: isize, y: isize, w: [i32; 4]) -> i32 {
    let p = |dx: isize, dy: isize| pixel_101(data, lw, lh, x + dx, y + dy);
    p(0, 0) * w[0] + p(1, 0) * w[1] + p(0, 1) * w[2] + p(1, 1) * w[3]
}

/// Q14 bilinear weights `[w00, w01, w10, w11]` for fractions `(a, b)`, rounded as OpenCV does.
#[inline]
fn q14_weights(a: f32, b: f32) -> [i32; 4] {
    let one = W_ONE as f32;
    let w00 = ((1.0 - a) * (1.0 - b) * one).round_ties_even() as i32;
    let w01 = (a * (1.0 - b) * one).round_ties_even() as i32;
    let w10 = ((1.0 - a) * b * one).round_ties_even() as i32;
    [w00, w01, w10, W_ONE - w00 - w01 - w10]
}

#[inline]
fn descale(v: i32, n: i32) -> i32 {
    (v + (1 << (n - 1))) >> n
}

/// Integer corner of a window whose top-left is at `p`, if the window still
/// overlaps the level as OpenCV requires (`-win <= floor(p) < size`).
#[inline]
fn corner(p: [f32; 2], win: usize, width: usize, height: usize) -> Option<[isize; 2]> {
    let (x, y) = (p[0].floor(), p[1].floor());
    let w = win as f32;
    if x < -w || x >= width as f32 || y < -w || y >= height as f32 {
        return None;
    }
    Some([x as isize, y as isize])
}

fn check_window(win: usize) -> Result<(), LkWindowError> {
    if win < 3 || win.is_multiple_of(2) {
        return Err(LkWindowError::InvalidWindow(win));
    }
    Ok(())
}

fn check_size(size: ImageSize, win: usize) -> Result<(), LkWindowError> {
    if size.width < win || size.height < win {
        return Err(LkWindowError::ImageTooSmall {
            width: size.width,
            height: size.height,
        });
    }
    Ok(())
}

fn copy_image(image: &Image<u8, 1>) -> Result<Image<u8, 1>, LkWindowError> {
    Image::new(image.size(), image.as_slice().to_vec()).map_err(image_error)
}

fn blank_image(size: ImageSize) -> Result<Image<u8, 1>, LkWindowError> {
    Image::from_size_val(size, 0u8).map_err(image_error)
}

fn image_error(e: ImageError) -> LkWindowError {
    LkWindowError::Image(e.to_string())
}

/// An image or window that [`LkFrame`] cannot hold.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum LkWindowError {
    /// Windows must be odd and at least 3.
    InvalidWindow(usize),
    /// The image is smaller than one window.
    ImageTooSmall { width: usize, height: usize },
    /// A level buffer could not be made.
    Image(String),
}

impl std::fmt::Display for LkWindowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidWindow(win) => write!(f, "LK window {win} must be odd and at least 3"),
            Self::ImageTooSmall { width, height } => {
                write!(f, "a {width}x{height} image is smaller than one LK window")
            }
            Self::Image(e) => write!(f, "LK level buffer: {e}"),
        }
    }
}

impl std::error::Error for LkWindowError {}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_imgproc::optical_flow_pyr_lk::{
        LkPyramidU8, PyrLKParams, calc_optical_flow_pyr_lk_u8,
    };

    /// A smooth textured scene shifted by `(dx, dy)`, with a flat square
    /// whose windows fail the eigenvalue test.
    fn scene(width: usize, height: usize, dx: f32, dy: f32) -> Image<u8, 1> {
        let data = (0..width * height)
            .map(|i| {
                let (x, y) = ((i % width) as f32 - dx, (i / width) as f32 - dy);
                if (40.0..70.0).contains(&x) && (40.0..70.0).contains(&y) {
                    return 90;
                }
                let v = 128.0
                    + 50.0 * (0.31 * x).sin() * (0.27 * y).cos()
                    + 35.0 * (0.13 * (x + 2.0 * y)).sin()
                    + 20.0 * (0.71 * x - 0.53 * y).cos();
                v.round().clamp(0.0, 255.0) as u8
            })
            .collect();
        Image::new(ImageSize { width, height }, data).unwrap()
    }

    /// Interior points, the flat square, and points whose windows leave the
    /// image (the border path) or the level (status 0).
    fn points(width: usize, height: usize) -> Vec<[f32; 2]> {
        let (w, h) = (width as f32, height as f32);
        let mut pts: Vec<[f32; 2]> = (0..12)
            .flat_map(|j| {
                (0..16).map(move |i| {
                    [
                        8.3 + i as f32 * (w - 16.0) / 15.0,
                        6.7 + j as f32 * (h - 13.0) / 11.0,
                    ]
                })
            })
            .collect();
        pts.extend([
            [55.0, 55.0],
            [0.0, 0.0],
            [1.5, h * 0.5],
            [w - 1.2, h * 0.5],
            [w * 0.5, h - 0.6],
            [w * 0.5, 2.25],
            [w - 3.0, h - 3.0],
            [-4.0, h * 0.5],
            [w + 30.0, 10.0],
        ]);
        pts
    }

    /// Tracks `pts` on one level with both kernels and asserts bit-equality:
    /// `prev_lk`/`next_lk` hold that level as their level 0 for kornia-rs.
    #[allow(clippy::too_many_arguments)]
    fn assert_matches_kornia(
        prev: &LkFrame,
        next: &LkFrame,
        prev_lk: &Image<u8, 1>,
        next_lk: &Image<u8, 1>,
        level: usize,
        win: usize,
        max_iter: usize,
        shift: [f32; 2],
    ) {
        let scale = (1u32 << level) as f32;
        let pts = points(prev.plane().width(), prev.plane().height());
        let guesses: Vec<[f32; 2]> = pts
            .iter()
            .map(|p| [p[0] + shift[0], p[1] + shift[1]])
            .collect();
        let down = |v: &[[f32; 2]]| {
            v.iter()
                .map(|p| [p[0] / scale, p[1] / scale])
                .collect::<Vec<_>>()
        };
        let params = PyrLKParams {
            win_size: win,
            max_level: 0,
            max_iter,
            use_initial_flow: true,
            ..PyrLKParams::default()
        };
        let p = LkPyramidU8::new(prev_lk, win, 0, true).unwrap();
        let q = LkPyramidU8::new(next_lk, win, 0, false).unwrap();
        let (their_pts, their_guesses) = (down(&pts), down(&guesses));
        let theirs = calc_optical_flow_pyr_lk_u8(
            &p,
            &q,
            &their_pts,
            Some(their_guesses.as_slice()),
            &params,
        )
        .unwrap();
        let mut template = LkTemplate::default();
        let mut tracked = 0;
        for (i, (&pt, &guess)) in pts.iter().zip(&guesses).enumerate() {
            if !extract_template(prev, pt, level, win, &mut template) {
                assert_eq!(theirs.status[i], 0, "point {i} {pt:?}");
                continue;
            }
            let ours = track_template(&template, next, guess, max_iter, params.epsilon);
            let their_pixel = [theirs.next_pts[i][0] * scale, theirs.next_pts[i][1] * scale];
            assert_eq!(ours.status, theirs.status[i] == 1, "point {i} {pt:?}");
            assert_eq!(ours.pixel, their_pixel, "point {i} {pt:?}");
            assert_eq!(ours.error, theirs.error[i], "point {i} {pt:?}");
            tracked += usize::from(ours.status);
        }
        assert!(
            tracked > pts.len() / 2,
            "{tracked} of {} tracked",
            pts.len()
        );
    }

    #[test]
    fn level_0_matches_kornia_bit_for_bit() {
        let (prev_img, next_img) = (scene(160, 120, 0.0, 0.0), scene(160, 120, 1.37, -0.81));
        for (win, max_iter) in [(11, 8), (21, 30), (7, 3)] {
            let prev = LkFrame::new(&prev_img, win, 0).unwrap();
            let next = LkFrame::new(&next_img, win, 1).unwrap();
            assert_matches_kornia(
                &prev,
                &next,
                &prev_img,
                &next_img,
                0,
                win,
                max_iter,
                [0.9, -0.4],
            );
        }
    }

    #[test]
    fn level_1_matches_kornia_on_its_pyrdown() {
        let (prev_img, next_img) = (scene(203, 151, 0.0, 0.0), scene(203, 151, 2.6, 1.9));
        let win = 11;
        let mut prev = LkFrame::new(&prev_img, win, 0).unwrap();
        let mut next = LkFrame::new(&next_img, win, 1).unwrap();
        prev.build_coarse(1).unwrap();
        next.build_coarse(1).unwrap();
        assert_eq!((prev.levels(), next.levels()), (2, 2));
        let half = |img: &Image<u8, 1>| {
            let size = ImageSize {
                width: img.width().div_ceil(2),
                height: img.height().div_ceil(2),
            };
            let mut dst = Image::from_size_val(size, 0u8).unwrap();
            kornia_imgproc::pyramid::pyrdown_u8(img, &mut dst).unwrap();
            dst
        };
        let (prev_1, next_1) = (half(&prev_img), half(&next_img));
        assert_eq!(prev.levels[1].as_slice(), prev_1.as_slice());
        assert_eq!(next.levels[1].as_slice(), next_1.as_slice());
        assert_matches_kornia(&prev, &next, &prev_1, &next_1, 1, win, 8, [2.0, 2.0]);
    }

    #[test]
    fn recovers_a_subpixel_shift() {
        let (prev_img, next_img) = (scene(160, 120, 0.0, 0.0), scene(160, 120, 0.6, -0.35));
        let prev = LkFrame::new(&prev_img, 11, 0).unwrap();
        let next = LkFrame::new(&next_img, 11, 1).unwrap();
        let mut template = LkTemplate::default();
        assert!(extract_template(&prev, [100.0, 80.0], 0, 11, &mut template));
        let track = track_template(&template, &next, [100.0, 80.0], 10, 0.01);
        assert!(track.status);
        assert!((track.pixel[0] - 100.6).abs() < 0.05, "{track:?}");
        assert!((track.pixel[1] - 79.65).abs() < 0.05, "{track:?}");
        assert!(track.iterations >= 1 && track.iterations <= 10);
    }

    #[test]
    fn coarse_levels_stop_at_the_window() {
        let mut frame = LkFrame::new(&scene(100, 60, 0.0, 0.0), 11, 0).unwrap();
        frame.build_coarse(5).unwrap();
        // 100x60, 50x30, 25x15; 13x8 would not hold an 11 window.
        assert_eq!(frame.levels(), 3);
        let mut template = LkTemplate::default();
        assert!(!extract_template(
            &frame,
            [50.0, 30.0],
            3,
            11,
            &mut template
        ));
    }

    #[test]
    fn rebuild_reuses_the_buffers_and_drops_coarse_levels() {
        let mut frame = LkFrame::new(&scene(100, 60, 0.0, 0.0), 11, 0).unwrap();
        frame.build_coarse(1).unwrap();
        let (plane, coarse) = (
            frame.levels[0].as_slice().as_ptr(),
            frame.levels[1].as_slice().as_ptr(),
        );
        let next = scene(100, 60, 1.0, 0.0);
        frame.rebuild(&next, 7).unwrap();
        assert_eq!((frame.frame_id(), frame.levels()), (7, 1));
        assert_eq!(frame.plane().as_slice(), next.as_slice());
        frame.build_coarse(1).unwrap();
        assert_eq!(
            (
                frame.levels[0].as_slice().as_ptr(),
                frame.levels[1].as_slice().as_ptr()
            ),
            (plane, coarse)
        );
    }

    #[test]
    fn rejects_bad_windows_and_small_images() {
        let img = scene(20, 20, 0.0, 0.0);
        assert_eq!(
            LkFrame::new(&img, 10, 0).err(),
            Some(LkWindowError::InvalidWindow(10))
        );
        assert_eq!(
            LkFrame::new(&img, 1, 0).err(),
            Some(LkWindowError::InvalidWindow(1))
        );
        assert_eq!(
            LkFrame::new(&img, 21, 0).err(),
            Some(LkWindowError::ImageTooSmall {
                width: 20,
                height: 20
            })
        );
        let frame = LkFrame::new(&img, 11, 0).unwrap();
        let mut template = LkTemplate::default();
        assert!(!extract_template(&frame, [10.0, 10.0], 0, 4, &mut template));
        let track = track_template(&template, &frame, [3.0, 4.0], 10, 0.01);
        assert_eq!(
            (track.status, track.pixel, track.iterations),
            (false, [3.0, 4.0], 0)
        );
    }

    #[test]
    #[ignore = "needs tests/fixtures/upstream/frontend_opencv.bin from benches/opencv_frontend_fixtures.py"]
    fn matches_opencv_within_a_hundredth_of_a_pixel() {
        use super::super::opencv_fixture::{self as cv, LK_MAX_ITER, LK_WIN};
        let fx = cv::load();
        let prev = LkFrame::new(&fx.frame1, LK_WIN, 0).unwrap();
        let next = LkFrame::new(&fx.frame2, LK_WIN, 1).unwrap();
        let mut template = LkTemplate::default();
        for (i, &p) in fx.refined.iter().enumerate() {
            let track = if extract_template(&prev, p, 0, LK_WIN, &mut template) {
                track_template(&template, &next, p, LK_MAX_ITER, cv::EPSILON)
            } else {
                LkTrack {
                    pixel: p,
                    status: false,
                    error: 0.0,
                    iterations: 0,
                }
            };
            assert_eq!(track.status, fx.status[i] == 1, "point {i} {p:?}");
            if track.status {
                let (dx, dy) = (
                    track.pixel[0] - fx.tracked[i][0],
                    track.pixel[1] - fx.tracked[i][1],
                );
                assert!(
                    dx.abs() <= 0.01 && dy.abs() <= 0.01,
                    "point {i}: {track:?} vs {:?}",
                    fx.tracked[i]
                );
            }
        }
    }
}
