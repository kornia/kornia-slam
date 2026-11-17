//! Upstream target: kornia_imgproc::pyramid. Kept here until upstreamed; no kornia-slam types.
//!
//! Single-channel u8 pyrDown on `wide` vectors. kornia-rs's `pyrdown_u8`
//! vectorises on x86_64 only, so aarch64 runs it scalar. The output is
//! byte-equal to it and to OpenCV's `pyrDown`: 1-4-6-4-1 taps on both axes,
//! reflect-101 borders, `(sum + 128) >> 8`. The 2D sum is an exact integer,
//! so the vertical pass runs first here, at full rate on whole rows.

use kornia_image::{Image, ImageError};
use wide::u16x8;

use super::lanes::{LANES, load16};

/// OpenCV's `BORDER_REFLECT_101` index of `p` in `0..len`.
#[inline]
pub(crate) fn reflect_101(p: isize, len: isize) -> isize {
    if len == 1 {
        return 0;
    }
    let period = 2 * (len - 1);
    let p = p.abs() % period;
    if p >= len { period - p } else { p }
}

/// Halves `src` into `dst` (`ceil(w / 2)` × `ceil(h / 2)`), reusing `rows`
/// for one row of vertical sums.
///
/// # Errors
///
/// [`ImageError::InvalidImageSize`] when `dst` is not half of `src`.
pub fn pyrdown_u8(
    src: &Image<u8, 1>,
    dst: &mut Image<u8, 1>,
    rows: &mut Vec<u16>,
) -> Result<(), ImageError> {
    let (w, h) = (src.width(), src.height());
    let (dw, dh) = (w.div_ceil(2), h.div_ceil(2));
    if dst.width() != dw || dst.height() != dh {
        return Err(ImageError::InvalidImageSize(
            dw,
            dh,
            dst.width(),
            dst.height(),
        ));
    }
    // Two reflected columns each side, plus lanes so the horizontal pass
    // reads whole vectors past the last output.
    rows.resize(w + 4 + 2 * LANES, 0);
    let src = src.as_slice();
    let out = dst.as_slice_mut();
    for dy in 0..dh {
        let taps = [-2isize, -1, 0, 1, 2].map(|k| {
            let y = reflect_101(2 * dy as isize + k, h as isize) as usize;
            &src[y * w..(y + 1) * w]
        });
        vertical_row(taps, &mut rows[2..2 + w]);
        for (pad, k) in [
            (0usize, -2isize),
            (1, -1),
            (w + 2, w as isize),
            (w + 3, w as isize + 1),
        ] {
            rows[pad] = rows[2 + reflect_101(k, w as isize) as usize];
        }
        horizontal_row(rows, &mut out[dy * dw..(dy + 1) * dw]);
    }
    Ok(())
}

/// `v[x] = r0 + r4 + 4 (r1 + r3) + 6 r2` per column; at most 4080.
fn vertical_row(r: [&[u8]; 5], v: &mut [u16]) {
    let w = v.len();
    let mut x = 0;
    while x + 2 * LANES <= w {
        let [a, b, c, d, e] = r.map(|row| load16(&row[x..]));
        let lo = u16x8::from_u8x16_low;
        let hi = u16x8::from_u8x16_high;
        let sum_lo = lo(a) + lo(e) + (lo(b) + lo(d)) * 4u16 + lo(c) * 6u16;
        let sum_hi = hi(a) + hi(e) + (hi(b) + hi(d)) * 4u16 + hi(c) * 6u16;
        v[x..x + LANES].copy_from_slice(&sum_lo.to_array());
        v[x + LANES..x + 2 * LANES].copy_from_slice(&sum_hi.to_array());
        x += 2 * LANES;
    }
    for x in x..w {
        let [a, b, c, d, e] = r.map(|row| u16::from(row[x]));
        v[x] = a + e + 4 * (b + d) + 6 * c;
    }
}

/// `out[dx] = (v[2dx - 2] + v[2dx + 2] + 4 (v[2dx ± 1]) + 6 v[2dx] + 128) >> 8`,
/// with `v` offset by the two reflected columns. Full-rate vectors, even
/// lanes kept; the sum is at most 65408, so u16 lanes hold it.
fn horizontal_row(v: &[u16], out: &mut [u8]) {
    let dw = out.len();
    let mut dx = 0;
    while dx + LANES / 2 <= dw && 2 * dx + 4 + LANES <= v.len() {
        let at = |k: usize| u16x8::new(load8_u16(&v[2 * dx + k..]));
        let sum = at(0) + at(4) + (at(1) + at(3)) * 4u16 + at(2) * 6u16 + 128u16;
        let lanes = (sum >> 8u32).to_array();
        for (k, o) in out[dx..dx + LANES / 2].iter_mut().enumerate() {
            *o = lanes[2 * k] as u8;
        }
        dx += LANES / 2;
    }
    for (dx, o) in out.iter_mut().enumerate().skip(dx) {
        let t = |k: usize| u32::from(v[2 * dx + k]);
        *o = ((t(0) + t(4) + 4 * (t(1) + t(3)) + 6 * t(2) + 128) >> 8) as u8;
    }
}

#[inline(always)]
fn load8_u16(s: &[u16]) -> [u16; 8] {
    let mut a = [0u16; 8];
    a.copy_from_slice(&s[..8]);
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::ImageSize;

    fn noise(width: usize, height: usize, seed: u32) -> Image<u8, 1> {
        let mut s = seed;
        let data = (0..width * height)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 24) as u8
            })
            .collect();
        Image::new(ImageSize { width, height }, data).unwrap()
    }

    #[test]
    fn matches_kornia_pyrdown_bytes() {
        for (w, h) in [(5, 5), (6, 7), (37, 21), (64, 48), (257, 161), (758, 572)] {
            let src = noise(w, h, (w * 31 + h) as u32);
            let size = ImageSize {
                width: w.div_ceil(2),
                height: h.div_ceil(2),
            };
            let mut ours = Image::from_size_val(size, 0u8).unwrap();
            let mut theirs = Image::from_size_val(size, 0u8).unwrap();
            pyrdown_u8(&src, &mut ours, &mut Vec::new()).unwrap();
            kornia_imgproc::pyramid::pyrdown_u8(&src, &mut theirs).unwrap();
            assert_eq!(ours.as_slice(), theirs.as_slice(), "{w}x{h}");
        }
    }

    #[test]
    fn rejects_a_wrong_destination() {
        let src = noise(10, 10, 1);
        let size = ImageSize {
            width: 4,
            height: 5,
        };
        let mut dst = Image::from_size_val(size, 0u8).unwrap();
        assert!(pyrdown_u8(&src, &mut dst, &mut Vec::new()).is_err());
    }

    #[test]
    fn reflect_101_matches_opencv() {
        assert_eq!(reflect_101(-1, 5), 1);
        assert_eq!(reflect_101(-2, 5), 2);
        assert_eq!(reflect_101(5, 5), 3);
        assert_eq!(reflect_101(6, 5), 2);
        assert_eq!(reflect_101(-9, 5), 1);
        assert_eq!(reflect_101(3, 1), 0);
    }
}
