//! Upstream target: kornia_imgproc::interpolation. Kept here until upstreamed; no kornia-slam types.
//!
//! Single-channel u8 bilinear remap through a precomputed table. kornia-rs's
//! mono `remap_u8` floors and splits every f32 map entry on every call, one
//! scalar pixel at a time. A rectification map is constant, so the integer
//! taps and fixed-point weights are computed once here and the blend runs on
//! `wide` lanes, on the calling thread.

use kornia_image::{Image, ImageError, ImageSize};
use wide::i32x8;

use super::lanes::{LANES, lane_starts};

/// How map coordinates become taps and weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemapQuantisation {
    /// kornia-rs `remap_u8`: `floor` taps, 10-bit truncated fractions,
    /// `(Σ w·p + 2^19) >> 20`. Byte-identical to it.
    Q10Trunc,
    /// OpenCV `convertMaps` to `CV_16SC2`: coordinates rounded to 1/32 px,
    /// `(Σ w·p + 512) >> 10`. OpenCV's 15-bit weights are these times 32, so
    /// the sum rounds the same.
    OpenCvQ5,
}

impl RemapQuantisation {
    const fn bits(self) -> u32 {
        match self {
            Self::Q10Trunc => 10,
            Self::OpenCvQ5 => 5,
        }
    }
}

/// A pixel whose taps the lane loop cannot read at fixed strides: the last
/// source row, where the clamped neighbours sit on the same row.
#[derive(Debug, Clone, Copy)]
struct EdgeTap {
    dst: u32,
    taps: [u32; 4],
    fx: u16,
    fy: u16,
}

/// Taps and weights of one map, for sources of `src_size`.
#[derive(Debug, Clone)]
pub struct RemapTable {
    src_size: ImageSize,
    dst_size: ImageSize,
    bits: u32,
    /// Top-left tap per output pixel; its neighbours are `+1`, `+w`, `+w+1`.
    offsets: Vec<u32>,
    /// `[fx, fy]` per output pixel; `fx = 0` where the right tap is clamped.
    weights: Vec<[u16; 2]>,
    /// Output pixels whose top-left tap is outside the source: zero.
    zeros: Vec<u32>,
    edges: Vec<EdgeTap>,
}

/// The top-left tap and fraction of one coordinate, `None` outside `0..len`.
fn tap(v: f32, len: i32, quant: RemapQuantisation) -> Option<(i32, u32)> {
    if !v.is_finite() {
        return None;
    }
    let (i, frac) = match quant {
        // The expressions of kornia-rs `remap_u8`, so every byte matches.
        RemapQuantisation::Q10Trunc => {
            let i = v.floor() as i32;
            (i, ((v - i as f32) * 1024.0) as u32)
        }
        RemapQuantisation::OpenCvQ5 => {
            let q = (v * 32.0).round_ties_even() as i32;
            (q >> 5, (q & 31) as u32)
        }
    };
    (0..len).contains(&i).then_some((i, frac))
}

impl RemapTable {
    /// Builds the table of `map_x`, `map_y` (source pixel per output pixel)
    /// for sources of `src_size`.
    ///
    /// # Errors
    ///
    /// [`ImageError::InvalidImageSize`] when the maps differ in size or the
    /// source is narrower or shorter than 2 pixels.
    pub fn from_maps(
        map_x: &Image<f32, 1>,
        map_y: &Image<f32, 1>,
        src_size: ImageSize,
        quant: RemapQuantisation,
    ) -> Result<Self, ImageError> {
        if map_x.size() != map_y.size() {
            return Err(ImageError::InvalidImageSize(
                map_x.width(),
                map_x.height(),
                map_y.width(),
                map_y.height(),
            ));
        }
        let (w, h) = (src_size.width, src_size.height);
        if w < 2 || h < 2 || w * h > u32::MAX as usize {
            return Err(ImageError::InvalidImageSize(w, h, 2, 2));
        }
        let n = map_x.width() * map_x.height();
        let mut table = Self {
            src_size,
            dst_size: map_x.size(),
            bits: quant.bits(),
            offsets: vec![0; n],
            weights: vec![[0; 2]; n],
            zeros: Vec::new(),
            edges: Vec::new(),
        };
        for (dst, (&xf, &yf)) in map_x.as_slice().iter().zip(map_y.as_slice()).enumerate() {
            let taps = tap(xf, w as i32, quant).zip(tap(yf, h as i32, quant));
            let Some(((xi, fx), (yi, fy))) = taps else {
                table.zeros.push(dst as u32);
                continue;
            };
            let (xi, yi) = (xi as usize, yi as usize);
            // A clamped neighbour is the same pixel, so its weight can move to
            // the near one: `p·(S - f) + p·f = p·S`.
            let (dx, fx) = if xi + 1 < w { (1, fx) } else { (0, 0) };
            let (dy, fy) = if yi + 1 < h { (w, fy) } else { (0, 0) };
            let (fx, fy) = (fx as u16, fy as u16);
            let off = yi * w + xi;
            // The lane loop reads `off ..= off + w + 1`; only the last row and
            // the last pixel of the row above it reach past the source.
            if off + w + 2 <= w * h {
                table.offsets[dst] = off as u32;
                table.weights[dst] = [fx, fy];
            } else {
                table.edges.push(EdgeTap {
                    dst: dst as u32,
                    taps: [off, off + dx, off + dy, off + dy + dx].map(|t| t as u32),
                    fx,
                    fy,
                });
            }
        }
        Ok(table)
    }

    pub fn src_size(&self) -> ImageSize {
        self.src_size
    }

    pub fn dst_size(&self) -> ImageSize {
        self.dst_size
    }
}

/// `(top·(S - fy) + bottom·fy + S²/2) >> 2·bits` with `top`, `bottom` the
/// horizontal blends; exact in i32 for 10-bit weights (≤ 255·2^20 + 2^19).
#[inline(always)]
fn blend(p: [i32x8; 4], fx: i32x8, fy: i32x8, bits: u32) -> i32x8 {
    let scale = i32x8::splat(1 << bits);
    let top = p[0] * (scale - fx) + p[1] * fx;
    let bottom = p[2] * (scale - fx) + p[3] * fx;
    (top * (scale - fy) + bottom * fy + i32x8::splat(1 << (2 * bits - 1))) >> (2 * bits) as i32
}

#[inline(always)]
fn blend_scalar(p: [i32; 4], fx: i32, fy: i32, bits: u32) -> u8 {
    let scale = 1 << bits;
    let top = p[0] * (scale - fx) + p[1] * fx;
    let bottom = p[2] * (scale - fx) + p[3] * fx;
    ((top * (scale - fy) + bottom * fy + (1 << (2 * bits - 1))) >> (2 * bits)) as u8
}

/// Samples `src` through `table` into `dst`. No allocation.
///
/// # Errors
///
/// [`ImageError::InvalidImageSize`] when `src` or `dst` is not the size the
/// table was built for.
pub fn remap_u8_table(
    src: &Image<u8, 1>,
    dst: &mut Image<u8, 1>,
    table: &RemapTable,
) -> Result<(), ImageError> {
    for (have, want) in [(src.size(), table.src_size), (dst.size(), table.dst_size)] {
        if have != want {
            return Err(ImageError::InvalidImageSize(
                have.width,
                have.height,
                want.width,
                want.height,
            ));
        }
    }
    let stride = table.src_size.width;
    let src = src.as_slice();
    let out = dst.as_slice_mut();
    let n = out.len();
    let lanes = |start: usize, out: &mut [u8]| {
        let mut p = [[0i32; LANES]; 4];
        let (mut fx, mut fy) = ([0i32; LANES], [0i32; LANES]);
        for k in 0..LANES {
            let off = table.offsets[start + k] as usize;
            // One bounds check for the four taps; `from_maps` keeps
            // `off + stride + 1` inside the source.
            let quad = &src[off..off + stride + 2];
            p[0][k] = i32::from(quad[0]);
            p[1][k] = i32::from(quad[1]);
            p[2][k] = i32::from(quad[stride]);
            p[3][k] = i32::from(quad[stride + 1]);
            let [wx, wy] = table.weights[start + k];
            (fx[k], fy[k]) = (i32::from(wx), i32::from(wy));
        }
        let v = blend(
            p.map(i32x8::new),
            i32x8::new(fx),
            i32x8::new(fy),
            table.bits,
        );
        for (o, &v) in out[start..start + LANES].iter_mut().zip(v.as_array()) {
            *o = v as u8;
        }
    };
    if n < LANES {
        for (i, o) in out.iter_mut().enumerate() {
            let off = table.offsets[i] as usize;
            let [wx, wy] = table.weights[i];
            let p = [off, off + 1, off + stride, off + stride + 1].map(|t| i32::from(src[t]));
            *o = blend_scalar(p, i32::from(wx), i32::from(wy), table.bits);
        }
    }
    for start in lane_starts(n) {
        lanes(start, out);
    }
    for e in &table.edges {
        let p = e.taps.map(|t| i32::from(src[t as usize]));
        out[e.dst as usize] = blend_scalar(p, i32::from(e.fx), i32::from(e.fy), table.bits);
    }
    for &z in &table.zeros {
        out[z as usize] = 0;
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use kornia_3d::stereo::{CameraCalib, StereoRectifier};
    use kornia_algebra::{Mat3F64, Vec3F64};
    use kornia_imgproc::interpolation::{InterpolationMode, remap_u8};

    use super::*;

    /// EuRoC MH_01 `cam0` / `cam1` (`mav0/cam*/sensor.yaml`): radtan, k1 −0.283.
    pub(crate) fn euroc_rectifier() -> StereoRectifier {
        let calib = |fx, fy, cx, cy, k1, k2, p1, p2| CameraCalib {
            width: 752,
            height: 480,
            fx,
            fy,
            cx,
            cy,
            distortion: kornia_imgproc::calibration::distortion::PolynomialDistortion {
                k1,
                k2,
                k3: 0.0,
                k4: 0.0,
                k5: 0.0,
                k6: 0.0,
                p1,
                p2,
            },
        };
        let left = calib(
            458.654,
            457.296,
            367.215,
            248.375,
            -0.28340811,
            0.07395907,
            0.00019359,
            1.76187114e-05,
        );
        let right = calib(
            457.587,
            456.134,
            379.999,
            255.238,
            -0.28368365,
            0.07451284,
            -0.00010473,
            -3.55590700e-05,
        );
        // Row-major rotation of `T_BS`.
        let t_bs = |r: [f64; 9], t: [f64; 3]| {
            let col = |c: usize| Vec3F64::new(r[c], r[3 + c], r[6 + c]);
            (
                Mat3F64::from_cols(col(0), col(1), col(2)),
                Vec3F64::new(t[0], t[1], t[2]),
            )
        };
        let (r_l, t_l) = t_bs(
            [
                0.0148655429818,
                -0.999880929698,
                0.00414029679422,
                0.999557249008,
                0.0149672133247,
                0.025715529948,
                -0.0257744366974,
                0.00375618835797,
                0.999660727178,
            ],
            [-0.0216401454975, -0.064676986768, 0.00981073058949],
        );
        let (r_r, t_r) = t_bs(
            [
                0.0125552670891,
                -0.999755099723,
                0.0182237714554,
                0.999598781151,
                0.0130119051815,
                0.0251588363115,
                -0.0253898008918,
                0.0179005838253,
                0.999517347078,
            ],
            [-0.0198435579556, 0.0453689425024, 0.00786212447038],
        );
        let r_rt = r_r.transpose();
        StereoRectifier::from_calib(&left, &right, r_rt * r_l, r_rt * (t_l - t_r)).unwrap()
    }

    /// A textured image with sharp steps, so every tap and weight matters.
    pub(crate) fn texture(size: ImageSize, seed: u32) -> Image<u8, 1> {
        let data = (0..size.width * size.height)
            .map(|i| {
                let (x, y) = ((i % size.width) as u32, (i / size.width) as u32);
                let h = (x.wrapping_mul(73_856_093) ^ y.wrapping_mul(19_349_663) ^ seed)
                    .wrapping_mul(2_654_435_761);
                if (x / 7 + y / 5) % 2 == 0 {
                    (h >> 24) as u8
                } else {
                    255 - ((h >> 26) as u8)
                }
            })
            .collect();
        Image::new(size, data).unwrap()
    }

    fn reference(src: &Image<u8, 1>, map_x: &Image<f32, 1>, map_y: &Image<f32, 1>) -> Vec<u8> {
        let mut dst = Image::from_size_val(map_x.size(), 0).unwrap();
        remap_u8(src, &mut dst, map_x, map_y, InterpolationMode::Bilinear).unwrap();
        dst.as_slice().to_vec()
    }

    fn ours(
        src: &Image<u8, 1>,
        map_x: &Image<f32, 1>,
        map_y: &Image<f32, 1>,
        quant: RemapQuantisation,
    ) -> Vec<u8> {
        let table = RemapTable::from_maps(map_x, map_y, src.size(), quant).unwrap();
        let mut dst = Image::from_size_val(map_x.size(), 0).unwrap();
        remap_u8_table(src, &mut dst, &table).unwrap();
        dst.as_slice().to_vec()
    }

    #[test]
    fn q10_equals_remap_u8_on_the_euroc_maps() {
        let rectifier = euroc_rectifier();
        let size = ImageSize {
            width: 752,
            height: 480,
        };
        for (seed, (map_x, map_y)) in [rectifier.left_maps(), rectifier.right_maps()]
            .into_iter()
            .enumerate()
        {
            let src = texture(size, seed as u32);
            assert_eq!(
                ours(&src, map_x, map_y, RemapQuantisation::Q10Trunc),
                reference(&src, map_x, map_y)
            );
        }
    }

    /// NaN, infinite, negative, exact-integer, last-row and last-column taps,
    /// the open `(w - 1, w)` band, and widths that are not a multiple of 8.
    #[test]
    fn q10_equals_remap_u8_on_the_edge_taps() {
        let src = texture(
            ImageSize {
                width: 13,
                height: 9,
            },
            7,
        );
        let (w, h) = (src.width() as f32, src.height() as f32);
        let specials = [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -1e-6,
            -0.5,
            -1.0,
            0.0,
            0.25,
            1.0,
            5.0,
            w - 2.0,
            w - 1.5,
            w - 1.0,
            w - 0.25,
            w - 1e-4,
            w,
            w + 0.5,
            h - 2.0,
            h - 1.0,
            h - 0.3,
            h,
            1e9,
            -1e9,
        ];
        // Every (x, y) pair, laid out at several output widths.
        let pairs: Vec<(f32, f32)> = specials
            .iter()
            .flat_map(|&y| specials.iter().map(move |&x| (x, y)))
            .collect();
        for dst_w in [1usize, 7, 8, 11, 23] {
            let size = ImageSize {
                width: dst_w,
                height: pairs.len().div_ceil(dst_w),
            };
            let at = |i: usize| pairs[i % pairs.len()];
            let n = size.width * size.height;
            let mx = Image::new(size, (0..n).map(|i| at(i).0).collect()).unwrap();
            let my = Image::new(size, (0..n).map(|i| at(i).1).collect()).unwrap();
            assert_eq!(
                ours(&src, &mx, &my, RemapQuantisation::Q10Trunc),
                reference(&src, &mx, &my),
                "dst width {dst_w}"
            );
        }
    }

    #[test]
    fn q5_samples_exact_positions_exactly() {
        let src = texture(
            ImageSize {
                width: 20,
                height: 10,
            },
            3,
        );
        let size = ImageSize {
            width: 18,
            height: 8,
        };
        let n = size.width * size.height;
        let mx = Image::new(size, (0..n).map(|i| (i % 18) as f32 + 0.5).collect()).unwrap();
        let my = Image::new(size, (0..n).map(|i| (i / 18) as f32).collect()).unwrap();
        let out = ours(&src, &mx, &my, RemapQuantisation::OpenCvQ5);
        for (i, &v) in out.iter().enumerate() {
            let (x, y) = (i % 18, i / 18);
            let s = |x: usize| u32::from(src.as_slice()[y * 20 + x]);
            assert_eq!(u32::from(v), (s(x) * 16 + s(x + 1) * 16 + 16) >> 5);
        }
    }

    #[test]
    fn wrong_sizes_are_rejected() {
        let size = ImageSize {
            width: 4,
            height: 4,
        };
        let map = Image::from_size_val(size, 1.0f32).unwrap();
        let table = RemapTable::from_maps(&map, &map, size, RemapQuantisation::Q10Trunc).unwrap();
        let src = Image::from_size_val(size, 9u8).unwrap();
        let mut small = Image::from_size_val(
            ImageSize {
                width: 3,
                height: 4,
            },
            0u8,
        )
        .unwrap();
        assert!(remap_u8_table(&src, &mut small, &table).is_err());
        assert!(remap_u8_table(&small.clone(), &mut src.clone(), &table).is_err());
        let one_row = ImageSize {
            width: 4,
            height: 1,
        };
        assert!(RemapTable::from_maps(&map, &map, one_row, RemapQuantisation::Q10Trunc).is_err());
    }

    /// OpenCV `convertMaps` + `remap` (`benches/opencv_remap_fixture.py`):
    /// within 1 LSB wherever all four taps are inside the image.
    #[test]
    #[ignore = "needs tests/fixtures/upstream/remap_opencv.bin from benches/opencv_remap_fixture.py"]
    fn q5_matches_opencv_remap() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/upstream/remap_opencv.bin"
        );
        let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as usize;
        let size = ImageSize {
            width: u32_at(0),
            height: u32_at(4),
        };
        let n = size.width * size.height;
        let floats = |o: usize| -> Image<f32, 1> {
            let v = (0..n)
                .map(|i| f32::from_le_bytes(b[o + 4 * i..o + 4 * i + 4].try_into().unwrap()))
                .collect();
            Image::new(size, v).unwrap()
        };
        let src = Image::new(size, b[8..8 + n].to_vec()).unwrap();
        let (mx, my) = (floats(8 + n), floats(8 + 5 * n));
        let opencv = &b[8 + 9 * n..8 + 10 * n];
        let got = ours(&src, &mx, &my, RemapQuantisation::OpenCvQ5);
        let inside = |v: f32, len: usize| {
            let i = ((v * 32.0).round_ties_even() as i32) >> 5;
            i >= 0 && (i as usize) + 1 < len
        };
        let (mut compared, mut worst) = (0usize, 0i32);
        for i in 0..n {
            let (x, y) = (mx.as_slice()[i], my.as_slice()[i]);
            if inside(x, size.width) && inside(y, size.height) {
                compared += 1;
                worst = worst.max((i32::from(got[i]) - i32::from(opencv[i])).abs());
            }
        }
        assert!(compared > n / 2, "{compared} of {n} pixels compared");
        assert!(worst <= 1, "max |diff| {worst}");
    }
}
