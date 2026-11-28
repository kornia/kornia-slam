//! Stereo rectification through precomputed fixed-point tables: the
//! [`StereoRectifier`]'s maps, sampled on the calling thread.

use kornia_3d::stereo::StereoRectifier;
use kornia_image::{Image, ImageError};

pub use kornia_imgproc::interpolation::RemapQuantisation;
use kornia_imgproc::interpolation::{RemapTable, remap_u8_table};

/// Both views' rectification tables of one [`StereoRectifier`].
#[derive(Debug, Clone)]
pub struct StereoRemap {
    left: RemapTable,
    right: RemapTable,
}

impl StereoRemap {
    /// Tables of `rectifier`'s maps. [`RemapQuantisation::Q10Trunc`] gives
    /// the bytes of [`StereoRectifier::rectify_left`] / `rectify_right`.
    ///
    /// # Errors
    ///
    /// [`ImageError::InvalidImageSize`] for maps smaller than 2×2.
    pub fn new(rectifier: &StereoRectifier, quant: RemapQuantisation) -> Result<Self, ImageError> {
        let table =
            |(x, y): (&Image<f32, 1>, &Image<f32, 1>)| RemapTable::from_maps(x, y, x.size(), quant);
        Ok(Self {
            left: table(rectifier.left_maps())?,
            right: table(rectifier.right_maps())?,
        })
    }

    /// Rectifies a raw pair into `left_dst`, `right_dst`. No allocation.
    ///
    /// # Errors
    ///
    /// [`ImageError::InvalidImageSize`] when an image is not the
    /// rectifier's size.
    pub fn rectify(
        &self,
        left: &Image<u8, 1>,
        right: &Image<u8, 1>,
        left_dst: &mut Image<u8, 1>,
        right_dst: &mut Image<u8, 1>,
    ) -> Result<(), ImageError> {
        remap_u8_table(left, left_dst, &self.left)?;
        remap_u8_table(right, right_dst, &self.right)
    }
}

#[cfg(test)]
mod tests {
    use kornia_3d::stereo::CameraCalib;
    use kornia_algebra::{Mat3F64, Vec3F64};
    use kornia_image::ImageSize;

    use super::*;

    /// EuRoC MH_01 `cam0` / `cam1` (`mav0/cam*/sensor.yaml`): radtan, k1 −0.283.
    fn euroc_rectifier() -> StereoRectifier {
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
    fn texture(size: ImageSize, seed: u32) -> Image<u8, 1> {
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

    #[test]
    fn q10_rectifies_like_the_rectifier() {
        let rectifier = euroc_rectifier();
        let remap = StereoRemap::new(&rectifier, RemapQuantisation::Q10Trunc).unwrap();
        let size = rectifier.left_maps().0.size();
        let (left, right) = (texture(size, 1), texture(size, 2));
        let blank = || Image::from_size_val(size, 0u8).unwrap();
        let (mut want_l, mut want_r, mut got_l, mut got_r) = (blank(), blank(), blank(), blank());
        rectifier.rectify_left(&left, &mut want_l).unwrap();
        rectifier.rectify_right(&right, &mut want_r).unwrap();
        remap
            .rectify(&left, &right, &mut got_l, &mut got_r)
            .unwrap();
        assert_eq!(got_l.as_slice(), want_l.as_slice());
        assert_eq!(got_r.as_slice(), want_r.as_slice());
    }
}
