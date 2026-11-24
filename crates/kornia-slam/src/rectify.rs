//! Stereo rectification through precomputed fixed-point tables: the
//! [`StereoRectifier`]'s maps, sampled on SIMD lanes on the calling thread.

use kornia_3d::stereo::StereoRectifier;
use kornia_image::{Image, ImageError};

pub use crate::upstream::kornia_imgproc::remap_u8::RemapQuantisation;
use crate::upstream::kornia_imgproc::remap_u8::{RemapTable, remap_u8_table};

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
    use super::*;
    use crate::upstream::kornia_imgproc::remap_u8::tests::{euroc_rectifier, texture};

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
