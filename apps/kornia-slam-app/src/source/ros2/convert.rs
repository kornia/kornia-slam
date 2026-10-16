//! ROS 2 message contents to kornia types, independent of the ROS bindings.

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, Vec3F64};
use kornia_image::{Image, ImageSize};
use kornia_imgproc::color::{bgr_from_rgb, gray_from_rgb_u8};

use super::Extrinsic;

/// The camera of a rectified image from its `CameraInfo` projection matrix `P`
/// (row-major 3×4). Rectified images carry no distortion.
pub fn camera_from_projection(p: &[f64; 12]) -> Result<PinholeCamera, String> {
    let camera = PinholeCamera {
        fx: p[0],
        fy: p[5],
        cx: p[2],
        cy: p[6],
        k1: 0.0,
        k2: 0.0,
        p1: 0.0,
        p2: 0.0,
    };
    if !(camera.fx > 0.0 && camera.fy > 0.0) {
        return Err(format!(
            "CameraInfo P has fx={} fy={}; is the camera calibrated?",
            camera.fx, camera.fy
        ));
    }
    Ok(camera)
}

/// Stereo baseline in metres from the right `CameraInfo` projection matrix,
/// whose `P[3]` is `-fx * baseline` (REP 104 / `sensor_msgs/CameraInfo`).
pub fn stereo_baseline_m(right_p: &[f64; 12]) -> Result<f64, String> {
    let baseline = -right_p[3] / right_p[0];
    if !(baseline.is_finite() && baseline > 0.0) {
        return Err(format!(
            "right CameraInfo P gives baseline {baseline} m; are the views rectified as a pair?"
        ));
    }
    Ok(baseline)
}

/// A grayscale copy of a `sensor_msgs/Image`. `step` is the row length in
/// bytes, which may include padding.
pub fn gray_image(
    encoding: &str,
    width: u32,
    height: u32,
    step: u32,
    data: &[u8],
) -> Result<Image<u8, 1>, String> {
    let size = ImageSize {
        width: width as usize,
        height: height as usize,
    };
    let channels = match encoding {
        "mono8" | "8UC1" => 1,
        "rgb8" | "bgr8" => 3,
        other => return Err(format!("unsupported image encoding `{other}`")),
    };
    let pixels = packed_rows(data, size, channels, step as usize)?;
    if channels == 1 {
        return Image::new(size, pixels).map_err(|e| e.to_string());
    }
    let mut rgb = Image::<u8, 3>::new(size, pixels).map_err(|e| e.to_string())?;
    if encoding == "bgr8" {
        let bgr = rgb;
        rgb = Image::from_size_val(size, 0).map_err(|e| e.to_string())?;
        bgr_from_rgb(&bgr, &mut rgb).map_err(|e| e.to_string())?;
    }
    let mut gray = Image::from_size_val(size, 0).map_err(|e| e.to_string())?;
    gray_from_rgb_u8(&rgb, &mut gray).map_err(|e| e.to_string())?;
    Ok(gray)
}

fn packed_rows(
    data: &[u8],
    size: ImageSize,
    channels: usize,
    step: usize,
) -> Result<Vec<u8>, String> {
    let row = size.width * channels;
    if step < row || data.len() < step * size.height.saturating_sub(1) + row {
        return Err(format!(
            "image of {}x{} with step {step} does not fit its {} data bytes",
            size.width,
            size.height,
            data.len()
        ));
    }
    if step == row {
        return Ok(data[..row * size.height].to_vec());
    }
    Ok(data
        .chunks(step)
        .take(size.height)
        .flat_map(|r| &r[..row])
        .copied()
        .collect())
}

impl Extrinsic {
    /// The transform as a pose; the run file checks the value counts.
    pub fn pose(&self) -> Pose3d {
        let r = &self.rotation;
        let rotation = Mat3F64::from_cols(
            Vec3F64::new(r[0], r[3], r[6]),
            Vec3F64::new(r[1], r[4], r[7]),
            Vec3F64::new(r[2], r[5], r[8]),
        );
        let t = &self.translation;
        Pose3d::from_rt(rotation, Vec3F64::new(t[0], t[1], t[2]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LEFT_P: [f64; 12] = [
        435.2, 0.0, 367.4, 0.0, //
        0.0, 435.2, 252.2, 0.0, //
        0.0, 0.0, 1.0, 0.0,
    ];

    #[test]
    fn the_projection_matrix_is_the_rectified_camera() {
        let camera = camera_from_projection(&LEFT_P).unwrap();
        assert_eq!(
            (camera.fx, camera.fy, camera.cx, camera.cy),
            (435.2, 435.2, 367.4, 252.2)
        );
        assert_eq!(
            (camera.k1, camera.k2, camera.p1, camera.p2),
            (0.0, 0.0, 0.0, 0.0)
        );
        assert!(camera_from_projection(&[0.0; 12]).is_err());
    }

    #[test]
    fn the_baseline_comes_from_the_right_projection() {
        let mut right = LEFT_P;
        right[3] = -435.2 * 0.11;
        assert!((stereo_baseline_m(&right).unwrap() - 0.11).abs() < 1e-12);
        assert!(stereo_baseline_m(&LEFT_P).is_err());
    }

    #[test]
    fn padded_mono_rows_are_packed() {
        let data = [1, 2, 0, 3, 4, 0];
        let image = gray_image("mono8", 2, 2, 3, &data).unwrap();
        assert_eq!(image.as_slice(), &[1, 2, 3, 4]);
    }

    #[test]
    fn color_is_converted_to_gray_in_either_channel_order() {
        let rgb = gray_image("rgb8", 1, 1, 3, &[255, 0, 0]).unwrap();
        let bgr = gray_image("bgr8", 1, 1, 3, &[0, 0, 255]).unwrap();
        assert_eq!(rgb.as_slice(), bgr.as_slice());
        assert!(rgb.as_slice()[0] > 0);
    }

    #[test]
    fn bad_images_are_rejected() {
        assert!(gray_image("mono16", 1, 1, 2, &[0, 0]).is_err());
        assert!(gray_image("mono8", 2, 2, 2, &[0; 3]).is_err());
        assert!(gray_image("mono8", 2, 1, 1, &[0; 2]).is_err());
    }

    #[test]
    fn the_extrinsic_maps_camera_points_into_the_imu_frame() {
        let extrinsic = Extrinsic {
            rotation: vec![0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            translation: vec![0.1, 0.2, 0.3],
        };
        let p = extrinsic
            .pose()
            .transform_point(&Vec3F64::new(1.0, 0.0, 0.0));
        assert!((p - Vec3F64::new(0.1, 1.2, 0.3)).length() < 1e-12);
    }
}
