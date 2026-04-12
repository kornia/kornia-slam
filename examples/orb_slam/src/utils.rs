//! Minimal Rerun visualization helpers for the ORB-SLAM example.

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::Mat3F64;
use kornia_image::{Image, ImageSize};
use kornia_sensors::imu::PreintegratedImu;
use kornia_slam::map::MapPoint;
use kornia_tensor::CpuAllocator;

use crate::datasets::euroc::ImuSample;

const CAMERA_IMAGE_PLANE_DISTANCE: f32 = 0.15;

#[derive(Debug, Clone, Copy, PartialEq)]
struct CameraVisualizationSpec {
    translation: [f32; 3],
    rotation_wxyz: [f32; 4],
    focal_length: [f32; 2],
    resolution: [f32; 2],
    principal_point: [f32; 2],
    image_plane_distance: f32,
}

pub fn log_frame_to_rerun(
    rec: &rerun::RecordingStream,
    image: &Image<u8, 1, CpuAllocator>,
    keypoints_xy: &[[f32; 2]],
) {
    rec.log(
        "image",
        &rerun::Image::from_elements(image.as_slice(), image.size().into(), rerun::ColorModel::L),
    )
    .ok();
    rec.log(
        "image/keypoints",
        &rerun::Points2D::new(keypoints_xy_to_rerun_points(keypoints_xy)).with_radii([1.5]),
    )
    .ok();
}

/// Convert feature keypoints to the point format expected by Rerun.
pub fn keypoints_xy_to_rerun_points(keypoints_xy: &[[f32; 2]]) -> Vec<[f32; 2]> {
    keypoints_xy.to_vec()
}

/// Log the estimated trajectory as a growing 3D line strip.
pub fn log_trajectory_to_rerun(rec: &rerun::RecordingStream, trajectory: &[[f32; 3]]) {
    if trajectory.len() >= 2 {
        rec.log("world/trajectory", &rerun::LineStrips3D::new([trajectory]))
            .ok();
    }
}

/// Log the current camera frustum in world coordinates.
pub fn log_camera_to_rerun(
    rec: &rerun::RecordingStream,
    pose_world_to_cam: &Pose3d,
    camera: &PinholeCamera,
    image_size: ImageSize,
) {
    let spec = camera_visualization_spec(pose_world_to_cam, camera, image_size);

    rec.log(
        "world/camera",
        &rerun::Transform3D::from_translation_rotation(
            spec.translation,
            rerun::Quaternion::from_wxyz(spec.rotation_wxyz),
        ),
    )
    .ok();
    rec.log(
        "world/camera/pinhole",
        &rerun::Pinhole::from_focal_length_and_resolution(spec.focal_length, spec.resolution)
            .with_principal_point(spec.principal_point)
            .with_image_plane_distance(spec.image_plane_distance),
    )
    .ok();
}

/// Log active map points as a 3D point cloud.
pub fn log_map_points_to_rerun(rec: &rerun::RecordingStream, map_points: &[MapPoint]) {
    let (positions, colors): (Vec<_>, Vec<_>) = map_points
        .iter()
        .filter(|mp| !mp.culled)
        .map(|mp| {
            (
                rerun::Position3D::new(
                    mp.position.x as f32,
                    mp.position.y as f32,
                    mp.position.z as f32,
                ),
                rerun::Color::from_rgb(mp.color[0], mp.color[1], mp.color[2]),
            )
        })
        .unzip();
    if !positions.is_empty() {
        rec.log(
            "world/map_points",
            &rerun::Points3D::new(positions)
                .with_colors(colors)
                .with_radii([rerun::Radius::new_scene_units(0.01)]),
        )
        .ok();
    }
}

/// Log a batch of IMU samples as scalar time series (gyro + accel, one point per axis).
pub fn log_imu_to_rerun(rec: &rerun::RecordingStream, samples: &[ImuSample]) {
    for s in samples {
        rec.set_time(
            "imu_time",
            rerun::TimeCell::from_duration_nanos((s.timestamp_sec * 1e9) as i64),
        );
        rec.log("imu/gyro/x", &rerun::Scalars::new([s.gyro[0]])).ok();
        rec.log("imu/gyro/y", &rerun::Scalars::new([s.gyro[1]])).ok();
        rec.log("imu/gyro/z", &rerun::Scalars::new([s.gyro[2]])).ok();
        rec.log("imu/accel/x", &rerun::Scalars::new([s.accel[0]])).ok();
        rec.log("imu/accel/y", &rerun::Scalars::new([s.accel[1]])).ok();
        rec.log("imu/accel/z", &rerun::Scalars::new([s.accel[2]])).ok();
    }
}

/// Log IMU preintegration covariances as 2D heatmap tensors.
///
/// Two tensors are logged:
///   - `imu/cov_nav`: 9×9 navigation covariance (tangent-space blocks: rot, vel, pos)
///   - `imu/cov_bias`: 6×6 bias covariance (blocks: bias_gyro, bias_accel)
///
/// Raw covariance entries span many orders of magnitude, so we also log a
/// signed-log version (`..._log`) which makes off-diagonal structure visible
/// in the Rerun viewer's default linear colormap.
pub fn log_imu_covariance_to_rerun(rec: &rerun::RecordingStream, pre: &PreintegratedImu) {
    // Rerun tensors are row-major; our covariances are stored column-major.
    // Since covariance matrices are symmetric the transpose equals itself,
    // so the flat buffer can be used directly without reordering.
    log_matrix_heatmap(rec, "imu/cov_nav", &pre.covariance, 9);
    log_matrix_heatmap(rec, "imu/cov_bias", &pre.bias_covariance, 6);
}

fn log_matrix_heatmap(rec: &rerun::RecordingStream, entity: &str, data: &[f64], n: usize) {
    let shape = vec![n as u64, n as u64];

    // Raw values.
    let raw = rerun::datatypes::TensorData::new(
        shape.clone(),
        rerun::datatypes::TensorBuffer::F64(data.to_vec().into()),
    );
    rec.log(entity, &rerun::Tensor::new(raw).with_dim_names(["row", "col"]))
        .ok();

    // Signed-log version: sign(x) * log10(1 + |x|/eps) — keeps sign but
    // compresses the dynamic range so off-diagonal structure is visible.
    const EPS: f64 = 1e-12;
    let slog: Vec<f64> = data
        .iter()
        .map(|&x| x.signum() * (1.0 + x.abs() / EPS).log10())
        .collect();
    let log_data = rerun::datatypes::TensorData::new(
        shape,
        rerun::datatypes::TensorBuffer::F64(slog.into()),
    );
    rec.log(
        format!("{entity}_log"),
        &rerun::Tensor::new(log_data).with_dim_names(["row", "col"]),
    )
    .ok();
}

/// Convert a world-to-camera pose into a camera-to-world trajectory point.
pub fn trajectory_point_from_pose(pose_world_to_cam: &Pose3d) -> [f32; 3] {
    let cam_to_world = pose_world_to_cam.inverse();
    [
        cam_to_world.translation.x as f32,
        cam_to_world.translation.y as f32,
        cam_to_world.translation.z as f32,
    ]
}

fn camera_visualization_spec(
    pose_world_to_cam: &Pose3d,
    camera: &PinholeCamera,
    image_size: ImageSize,
) -> CameraVisualizationSpec {
    let cam_to_world = pose_world_to_cam.inverse();
    let t = cam_to_world.translation;
    let (qx, qy, qz, qw) = quat_xyzw_from_matrix(&cam_to_world.rotation);

    CameraVisualizationSpec {
        translation: [t.x as f32, t.y as f32, t.z as f32],
        rotation_wxyz: [qw as f32, qx as f32, qy as f32, qz as f32],
        focal_length: [camera.fx as f32, camera.fy as f32],
        resolution: [image_size.width as f32, image_size.height as f32],
        principal_point: [camera.cx as f32, camera.cy as f32],
        image_plane_distance: CAMERA_IMAGE_PLANE_DISTANCE,
    }
}

fn quat_xyzw_from_matrix(r: &Mat3F64) -> (f64, f64, f64, f64) {
    let trace = r.x_axis.x + r.y_axis.y + r.z_axis.z;

    if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        let w = 0.25 * s;
        let x = (r.y_axis.z - r.z_axis.y) / s;
        let y = (r.z_axis.x - r.x_axis.z) / s;
        let z = (r.x_axis.y - r.y_axis.x) / s;
        (x, y, z, w)
    } else if r.x_axis.x > r.y_axis.y && r.x_axis.x > r.z_axis.z {
        let s = (1.0 + r.x_axis.x - r.y_axis.y - r.z_axis.z).sqrt() * 2.0;
        let w = (r.y_axis.z - r.z_axis.y) / s;
        let x = 0.25 * s;
        let y = (r.x_axis.y + r.y_axis.x) / s;
        let z = (r.z_axis.x + r.x_axis.z) / s;
        (x, y, z, w)
    } else if r.y_axis.y > r.z_axis.z {
        let s = (1.0 + r.y_axis.y - r.x_axis.x - r.z_axis.z).sqrt() * 2.0;
        let w = (r.z_axis.x - r.x_axis.z) / s;
        let x = (r.x_axis.y + r.y_axis.x) / s;
        let y = 0.25 * s;
        let z = (r.y_axis.z + r.z_axis.y) / s;
        (x, y, z, w)
    } else {
        let s = (1.0 + r.z_axis.z - r.x_axis.x - r.y_axis.y).sqrt() * 2.0;
        let w = (r.x_axis.y - r.y_axis.x) / s;
        let x = (r.z_axis.x + r.x_axis.z) / s;
        let y = (r.y_axis.z + r.z_axis.y) / s;
        let z = 0.25 * s;
        (x, y, z, w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_algebra::Vec3F64;

    #[test]
    fn keypoints_xy_to_rerun_points_preserves_xy_order() {
        let rerun_points = keypoints_xy_to_rerun_points(&[[10.0, 20.0], [30.5, 40.5]]);
        assert_eq!(rerun_points, vec![[10.0, 20.0], [30.5, 40.5]]);
    }

    #[test]
    fn trajectory_point_from_pose_uses_camera_to_world_translation() {
        let pose_world_to_cam = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(1.0, -2.0, 3.0));
        assert_eq!(
            trajectory_point_from_pose(&pose_world_to_cam),
            [-1.0, 2.0, -3.0]
        );
    }

    #[test]
    fn camera_visualization_spec_uses_camera_to_world_pose() {
        let pose_world_to_cam = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(1.0, -2.0, 3.0));
        let camera = PinholeCamera {
            fx: 458.654,
            fy: 457.296,
            cx: 367.215,
            cy: 248.375,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        };
        let image_size = ImageSize {
            width: 752,
            height: 480,
        };
        let spec = camera_visualization_spec(&pose_world_to_cam, &camera, image_size);

        assert_eq!(spec.translation, [-1.0, 2.0, -3.0]);
        assert_eq!(spec.rotation_wxyz, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(spec.focal_length, [458.654, 457.296]);
        assert_eq!(spec.resolution, [752.0, 480.0]);
        assert_eq!(spec.principal_point, [367.215, 248.375]);
        assert_eq!(spec.image_plane_distance, CAMERA_IMAGE_PLANE_DISTANCE);
    }
}
