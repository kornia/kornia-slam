//! Publishes the tracked camera pose to ROS 2 as `nav_msgs/Odometry` on
//! `kornia_slam/odometry` and as the TF transform `map` → `kornia_slam_camera`.
//!
//! `map` is the SLAM world frame, which inertial initialization may rotate and
//! rescale. `kornia_slam_camera` is the (left) camera's optical frame: x right,
//! y down, z forward. Link both into an existing TF tree with static transforms.

use kornia_3d::pose::Pose3d;
use kornia_algebra::QuatF64;
use rclrs::{
    Context, CreateBasicExecutor, InitOptions, IntoPrimitiveOptions, Publisher, RclrsError,
};
use ros_env::builtin_interfaces::msg::Time;
use ros_env::geometry_msgs::msg::{Point, Pose, Quaternion, Transform, TransformStamped, Vector3};
use ros_env::nav_msgs::msg::Odometry;
use ros_env::std_msgs::msg::Header;
use ros_env::tf2_msgs::msg::TFMessage;

pub const WORLD_FRAME: &str = "map";
pub const CAMERA_FRAME: &str = "kornia_slam_camera";

pub struct PosePublisher {
    odometry: Publisher<Odometry>,
    tf: Publisher<TFMessage>,
    _node: rclrs::Node,
}

impl PosePublisher {
    pub fn new() -> Result<Self, RclrsError> {
        let context = Context::new(std::iter::empty::<String>(), InitOptions::default())?;
        let executor = context.create_basic_executor();
        let node = executor.create_node("kornia_slam_output")?;
        Ok(Self {
            odometry: node.create_publisher("kornia_slam/odometry")?,
            tf: node.create_publisher("/tf".keep_last(100))?,
            _node: node,
        })
    }

    /// Publishes the camera pose of the frame captured at `stamp`.
    pub fn publish(&self, stamp: Time, pose_world_to_cam: &Pose3d) -> Result<(), RclrsError> {
        let header = Header {
            stamp,
            frame_id: WORLD_FRAME.into(),
        };
        let (t, q) = camera_in_world(pose_world_to_cam);
        let rotation = Quaternion {
            x: q[0],
            y: q[1],
            z: q[2],
            w: q[3],
        };
        let mut odometry = Odometry {
            header: header.clone(),
            child_frame_id: CAMERA_FRAME.into(),
            ..Default::default()
        };
        odometry.pose.pose = Pose {
            position: Point {
                x: t[0],
                y: t[1],
                z: t[2],
            },
            orientation: rotation.clone(),
        };
        self.odometry.publish(odometry)?;
        self.tf.publish(TFMessage {
            transforms: vec![TransformStamped {
                header,
                child_frame_id: CAMERA_FRAME.into(),
                transform: Transform {
                    translation: Vector3 {
                        x: t[0],
                        y: t[1],
                        z: t[2],
                    },
                    rotation,
                },
            }],
        })
    }
}

/// A frame's stamp: exact when the source kept it in nanoseconds, otherwise
/// rounded from `timestamp_sec`, which holds stamps since 1970 only to about
/// 0.2 µs.
pub fn frame_stamp(stamp_ns: Option<i64>, timestamp_sec: f64) -> Time {
    let ns = stamp_ns.unwrap_or_else(|| {
        let sec = timestamp_sec.floor();
        sec as i64 * 1_000_000_000 + ((timestamp_sec - sec) * 1e9).round() as i64
    });
    Time {
        sec: ns.div_euclid(1_000_000_000) as i32,
        nanosec: ns.rem_euclid(1_000_000_000) as u32,
    }
}

/// Camera position and orientation (`[x, y, z, w]`) in the world frame.
fn camera_in_world(pose_world_to_cam: &Pose3d) -> ([f64; 3], [f64; 4]) {
    let cam_to_world = pose_world_to_cam.inverse();
    let t = cam_to_world.translation;
    let q = QuatF64::from_mat3(&cam_to_world.rotation).normalize();
    ([t.x, t.y, t.z], q.to_array())
}

#[cfg(test)]
mod tests {
    use kornia_algebra::{Mat3F64, Vec3F64};

    use super::*;

    #[test]
    fn an_exact_stamp_is_published_unchanged() {
        let ns = 1_403_715_273_262_142_976;
        let time = frame_stamp(Some(ns), ns as f64 * 1e-9);
        assert_eq!((time.sec, time.nanosec), (1_403_715_273, 262_142_976));
    }

    #[test]
    fn a_stamp_in_seconds_rounds_to_the_nanosecond() {
        let time = frame_stamp(None, 1_403_715_273.25);
        assert_eq!((time.sec, time.nanosec), (1_403_715_273, 250_000_000));
        let time = frame_stamp(None, 4.999_999_999_9);
        assert_eq!((time.sec, time.nanosec), (5, 0));
    }

    #[test]
    fn the_camera_pose_is_its_world_position_and_orientation() {
        // Camera at (1, 2, 3), turned 90° about the world z axis.
        let cam_to_world = Pose3d::from_rt(
            Mat3F64::from_cols(
                Vec3F64::new(0.0, 1.0, 0.0),
                Vec3F64::new(-1.0, 0.0, 0.0),
                Vec3F64::new(0.0, 0.0, 1.0),
            ),
            Vec3F64::new(1.0, 2.0, 3.0),
        );
        let (t, q) = camera_in_world(&cam_to_world.inverse());
        assert!(
            t.iter()
                .zip([1.0, 2.0, 3.0])
                .all(|(a, b)| (a - b).abs() < 1e-12)
        );
        let half = std::f64::consts::FRAC_1_SQRT_2;
        assert!(
            q.iter()
                .zip([0.0, 0.0, half, half])
                .all(|(a, b)| (a - b).abs() < 1e-12)
        );
    }
}
