//! A stereo-inertial head walk with exact pixels, for the window's tests and
//! its bench: `HeadWalk` turned into the window's z-up world, a stereo pair
//! looking along the body's +Z, and points 2-8 m around the path.

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, Vec3F64};
use kornia_sensors::imu::{ImuBias, ImuCalib, ImuMeasurement};

use super::state::FrameId;
use super::window::{Observation, StereoView};
use crate::mapping::synthetic_walk::{BodyState, Gauss, HEAD_WALK_IMU_DT, HeadWalk};
use crate::tracking::pose_estimation::pose_inertial::FrameNavState;

pub(crate) const FRAME_DT: f64 = 0.05;
pub(crate) const WIDTH: f64 = 752.0;
pub(crate) const HEIGHT: f64 = 480.0;
pub(crate) const BASELINE_M: f64 = 0.11;
/// `HeadWalk`'s gravity, on +Y.
const WALK_GRAVITY: [f64; 3] = [0.0, 9.81, 0.0];
const POINTS: usize = 2000;
const BORDER_PX: f64 = 10.0;

pub(crate) fn camera() -> PinholeCamera {
    PinholeCamera {
        fx: 435.0,
        fy: 435.0,
        cx: 376.0,
        cy: 240.0,
        k1: 0.0,
        k2: 0.0,
        p1: 0.0,
        p2: 0.0,
    }
}

/// The left and right views of the rectified pair.
pub(crate) fn stereo_views() -> [StereoView; 2] {
    [
        StereoView {
            camera: camera(),
            camera_from_primary: Pose3d::IDENTITY,
        },
        StereoView {
            camera: camera(),
            camera_from_primary: Pose3d::new(
                Mat3F64::IDENTITY,
                Vec3F64::new(-BASELINE_M, 0.0, 0.0),
            ),
        },
    ]
}

/// `T_bc`: the camera a few centimetres off the body origin, axes aligned.
pub(crate) fn camera_to_body() -> Pose3d {
    Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.03, -0.01, 0.02))
}

pub(crate) fn gravity() -> Vec3F64 {
    Vec3F64::new(0.0, 0.0, -9.81)
}

/// `HeadWalk`'s world (gravity on +Y) into the window's (gravity on −Z):
/// `(x, y, z) → (x, z, −y)`.
fn to_z_up() -> Mat3F64 {
    Mat3F64::from_cols(
        Vec3F64::new(1.0, 0.0, 0.0),
        Vec3F64::new(0.0, 0.0, -1.0),
        Vec3F64::new(0.0, 1.0, 0.0),
    )
}

pub(crate) struct StereoWalk {
    pub states: Vec<BodyState>,
    pub readings: Vec<ImuMeasurement>,
    pub points: Vec<Vec3F64>,
    pub bias: ImuBias,
}

impl StereoWalk {
    /// `seconds` of the walk at 1 kHz with constant biases; readings with
    /// white noise at `noise` when given.
    pub(crate) fn new(seconds: f64, bias: ImuBias, noise: Option<(ImuCalib, u64)>) -> Self {
        let walk = HeadWalk {
            origin: Vec3F64::ZERO,
            accel_bias: bias.accel,
            gyro_bias: bias.gyro,
        };
        let steps = (seconds / HEAD_WALK_IMU_DT).round() as usize;
        let (states, mut readings) = walk.simulate(steps, Vec3F64::from_array(WALK_GRAVITY));
        let fix = to_z_up();
        let states = states
            .into_iter()
            .map(|s| BodyState {
                r_wb: fix * s.r_wb,
                position: fix * s.position,
                velocity: fix * s.velocity,
            })
            .collect();
        if let Some((calib, seed)) = noise {
            let mut gauss = Gauss(seed);
            let scale = 1.0 / HEAD_WALK_IMU_DT.sqrt();
            for r in &mut readings {
                r.gyro += gauss.vec3(calib.gyro_noise * scale);
                r.accel += gauss.vec3(calib.accel_noise * scale);
            }
        }
        Self {
            states,
            readings,
            points: points(),
            bias,
        }
    }

    pub(crate) fn frames(&self) -> usize {
        ((self.states.len() - 1) as f64 * HEAD_WALK_IMU_DT / FRAME_DT).floor() as usize + 1
    }

    pub(crate) fn time(frame: usize) -> f64 {
        frame as f64 * FRAME_DT
    }

    pub(crate) fn id(frame: usize) -> FrameId {
        (Self::time(frame) * 1e9).round() as FrameId
    }

    fn body(&self, frame: usize) -> &BodyState {
        let index = (Self::time(frame) / HEAD_WALK_IMU_DT).round() as usize;
        &self.states[index.min(self.states.len() - 1)]
    }

    /// The true camera-0 state at `frame`, biases included.
    pub(crate) fn truth(&self, frame: usize) -> FrameNavState {
        let body = self.body(frame);
        let pose_wb = Pose3d::new(body.r_wb, body.position);
        FrameNavState {
            pose_cw: camera_to_body().inverse().compose(&pose_wb.inverse()),
            velocity: body.velocity,
            bias: self.bias,
        }
    }

    /// Readings stamped in `(t0, t1]`.
    pub(crate) fn samples(&self, t0: f64, t1: f64) -> &[ImuMeasurement] {
        let start = self.readings.partition_point(|r| r.timestamp <= t0 + 1e-9);
        let end = self.readings.partition_point(|r| r.timestamp <= t1 + 1e-9);
        &self.readings[start..end]
    }

    /// The first reading at or after `t`.
    pub(crate) fn accel_at(&self, t: f64) -> Vec3F64 {
        let k = self.readings.partition_point(|r| r.timestamp < t - 1e-9);
        self.readings[k.min(self.readings.len() - 1)].accel
    }

    /// Every point the left camera sees inside the border, and its right
    /// pixel when the right camera sees it too; sorted by (camera, id).
    pub(crate) fn observations(
        &self,
        frame: usize,
        noise: Option<(&mut Gauss, f64)>,
    ) -> Vec<Observation> {
        let truth = self.truth(frame);
        let views = stereo_views();
        let mut noise = noise;
        let mut out: Vec<Observation> = Vec::new();
        let mut right: Vec<Observation> = Vec::new();
        for (id, point) in self.points.iter().enumerate() {
            let Some(left) = project(&views[0], &truth.pose_cw, point) else {
                continue;
            };
            let mut push = |list: &mut Vec<Observation>, camera: u8, pixel: [f64; 2]| {
                let mut pixel = pixel;
                if let Some((gauss, sigma)) = noise.as_mut() {
                    pixel[0] += gauss.normal() * *sigma;
                    pixel[1] += gauss.normal() * *sigma;
                }
                list.push(Observation {
                    landmark: id as u64,
                    camera,
                    pixel,
                });
            };
            push(&mut out, 0, left);
            if let Some(pixel) = project(&views[1], &truth.pose_cw, point) {
                push(&mut right, 1, pixel);
            }
        }
        out.extend(right);
        out
    }
}

fn project(view: &StereoView, pose_cw: &Pose3d, point: &Vec3F64) -> Option<[f64; 2]> {
    let p = view
        .camera_from_primary
        .compose(pose_cw)
        .transform_point(point);
    if p.z < 0.5 {
        return None;
    }
    let c = &view.camera;
    let (u, v) = (c.fx * p.x / p.z + c.cx, c.fy * p.y / p.z + c.cy);
    (u >= BORDER_PX && v >= BORDER_PX && u <= WIDTH - BORDER_PX && v <= HEIGHT - BORDER_PX)
        .then_some([u, v])
}

/// Points 2-8 m off the walk's 6 m circle, inside and outside it, within
/// 1.5 m of head height.
fn points() -> Vec<Vec3F64> {
    let mut gauss = Gauss(0x9e37_79b9_7f4a_7c15);
    let fix = to_z_up();
    // The circle's centre in HeadWalk's world is (6, 0, 0).
    let centre = fix * Vec3F64::new(6.0, 0.0, 0.0);
    (0..POINTS)
        .map(|k| {
            let angle = gauss.uniform() * std::f64::consts::TAU;
            let off = 2.0 + 6.0 * gauss.uniform();
            let radius = if k % 3 == 0 {
                (6.0 - off).max(0.3)
            } else {
                6.0 + off
            };
            let height = 3.0 * gauss.uniform() - 1.5;
            centre + Vec3F64::new(radius * angle.cos(), radius * angle.sin(), height)
        })
        .collect()
}
