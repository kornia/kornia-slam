//! The offline solve's variables and visual measurements, built from a
//! session dump.
//!
//! States live in the body (IMU) frame, as ORB-SLAM3's `VertexPose` with a
//! per-camera `Tcb` (`G2oTypes.cc:25-68`): `T_cw = R_off · T_cb · T_bw`,
//! where `R_off` is the camera's calibration rotation offset for the
//! keyframe's time slice. A right perturbation `T_bw · Exp([ρ; ω])` is the
//! same right perturbation of every camera's `T_cw`, so each camera's
//! Jacobian reads the body state directly.

use std::borrow::Cow;
use std::cell::Cell;
use std::collections::HashMap;

use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, SE3F64, SMatF64, SO3F64, SVecF64, Vec3F64};
use kornia_sensors::imu::{ImuBias, ImuCalib};

use super::OfflineBaError;
use super::calibration::{Calibration, Layout};
use crate::mapping::SegmentId;
use crate::mapping::bundle_adjustment::STEREO_DEPTH_MIN_SIGMA;
use crate::mapping::session::{ImuSample, SESSION_DUMP_VERSION, SessionDump};
use crate::mapping::vi_ba_schur::{ViBaParams, visual_huber};
use crate::mapping::visual_information::VisualInformation;
use crate::system::StereoDepthSigma;
use crate::upstream::kornia_3d::refine_analytic::pinhole_point_jacobian;

/// Depth clamp of a projection, as the online VI-BA's `MIN_Z`, so a point
/// crossing the image plane mid-solve keeps a finite residual.
const MIN_DEPTH: f64 = 1e-3;
/// The gauge prior on each connected component's first keyframe: position
/// (m) and yaw about gravity (rad). Roll and pitch stay free; gravity
/// observes them.
const GAUGE_POSITION_SIGMA: f64 = 1e-3;
const GAUGE_YAW_SIGMA: f64 = 1e-4;

/// One camera of the rig: pinhole intrinsics of its undistorted pixels and
/// `T_cb` (`X_c = T_cb · X_b`).
pub(super) struct Camera {
    pub(super) fx: f64,
    pub(super) fy: f64,
    pub(super) cx: f64,
    pub(super) cy: f64,
    pub(super) camera_from_body: Pose3d,
}

/// A keyframe's 15-DoF state: `T_bw`, world velocity (m/s) and IMU bias.
#[derive(Debug, Clone, Copy)]
pub(super) struct KeyframeState {
    pub(super) pose_bw: Pose3d,
    pub(super) velocity: Vec3F64,
    pub(super) bias: ImuBias,
}

/// Every variable of the solve.
#[derive(Debug, Clone)]
pub(super) struct State {
    pub(super) keyframes: Vec<KeyframeState>,
    pub(super) landmarks: Vec<Vec3F64>,
    pub(super) calibration: Calibration,
}

impl State {
    /// `step` applied: poses on the right, `[ρ ω v b_g b_a]` per keyframe;
    /// everything else additively.
    pub(super) fn retracted(
        &self,
        keyframes: &[f64],
        landmarks: &[Vec3F64],
        calibration: &[f64],
    ) -> Self {
        let keyframes = self
            .keyframes
            .iter()
            .zip(keyframes.chunks_exact(15))
            .map(|(kf, d)| {
                let e = SE3F64::exp(
                    Vec3F64::new(d[0], d[1], d[2]),
                    Vec3F64::new(d[3], d[4], d[5]),
                );
                KeyframeState {
                    pose_bw: kf.pose_bw.compose(&Pose3d::new(e.r.matrix(), e.t)),
                    velocity: kf.velocity + Vec3F64::new(d[6], d[7], d[8]),
                    bias: ImuBias {
                        gyro: kf.bias.gyro + Vec3F64::new(d[9], d[10], d[11]),
                        accel: kf.bias.accel + Vec3F64::new(d[12], d[13], d[14]),
                    },
                }
            })
            .collect();
        Self {
            keyframes,
            landmarks: self
                .landmarks
                .iter()
                .zip(landmarks)
                .map(|(p, d)| *p + *d)
                .collect(),
            calibration: self.calibration.retracted(calibration),
        }
    }
}

/// One keypoint of a keyframe seen as a landmark.
pub(super) struct Observation {
    pub(super) keyframe: usize,
    pub(super) landmark: usize,
    pub(super) camera: usize,
    /// Undistorted pixel.
    pub(super) pixel: [f64; 2],
    /// `1 / σ_px` of its octave.
    pub(super) inv_sigma: f64,
    /// Primary-camera stereo depth (m) and its `1 / σ`.
    pub(super) depth: Option<(f64, f64)>,
    /// Cleared once an outlier stage drops it; it never comes back.
    pub(super) active: Cell<bool>,
}

/// Whitened, kernel-weighted rows of one observation: `u`, `v` and, with a
/// stereo depth, `z`. Rows it does not have stay zero.
pub(super) struct ObservationRows {
    pub(super) residual: SVecF64<3>,
    /// Columns `[ρ ω]` of the keyframe state.
    pub(super) j_pose: SMatF64<3, 6>,
    pub(super) j_point: SMatF64<3, 3>,
    /// The calibration columns it reads: three rotation-offset ones, then focal.
    j_calibration: [Option<(usize, SVecF64<3>)>; 4],
}

impl Default for ObservationRows {
    fn default() -> Self {
        Self {
            residual: SVecF64::zeros(),
            j_pose: SMatF64::zeros(),
            j_point: SMatF64::zeros(),
            j_calibration: [None; 4],
        }
    }
}

impl ObservationRows {
    pub(super) fn calibration(&self) -> impl Iterator<Item = (usize, &SVecF64<3>)> + '_ {
        self.j_calibration
            .iter()
            .flatten()
            .map(|(column, j)| (*column, j))
    }
}

/// A landmark projected into an observing camera.
struct Projection {
    /// After the rotation offset; `z` clamped away from 0, keeping its sign.
    point_cam: Vec3F64,
    /// Rotation of the full world-to-camera chain.
    rotation_cw: Mat3F64,
    /// Rotation-offset column and value `θ`.
    offset: Option<(usize, Vec3F64)>,
    focal: Option<usize>,
    focal_scale: f64,
    pixel: [f64; 2],
}

/// The first keyframe of a connected component and its starting pose, which
/// the gauge prior holds in position and yaw.
pub(super) struct GaugeAnchor {
    keyframe: usize,
    position: Vec3F64,
    rotation_bw: Mat3F64,
}

pub(super) struct Problem<'a> {
    pub(super) cameras: Vec<Camera>,
    /// `T_bc` of the primary camera.
    pub(super) body_from_primary: Pose3d,
    pub(super) noise: ImuCalib,
    pub(super) gravity: Vec3F64,
    /// Ordered by time.
    pub(super) imu: Cow<'a, [ImuSample]>,
    /// Dump position of each solved keyframe; the solve orders them by time.
    pub(super) dump_index: Vec<usize>,
    pub(super) timestamps: Vec<f64>,
    pub(super) segments: Vec<SegmentId>,
    /// Calibration time slice of each keyframe.
    pub(super) slices: Vec<usize>,
    pub(super) observations: Vec<Observation>,
    /// Observations of each landmark.
    pub(super) by_landmark: Vec<Vec<usize>>,
    huber_reproj: f64,
    huber_depth: f64,
}

impl<'a> Problem<'a> {
    /// The problem and its starting state. A keyframe the dump could not
    /// stamp (tracked before recording began) is left out of the solve.
    pub(super) fn new(
        session: &'a SessionDump,
        visual: VisualInformation,
        stereo_depth_sigma: StereoDepthSigma,
        layout: &Layout,
    ) -> Result<(Self, State), OfflineBaError> {
        if session.version != SESSION_DUMP_VERSION {
            return Err(OfflineBaError::VersionMismatch(session.version));
        }
        if session.imu.len() < 2 {
            return Err(OfflineBaError::NoImu);
        }
        let imu = if session.imu.windows(2).all(|w| w[0].t < w[1].t) {
            Cow::Borrowed(session.imu.as_slice())
        } else {
            let mut sorted = session.imu.clone();
            sorted.sort_by(|a, b| a.t.total_cmp(&b.t));
            sorted.dedup_by(|a, b| a.t == b.t);
            Cow::Owned(sorted)
        };
        let body_from_primary = session.rig.camera_to_body.to_pose();
        let primary_from_body = body_from_primary.inverse();
        let cameras: Vec<Camera> = session
            .rig
            .cameras
            .iter()
            .map(|c| Camera {
                fx: c.fx,
                fy: c.fy,
                cx: c.cx,
                cy: c.cy,
                camera_from_body: c
                    .primary_from_camera
                    .to_pose()
                    .inverse()
                    .compose(&primary_from_body),
            })
            .collect();
        let [gyro_noise, accel_noise, gyro_bias_noise, accel_bias_noise] = session.rig.imu_noise;

        let mut dump_index: Vec<usize> = (0..session.keyframes.len())
            .filter(|&i| session.keyframes[i].timestamp_sec.is_finite())
            .collect();
        dump_index.sort_by(|&a, &b| {
            session.keyframes[a]
                .timestamp_sec
                .total_cmp(&session.keyframes[b].timestamp_sec)
        });
        let keyframes: Vec<_> = dump_index.iter().map(|&i| &session.keyframes[i]).collect();
        let timestamps: Vec<f64> = keyframes.iter().map(|kf| kf.timestamp_sec).collect();
        let slices = time_slices(&timestamps, layout.slices());

        let mut states: Vec<KeyframeState> = keyframes
            .iter()
            .map(|kf| KeyframeState {
                pose_bw: body_from_primary.compose(&kf.pose_cw.to_pose()),
                velocity: Vec3F64::from_array(kf.velocity_world),
                bias: ImuBias {
                    gyro: Vec3F64::from_array(kf.bias_gyro),
                    accel: Vec3F64::from_array(kf.bias_accel),
                },
            })
            .collect();
        seed_uninitialized_velocities(&mut states, &timestamps);

        let landmark_slot: HashMap<u32, usize> = session
            .landmarks
            .iter()
            .enumerate()
            .map(|(slot, l)| (l.idx, slot))
            .collect();
        let focal_baseline = session
            .rig
            .stereo_baseline_m
            .zip(cameras.first())
            .map(|(baseline, primary)| baseline * primary.fx);
        let mut observations = Vec::new();
        let mut by_landmark = vec![Vec::new(); session.landmarks.len()];
        for (keyframe, kf) in keyframes.iter().enumerate() {
            for obs in &kf.observations {
                let camera = usize::from(obs.camera);
                let Some(&landmark) = landmark_slot.get(&obs.landmark) else {
                    continue;
                };
                if camera >= cameras.len() {
                    continue;
                }
                let pixel_sigma = f64::from(visual.sigma_px(obs.octave));
                let depth = obs.depth.filter(|_| camera == 0).map(|z| {
                    let z = f64::from(z);
                    let sigma = depth_sigma(stereo_depth_sigma, z, pixel_sigma, focal_baseline);
                    (z, 1.0 / sigma)
                });
                by_landmark[landmark].push(observations.len());
                observations.push(Observation {
                    keyframe,
                    landmark,
                    camera,
                    pixel: [f64::from(obs.u), f64::from(obs.v)],
                    inv_sigma: 1.0 / pixel_sigma,
                    depth,
                    active: Cell::new(true),
                });
            }
        }
        // The online VI-BA's kernels, so both weight an observation alike.
        let online = ViBaParams::default();
        let state = State {
            keyframes: states,
            landmarks: session
                .landmarks
                .iter()
                .map(|l| Vec3F64::from_array(l.position_world))
                .collect(),
            calibration: Calibration::zero(layout),
        };
        let problem = Self {
            cameras,
            body_from_primary,
            noise: ImuCalib {
                gyro_noise,
                accel_noise,
                gyro_bias_noise,
                accel_bias_noise,
            },
            gravity: Vec3F64::from_array(session.gravity_world),
            imu,
            dump_index,
            timestamps,
            segments: keyframes.iter().map(|kf| kf.segment).collect(),
            slices,
            observations,
            by_landmark,
            huber_reproj: online.huber_reproj_px_sq,
            huber_depth: online.huber_depth_sigma_sq,
        };
        Ok((problem, state))
    }

    fn project(&self, obs: &Observation, state: &State, layout: &Layout) -> Projection {
        let camera = &self.cameras[obs.camera];
        let slice = self.slices[obs.keyframe];
        let pose_cw = camera
            .camera_from_body
            .compose(&state.keyframes[obs.keyframe].pose_bw);
        let offset = layout
            .rotation(obs.camera, slice)
            .map(|column| (column, state.calibration.vec3(column)));
        let r_off = offset.map_or(Mat3F64::IDENTITY, |(_, theta)| SO3F64::exp(theta).matrix());
        let mut point_cam = r_off * pose_cw.transform_point(&state.landmarks[obs.landmark]);
        if point_cam.z.abs() < MIN_DEPTH {
            point_cam.z = MIN_DEPTH.copysign(point_cam.z);
        }
        let focal = layout.focal(obs.camera, slice);
        let focal_scale = focal.map_or(1.0, |column| 1.0 + state.calibration.values[column]);
        let pixel = [
            focal_scale * camera.fx * point_cam.x / point_cam.z + camera.cx,
            focal_scale * camera.fy * point_cam.y / point_cam.z + camera.cy,
        ];
        Projection {
            point_cam,
            rotation_cw: r_off * pose_cw.rotation,
            offset,
            focal,
            focal_scale,
            pixel,
        }
    }

    /// Pixel residual (px) and depth residual (m) of an observation.
    fn residuals(obs: &Observation, projection: &Projection) -> ([f64; 2], Option<f64>) {
        (
            [
                projection.pixel[0] - obs.pixel[0],
                projection.pixel[1] - obs.pixel[1],
            ],
            obs.depth.map(|(z, _)| projection.point_cam.z - z),
        )
    }

    /// Kernel-weighted cost and the square roots of the kernel weights,
    /// reprojection then depth.
    fn kernel(&self, obs: &Observation, pixel: [f64; 2], depth: Option<f64>) -> (f64, f64, f64) {
        let chi2 = (pixel[0] * pixel[0] + pixel[1] * pixel[1]) * obs.inv_sigma * obs.inv_sigma;
        let (w, mut cost) = visual_huber(chi2, self.huber_reproj);
        let mut w_depth = 0.0;
        if let (Some(r), Some((_, inv_sigma))) = (depth, obs.depth) {
            let (w, c) = visual_huber(r * r * inv_sigma * inv_sigma, self.huber_depth);
            cost += c;
            w_depth = w;
        }
        (cost, w.sqrt(), w_depth.sqrt())
    }

    /// Drops every active observation whose reprojection error at `state`
    /// exceeds `threshold_px`, or that is behind its camera; returns how many.
    pub(super) fn drop_outliers(&self, state: &State, layout: &Layout, threshold_px: f64) -> usize {
        let mut dropped = 0;
        for (o, obs) in self.observations.iter().enumerate() {
            if !obs.active.get() {
                continue;
            }
            let error = self.pixel_error(o, state, layout);
            if error > threshold_px || error.is_nan() {
                obs.active.set(false);
                dropped += 1;
            }
        }
        dropped
    }

    /// `T_bw` of every dump keyframe, in dump order: solved, or the start for
    /// one left out of the solve.
    pub(super) fn dump_body_poses(&self, session: &SessionDump, state: &State) -> Vec<Pose3d> {
        let mut poses: Vec<Pose3d> = session
            .keyframes
            .iter()
            .map(|kf| self.body_from_primary.compose(&kf.pose_cw.to_pose()))
            .collect();
        for (k, &slot) in self.dump_index.iter().enumerate() {
            poses[slot] = state.keyframes[k].pose_bw;
        }
        poses
    }

    /// Reprojection error (px) at `state`; infinite behind the camera.
    pub(super) fn pixel_error(&self, o: usize, state: &State, layout: &Layout) -> f64 {
        let obs = &self.observations[o];
        let projection = self.project(obs, state, layout);
        if projection.point_cam.z < 0.0 {
            return f64::INFINITY;
        }
        let ([du, dv], _) = Self::residuals(obs, &projection);
        du.hypot(dv)
    }

    pub(super) fn observation_cost(&self, o: usize, state: &State, layout: &Layout) -> f64 {
        let obs = &self.observations[o];
        let projection = self.project(obs, state, layout);
        let (pixel, depth) = Self::residuals(obs, &projection);
        self.kernel(obs, pixel, depth).0
    }

    /// The observation's rows at `state` (ORB-SLAM3 `EdgeMono`/`EdgeStereo`
    /// through `Rcb`, `G2oTypes.cc:349`).
    pub(super) fn linearize_observation(
        &self,
        o: usize,
        state: &State,
        layout: &Layout,
    ) -> ObservationRows {
        let obs = &self.observations[o];
        let camera = &self.cameras[obs.camera];
        let projection = self.project(obs, state, layout);
        let (pixel, depth) = Self::residuals(obs, &projection);
        let (_, sw_pixel, sw_depth) = self.kernel(obs, pixel, depth);
        let mut rows = ObservationRows::default();
        if let Some((column, _)) = projection.offset {
            for d in 0..3 {
                rows.j_calibration[d] = Some((column + d, SVecF64::zeros()));
            }
        }
        if let Some(column) = projection.focal {
            rows.j_calibration[3] = Some((column, SVecF64::zeros()));
        }
        let p = projection.point_cam;
        let [du, dv] = pinhole_point_jacobian(
            projection.focal_scale * camera.fx,
            projection.focal_scale * camera.fy,
            p,
        );
        let s = sw_pixel * obs.inv_sigma;
        // ∂u/∂s = fx·x/z for the focal scale `fx (1 + s)`.
        let d_focal = [camera.fx * p.x / p.z, camera.fy * p.y / p.z];
        rows.set(
            0,
            du * s,
            pixel[0] * s,
            d_focal[0] * s,
            &projection,
            state.landmarks[obs.landmark],
        );
        rows.set(
            1,
            dv * s,
            pixel[1] * s,
            d_focal[1] * s,
            &projection,
            state.landmarks[obs.landmark],
        );
        if let (Some(r), Some((_, inv_sigma))) = (depth, obs.depth) {
            let s = sw_depth * inv_sigma;
            rows.set(
                2,
                Vec3F64::new(0.0, 0.0, s),
                r * s,
                0.0,
                &projection,
                state.landmarks[obs.landmark],
            );
        }
        rows
    }

    /// Per landmark, the keyframes of its active observations, ascending and
    /// distinct.
    pub(super) fn observers(&self) -> Vec<Vec<usize>> {
        self.by_landmark
            .iter()
            .map(|obs| {
                let mut keyframes: Vec<usize> = obs
                    .iter()
                    .map(|&o| &self.observations[o])
                    .filter(|o| o.active.get())
                    .map(|o| o.keyframe)
                    .collect();
                keyframes.sort_unstable();
                keyframes.dedup();
                keyframes
            })
            .collect()
    }

    /// The gauge anchor of every connected component: keyframes joined by an
    /// IMU edge or a shared landmark.
    pub(super) fn gauge_anchors(
        &self,
        imu_pairs: &[(usize, usize)],
        observers: &[Vec<usize>],
        state: &State,
    ) -> Vec<GaugeAnchor> {
        fn root(parent: &mut [usize], mut k: usize) -> usize {
            while parent[k] != k {
                parent[k] = parent[parent[k]];
                k = parent[k];
            }
            k
        }
        let mut parent: Vec<usize> = (0..state.keyframes.len()).collect();
        let pairs = imu_pairs.iter().copied().chain(
            observers
                .iter()
                .flat_map(|kfs| kfs.windows(2).map(|w| (w[0], w[1]))),
        );
        for (a, b) in pairs {
            let (ra, rb) = (root(&mut parent, a), root(&mut parent, b));
            // The earlier keyframe stays the root, so it anchors the component.
            parent[ra.max(rb)] = ra.min(rb);
        }
        (0..state.keyframes.len())
            .filter(|&k| root(&mut parent, k) == k)
            .map(|k| {
                let pose = &state.keyframes[k].pose_bw;
                GaugeAnchor {
                    keyframe: k,
                    position: pose.inverse().translation,
                    rotation_bw: pose.rotation,
                }
            })
            .collect()
    }

    /// The gauge prior rows of `anchor` at `state`: whitened residual and its
    /// derivative over the keyframe's `[ρ ω]`. Under `T_bw · Exp([ρ; ω])`
    /// the body position moves by `-ρ + [t_wb]× ω` and the attitude by a
    /// world-frame `ω`, whose yaw is its component along gravity.
    pub(super) fn gauge_rows(
        &self,
        anchor: &GaugeAnchor,
        state: &State,
    ) -> (usize, [(f64, [f64; 6]); 4]) {
        let pose = &state.keyframes[anchor.keyframe].pose_bw;
        let t = pose.inverse().translation;
        let dp = (t - anchor.position) * (1.0 / GAUGE_POSITION_SIGMA);
        let w = 1.0 / GAUGE_POSITION_SIGMA;
        let up = self.gravity.normalize();
        let phi = SO3F64::from_matrix(&(anchor.rotation_bw.transpose() * pose.rotation)).log();
        let wy = 1.0 / GAUGE_YAW_SIGMA;
        (
            anchor.keyframe,
            [
                (dp.x, [-w, 0.0, 0.0, 0.0, -t.z * w, t.y * w]),
                (dp.y, [0.0, -w, 0.0, t.z * w, 0.0, -t.x * w]),
                (dp.z, [0.0, 0.0, -w, -t.y * w, t.x * w, 0.0]),
                (
                    up.dot(phi) * wy,
                    [0.0, 0.0, 0.0, up.x * wy, up.y * wy, up.z * wy],
                ),
            ],
        )
    }

    pub(super) fn gauge_cost(&self, anchors: &[GaugeAnchor], state: &State) -> f64 {
        anchors
            .iter()
            .flat_map(|anchor| self.gauge_rows(anchor, state).1)
            .map(|(r, _)| 0.5 * r * r)
            .sum()
    }
}

impl ObservationRows {
    /// Row `row` from `a = ∂r/∂p_c` (whitened, weighted): `∂r/∂p_w = R_cwᵀ a`
    /// is also the `ρ` part, `p_w × R_cwᵀ a` the `ω` part, and the rotation
    /// offset reads `J_l(θ)ᵀ (p_c × a)` (`p_c = Exp(θ) q`).
    fn set(
        &mut self,
        row: usize,
        a: Vec3F64,
        residual: f64,
        d_focal: f64,
        projection: &Projection,
        point_world: Vec3F64,
    ) {
        let b = projection.rotation_cw.transpose() * a;
        let omega = point_world.cross(b);
        self.residual[row] = residual;
        for (d, (vb, vo)) in b.to_array().into_iter().zip(omega.to_array()).enumerate() {
            self.j_point[(row, d)] = vb;
            self.j_pose[(row, d)] = vb;
            self.j_pose[(row, d + 3)] = vo;
        }
        if let Some((_, theta)) = projection.offset {
            let g = SO3F64::left_jacobian(theta).transpose() * projection.point_cam.cross(a);
            for (d, v) in g.to_array().into_iter().enumerate() {
                if let Some((_, j)) = self.j_calibration[d].as_mut() {
                    j[row] = v;
                }
            }
        }
        if let Some((_, j)) = self.j_calibration[3].as_mut() {
            j[row] = d_focal;
        }
    }
}

/// Standard deviation (m) of a stereo depth `z`, as the online BA weights it
/// (`DepthSigma::sigma_at`), floored at [`STEREO_DEPTH_MIN_SIGMA`].
fn depth_sigma(
    model: StereoDepthSigma,
    z: f64,
    pixel_sigma: f64,
    focal_baseline: Option<f64>,
) -> f64 {
    let sigma = match model {
        StereoDepthSigma::RelativeDepth(fraction) => fraction * z,
        // Without a baseline there is no disparity to propagate.
        StereoDepthSigma::Disparity { px } => {
            focal_baseline.map_or(0.0, |fb| px / fb) * pixel_sigma * z * z
        }
    };
    sigma.max(f64::from(STEREO_DEPTH_MIN_SIGMA))
}

/// Slice of each keyframe when the recording is cut into `slices` equal spans.
fn time_slices(timestamps: &[f64], slices: usize) -> Vec<usize> {
    let (Some(&first), Some(&last)) = (timestamps.first(), timestamps.last()) else {
        return Vec::new();
    };
    let span = last - first;
    timestamps
        .iter()
        .map(|&t| {
            if span > 0.0 {
                (((t - first) / span * slices as f64) as usize).min(slices - 1)
            } else {
                0
            }
        })
        .collect()
}

/// A keyframe whose segment never initialised the IMU has zero velocity: it
/// starts at the central difference of its neighbours' positions, zero bias.
fn seed_uninitialized_velocities(states: &mut [KeyframeState], timestamps: &[f64]) {
    let positions: Vec<Vec3F64> = states
        .iter()
        .map(|s| s.pose_bw.inverse().translation)
        .collect();
    let last = states.len().saturating_sub(1);
    for (k, state) in states.iter_mut().enumerate() {
        if state.velocity != Vec3F64::ZERO {
            continue;
        }
        let (prev, next) = (k.saturating_sub(1), (k + 1).min(last));
        let dt = timestamps[next] - timestamps[prev];
        if next != prev && dt > 0.0 {
            state.velocity = (positions[next] - positions[prev]) * (1.0 / dt);
            state.bias = ImuBias::default();
        }
    }
}
