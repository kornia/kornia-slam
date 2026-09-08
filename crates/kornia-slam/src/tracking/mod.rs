//! Frame-to-map tracking, motion propagation, and tracking policies.

pub mod optical_flow;
mod policy;
pub mod pose_estimation;
mod state;

pub use policy::{KeyframePolicy, TrackingLossRecoveryPolicy};
pub use state::{SystemMode, SystemState, TrackingResult, TrackingStatus};

use crate::{Frame, map::Map};
use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;
use kornia_image::Image;
use kornia_sensors::imu::PreintegratedImu;
use optical_flow::{FlowSurvivor, KltTracker, MapKeypointMatch, TrackSet, snap_unique};
use pose_estimation::MapProjectionEstimator;
use pose_estimation::map_projection::{MapProjectionConfig, MapProjectionRejectReason};

/// Preintegration is prepared by the system, which owns bias and sample buffering.
pub(crate) struct InertialPrediction<'a> {
    pub preintegrated: &'a PreintegratedImu,
    pub t_bc: Option<Pose3d>,
    pub gravity_world: Vec3F64,
}

pub(crate) struct TrackingInput<'a> {
    pub frame: &'a Frame,
    pub previous_image: Option<&'a Image<u8, 1>>,
    pub current_image: &'a Image<u8, 1>,
    pub timestamp_sec: f64,
    pub inertial: Option<InertialPrediction<'a>>,
}

#[derive(Debug, PartialEq)]
pub(crate) enum RecoveryDecision {
    Continue,
    RestartBootstrap,
}

pub(crate) struct TrackingOutcome {
    pub candidate_pose: Pose3d,
    pub matches: Vec<(usize, usize)>,
    pub inliers: usize,
    pub rejection: Option<MapProjectionRejectReason>,
    pub recovery: RecoveryDecision,
}

/// Owns per-frame tracking; the system owns map publication and mode transitions.
pub(crate) struct Tracker {
    estimator: MapProjectionEstimator,
    klt_tracker: KltTracker,
    track_set: TrackSet,
    loss_policy: TrackingLossRecoveryPolicy,
    // One authoritative aggregate for now. Mode/bootstrap fields remain system
    // coordinated until the separate lifecycle ownership refactor.
    pub(crate) state: SystemState,
}

impl Tracker {
    pub(crate) fn new(
        config: MapProjectionConfig,
        loss_policy: TrackingLossRecoveryPolicy,
    ) -> Self {
        Self {
            estimator: MapProjectionEstimator::new(config),
            klt_tracker: KltTracker::default(),
            track_set: TrackSet::new(),
            loss_policy,
            state: SystemState::new(),
        }
    }

    pub(crate) fn track(
        &mut self,
        input: TrackingInput<'_>,
        map: &Map,
        camera: &PinholeCamera,
    ) -> TrackingOutcome {
        let TrackingInput {
            frame,
            previous_image,
            current_image,
            timestamp_sec,
            inertial,
        } = input;
        let pose_before = self.state.pose_world_to_cam;
        let candidate_pose = self.predict_pose(inertial);
        let currently_lost_for = self
            .state
            .lost_since_sec
            .map_or(0.0, |t0| timestamp_sec - t0);
        let search_scale = self.estimator.config().search_scale_for(currently_lost_for);

        let klt_survivors = if self.track_set.is_empty() {
            None
        } else {
            previous_image.and_then(|previous_image| {
                self.klt_tracker
                    .track(self.track_set.tracks(), previous_image, current_image)
                    .ok()
            })
        };
        let pre_seeded = klt_survivors
            .as_ref()
            .and_then(|survivors| {
                snap_unique(
                    &self.track_set,
                    survivors,
                    &frame.features.keypoints_xy,
                    3.0,
                )
                .ok()
            })
            .map(|matches| {
                matches
                    .into_iter()
                    .map(|matched| (matched.map_point_idx, matched.keypoint_idx))
                    .collect()
            });

        let result = self.estimator.estimate_pose(
            frame,
            &candidate_pose,
            &pose_before,
            map,
            camera,
            self.state.current_keyframe_idx,
            search_scale,
            pre_seeded,
        );

        let (status, matches, tracked_inliers, reject_reason) = match result {
            Ok(estimate) => {
                self.state.pose_world_to_cam = estimate.pose;

                // When IMU-initialized, velocity_world was already updated by IMU
                // preintegration (predict_pose_imu → pred_vel) before PnP ran; don't
                // overwrite it with a visual finite-difference, since at 30 fps the
                // inter-frame displacement is noise-dominated during low-translation
                // segments, which would collapse velocity to zero and permanently
                // freeze the IMU pose prediction.
                if !self.state.imu_initialized {
                    self.state.velocity = Some(Pose3d::between(&pose_before, &estimate.pose));
                }

                let track_matches: Vec<MapKeypointMatch> = estimate
                    .matches
                    .iter()
                    .map(|&(map_point_idx, keypoint_idx)| MapKeypointMatch {
                        map_point_idx,
                        keypoint_idx,
                    })
                    .collect();
                if self
                    .track_set
                    .reconcile_from_matches(&track_matches, &frame.features.keypoints_xy)
                    .is_err()
                {
                    self.track_set = TrackSet::new();
                }

                (
                    TrackingStatus::Tracked,
                    estimate.matches,
                    estimate.inliers,
                    None,
                )
            }
            Err(reason) => {
                carry_klt_survivors(&mut self.track_set, klt_survivors);

                // Carry the predicted pose forward instead of freezing at
                // pose_before. state.velocity_world was already advanced by
                // predict_pose_imu above regardless of visual outcome, so
                // anchoring the next frame's prediction on a stale position
                // would desync position/rotation from velocity: every
                // subsequent frame's candidate pose would drift further from
                // reality, making the projection search miss again and
                // compounding a single bad frame into a full tracking loss.
                self.state.pose_world_to_cam = candidate_pose;
                (TrackingStatus::Skipped, Vec::new(), 0, Some(reason))
            }
        };

        let recovery = if status == TrackingStatus::Skipped {
            self.recovery_after_rejection(timestamp_sec, map.keyframes().len())
        } else {
            RecoveryDecision::Continue
        };
        TrackingOutcome {
            candidate_pose,
            matches,
            inliers: tracked_inliers,
            rejection: reject_reason,
            recovery,
        }
    }

    fn predict_pose(&mut self, inertial: Option<InertialPrediction<'_>>) -> Pose3d {
        let pose_before = self.state.pose_world_to_cam;
        if self.state.imu_initialized
            && self.state.last_frame_timestamp_sec > 0.0
            && let Some(input) = inertial
            && input.preintegrated.dt > 0.0
        {
            let (pose, velocity) = predict_pose_imu(
                pose_before,
                self.state.velocity_world,
                input.gravity_world,
                input.preintegrated,
                input.t_bc,
            );
            self.state.velocity_world = velocity;
            return pose;
        }
        self.state
            .velocity
            .map(|v| v.compose(&pose_before))
            .unwrap_or(pose_before)
    }

    fn recovery_after_rejection(
        &mut self,
        timestamp_sec: f64,
        map_size: usize,
    ) -> RecoveryDecision {
        let policy = &self.loss_policy;
        let lost_since = *self.state.lost_since_sec.get_or_insert(timestamp_sec);
        let lost_for_sec = timestamp_sec - lost_since;
        let imu_confident = self.state.imu_initialized
            && self
                .state
                .imu_init_timestamp_sec
                .is_some_and(|t0| timestamp_sec - t0 >= policy.min_imu_confidence_sec);
        let map_established = map_size > policy.min_keyframes_for_grace;
        if !map_established || lost_for_sec >= policy.grace_period_sec(imu_confident) {
            RecoveryDecision::RestartBootstrap
        } else {
            RecoveryDecision::Continue
        }
    }

    pub(crate) fn restart_bootstrap(&mut self) {
        self.track_set = TrackSet::new();
        self.state.reset();
    }

    pub(crate) fn apply_local_ba_correction(
        &mut self,
        before: Pose3d,
        after: Pose3d,
        velocity: Vec3F64,
    ) {
        self.state.pose_world_to_cam =
            apply_reference_pose_correction(self.state.pose_world_to_cam, before, after);
        self.state.velocity_world = velocity;
    }

    pub(crate) fn apply_loop_correction(&mut self, before: Pose3d, after: Pose3d, world: Pose3d) {
        self.state.pose_world_to_cam =
            apply_reference_pose_correction(self.state.pose_world_to_cam, before, after);
        self.state.velocity_world = world.rotation * self.state.velocity_world;
        self.track_set = TrackSet::new();
    }

    pub(crate) fn adopt_inertial_alignment(&mut self, pose: Pose3d, velocity: Vec3F64) {
        self.state.pose_world_to_cam = pose;
        self.state.velocity_world = velocity;
        self.state.velocity = None;
        self.state.imu_initialized = true;
    }
}

/// Body-to-world pose `T_WB` for a world-to-camera pose, via
/// `T_WB = T_WC ∘ T_CB`. Treats camera == body when no extrinsic is set.
fn body_to_world(pose_w2c: &Pose3d, t_bc: Option<Pose3d>) -> Pose3d {
    let cam_to_world = pose_w2c.inverse();
    match &t_bc {
        Some(t_bc) => cam_to_world.compose(&t_bc.inverse()),
        None => cam_to_world,
    }
}

/// Propagates the camera pose and body velocity through one preintegrated
/// IMU window.
fn predict_pose_imu(
    pose_w2c: Pose3d,
    vel_world: Vec3F64,
    gravity_world: Vec3F64,
    preint: &PreintegratedImu,
    t_bc: Option<Pose3d>,
) -> (Pose3d, Vec3F64) {
    let body_to_world = body_to_world(&pose_w2c, t_bc);
    let (r_j, v_j, p_j) = preint.predict(
        &body_to_world.rotation,
        &vel_world,
        &body_to_world.translation,
        &gravity_world,
    );

    let pred_body_to_world = Pose3d::from_rt(r_j, p_j);
    let pred_cam_to_world = match &t_bc {
        Some(t_bc) => pred_body_to_world.compose(t_bc),
        None => pred_body_to_world,
    };
    (pred_cam_to_world.inverse(), v_j)
}

fn carry_klt_survivors(track_set: &mut TrackSet, survivors: Option<Vec<FlowSurvivor>>) {
    if survivors.is_none_or(|survivors| track_set.advance(survivors).is_err()) {
        *track_set = TrackSet::new();
    }
}

/// Carries a reference-keyframe BA correction into the current tracking pose
/// while preserving the current camera's pose relative to that reference.
fn apply_reference_pose_correction(
    current_pose: Pose3d,
    reference_before: Pose3d,
    reference_after: Pose3d,
) -> Pose3d {
    let current_from_reference = Pose3d::between(&reference_before, &current_pose);
    current_from_reference.compose(&reference_after)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    #[test]
    fn klt_tracks_survive_skipped_frame_and_clear_without_survivors() {
        let mut tracks = TrackSet::new();
        tracks
            .reconcile_from_matches(
                &[MapKeypointMatch {
                    map_point_idx: 42,
                    keypoint_idx: 0,
                }],
                &[[10.0, 20.0]],
            )
            .unwrap();
        let track_id = tracks.tracks()[0].id();

        carry_klt_survivors(
            &mut tracks,
            Some(vec![FlowSurvivor {
                track_id,
                pixel: [12.0, 21.0],
                error: 0.5,
            }]),
        );

        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks.tracks()[0].id(), track_id);
        assert_eq!(tracks.tracks()[0].map_point_idx(), Some(42));
        assert_eq!(tracks.tracks()[0].pixel(), [12.0, 21.0]);
        assert_eq!(tracks.tracks()[0].age(), 2);

        carry_klt_survivors(&mut tracks, None);
        assert!(tracks.is_empty());
    }
    use crate::map::{Keyframe, MapPoint};
    use kornia_algebra::{Mat3F64, SO3F64};
    use kornia_image::ImageSize;
    use kornia_imgproc::features::OrbFeatures;
    use kornia_sensors::imu::{ImuBias, ImuCalib};

    fn tracker() -> Tracker {
        Tracker::new(
            MapProjectionConfig::default(),
            TrackingLossRecoveryPolicy::default(),
        )
    }

    fn preintegration(dt: f64) -> PreintegratedImu {
        let mut pre = PreintegratedImu::new(
            ImuBias::default(),
            ImuCalib {
                gyro_noise: 1e-4,
                accel_noise: 1e-3,
                gyro_bias_noise: 1e-5,
                accel_bias_noise: 1e-4,
            },
        );
        pre.dt = dt;
        pre
    }

    pub(crate) fn synthetic_scene() -> (Map, Frame, PinholeCamera) {
        let camera = PinholeCamera {
            fx: 400.0,
            fy: 400.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        };
        let pose = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.03, -0.01, 0.02));
        let points: Vec<_> = (0..64)
            .map(|i| {
                Vec3F64::new(
                    (i % 8) as f64 * 0.3 - 1.0,
                    (i / 8) as f64 * 0.3 - 1.0,
                    4.0 + (i % 3) as f64 * 0.2,
                )
            })
            .collect();
        let make_frame = |idx, pose: Pose3d| {
            let pixels: Vec<_> = points
                .iter()
                .map(|point| {
                    let p = pose.transform_point(point);
                    [
                        (camera.fx * p.x / p.z + camera.cx) as f32,
                        (camera.fy * p.y / p.z + camera.cy) as f32,
                    ]
                })
                .collect();
            Frame {
                idx,
                features: OrbFeatures {
                    keypoints_xy: pixels.clone(),
                    orientations: vec![0.0; points.len()],
                    descriptors: (0..points.len()).map(|i| [i as u8; 32]).collect(),
                    octaves: vec![0; points.len()],
                },
                pose_world_to_cam: pose,
                image_size: ImageSize {
                    width: 640,
                    height: 480,
                },
                keypoint_colors: vec![[0; 3]; points.len()],
                u_right: Vec::new(),
                depth: Vec::new(),
                keypoints_undist: pixels,
            }
        };
        let current = make_frame(8, pose);
        let mut reference = Keyframe::from_frame(make_frame(0, Pose3d::IDENTITY));
        let mut map = Map::new();
        for (idx, point) in points.into_iter().enumerate() {
            let mp = map.push_map_point(MapPoint::new(point, [idx as u8; 32], 0, [0; 3], 0));
            reference.associate_map_point(idx, mp);
        }
        map.upsert_keyframe(reference);
        (map, current, camera)
    }

    #[test]
    fn clean_tracking_recovers_pose_and_associations() {
        let (map, frame, camera) = synthetic_scene();
        let image = Image::from_size_val(frame.image_size, 0u8).unwrap();
        let mut tracker = tracker();
        tracker.state.current_keyframe_idx = Some(0);
        tracker.state.lost_since_sec = Some(1.0);
        let result = tracker.track(
            TrackingInput {
                frame: &frame,
                previous_image: None,
                current_image: &image,
                timestamp_sec: 1.1,
                inertial: None,
            },
            &map,
            &camera,
        );
        assert_eq!(result.rejection, None);
        assert_eq!(result.matches.len(), 64);
        assert!(
            result
                .matches
                .iter()
                .all(|&(point, descriptor)| point == descriptor)
        );
        assert!(
            (tracker.state.pose_world_to_cam.translation - frame.pose_world_to_cam.translation)
                .length()
                < 1e-4
        );
        assert_eq!(tracker.track_set.len(), 64);
        assert_eq!(tracker.state.lost_since_sec, Some(1.0)); // publication has not happened
    }

    #[test]
    fn visual_success_preserves_imu_velocity_and_visual_model() {
        let (map, frame, camera) = synthetic_scene();
        let image = Image::from_size_val(frame.image_size, 0u8).unwrap();
        let mut tracker = tracker();
        tracker.state.current_keyframe_idx = Some(0);
        tracker.state.imu_initialized = true;
        tracker.state.last_frame_timestamp_sec = 1.0;
        let visual_model = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.7, 0.0, 0.0));
        tracker.state.velocity = Some(visual_model);
        let mut pre = preintegration(0.1);
        pre.delta_velocity = Vec3F64::new(0.02, -0.03, 0.01);
        let result = tracker.track(
            TrackingInput {
                frame: &frame,
                previous_image: None,
                current_image: &image,
                timestamp_sec: 1.1,
                inertial: Some(InertialPrediction {
                    preintegrated: &pre,
                    t_bc: None,
                    gravity_world: Vec3F64::ZERO,
                }),
            },
            &map,
            &camera,
        );
        assert_eq!(result.rejection, None);
        assert_eq!(tracker.state.velocity_world, pre.delta_velocity);
        assert_eq!(tracker.state.velocity, Some(visual_model));
    }

    #[test]
    fn rejected_frame_carries_inertial_prediction_with_camera_body_offset() {
        let (map, mut frame, camera) = synthetic_scene();
        frame.features = OrbFeatures {
            keypoints_xy: Vec::new(),
            orientations: Vec::new(),
            descriptors: Vec::new(),
            octaves: Vec::new(),
        };
        frame.keypoints_undist.clear();
        let image = Image::from_size_val(frame.image_size, 0u8).unwrap();
        let mut tracker = tracker();
        tracker.state.imu_initialized = true;
        tracker.state.last_frame_timestamp_sec = 1.0;
        let mut pre = preintegration(1.0);
        pre.delta_rotation =
            SO3F64::exp(Vec3F64::new(0.0, 0.0, std::f64::consts::FRAC_PI_2)).matrix();
        let result = tracker.track(
            TrackingInput {
                frame: &frame,
                previous_image: None,
                current_image: &image,
                timestamp_sec: 2.0,
                inertial: Some(InertialPrediction {
                    preintegrated: &pre,
                    t_bc: Some(Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(1.0, 0.0, 0.0))),
                    gravity_world: Vec3F64::ZERO,
                }),
            },
            &map,
            &camera,
        );
        assert!(result.rejection.is_some());
        let expected = Pose3d::new(
            pre.delta_rotation.transpose(),
            Vec3F64::new(-1.0, -1.0, 0.0),
        );
        assert_pose_close(tracker.state.pose_world_to_cam, expected);
        assert_pose_close(result.candidate_pose, expected);
        assert!(matches!(
            result.recovery,
            RecoveryDecision::RestartBootstrap
        ));
        assert!(tracker.state.imu_initialized); // Tracker reports restart, it does not execute it.
        assert_eq!(tracker.state.last_frame_timestamp_sec, 1.0);
        assert_eq!(map.keyframes().len(), 1);
    }

    #[test]
    fn missing_or_empty_imu_window_uses_visual_motion() {
        for (initialized, previous_timestamp, dt) in
            [(false, 1.0, 1.0), (true, 0.0, 1.0), (true, 1.0, 0.0)]
        {
            let mut tracker = tracker();
            tracker.state.imu_initialized = initialized;
            tracker.state.last_frame_timestamp_sec = previous_timestamp;
            let visual = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.1, 0.0, 0.0));
            tracker.state.velocity = Some(visual);
            let mut pre = preintegration(dt);
            pre.delta_position = Vec3F64::new(100.0, 0.0, 0.0);
            assert_eq!(
                tracker.predict_pose(Some(InertialPrediction {
                    preintegrated: &pre,
                    t_bc: None,
                    gravity_world: Vec3F64::ZERO
                })),
                visual
            );
            assert_eq!(tracker.predict_pose(None), visual);
        }
    }

    #[test]
    fn missing_previous_image_and_failed_klt_clear_tracks_on_rejection() {
        let (map, mut frame, camera) = synthetic_scene();
        frame.features = OrbFeatures {
            keypoints_xy: Vec::new(),
            orientations: Vec::new(),
            descriptors: Vec::new(),
            octaves: Vec::new(),
        };
        frame.keypoints_undist.clear();
        let image = Image::from_size_val(frame.image_size, 0u8).unwrap();
        let wrong_size = Image::from_size_val(
            ImageSize {
                width: 32,
                height: 32,
            },
            0u8,
        )
        .unwrap();
        for previous_image in [None, Some(&wrong_size)] {
            let mut tracker = tracker();
            seed_track(&mut tracker);
            let outcome = tracker.track(
                TrackingInput {
                    frame: &frame,
                    previous_image,
                    current_image: &image,
                    timestamp_sec: 2.0,
                    inertial: None,
                },
                &map,
                &camera,
            );
            assert!(outcome.rejection.is_some());
            assert!(tracker.track_set.is_empty());
        }
    }

    fn seed_track(tracker: &mut Tracker) {
        tracker
            .track_set
            .reconcile_from_matches(
                &[MapKeypointMatch {
                    map_point_idx: 42,
                    keypoint_idx: 0,
                }],
                &[[10.0, 20.0]],
            )
            .unwrap();
    }

    fn assert_pose_close(actual: Pose3d, expected: Pose3d) {
        assert!((actual.translation - expected.translation).length() < 1e-9);
        for (a, b) in actual
            .rotation
            .to_cols_array()
            .iter()
            .zip(expected.rotation.to_cols_array())
        {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn correction_operations_preserve_their_distinct_velocity_and_track_contracts() {
        let before = Pose3d::new(
            SO3F64::exp(Vec3F64::new(0.1, -0.2, 0.3)).matrix(),
            Vec3F64::new(-1.0, 0.5, 0.2),
        );
        let after = Pose3d::new(
            SO3F64::exp(Vec3F64::new(0.25, 0.1, -0.1)).matrix(),
            Vec3F64::new(-2.0, -0.3, 0.8),
        );
        let relative = Pose3d::new(
            SO3F64::exp(Vec3F64::new(-0.15, 0.05, 0.2)).matrix(),
            Vec3F64::new(-0.5, 0.1, 0.3),
        );
        let velocity = Vec3F64::new(1.0, 0.2, -0.5);
        let mut tracker = tracker();
        tracker.state.pose_world_to_cam = relative.compose(&before);
        tracker.state.velocity = Some(relative);
        seed_track(&mut tracker);
        tracker.apply_local_ba_correction(before, after, velocity);
        assert_pose_close(
            Pose3d::between(&after, &tracker.state.pose_world_to_cam),
            relative,
        );
        assert_eq!(tracker.state.velocity_world, velocity);
        assert_eq!(tracker.state.velocity, Some(relative));
        assert_eq!(tracker.track_set.len(), 1);

        let world = before.inverse().compose(&after);
        tracker.apply_loop_correction(after, before, world);
        assert_pose_close(
            Pose3d::between(&before, &tracker.state.pose_world_to_cam),
            relative,
        );
        assert!((tracker.state.velocity_world - world.rotation * velocity).length() < 1e-12);
        assert!(tracker.track_set.is_empty());
        assert_eq!(tracker.state.velocity, Some(relative));
        seed_track(&mut tracker);
        tracker.adopt_inertial_alignment(after, velocity);
        assert_eq!(tracker.state.pose_world_to_cam, after);
        assert_eq!(tracker.state.velocity_world, velocity);
        assert!(tracker.state.velocity.is_none());
        assert!(tracker.state.imu_initialized);
        assert_eq!(tracker.track_set.len(), 1);
        tracker.state.last_frame_timestamp_sec = 2.0;
        tracker.restart_bootstrap();
        assert_eq!(tracker.state.pose_world_to_cam, after);
        assert_eq!(tracker.state.last_frame_timestamp_sec, 2.0);
        assert_eq!(tracker.state.velocity_world, Vec3F64::ZERO);
        assert!(tracker.track_set.is_empty());
    }
}
