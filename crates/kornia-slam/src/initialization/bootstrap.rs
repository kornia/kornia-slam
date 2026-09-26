//! Bootstrap: deciding what a frame offered to initialization is worth, and
//! building the initial map from it.
//!
//! The decision is separated from acting on it: [`evaluate_bootstrap`] reads
//! the reference frame and the incoming one and says which case this is, and
//! the caller performs the state changes each case implies. Keeping the two
//! apart matters here because the cases differ in how they treat the stored
//! reference — one drops it, one keeps it, one consumes it.
//!
//! [`stereo_initial_map`] and [`two_view_initial_map`] build the initial map
//! as one [`MapInsertion`] without touching the map; publishing it, and the
//! state that follows, stay with the caller.

use std::collections::HashSet;

use kornia_3d::camera::PinholeCamera;

use crate::Frame;
use crate::initialization::two_view::{
    TwoViewEstimate, TwoViewInitConfig, TwoViewRejectReason, try_initialize_two_view,
};
use crate::mapping::Map;
use crate::mapping::growth::{accepted_pair_claims, stereo_seeds};
use crate::mapping::map::{
    Keyframe, LandmarkSeed, LandmarkTarget, MapInsertion, ObservationKey, ObservationLink,
};
use kornia_algebra::Vec3F64;

/// A frame with fewer keypoints than this is neither a viable reference nor a
/// viable current frame (mirrors ORB-SLAM3's `MonocularInitialization`).
pub(crate) const MIN_KEYPOINTS_FOR_BOOTSTRAP: usize = 100;

/// Fewest stereo points a single keyframe needs to seed a metric map.
pub(crate) const MIN_STEREO_POINTS: usize = 50;

/// Fewest landmarks with positive depth in both bootstrap keyframes for the
/// two-view map to be kept (ORB-SLAM3's `CreateInitialMapMonocular`).
pub(crate) const MIN_VALID_POINTS: usize = 50;

/// What the caller should do with the frame it offered.
pub(crate) enum BootstrapDecision {
    /// Too few keypoints to use at all. Any stored reference is stale and the
    /// caller drops it, waiting for a feature-rich frame to start over.
    Unusable { keypoints: usize },
    /// No reference was held, and this frame becomes it. The caller stores the
    /// frame and starts the inertial window at its timestamp.
    StoreAsReference,
    /// Two-view estimation failed against the held reference. The caller keeps
    /// the reference and waits for a better second frame.
    Rejected {
        reference_idx: usize,
        reason: TwoViewRejectReason,
    },
    /// Two-view estimation succeeded. The caller consumes the reference and
    /// publishes the initial map.
    Initialized(Box<TwoViewEstimate>),
}

/// Decides what the incoming frame is worth, without changing anything.
///
/// `reference` is the frame previously stored by [`BootstrapDecision::StoreAsReference`],
/// if any. Passing it by reference is deliberate: whether it survives is the
/// caller's decision, driven by which variant comes back.
pub(crate) fn evaluate_bootstrap(
    reference: Option<&Frame>,
    current: &Frame,
    camera: &PinholeCamera,
    config: &TwoViewInitConfig,
) -> BootstrapDecision {
    let keypoints = current.features.keypoints_xy.len();
    if keypoints <= MIN_KEYPOINTS_FOR_BOOTSTRAP {
        return BootstrapDecision::Unusable { keypoints };
    }

    let Some(reference) = reference else {
        return BootstrapDecision::StoreAsReference;
    };

    match try_initialize_two_view(
        &reference.features,
        &reference.pose_world_to_cam,
        &current.features,
        camera,
        config,
    ) {
        Ok(estimate) => BootstrapDecision::Initialized(Box::new(estimate)),
        Err(reason) => BootstrapDecision::Rejected {
            reference_idx: reference.idx,
            reason,
        },
    }
}

/// A stereo keyframe with too few valid depths to seed the initial map.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TooFewStereoPoints {
    pub(crate) found: usize,
}

/// The initial map from a single stereo keyframe: every keypoint with valid
/// depth becomes a metric landmark (ORB-SLAM3's `StereoInitialization`).
///
/// `camera` must be the rectified camera the depths were computed against.
pub(crate) fn stereo_initial_map(
    keyframe: Keyframe,
    camera: &PinholeCamera,
) -> Result<MapInsertion, TooFewStereoPoints> {
    let landmarks = stereo_seeds(&keyframe.frame, camera, f64::INFINITY, &[]);
    if landmarks.len() < MIN_STEREO_POINTS {
        return Err(TooFewStereoPoints {
            found: landmarks.len(),
        });
    }
    Ok(MapInsertion {
        keyframes: vec![keyframe],
        landmarks,
        ..Default::default()
    })
}

/// The initial map from an accepted two-view estimate: both keyframes, the
/// triangulated landmarks, and each landmark's second observation.
///
/// Points are rescaled so the median depth is one, and landmarks are
/// referenced to the newer keyframe, as ORB-SLAM3 creates them.
pub(crate) fn two_view_initial_map(
    reference: Keyframe,
    current: Keyframe,
    estimate: &TwoViewEstimate,
) -> MapInsertion {
    let depth_scale = estimate.median_depth.filter(|&d| d > 1e-6).unwrap_or(1.0);
    let reference_pose_inv = reference.frame.pose_world_to_cam.inverse();
    let reference_kf_idx = reference.frame.idx;
    let current_kf_idx = current.frame.idx;
    let matches = &estimate.estimate.matches;

    let mut candidates: Vec<(Vec3F64, [u8; 3], usize, usize)> = Vec::new();
    for (p_cam, &match_idx) in estimate.points3d.iter().zip(estimate.inlier_indices.iter()) {
        let Some(&(ref_desc_idx, curr_desc_idx)) = matches.get(match_idx) else {
            continue;
        };
        if ref_desc_idx >= reference.map_point_by_desc_idx.len()
            || curr_desc_idx >= current.map_point_by_desc_idx.len()
        {
            continue;
        }
        if current
            .frame
            .features
            .descriptors
            .get(curr_desc_idx)
            .or_else(|| reference.frame.features.descriptors.get(ref_desc_idx))
            .is_none()
        {
            continue;
        }
        let color = current
            .frame
            .keypoint_colors
            .get(curr_desc_idx)
            .copied()
            .unwrap_or([128; 3]);
        let position = reference_pose_inv.transform_point(&(*p_cam / depth_scale));
        candidates.push((position, color, ref_desc_idx, curr_desc_idx));
    }

    // The two-view result can name one feature twice — two triangulated
    // points landing on the same keypoint in either view. A feature holds
    // at most one landmark, so the first claim wins and the rest are
    // dropped, as in pair growth; refusing the whole pair instead would
    // throw away an otherwise good bootstrap for one duplicate.
    let claims: Vec<(usize, usize)> = candidates
        .iter()
        .map(|&(_, _, ref_desc_idx, curr_desc_idx)| (ref_desc_idx, curr_desc_idx))
        .collect();

    // Both keyframes publish in the same request as the landmarks, so the
    // reference keyframe each second observation points at arrives with it.
    let mut request = MapInsertion::default();
    for index in accepted_pair_claims(&claims) {
        let (position, color, ref_desc_idx, curr_desc_idx) = candidates[index];
        let new_index = request.landmarks.len();
        request.landmarks.push(LandmarkSeed {
            position,
            color,
            reference: ObservationKey {
                keyframe_idx: current_kf_idx,
                feature_idx: curr_desc_idx,
            },
        });
        request.observations.push(ObservationLink {
            observation: ObservationKey {
                keyframe_idx: reference_kf_idx,
                feature_idx: ref_desc_idx,
            },
            landmark: LandmarkTarget::New(new_index),
        });
    }
    request.keyframes = vec![reference, current];
    request
}

/// Quality metrics for a freshly-bootstrapped 2-keyframe map.
///
/// Used as the gate for accepting a bootstrap result. Mirrors
/// ORB-SLAM3's reset criteria in `CreateInitialMapMonocular`:
/// `medianDepth < 0 || TrackedMapPoints(1) < 50`.
#[derive(Debug, Clone, Copy, Default)]
pub struct InitialMapHealth {
    /// Number of map points with positive depth in both bootstrap KFs.
    pub valid_in_both: usize,
    /// Median depth of valid points in the older KF's frame.
    pub median_depth_older_kf: f64,
}

impl InitialMapHealth {
    /// Whether the map meets ORB-SLAM3's reset criteria and must be discarded.
    pub fn is_degenerate(&self) -> bool {
        self.valid_in_both < MIN_VALID_POINTS || self.median_depth_older_kf <= 0.0
    }
}

/// Health metrics for the just-bootstrapped pair of keyframes.
///
/// Inspects the last two keyframes in insertion order and reports how
/// many associated map points still have positive depth in both KFs,
/// plus the median depth in the older KF's frame. Used to decide whether
/// a freshly-bootstrapped map is safe to commit. The gate is a bootstrap
/// policy, so it reads the map rather than living on it.
pub fn initial_map_health(map: &Map) -> InitialMapHealth {
    let keyframes = map.keyframes();
    let n = keyframes.len();
    if n < 2 {
        return InitialMapHealth::default();
    }
    let kf_older = &keyframes[n - 2];
    let kf_newer = &keyframes[n - 1];
    let pose_older = kf_older.frame.pose_world_to_cam;
    let pose_newer = kf_newer.frame.pose_world_to_cam;

    // Collect MPs observed by either KF, dedup.
    let mut seen: HashSet<usize> = HashSet::new();
    for mp_idx in kf_older
        .map_point_by_desc_idx
        .iter()
        .chain(kf_newer.map_point_by_desc_idx.iter())
        .flatten()
    {
        seen.insert(*mp_idx);
    }

    let mut depths_older: Vec<f64> = Vec::with_capacity(seen.len());
    let mut valid_in_both = 0usize;
    for idx in seen {
        let Some(mp) = map.map_points().get(idx) else {
            continue;
        };
        if mp.culled {
            continue;
        }
        let z_older = pose_older.transform_point(&mp.position).z;
        let z_newer = pose_newer.transform_point(&mp.position).z;
        if z_older > 0.0 && z_newer > 0.0 {
            valid_in_both += 1;
            depths_older.push(z_older);
        }
    }

    let median_depth = if depths_older.is_empty() {
        0.0
    } else {
        let mid = depths_older.len() / 2;
        depths_older.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
        depths_older[mid]
    };

    InitialMapHealth {
        valid_in_both,
        median_depth_older_kf: median_depth,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BootstrapDecision, MIN_KEYPOINTS_FOR_BOOTSTRAP, MIN_STEREO_POINTS, evaluate_bootstrap,
        stereo_initial_map, two_view_initial_map,
    };
    use crate::Frame;
    use crate::initialization::two_view::TwoViewEstimate;
    use crate::mapping::map::{Keyframe, LandmarkTarget};
    use crate::tracking::pose_estimation::Estimate;
    use kornia_3d::camera::PinholeCamera;
    use kornia_3d::pose::Pose3d;
    use kornia_algebra::Vec3F64;
    use kornia_image::ImageSize;
    use kornia_imgproc::features::OrbFeatures;

    fn frame(idx: usize, keypoints: usize) -> Frame {
        Frame {
            idx,
            features: OrbFeatures {
                keypoints_xy: (0..keypoints).map(|i| [i as f32, i as f32]).collect(),
                orientations: vec![0.0; keypoints],
                descriptors: vec![[0u8; 32]; keypoints],
                octaves: vec![0; keypoints],
            },
            pose_world_to_cam: Pose3d::IDENTITY,
            image_size: ImageSize {
                width: 640,
                height: 480,
            },
            keypoint_colors: Vec::new(),
            u_right: Vec::new(),
            depth: Vec::new(),
            keypoints_undist: Vec::new(),
        }
    }

    fn camera() -> PinholeCamera {
        PinholeCamera {
            fx: 400.0,
            fy: 400.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

    /// A feature-poor frame is unusable whether or not a reference is held —
    /// and the caller is told to drop the reference with it.
    #[test]
    fn a_feature_poor_frame_is_unusable_with_or_without_a_reference() {
        let poor = frame(1, MIN_KEYPOINTS_FOR_BOOTSTRAP);
        let reference = frame(0, 500);

        for held in [None, Some(&reference)] {
            assert!(matches!(
                evaluate_bootstrap(held, &poor, &camera(), &Default::default()),
                BootstrapDecision::Unusable { keypoints } if keypoints == MIN_KEYPOINTS_FOR_BOOTSTRAP
            ));
        }
    }

    /// The threshold is exclusive: exactly the minimum is still unusable, one
    /// more is not.
    #[test]
    fn the_keypoint_threshold_is_exclusive() {
        assert!(matches!(
            evaluate_bootstrap(
                None,
                &frame(1, MIN_KEYPOINTS_FOR_BOOTSTRAP),
                &camera(),
                &Default::default()
            ),
            BootstrapDecision::Unusable { .. }
        ));
        assert!(matches!(
            evaluate_bootstrap(
                None,
                &frame(1, MIN_KEYPOINTS_FOR_BOOTSTRAP + 1),
                &camera(),
                &Default::default()
            ),
            BootstrapDecision::StoreAsReference
        ));
    }

    /// With no reference held, a usable frame becomes one.
    #[test]
    fn a_usable_frame_without_a_reference_becomes_the_reference() {
        assert!(matches!(
            evaluate_bootstrap(None, &frame(7, 500), &camera(), &Default::default()),
            BootstrapDecision::StoreAsReference
        ));
    }

    /// Two-view failure names the reference it failed against, so the caller
    /// keeps that reference rather than starting over.
    #[test]
    fn two_view_failure_reports_the_reference_it_failed_against() {
        let reference = frame(3, 500);
        let current = frame(4, 500);

        match evaluate_bootstrap(Some(&reference), &current, &camera(), &Default::default()) {
            BootstrapDecision::Rejected { reference_idx, .. } => {
                assert_eq!(reference_idx, 3, "the held reference must be identified");
            }
            _ => panic!("degenerate identical features cannot initialize"),
        }
    }

    /// Evaluation is a decision, not an action: it must not touch the frames.
    #[test]
    fn evaluation_leaves_both_frames_untouched() {
        let reference = frame(3, 500);
        let current = frame(4, 500);
        let reference_before = reference.features.keypoints_xy.len();
        let current_before = current.features.keypoints_xy.len();

        let _ = evaluate_bootstrap(Some(&reference), &current, &camera(), &Default::default());

        assert_eq!(reference.features.keypoints_xy.len(), reference_before);
        assert_eq!(current.features.keypoints_xy.len(), current_before);
        assert_eq!(reference.pose_world_to_cam, Pose3d::IDENTITY);
    }

    fn stereo_frame(idx: usize, valid_depths: usize) -> Frame {
        let mut frame = frame(idx, valid_depths + 5);
        frame.depth = (0..valid_depths + 5)
            .map(|i| if i < valid_depths { 2.0 } else { -1.0 })
            .collect();
        frame
    }

    #[test]
    fn a_stereo_frame_needs_enough_valid_depths_to_seed_a_map() {
        let too_few = stereo_frame(1, MIN_STEREO_POINTS - 1);
        let error = stereo_initial_map(Keyframe::from_frame(too_few), &camera()).unwrap_err();
        assert_eq!(error.found, MIN_STEREO_POINTS - 1);

        let insertion = stereo_initial_map(
            Keyframe::from_frame(stereo_frame(1, MIN_STEREO_POINTS)),
            &camera(),
        )
        .unwrap();
        assert_eq!(insertion.keyframes.len(), 1);
        assert_eq!(insertion.landmarks.len(), MIN_STEREO_POINTS);
        assert!(insertion.observations.is_empty());
    }

    #[test]
    fn a_two_view_map_keeps_the_first_claim_on_each_feature_and_normalizes_depth() {
        let reference = Keyframe::from_frame(frame(3, 500));
        let current = Keyframe::from_frame(frame(4, 500));
        let estimate = TwoViewEstimate {
            estimate: Estimate {
                pose: Pose3d::IDENTITY,
                // Match 1 reuses reference feature 10; match 2 is out of range.
                matches: vec![(10, 20), (10, 21), (11, 999), (12, 22)],
                inliers: 4,
            },
            points3d: vec![
                Vec3F64::new(0.0, 0.0, 4.0),
                Vec3F64::new(0.0, 0.0, 4.0),
                Vec3F64::new(0.0, 0.0, 4.0),
                Vec3F64::new(0.0, 0.0, 8.0),
            ],
            inlier_indices: vec![0, 1, 2, 3],
            median_depth: Some(4.0),
            model_kind: 'F',
        };

        let insertion = two_view_initial_map(reference, current, &estimate);

        assert_eq!(insertion.keyframes.len(), 2);
        let features: Vec<usize> = insertion
            .landmarks
            .iter()
            .map(|seed| seed.reference.feature_idx)
            .collect();
        assert_eq!(features, vec![20, 22]);
        assert!(
            insertion
                .landmarks
                .iter()
                .all(|seed| seed.reference.keyframe_idx == 4)
        );
        assert_eq!(insertion.landmarks[0].position.z, 1.0);
        assert_eq!(insertion.landmarks[1].position.z, 2.0);
        assert!(insertion.observations.iter().all(|link| {
            link.observation.keyframe_idx == 3 && matches!(link.landmark, LandmarkTarget::New(_))
        }));
    }
}
