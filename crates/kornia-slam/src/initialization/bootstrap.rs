//! What to do with a frame offered to monocular bootstrap.
//!
//! The decision is separated from acting on it: [`evaluate_bootstrap`] reads
//! the reference frame and the incoming one and says which case this is, and
//! the caller performs the state changes each case implies. Keeping the two
//! apart matters here because the cases differ in how they treat the stored
//! reference — one drops it, one keeps it, one consumes it — and that was
//! previously expressed as scattered assignments amongst early returns.

use kornia_3d::camera::PinholeCamera;

use crate::Frame;
use crate::initialization::two_view::{
    TwoViewEstimate, TwoViewInitConfig, TwoViewRejectReason, try_initialize_two_view,
};

/// A frame with fewer keypoints than this is neither a viable reference nor a
/// viable current frame (mirrors ORB-SLAM3's `MonocularInitialization`).
pub(crate) const MIN_KEYPOINTS_FOR_BOOTSTRAP: usize = 100;

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

#[cfg(test)]
mod tests {
    use super::{BootstrapDecision, MIN_KEYPOINTS_FOR_BOOTSTRAP, evaluate_bootstrap};
    use crate::Frame;
    use kornia_3d::camera::PinholeCamera;
    use kornia_3d::pose::Pose3d;
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
}
