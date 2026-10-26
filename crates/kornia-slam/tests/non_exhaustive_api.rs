//! The `#[non_exhaustive]` types a downstream crate builds (cu-kornia-vio
//! does each of these) can be built from outside this crate: a struct
//! literal of one is E0639 here, so every value below goes through a
//! constructor, `Default` or a public field assignment.

use kornia_slam::mapping::culling::KeyframeCullingConfig;
use kornia_slam::mapping::map::{StillNeighbour, StillSide};
use kornia_slam::tracking::{KeyframeEvidence, KeyframePolicy};

#[test]
fn keyframe_evidence_and_culling_config_from_outside() {
    let e = KeyframeEvidence::new(10, Some(4), 120, 200).with_median_parallax_px(Some(7.5));
    assert_eq!(e.median_parallax_px, Some(7.5));
    let mut set = KeyframeEvidence::new(10, Some(4), 120, 200);
    set.median_parallax_px = Some(7.5);
    assert_eq!(set, e);
    // KeyframePolicy is exhaustive: functional update still works.
    let policy = KeyframePolicy {
        max_frames_between: usize::MAX,
        ..KeyframePolicy::default()
    };
    let _ = policy.should_insert_with(&e);

    let mut cfg = KeyframeCullingConfig::default();
    cfg.min_keyframes = 4;
    cfg.max_removals = 1;
    cfg.protect_newest = 5;
    cfg.max_merged_imu_span_sec = 11.0;
    cfg.merge_imu_edges = false;
    let copy = cfg.clone();
    assert_eq!(
        (copy.min_keyframes, copy.max_removals, copy.protect_newest),
        (4, 1, 5)
    );
}

#[test]
fn still_neighbour_from_outside() {
    let n = StillNeighbour::new(42, StillSide::After, 3.5);
    assert_eq!(
        (n.keyframe_idx, n.side, n.span_s),
        (42, StillSide::After, 3.5)
    );
    assert_eq!(n, StillNeighbour::new(42, StillSide::After, 3.5));
    assert_ne!(n, StillNeighbour::new(42, StillSide::Before, 3.5));
}
