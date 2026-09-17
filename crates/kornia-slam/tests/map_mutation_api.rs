//! The canonical mutation API must be sufficient on its own.
//!
//! Everything here goes through `kornia_slam::map`'s public exports — no
//! private access, no mutable storage handles, no test-only bypass. If a
//! caller outside the crate cannot build and read a map with these alone, the
//! API is incomplete.

use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;
use kornia_image::ImageSize;
use kornia_slam::Frame;
use kornia_slam::map::{
    Keyframe, LandmarkSeed, LandmarkTarget, Map, MapInsertion, MapMutationError, ObservationKey,
    ObservationLink,
};
use kornia_slam::{OrbFeatures, tracking::TrackingStatus};

fn frame(idx: usize, n_features: usize) -> Frame {
    Frame {
        idx,
        features: OrbFeatures {
            keypoints_xy: (0..n_features).map(|i| [i as f32, i as f32]).collect(),
            orientations: vec![0.0; n_features],
            descriptors: (0..n_features).map(|i| [i as u8; 32]).collect(),
            octaves: vec![0; n_features],
        },
        pose_world_to_cam: Pose3d::IDENTITY,
        image_size: ImageSize {
            width: 640,
            height: 480,
        },
        keypoint_colors: vec![[0; 3]; n_features],
        u_right: Vec::new(),
        depth: Vec::new(),
        keypoints_undist: Vec::new(),
    }
}

fn key(keyframe_idx: usize, feature_idx: usize) -> ObservationKey {
    ObservationKey {
        keyframe_idx,
        feature_idx,
    }
}

#[test]
fn a_map_can_be_built_and_read_through_public_exports_alone() {
    let mut map = Map::new();

    // Two keyframes and a landmark seen by both, published as one batch.
    let result = map
        .apply_insertion(MapInsertion {
            keyframes: vec![
                Keyframe::from_frame(frame(0, 3)),
                Keyframe::from_frame(frame(1, 3)),
            ],
            landmarks: vec![LandmarkSeed {
                position: Vec3F64::new(0.0, 0.0, 5.0),
                color: [10, 20, 30],
                reference: key(0, 0),
            }],
            observations: vec![ObservationLink {
                observation: key(1, 0),
                landmark: LandmarkTarget::New(0),
            }],
            ..Default::default()
        })
        .expect("a well-formed batch is accepted");

    assert_eq!(result.keyframe_ids, vec![0, 1]);
    assert_eq!(result.observations_added, 2);
    let landmark = result.landmark_ids[0];

    // Read it all back through the public accessors.
    assert_eq!(map.keyframes().len(), 2);
    assert_eq!(map.num_map_points(), 1);
    assert_eq!(map.num_active_map_points(), 1);
    assert_eq!(map.get_keyframe(0).unwrap().map_point(0), Some(landmark));
    assert_eq!(map.get_keyframe(1).unwrap().map_point(0), Some(landmark));

    let point = &map.map_points()[landmark];
    assert_eq!(point.color, [10, 20, 30]);
    assert!(point.is_observed_by(0) && point.is_observed_by(1));
    assert_eq!(
        point.observer_keyframes().collect::<Vec<_>>(),
        vec![0, 1],
        "records keep insertion order"
    );
    assert_eq!(point.observations()[0].key, key(0, 0));
    // Geometry is valid on return, not at some later refresh.
    assert!(point.max_distance > 0.0);

    // Covisibility is derived from those records.
    assert_eq!(map.covisible_keyframes(0), vec![(1, 1)]);
}

#[test]
fn incremental_operations_compose_with_the_batch_api() {
    let mut map = Map::new();
    map.insert_keyframe(Keyframe::from_frame(frame(0, 3)))
        .unwrap();
    map.insert_keyframe(Keyframe::from_frame(frame(1, 3)))
        .unwrap();

    let landmark = map
        .insert_landmark(LandmarkSeed {
            position: Vec3F64::new(0.0, 0.0, 5.0),
            color: [0; 3],
            reference: key(0, 0),
        })
        .unwrap();

    assert!(map.link_observation(1, 0, landmark).unwrap(), "a new link");
    assert!(
        !map.link_observation(1, 0, landmark).unwrap(),
        "the identical link again is a no-op"
    );

    // Unlinking a non-reference observation leaves the reference alone.
    assert_eq!(map.unlink_observation(1, 0).unwrap(), Some(landmark));
    assert_eq!(map.map_points()[landmark].keyframe_idx, 0);
    assert_eq!(map.map_points()[landmark].observations().len(), 1);

    // The last unlink retires the landmark and clears its geometry.
    assert_eq!(map.unlink_observation(0, 0).unwrap(), Some(landmark));
    assert!(map.map_points()[landmark].culled);
    assert!(map.map_points()[landmark].observations().is_empty());
    assert_eq!(map.num_active_map_points(), 0);

    // The retired slot is never reused.
    let next = map
        .insert_landmark(LandmarkSeed {
            position: Vec3F64::new(0.0, 0.0, 6.0),
            color: [0; 3],
            reference: key(0, 1),
        })
        .unwrap();
    assert_ne!(next, landmark);
}

#[test]
fn conflicts_are_typed_and_leave_the_map_unchanged() {
    let mut map = Map::new();
    map.insert_keyframe(Keyframe::from_frame(frame(0, 2)))
        .unwrap();
    let first = map
        .insert_landmark(LandmarkSeed {
            position: Vec3F64::new(0.0, 0.0, 5.0),
            color: [0; 3],
            reference: key(0, 0),
        })
        .unwrap();

    let before_points = map.num_map_points();
    let before_keyframes = map.keyframes().len();

    // Feature 0 is taken, and the offending claim sits last in the request.
    let error = map
        .apply_insertion(MapInsertion {
            keyframes: vec![Keyframe::from_frame(frame(1, 2))],
            landmarks: vec![LandmarkSeed {
                position: Vec3F64::new(0.0, 0.0, 7.0),
                color: [0; 3],
                reference: key(1, 0),
            }],
            observations: vec![ObservationLink {
                observation: key(0, 0),
                landmark: LandmarkTarget::New(0),
            }],
            ..Default::default()
        })
        .unwrap_err();

    assert_eq!(
        error,
        MapMutationError::FeatureOccupied {
            keyframe_idx: 0,
            feature_idx: 0,
            holder: first,
        }
    );
    assert_eq!(map.num_map_points(), before_points);
    assert_eq!(map.keyframes().len(), before_keyframes);
    assert!(map.get_keyframe(1).is_none());
    assert_eq!(map.get_keyframe(0).unwrap().map_point(0), Some(first));

    // A duplicate keyframe id is refused rather than replacing the stored one.
    assert_eq!(
        map.insert_keyframe(Keyframe::from_frame(frame(0, 2)))
            .unwrap_err(),
        MapMutationError::DuplicateKeyframe(0)
    );
    assert_eq!(map.keyframes().len(), before_keyframes);
}

#[test]
fn tracking_status_remains_publicly_matchable() {
    // Guards the facade the app matches on alongside the map API.
    let status = TrackingStatus::Tracked;
    assert!(matches!(status, TrackingStatus::Tracked));
}
