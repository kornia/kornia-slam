//! Map: keyframes, map points, local map selection, and culling.
//!
//! ```text
//!    +--------+
//!    | Frame  |
//!    +--------+
//!         |
//!         v
//!    +----------------------+
//!    | Keyframe             |
//!    | frame + desc -> mp   |
//!    +----------------------+
//!         |
//!         v
//!    +----------------------+
//!    | Map                  |
//!    | keyframes + points   |
//!    +----------------------+
//!
//!    ops:
//!      * upsert_keyframe
//!      * push_map_point
//!      * build_local_map_points
//!      * cull
//!      * run_local_ba
//! ```

mod local_mapping;

pub use local_mapping::{KeyframeJob, LocalMapping, LocalMappingMode};

mod keyframe;
mod map_point;
pub mod ops;

#[cfg(test)]
use crate::frame::Frame;
#[cfg(test)]
use kornia_algebra::Vec3F64;
#[cfg(test)]
use std::collections::HashSet;

pub(crate) use keyframe::stereo_depth_obs;
pub use keyframe::{ImuFactor, Keyframe, STEREO_DEPTH_MIN_SIGMA, STEREO_DEPTH_REL_SIGMA};
pub use map_point::{
    LandmarkObservation, MapPoint, ORB_N_LEVELS, ORB_SCALE_FACTOR, ObservationKey,
    TriangulatedPoint,
};
pub use ops::{
    InitialMapHealth, KeyframeBaCorrection, LocalBaMergeResult, LocalBaSnapshot,
    MapPointMergeResult, PoseGraphCorrectionError, PoseGraphCorrectionResult,
};

/// In-memory map storage for keyframes and persistent map points.
#[derive(Debug, Clone, Default)]
pub struct Map {
    keyframes: Vec<Keyframe>,
    map_points: Vec<MapPoint>,
    imu_factors: Vec<ImuFactor>,
    // Incremented whenever the map's world coordinate frame changes. Local BA
    // snapshots from an older epoch must never be merged into the new frame.
    world_epoch: u64,
}

impl Map {
    /// Creates an empty map.
    pub fn new() -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_3d::pose::Pose3d;
    use kornia_image::ImageSize;
    use kornia_imgproc::features::OrbFeatures;

    fn test_frame(idx: usize, descriptors: Vec<[u8; 32]>) -> Frame {
        test_frame_with_pose(idx, descriptors, Pose3d::IDENTITY)
    }

    fn test_frame_with_pose(idx: usize, descriptors: Vec<[u8; 32]>, pose: Pose3d) -> Frame {
        let n = descriptors.len();
        Frame {
            idx,
            features: OrbFeatures {
                keypoints_xy: (0..n).map(|i| [i as f32, i as f32]).collect(),
                orientations: vec![0.0; n],
                descriptors,
                octaves: vec![0; n],
            },
            pose_world_to_cam: pose,
            image_size: ImageSize {
                width: 640,
                height: 480,
            },
            keypoint_colors: vec![[0; 3]; n],
            u_right: Vec::new(),
            depth: Vec::new(),
            keypoints_undist: Vec::new(),
        }
    }

    #[test]
    fn keyframe_from_frame_initializes_map_point_slots() {
        let keyframe = Keyframe::from_frame(test_frame(7, vec![[0u8; 32], [1u8; 32], [2u8; 32]]));

        assert_eq!(keyframe.frame.idx, 7);
        assert_eq!(keyframe.map_point_by_desc_idx.len(), 3);
        assert!(
            keyframe
                .map_point_by_desc_idx
                .iter()
                .all(|slot| slot.is_none())
        );
    }

    #[test]
    fn keyframe_association_helpers_work() {
        let mut keyframe = Keyframe::from_frame(test_frame(1, vec![[0u8; 32], [1u8; 32]]));

        keyframe.associate_map_point(1, 42);
        assert_eq!(keyframe.map_point(1), Some(42));
        assert_eq!(keyframe.num_associated_points(), 1);

        keyframe.clear_map_point(1);
        assert_eq!(keyframe.map_point(1), None);
        assert_eq!(keyframe.num_associated_points(), 0);
    }

    #[test]
    fn map_point_new_sets_active_defaults() {
        let mp = MapPoint::new(Vec3F64::new(1.0, 2.0, 3.0), [9u8; 32], 0, [0; 3], 5, 0);

        assert_eq!(mp.position, Vec3F64::new(1.0, 2.0, 3.0));
        assert_eq!(mp.descriptor, [9u8; 32]);
        assert_eq!(mp.keyframe_idx, 5);
        assert_eq!(mp.n_visible, 1);
        assert_eq!(mp.n_found, 1);
        assert!(!mp.culled);
    }

    #[test]
    fn map_point_tracking_helpers_work() {
        let mut mp = MapPoint::new(Vec3F64::new(0.0, 0.0, 1.0), [0u8; 32], 0, [0; 3], 0, 0);
        mp.n_visible = 10;
        mp.n_found = 4;

        assert!((mp.found_ratio() - 0.4).abs() < 1e-9);
        mp.mark_culled();
        assert!(mp.culled);
    }

    #[test]
    fn covisible_keyframes_weights_by_shared_points() {
        // Three keyframes; map points observed by overlapping subsets:
        //   A: KF0, KF1, KF2   B: KF0, KF1   C: KF0, KF2
        // From KF0's view: KF1 shares {A,B}=2, KF2 shares {A,C}=2.
        let mut map = Map::new();
        for idx in 0..3 {
            // Four descriptor slots so KF0 can hold its three observations.
            map.upsert_keyframe(Keyframe::from_frame(test_frame(
                idx,
                (0..4).map(|d| [(idx * 10 + d) as u8; 32]).collect(),
            )));
        }

        // (observers, name) -> push a map point and associate desc slot 0/1.
        let specs: [(&[usize], usize); 3] = [(&[0, 1, 2], 0), (&[0, 1], 1), (&[0, 2], 1)];
        for (observers, _) in specs {
            let mp_idx = map.push_map_point(MapPoint::new(
                Vec3F64::new(0.0, 0.0, 1.0),
                [0u8; 32],
                0,
                [0; 3],
                observers[0],
                0,
            ));
            for (slot, &kf_idx) in observers.iter().enumerate() {
                if slot > 0
                    && let Some(mp) = map.map_points.get_mut(mp_idx)
                {
                    mp.add_observation(
                        ObservationKey {
                            keyframe_idx: kf_idx,
                            feature_idx: 0,
                        },
                        [0u8; 32],
                    );
                }
                // Associate a free descriptor slot in that keyframe.
                let kf = map.get_keyframe_mut(kf_idx).unwrap();
                let free = kf.map_point_by_desc_idx.iter().position(|s| s.is_none());
                kf.associate_map_point(free.unwrap_or(0), mp_idx);
            }
        }

        // min_weight 1 keeps both neighbors; sorted by descending weight.
        let covis = map.covisible_keyframes(0, 1);
        assert_eq!(covis.len(), 2);
        assert!(covis.iter().all(|&(_, w)| w == 2));
        assert_eq!(
            covis.iter().map(|&(k, _)| k).collect::<HashSet<_>>(),
            HashSet::from([1, 2])
        );

        // High threshold drops everything but the strongest link is retained.
        let fallback = map.covisible_keyframes(0, 99);
        assert_eq!(fallback.len(), 1);
        assert_eq!(fallback[0].1, 2);

        // Unknown keyframe yields no connections.
        assert!(map.covisible_keyframes(999, 1).is_empty());
    }

    #[test]
    fn stereo_depth_obs_uses_proportional_sigma_with_floor() {
        let mut frame = test_frame(0, vec![[0u8; 32], [1u8; 32]]);
        frame.depth = vec![10.0, -1.0];
        frame.u_right = vec![5.0, -1.0];
        let kf = Keyframe::from_frame(frame);

        // Valid depth: sigma = 0.05 * 10 = 0.5.
        let (d, s) = stereo_depth_obs(&kf, 0);
        assert_eq!(d, Some(10.0));
        assert!((s - 0.5).abs() < 1e-6);

        // Sentinel depth: no measurement.
        let (d1, s1) = stereo_depth_obs(&kf, 1);
        assert_eq!(d1, None);
        assert!((s1 - 1.0).abs() < 1e-9);

        // Very near depth clamps to the sigma floor.
        let mut near = test_frame(1, vec![[0u8; 32]]);
        near.depth = vec![0.1]; // 0.05 * 0.1 = 0.005 < floor
        let kf_near = Keyframe::from_frame(near);
        let (_, s2) = stereo_depth_obs(&kf_near, 0);
        assert!((s2 - STEREO_DEPTH_MIN_SIGMA).abs() < 1e-9);
    }

    #[test]
    fn upsert_keyframe_replaces_existing_idx() {
        let mut map = Map::new();

        map.upsert_keyframe(Keyframe::from_frame(test_frame(
            10,
            vec![[0u8; 32], [1u8; 32]],
        )));
        assert_eq!(map.keyframes().len(), 1);

        map.upsert_keyframe(Keyframe::from_frame(test_frame(10, vec![[2u8; 32]])));

        assert_eq!(map.keyframes().len(), 1);
        assert_eq!(
            map.get_keyframe(10)
                .expect("expected keyframe with idx 10")
                .frame
                .features
                .descriptors
                .len(),
            1
        );
    }

    #[test]
    fn push_map_point_returns_sequential_index() {
        let mut map = Map::new();

        let first_idx = map.push_map_point(MapPoint::new(
            Vec3F64::new(0.0, 0.0, 1.0),
            [0u8; 32],
            0,
            [0; 3],
            0,
            0,
        ));
        let second_idx = map.push_map_point(MapPoint::new(
            Vec3F64::new(1.0, 0.0, 1.0),
            [1u8; 32],
            0,
            [0; 3],
            0,
            0,
        ));

        assert_eq!(first_idx, 0);
        assert_eq!(second_idx, 1);
        assert_eq!(map.num_map_points(), 2);
    }

    #[test]
    fn cull_map_points_removes_low_ratio() {
        let mut map = Map::new();

        let first_idx = map.push_map_point(MapPoint::new(
            Vec3F64::new(0.0, 0.0, 5.0),
            [0u8; 32],
            0,
            [0; 3],
            0,
            0,
        ));
        let second_idx = map.push_map_point(MapPoint::new(
            Vec3F64::new(1.0, 0.0, 5.0),
            [1u8; 32],
            0,
            [0; 3],
            0,
            0,
        ));
        map.map_points_mut()[first_idx].n_visible = 10;
        map.map_points_mut()[first_idx].n_found = 1;
        map.map_points_mut()[second_idx].n_visible = 10;
        map.map_points_mut()[second_idx].n_found = 5;

        map.cull();

        assert!(map.map_points()[first_idx].culled);
        assert!(!map.map_points()[second_idx].culled);
    }

    #[test]
    fn local_ba_snapshot_merge_updates_only_snapshot_entities() {
        let mut map = Map::new();
        map.upsert_keyframe(Keyframe::from_frame(test_frame(0, vec![[0u8; 32]])));
        map.push_map_point(MapPoint::new(
            Vec3F64::new(0.0, 0.0, 5.0),
            [0u8; 32],
            0,
            [0; 3],
            0,
            0,
        ));

        let mut snapshot = map.local_ba_snapshot();
        snapshot.optimized.keyframes[0]
            .frame
            .pose_world_to_cam
            .translation
            .x = 2.0;
        snapshot.optimized.map_points[0].position.x = 3.0;

        map.upsert_keyframe(Keyframe::from_frame(test_frame(1, vec![[1u8; 32]])));
        let later_point = map.push_map_point(MapPoint::new(
            Vec3F64::new(9.0, 0.0, 5.0),
            [1u8; 32],
            0,
            [0; 3],
            1,
            0,
        ));

        let merged = map
            .merge_local_ba_snapshot(snapshot)
            .expect("snapshot should still use the live world frame");

        assert_eq!(merged.keyframe_corrections.len(), 1);
        assert_eq!(merged.keyframe_corrections[0].kf_idx, 0);
        assert_eq!(
            map.get_keyframe(0)
                .unwrap()
                .frame
                .pose_world_to_cam
                .translation
                .x,
            2.0
        );
        assert_eq!(map.map_points()[0].position.x, 3.0);
        assert_eq!(
            map.get_keyframe(1).unwrap().frame.pose_world_to_cam,
            Pose3d::IDENTITY
        );
        assert_eq!(map.map_points()[later_point].position.x, 9.0);
    }

    #[test]
    fn local_ba_snapshot_merge_rejects_an_obsolete_world_frame() {
        let mut map = Map::new();
        map.upsert_keyframe(Keyframe::from_frame(test_frame(0, vec![[0u8; 32]])));
        map.push_map_point(MapPoint::new(
            Vec3F64::new(1.0, 0.0, 5.0),
            [0u8; 32],
            0,
            [0; 3],
            0,
            0,
        ));

        let mut snapshot = map.local_ba_snapshot();
        snapshot.optimized.map_points[0].position.x = 7.0;
        map.scale_world(2.0);

        assert!(map.merge_local_ba_snapshot(snapshot).is_none());
        assert_eq!(map.map_points()[0].position.x, 2.0);
    }

    #[test]
    fn pose_graph_correction_preserves_point_in_reference_camera() {
        let before = [
            Pose3d::IDENTITY,
            Pose3d::new(
                kornia_algebra::Mat3F64::IDENTITY,
                Vec3F64::new(-1.0, 0.0, 0.0),
            ),
        ];
        let after = [
            Pose3d::IDENTITY,
            Pose3d::new(
                kornia_algebra::Mat3F64::IDENTITY,
                Vec3F64::new(-2.0, 0.0, 0.0),
            ),
        ];
        let mut map = Map::new();
        map.upsert_keyframe(Keyframe::from_frame(test_frame_with_pose(
            10,
            vec![[0; 32]],
            before[0],
        )));
        map.upsert_keyframe(Keyframe::from_frame(test_frame_with_pose(
            20,
            vec![[1; 32]],
            before[1],
        )));
        let point_before = Vec3F64::new(1.0, 0.0, 5.0);
        let point_idx = map.push_map_point(MapPoint::new(point_before, [1; 32], 0, [0; 3], 20, 0));
        map.get_keyframe_mut(20)
            .unwrap()
            .associate_map_point(0, point_idx);
        let point_in_reference_before = before[1].transform_point(&point_before);

        let result = map
            .apply_pose_graph_correction(&[10, 20], &before, &after)
            .unwrap();
        let point_in_reference_after =
            after[1].transform_point(&map.map_points()[point_idx].position);

        assert!((point_in_reference_after - point_in_reference_before).length() < 1e-10);
        assert_eq!(result.keyframes_corrected, 1);
        assert_eq!(result.map_points_corrected, 1);
    }

    #[test]
    fn pose_graph_correction_rejects_stale_snapshot_without_mutation() {
        let mut map = Map::new();
        map.upsert_keyframe(Keyframe::from_frame(test_frame(7, vec![[0; 32]])));
        let point_idx = map.push_map_point(MapPoint::new(
            Vec3F64::new(0.0, 0.0, 5.0),
            [0; 32],
            0,
            [0; 3],
            7,
            0,
        ));
        let live_pose = map.get_keyframe(7).unwrap().frame.pose_world_to_cam;
        let live_point = map.map_points()[point_idx].position;
        let stale = Pose3d::new(
            kornia_algebra::Mat3F64::IDENTITY,
            Vec3F64::new(1.0, 0.0, 0.0),
        );

        assert!(
            map.apply_pose_graph_correction(&[7], &[stale], &[Pose3d::IDENTITY])
                .is_err()
        );
        assert_eq!(
            map.get_keyframe(7).unwrap().frame.pose_world_to_cam,
            live_pose
        );
        assert_eq!(map.map_points()[point_idx].position, live_point);
    }

    #[test]
    fn pose_graph_correction_invalidates_older_local_ba_snapshot() {
        let mut map = Map::new();
        map.upsert_keyframe(Keyframe::from_frame(test_frame(0, vec![[0; 32]])));
        let mut snapshot = map.local_ba_snapshot();
        snapshot.optimized.keyframes[0]
            .frame
            .pose_world_to_cam
            .translation
            .x = 9.0;
        let corrected = Pose3d::new(
            kornia_algebra::Mat3F64::IDENTITY,
            Vec3F64::new(-1.0, 0.0, 0.0),
        );

        map.apply_pose_graph_correction(&[0], &[Pose3d::IDENTITY], &[corrected])
            .unwrap();

        assert!(map.merge_local_ba_snapshot(snapshot).is_none());
        assert_eq!(
            map.get_keyframe(0).unwrap().frame.pose_world_to_cam,
            corrected
        );
    }

    #[test]
    fn merge_map_points_redirects_associations_and_deduplicates_observers() {
        let mut map = Map::new();
        for idx in 0..3 {
            map.upsert_keyframe(Keyframe::from_frame(test_frame(
                idx,
                vec![[idx as u8; 32], [10 + idx as u8; 32]],
            )));
        }

        let survivor = map.push_map_point(MapPoint::new(
            Vec3F64::new(0.0, 0.0, 5.0),
            [0; 32],
            0,
            [0; 3],
            0,
            0,
        ));
        map.map_points_mut()[survivor].add_observation(
            ObservationKey {
                keyframe_idx: 1,
                feature_idx: 0,
            },
            [1; 32],
        );
        map.map_points_mut()[survivor].n_visible = 7;
        map.map_points_mut()[survivor].n_found = 5;
        map.get_keyframe_mut(0)
            .unwrap()
            .associate_map_point(0, survivor);
        map.get_keyframe_mut(1)
            .unwrap()
            .associate_map_point(0, survivor);

        let replaced = map.push_map_point(MapPoint::new(
            Vec3F64::new(0.01, 0.0, 5.0),
            [2; 32],
            0,
            [0; 3],
            2,
            0,
        ));
        map.map_points_mut()[replaced].add_observation(
            ObservationKey {
                keyframe_idx: 1,
                feature_idx: 1,
            },
            [11; 32],
        );
        map.map_points_mut()[replaced].n_visible = 4;
        map.map_points_mut()[replaced].n_found = 3;
        map.get_keyframe_mut(2)
            .unwrap()
            .associate_map_point(0, replaced);
        map.get_keyframe_mut(1)
            .unwrap()
            .associate_map_point(1, replaced);

        let result = map.merge_map_points(survivor, replaced).unwrap();

        assert_eq!(result.survivor, survivor);
        assert_eq!(result.replaced, replaced);
        assert_eq!(result.redirected_associations, 1);
        assert!(map.map_points()[replaced].culled);
        assert_eq!(map.map_points()[survivor].n_visible, 11);
        assert_eq!(map.map_points()[survivor].n_found, 8);
        assert_eq!(
            map.map_points()[survivor]
                .observer_keyframes()
                .collect::<HashSet<_>>(),
            HashSet::from([0, 1, 2])
        );
        assert_eq!(map.get_keyframe(2).unwrap().map_point(0), Some(survivor));
        assert_eq!(map.get_keyframe(1).unwrap().map_point(0), Some(survivor));
        assert_eq!(map.get_keyframe(1).unwrap().map_point(1), None);
        for keyframe in map.keyframes() {
            assert_eq!(
                keyframe
                    .map_point_by_desc_idx
                    .iter()
                    .filter(|&&point| point == Some(survivor))
                    .count(),
                1
            );
        }
    }

    #[test]
    fn merge_map_points_rejects_invalid_or_culled_inputs() {
        let mut map = Map::new();
        map.upsert_keyframe(Keyframe::from_frame(test_frame(0, vec![[0; 32]])));
        let first = map.push_map_point(MapPoint::new(
            Vec3F64::new(0.0, 0.0, 5.0),
            [0; 32],
            0,
            [0; 3],
            0,
            0,
        ));
        let second = map.push_map_point(MapPoint::new(
            Vec3F64::new(0.0, 0.0, 5.0),
            [1; 32],
            0,
            [0; 3],
            0,
            0,
        ));

        assert!(map.merge_map_points(first, first).is_none());
        assert!(map.merge_map_points(first, usize::MAX).is_none());
        map.map_points_mut()[second].mark_culled();
        assert!(map.merge_map_points(first, second).is_none());
    }

    // ── Scale-invariance state (T1: deterministic, cross-checked vs ORB-SLAM3
    //    formulas; ORB-SLAM3 itself not needed since these are closed-form). ──

    /// T1a: `predict_scale` matches ORB-SLAM3's
    /// `nScale = clamp(ceil(log(maxDist/dist) / log(scaleFactor)), 0, nLevels-1)`.
    #[test]
    fn predict_scale_matches_orbslam3_closed_form() {
        let mut mp = MapPoint::new(Vec3F64::new(0.0, 0.0, 1.0), [0u8; 32], 0, [0; 3], 0, 0);
        mp.max_distance = 10.0;
        let sf = ORB_SCALE_FACTOR;
        let n = ORB_N_LEVELS;

        for &dist in &[40.0_f64, 12.0, 10.0, 8.0, 3.7, 0.9, 0.01] {
            let want = {
                let ratio = mp.max_distance / dist;
                let lvl = (ratio.ln() / sf.ln()).ceil();
                if lvl.is_nan() || lvl < 0.0 {
                    0
                } else {
                    (lvl as usize).min(n - 1)
                }
            };
            assert_eq!(mp.predict_scale(dist, sf, n), want, "dist={dist}");
        }

        // Degenerate inputs return level 0.
        assert_eq!(mp.predict_scale(0.0, sf, n), 0);
        let mut unset = MapPoint::new(Vec3F64::ZERO, [0u8; 32], 0, [0; 3], 0, 0);
        unset.max_distance = 0.0;
        assert_eq!(unset.predict_scale(5.0, sf, n), 0);
    }

    /// T1b: after `update_map_point_geometry`,
    ///   max_distance == dist_to_ref_kf * scaleFactor^reference_octave
    ///   max_distance / min_distance == scaleFactor^(n_levels - 1)
    #[test]
    fn scale_geometry_distance_invariants() {
        let mut map = Map::new();
        // Reference keyframe 0 at the world origin (identity pose => camera
        // center at origin).
        map.upsert_keyframe(Keyframe::from_frame(test_frame(0, vec![[0u8; 32]])));

        // Point referenced to KF 0, keypoint detected at octave 2, world (0,0,5).
        let mp_idx = map.push_map_point(MapPoint::new(
            Vec3F64::new(0.0, 0.0, 5.0),
            [0u8; 32],
            2,
            [0; 3],
            0,
            0,
        ));
        map.update_map_point_geometry(mp_idx, ORB_SCALE_FACTOR, ORB_N_LEVELS);

        let mp = &map.map_points()[mp_idx];
        let expected_max = 5.0 * ORB_SCALE_FACTOR.powi(2);
        assert!((mp.max_distance - expected_max).abs() < 1e-9);
        assert!(
            (mp.max_distance / mp.min_distance - ORB_SCALE_FACTOR.powi(ORB_N_LEVELS as i32 - 1))
                .abs()
                < 1e-9
        );
        // Margined bounds carry the 0.8 / 1.2 factors.
        assert!((mp.min_distance_invariance() - 0.8 * mp.min_distance).abs() < 1e-12);
        assert!((mp.max_distance_invariance() - 1.2 * mp.max_distance).abs() < 1e-12);
    }

    /// T1c: mean viewing direction is the average of unit (point - cam_center)
    /// over observing keyframes; ‖normal‖ <= 1, and for a single forward-facing
    /// observation it's exactly (point - cam_center) normalized.
    #[test]
    fn mean_viewing_direction_averages_observations() {
        let mut map = Map::new();
        map.upsert_keyframe(Keyframe::from_frame(test_frame(0, vec![[0u8; 32]])));

        let mp_idx = map.push_map_point(MapPoint::new(
            Vec3F64::new(0.0, 0.0, 5.0),
            [0u8; 32],
            0,
            [0; 3],
            0,
            0,
        ));
        map.update_map_point_geometry(mp_idx, ORB_SCALE_FACTOR, ORB_N_LEVELS);

        // Single observation from the origin looking at (0,0,5): normal = +z.
        let n0 = map.map_points()[mp_idx].mean_viewing_direction;
        assert!(n0.x.abs() < 1e-9 && n0.y.abs() < 1e-9 && (n0.z - 1.0).abs() < 1e-9);

        // Add a second keyframe translated along +x by 1 (world->cam pose has
        // translation -1 along x, so camera center is at (1,0,0)).
        let kf1 = Keyframe::from_frame(test_frame_with_pose(
            1,
            vec![[0u8; 32]],
            Pose3d::new(
                kornia_algebra::Mat3F64::IDENTITY,
                Vec3F64::new(-1.0, 0.0, 0.0),
            ),
        ));
        map.register_observation(mp_idx, &kf1, 0);
        map.upsert_keyframe(kf1);
        map.update_map_point_geometry(mp_idx, ORB_SCALE_FACTOR, ORB_N_LEVELS);

        let n = map.map_points()[mp_idx].mean_viewing_direction;
        // Average of (0,0,1) and (point-(1,0,0)) normalized = (-1,0,5)/sqrt(26).
        let d2 = Vec3F64::new(-1.0, 0.0, 5.0);
        let d2n = d2 / d2.length();
        let expected = Vec3F64::new(
            (n0.x + d2n.x) / 2.0,
            (n0.y + d2n.y) / 2.0,
            (n0.z + d2n.z) / 2.0,
        );
        assert!((n.x - expected.x).abs() < 1e-9);
        assert!((n.y - expected.y).abs() < 1e-9);
        assert!((n.z - expected.z).abs() < 1e-9);
        assert!(n.length() <= 1.0 + 1e-12);
    }
}
