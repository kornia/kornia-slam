use super::*;
use crate::Frame;
use crate::loop_closure::fusion_plan::{FusionOp, FusionSnapshot, apply_fusion};
use crate::loop_closure::sparse_pgo::{Se3Manifold, sparse_pose_graph_optimize};
use crate::mapping::map::{Keyframe, LandmarkSeed, Map, ObservationKey};
use crate::pose_conversion::pose_to_se3;
use kornia_3d::camera::PinholeCamera;
use kornia_3d::pgo::{PgoEdge, PgoParams, pose_graph_optimize};
use kornia_3d::pnp::RansacParams;
use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, SO3F64, Vec3F64};
use kornia_image::ImageSize;
use kornia_imgproc::features::OrbFeatures;

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

fn frame(idx: usize, pose: Pose3d, points: &[[f32; 2]], descriptors: Vec<[u8; 32]>) -> Frame {
    Frame {
        idx,
        features: OrbFeatures {
            keypoints_xy: points.to_vec(),
            orientations: vec![0.0; points.len()],
            descriptors,
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
        keypoints_undist: points.to_vec(),
        rig: Default::default(),
    }
}

fn descriptor(index: usize) -> [u8; 32] {
    let mut descriptor = [0_u8; 32];
    for (byte_index, value) in descriptor.iter_mut().enumerate() {
        *value = (index.wrapping_mul(37).wrapping_add(byte_index * 13)) as u8;
    }
    descriptor
}

fn verified_edge(
    query_kf_idx: usize,
    candidate_kf_idx: usize,
    inliers: usize,
    inlier_ratio: f32,
    reprojection_rmse_px: f32,
    occupied_cells: usize,
) -> VerifiedLoopEdge {
    VerifiedLoopEdge {
        query_kf_idx,
        candidate_kf_idx,
        candidate_to_query: Pose3d::IDENTITY,
        correspondences: inliers + 5,
        inliers,
        inlier_ratio,
        reprojection_rmse_px,
        occupied_cells,
    }
}

fn synthetic_loop_map() -> (Map, Pose3d) {
    let camera = camera();
    let query_pose = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.15, -0.05, 0.1));
    let mut map = Map::new();
    let mut world = Vec::new();
    let mut query_pixels = Vec::new();
    let mut candidate_pixels = Vec::new();
    let mut descriptors = Vec::new();
    for index in 0..36 {
        let x = (index % 6) as f64 * 0.35 - 0.875;
        let y = (index / 6) as f64 * 0.28 - 0.7;
        let point = Vec3F64::new(x, y, 4.0 + (index % 3) as f64 * 0.2);
        let query_point = query_pose.transform_point(&point);
        query_pixels.push([
            (camera.fx * query_point.x / query_point.z + camera.cx) as f32,
            (camera.fy * query_point.y / query_point.z + camera.cy) as f32,
        ]);
        candidate_pixels.push([
            (camera.fx * point.x / point.z + camera.cx) as f32,
            (camera.fy * point.y / point.z + camera.cy) as f32,
        ]);
        world.push(point);
        descriptors.push(descriptor(index));
    }
    let candidate = Keyframe::from_frame(frame(
        0,
        Pose3d::IDENTITY,
        &candidate_pixels,
        descriptors.clone(),
    ));
    let query = Keyframe::from_frame(frame(10, query_pose, &query_pixels, descriptors.clone()));
    map.insert_keyframe(candidate).unwrap();
    map.insert_keyframe(query).unwrap();
    for (index, point) in world.into_iter().enumerate() {
        map.insert_landmark(LandmarkSeed {
            position: point,
            color: [0; 3],
            reference: ObservationKey {
                keyframe_idx: 0,
                feature_idx: index,
            },
        })
        .unwrap();
    }
    (map, query_pose)
}

#[test]
fn correspondence_input_filters_unassociated_and_culled_points() {
    let (mut map, _) = synthetic_loop_map();
    map.unlink_observation(0, 0).unwrap();
    let culled = map.get_keyframe(0).unwrap().map_point(1).unwrap();
    map.remove_landmark(culled).unwrap();
    let config = LoopVerificationConfig {
        min_correspondences: 1,
        ..LoopVerificationConfig::default()
    };
    let input = verification_input(&map, &camera(), 10, 0, &config).unwrap();
    assert_eq!(input.world_points.len(), 34);
    assert_eq!(input.image_points.len(), 34);
}

#[test]
fn default_verification_sampling_is_reproducible() {
    assert!(
        LoopVerificationConfig::default()
            .pnp_ransac
            .random_seed
            .is_some()
    );
}

#[test]
fn verify_loop_recovers_metric_relative_pose() {
    let (map, expected_query_pose) = synthetic_loop_map();
    let config = LoopVerificationConfig {
        min_correspondences: 20,
        min_inliers: 20,
        min_inlier_ratio: 0.8,
        min_occupied_cells: 4,
        pnp_ransac: RansacParams {
            random_seed: Some(7),
            ..LoopVerificationConfig::default().pnp_ransac
        },
        ..LoopVerificationConfig::default()
    };
    let verified = verify_loop_candidate(&map, &camera(), 10, 0, &config).unwrap();
    assert!(verified.inliers >= 20);
    assert!(verified.reprojection_rmse_px < 1e-2);
    assert!(
        (verified.candidate_to_query.translation - expected_query_pose.translation).length() < 1e-2
    );
}

#[test]
fn loop_episode_requires_consistency_and_selects_strongest_edge() {
    let mut tracker = LoopEpisodeTracker::new(LoopEpisodeConfig::default());
    let first = verified_edge(100, 20, 31, 0.70, 1.6, 6);
    let strongest = verified_edge(108, 24, 42, 0.84, 1.4, 8);
    let third = verified_edge(116, 16, 36, 0.90, 1.1, 9);

    assert!(matches!(
        tracker.observe(20, 5, first),
        LoopEpisodeDecision::Pending {
            hits: 1,
            required: 3
        }
    ));
    assert!(matches!(
        tracker.observe(21, 6, strongest.clone()),
        LoopEpisodeDecision::Pending {
            hits: 2,
            required: 3
        }
    ));
    assert_eq!(
        tracker.observe(23, 4, third),
        LoopEpisodeDecision::Ready {
            representative: strongest,
            hits: 3,
        }
    );
}

#[test]
fn loop_episode_suppresses_redundant_edges_after_ready() {
    let mut tracker = LoopEpisodeTracker::new(LoopEpisodeConfig {
        min_consistent_edges: 2,
        ..LoopEpisodeConfig::default()
    });
    let representative = verified_edge(100, 20, 35, 0.8, 1.2, 8);
    tracker.observe(20, 5, representative.clone());
    assert!(matches!(
        tracker.observe(21, 6, verified_edge(108, 24, 32, 0.8, 1.3, 7)),
        LoopEpisodeDecision::Ready { .. }
    ));
    assert_eq!(
        tracker.observe(24, 7, verified_edge(120, 28, 45, 0.9, 1.0, 9)),
        LoopEpisodeDecision::Suppressed {
            representative_query_kf_idx: representative.query_kf_idx,
            representative_candidate_kf_idx: representative.candidate_kf_idx,
        }
    );
}

#[test]
fn loop_episode_resets_after_query_gap_or_candidate_jump() {
    let config = LoopEpisodeConfig {
        min_consistent_edges: 3,
        max_query_gap: 3,
        candidate_neighborhood_radius: 4,
    };
    let mut tracker = LoopEpisodeTracker::new(config);
    tracker.observe(10, 5, verified_edge(50, 20, 35, 0.8, 1.2, 8));
    assert!(matches!(
        tracker.observe(14, 6, verified_edge(70, 24, 36, 0.8, 1.2, 8)),
        LoopEpisodeDecision::Pending { hits: 1, .. }
    ));
    assert!(matches!(
        tracker.observe(15, 20, verified_edge(75, 100, 37, 0.8, 1.2, 8)),
        LoopEpisodeDecision::Pending { hits: 1, .. }
    ));
}

fn direct_loop_edge() -> VerifiedLoopEdge {
    verified_edge(10, 0, 35, 0.8, 1.0, 8)
}

#[test]
fn sparse_se3_pgo_matches_dense_on_rotated_drifting_loop() {
    let poses = (0..5)
        .map(|index| {
            let progress = index as f64;
            let rotation = SO3F64::exp(Vec3F64::new(
                0.015 * progress,
                -0.01 * progress,
                0.04 * progress,
            ))
            .matrix();
            let center = Vec3F64::new(progress * 0.8, progress * 0.12, progress * -0.05);
            Pose3d::new(rotation, -(rotation * center))
        })
        .collect::<Vec<_>>();
    let mut edges = (0..poses.len() - 1)
        .map(|index| {
            let measurement = Pose3d::between(&poses[index], &poses[index + 1]);
            PgoEdge {
                pose_a: index,
                pose_b: index + 1,
                t_ab_meas: pose_to_se3(&measurement),
                weight: 1.0,
            }
        })
        .collect::<Vec<_>>();
    edges.push(PgoEdge {
        pose_a: 0,
        pose_b: poses.len() - 1,
        t_ab_meas: pose_to_se3(&Pose3d::IDENTITY),
        weight: 0.5,
    });
    let params = PgoParams::default();
    let initial_cost = pose_graph_cost(&poses, &edges);

    let dense = pose_graph_optimize(&poses, &edges, &[0], &params).unwrap();
    let sparse = sparse_pose_graph_optimize(&poses, &edges, &[0], &params, &Se3Manifold).unwrap();

    assert!(dense.converged);
    assert!(sparse.converged);
    assert_eq!(dense.poses.len(), sparse.poses.len());
    assert!(pose_graph_cost(&dense.poses, &edges) < initial_cost);
    assert!(pose_graph_cost(&sparse.poses, &edges) < initial_cost);
    assert_eq!(dense.poses[0], poses[0]);
    assert_eq!(sparse.poses[0], poses[0]);
    for (dense_pose, sparse_pose) in dense.poses.iter().zip(&sparse.poses) {
        let translation_difference = (dense_pose.translation - sparse_pose.translation).length();
        let dense_rotation = SO3F64::from_matrix(&dense_pose.rotation);
        let sparse_rotation = SO3F64::from_matrix(&sparse_pose.rotation);
        let rotation_difference = dense_rotation.rminus(&sparse_rotation).length();
        assert!(
            translation_difference <= 2e-3,
            "translation difference {translation_difference} exceeds tolerance"
        );
        assert!(
            rotation_difference <= 2e-3,
            "rotation difference {rotation_difference} exceeds tolerance"
        );
    }
}

#[test]
fn loop_fusion_attaches_a_point_to_an_unassociated_loop_keypoint() {
    let mut map = Map::new();
    let descriptor = [7; 32];
    map.insert_keyframe(Keyframe::from_frame(frame(
        0,
        Pose3d::IDENTITY,
        &[[320.0, 240.0]],
        vec![descriptor],
    )))
    .unwrap();
    map.insert_keyframe(Keyframe::from_frame(frame(
        10,
        Pose3d::IDENTITY,
        &[[320.0, 240.0]],
        vec![descriptor],
    )))
    .unwrap();
    let point = map
        .insert_landmark(LandmarkSeed {
            position: Vec3F64::new(0.0, 0.0, 5.0),
            color: [0; 3],
            reference: ObservationKey {
                keyframe_idx: 0,
                feature_idx: 0,
            },
        })
        .unwrap();

    let stats = fuse_verified_loop(
        &mut map,
        &camera(),
        &direct_loop_edge(),
        &LoopFusionConfig {
            max_neighbors_per_side: 0,
            ..LoopFusionConfig::default()
        },
    );

    assert_eq!(stats.observations_added, 1);
    assert_eq!(stats.map_points_merged, 0);
    assert_eq!(map.get_keyframe(10).unwrap().map_point(0), Some(point));
    assert!(map.map_points()[point].is_observed_by(10));
}

#[test]
fn loop_fusion_merges_consistent_duplicate_points() {
    let mut map = Map::new();
    let descriptor = [9; 32];
    map.insert_keyframe(Keyframe::from_frame(frame(
        0,
        Pose3d::IDENTITY,
        &[[320.0, 240.0]],
        vec![descriptor],
    )))
    .unwrap();
    map.insert_keyframe(Keyframe::from_frame(frame(
        10,
        Pose3d::IDENTITY,
        &[[320.8, 240.0]],
        vec![descriptor],
    )))
    .unwrap();
    // Seeded through the canonical API so each association has a matching
    // observation record; merge redirects records, not bare slots.
    let candidate_point = map
        .insert_landmark(LandmarkSeed {
            position: Vec3F64::new(0.0, 0.0, 5.0),
            color: [0; 3],
            reference: ObservationKey {
                keyframe_idx: 0,
                feature_idx: 0,
            },
        })
        .unwrap();
    let query_point = map
        .insert_landmark(LandmarkSeed {
            position: Vec3F64::new(0.01, 0.0, 5.0),
            color: [0; 3],
            reference: ObservationKey {
                keyframe_idx: 10,
                feature_idx: 0,
            },
        })
        .unwrap();

    let stats = fuse_verified_loop(
        &mut map,
        &camera(),
        &direct_loop_edge(),
        &LoopFusionConfig {
            max_neighbors_per_side: 0,
            ..LoopFusionConfig::default()
        },
    );

    assert_eq!(stats.observations_added, 0);
    assert_eq!(stats.map_points_merged, 1);
    assert!(!map.map_points()[candidate_point].culled);
    assert!(map.map_points()[query_point].culled);
    assert_eq!(
        map.get_keyframe(10).unwrap().map_point(0),
        Some(candidate_point)
    );
}

#[test]
fn loop_fusion_rejects_descriptor_mismatch() {
    let mut map = Map::new();
    map.insert_keyframe(Keyframe::from_frame(frame(
        0,
        Pose3d::IDENTITY,
        &[[320.0, 240.0]],
        vec![[0; 32]],
    )))
    .unwrap();
    map.insert_keyframe(Keyframe::from_frame(frame(
        10,
        Pose3d::IDENTITY,
        &[[320.0, 240.0]],
        vec![[u8::MAX; 32]],
    )))
    .unwrap();
    map.insert_landmark(LandmarkSeed {
        position: Vec3F64::new(0.0, 0.0, 5.0),
        color: [0; 3],
        reference: ObservationKey {
            keyframe_idx: 0,
            feature_idx: 0,
        },
    })
    .unwrap();

    let stats = fuse_verified_loop(
        &mut map,
        &camera(),
        &direct_loop_edge(),
        &LoopFusionConfig {
            max_neighbors_per_side: 0,
            ..LoopFusionConfig::default()
        },
    );

    assert_eq!(stats, LoopFusionStats::default());
    assert_eq!(map.get_keyframe(10).unwrap().map_point(0), None);
}

#[test]
fn loop_fusion_rejects_duplicate_with_inconsistent_reciprocal_projection() {
    let mut map = Map::new();
    let descriptor = [5; 32];
    map.insert_keyframe(Keyframe::from_frame(frame(
        0,
        Pose3d::IDENTITY,
        &[[320.0, 240.0]],
        vec![descriptor],
    )))
    .unwrap();
    map.insert_keyframe(Keyframe::from_frame(frame(
        10,
        Pose3d::IDENTITY,
        &[[320.0, 240.0]],
        vec![descriptor],
    )))
    .unwrap();
    let source_point = map
        .insert_landmark(LandmarkSeed {
            position: Vec3F64::new(0.0, 0.0, 5.0),
            color: [0; 3],
            reference: ObservationKey {
                keyframe_idx: 0,
                feature_idx: 0,
            },
        })
        .unwrap();
    let inconsistent_target = map
        .insert_landmark(LandmarkSeed {
            position: Vec3F64::new(1.0, 0.0, 5.0),
            color: [0; 3],
            reference: ObservationKey {
                keyframe_idx: 10,
                feature_idx: 0,
            },
        })
        .unwrap();

    let stats = fuse_verified_loop(
        &mut map,
        &camera(),
        &direct_loop_edge(),
        &LoopFusionConfig {
            max_neighbors_per_side: 0,
            ..LoopFusionConfig::default()
        },
    );

    assert_eq!(stats, LoopFusionStats::default());
    assert!(!map.map_points()[source_point].culled);
    assert!(!map.map_points()[inconsistent_target].culled);
}

#[test]
fn pgo_reduces_terminal_loop_gap_without_moving_anchor() {
    let mut map = Map::new();
    for index in 0..5 {
        let pose = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(-(index as f64), 0.0, 0.0));
        map.insert_keyframe(Keyframe::from_frame(frame(index, pose, &[], Vec::new())))
            .unwrap();
    }
    let verified = VerifiedLoopEdge {
        query_kf_idx: 4,
        candidate_kf_idx: 0,
        candidate_to_query: Pose3d::IDENTITY,
        correspondences: 40,
        inliers: 35,
        inlier_ratio: 0.875,
        reprojection_rmse_px: 1.0,
        occupied_cells: 8,
    };
    let result = optimize_pose_graph(&map, &[verified], &PgoConfig::default(), None).unwrap();
    assert_eq!(result.optimized_poses[0], result.original_poses[0]);
    assert!(result.iterations > 0);
    assert!(result.usable);
    let before = result.original_poses[4].inverse().translation.length();
    let after = result.optimized_poses[4].inverse().translation.length();
    assert!(after < before, "expected {after} < {before}");
}

#[test]
fn inertial_pgo_uses_four_dof_and_preserves_gravity() {
    let gravity = Vec3F64::new(0.3, -9.4, 1.1);
    let gravity_axis = gravity.normalize();
    let tilt = kornia_algebra::SO3F64::exp(Vec3F64::new(0.2, -0.1, 0.05)).matrix();
    let mut map = Map::new();
    for index in 0..5 {
        let yaw = index as f64 * 0.03;
        let yaw_world = kornia_algebra::SO3F64::exp(gravity_axis * -yaw).matrix();
        let rotation = tilt * yaw_world;
        let center = Vec3F64::new(index as f64, index as f64 * 0.05, 0.0);
        let pose = Pose3d::new(rotation, -(rotation * center));
        map.insert_keyframe(Keyframe::from_frame(frame(index, pose, &[], Vec::new())))
            .unwrap();
    }
    let verified = VerifiedLoopEdge {
        query_kf_idx: 4,
        candidate_kf_idx: 0,
        candidate_to_query: Pose3d::IDENTITY,
        correspondences: 40,
        inliers: 35,
        inlier_ratio: 0.875,
        reprojection_rmse_px: 1.0,
        occupied_cells: 8,
    };

    let result = optimize_pose_graph(
        &map,
        &[verified],
        &PgoConfig::default(),
        Some(InertialPgoContext {
            gravity_world: gravity,
        }),
    )
    .unwrap();

    assert_eq!(result.optimized_poses[0], result.original_poses[0]);
    assert!(result.iterations > 0);
    assert!(result.usable);
    assert!(
        max_gravity_alignment_error(&result.original_poses, &result.optimized_poses, gravity,)
            <= 1e-4
    );
}

fn closing_config() -> LoopClosingConfig {
    LoopClosingConfig {
        episode: LoopEpisodeConfig {
            min_consistent_edges: 1,
            ..LoopEpisodeConfig::default()
        },
        verification: LoopVerificationConfig {
            min_correspondences: 20,
            min_inliers: 20,
            min_inlier_ratio: 0.8,
            min_occupied_cells: 4,
            pnp_ransac: RansacParams {
                random_seed: Some(7),
                ..LoopVerificationConfig::default().pnp_ransac
            },
            ..LoopVerificationConfig::default()
        },
        // The fixture drifts 0.6 m over a 0.76 m path, far beyond odometry drift.
        odometry_gate: LoopOdometryGate::Disabled,
        ..LoopClosingConfig::default()
    }
}

/// Tracking state at the drifted query keyframe 10.
fn closing_context(map: &Map) -> LoopClosingContext {
    LoopClosingContext {
        pose_world_to_cam: map.get_keyframe(10).unwrap().frame.pose_world_to_cam,
        velocity_world: Vec3F64::ZERO,
        current_keyframe_idx: Some(10),
        imu_initialized: false,
        gravity_world: Vec3F64::ZERO,
    }
}

/// The synthetic loop with keyframe 10 drifted, a vocabulary trained on its
/// descriptors, and keyframe 0 already indexed.
fn closing_fixture(config: LoopClosingConfig, require_imu_initialized: bool) -> (LoopCloser, Map) {
    let (mut map, _) = synthetic_loop_map();
    // The observed pixels still describe the true pose, but odometry has drifted.
    let mut drifted = map.get_keyframe(10).unwrap().frame.pose_world_to_cam;
    drifted.translation.x += 0.6;
    map.set_keyframe_pose_for_test(10, drifted);
    let descriptors: Vec<_> = map
        .get_keyframe(0)
        .unwrap()
        .frame
        .features
        .descriptors
        .iter()
        .map(kornia_bow::orb_slam3::pack_orb_descriptor)
        .collect();
    let vocabulary =
        crate::loop_closure::place_recognition::Vocabulary::train(&descriptors, 1).unwrap();
    let mut closer = LoopCloser::new(vocabulary, config, require_imu_initialized);
    let context = closing_context(&map);
    let first = closer.on_keyframe(&mut map, &camera(), 0, context);
    assert!(first.events.is_empty());
    assert!(first.debug_message.is_none());
    assert_eq!(closer.indexed_keyframes(), 1);
    (closer, map)
}

#[test]
fn loop_closer_imu_gate_preserves_indexing_without_accepting_loop() {
    let (mut closer, mut map) = closing_fixture(closing_config(), true);
    let context = closing_context(&map);
    let outcome = closer.on_keyframe(&mut map, &camera(), 10, context);
    assert_eq!(closer.indexed_keyframes(), 2);
    assert!(outcome.debug_message.is_some());
    assert!(outcome.events.is_empty());
    assert!(outcome.tracking_correction.is_none());
    assert_eq!(closer.verified_loop_count(), 0);
}

#[test]
fn loop_closer_applies_map_correction_and_returns_tracking_correction() {
    let mut config = closing_config();
    // Make this synthetic loop strong enough to bring projections within the
    // unchanged fusion gates; the default weight only partially corrects drift.
    config.optimizer.loop_edge_weight = 100.0;
    config.information = LoopGraphInformation::ScalarWeights;
    let (mut closer, mut map) = closing_fixture(config, false);
    let before = map.get_keyframe(10).unwrap().frame.pose_world_to_cam;
    let anchor = map.get_keyframe(0).unwrap().frame.pose_world_to_cam;
    let context = closing_context(&map);
    let outcome = closer.on_keyframe(&mut map, &camera(), 10, context);
    assert!(matches!(
        outcome.events.as_slice(),
        [LoopClosureEvent::Accepted { applied: true, .. }]
    ));
    assert!(outcome.pgo_applied);
    let (corrected_pose, _) = outcome.tracking_correction.unwrap();
    let after = map.get_keyframe(10).unwrap().frame.pose_world_to_cam;
    // The tracking pose sat on keyframe 10, so it follows the keyframe.
    assert!((corrected_pose.translation - after.translation).length() < 1e-9);
    assert!((after.translation - before.translation).length() > 0.01);
    assert_eq!(map.get_keyframe(0).unwrap().frame.pose_world_to_cam, anchor);
    assert!(map.get_keyframe(10).unwrap().num_associated_points() > 0);
    assert_eq!(closer.verified_loop_count(), 1);
}

#[test]
fn loop_closer_skips_an_already_verified_pair() {
    let (mut closer, mut map) = closing_fixture(closing_config(), false);
    closer.mark_verified_for_test(0, 10);
    let before = map.get_keyframe(10).unwrap().frame.pose_world_to_cam;
    let context = closing_context(&map);
    let outcome = closer.on_keyframe(&mut map, &camera(), 10, context);
    assert!(outcome.debug_message.is_some());
    assert!(outcome.events.is_empty());
    assert!(outcome.tracking_correction.is_none());
    assert_eq!(closer.verified_loop_count(), 0);
    assert_eq!(
        map.get_keyframe(10).unwrap().frame.pose_world_to_cam,
        before
    );
}

#[test]
fn loop_closer_missing_reference_records_acceptance_without_mutating_map() {
    let (mut closer, mut map) = closing_fixture(closing_config(), false);
    let before = map.get_keyframe(10).unwrap().frame.pose_world_to_cam;
    let context = LoopClosingContext {
        current_keyframe_idx: Some(999),
        ..closing_context(&map)
    };
    let outcome = closer.on_keyframe(&mut map, &camera(), 10, context);
    assert!(matches!(outcome.events.as_slice(), [
        LoopClosureEvent::PgoFailed { reason, .. },
        LoopClosureEvent::Accepted { applied: false, .. }
    ] if reason.contains("reference keyframe")));
    assert!(outcome.tracking_correction.is_none());
    assert_eq!(
        map.get_keyframe(10).unwrap().frame.pose_world_to_cam,
        before
    );
    assert_eq!(map.get_keyframe(10).unwrap().num_associated_points(), 0);
    assert_eq!(closer.verified_loop_count(), 1);
}

/// A 24-keyframe square lap with yaw and translation drift and two loop edges, for the golden test.
fn drifting_lap_map(gravity_axis: Vec3F64) -> (Map, Vec<VerifiedLoopEdge>) {
    let tilt = SO3F64::exp(Vec3F64::new(0.04, -0.03, 0.02)).matrix();
    let mut map = Map::new();
    for index in 0..24 {
        let side = index / 6;
        let along = (index % 6) as f64 * 0.5;
        let (x, y) = match side {
            0 => (along, 0.0),
            1 => (3.0, along),
            2 => (3.0 - along, 3.0),
            _ => (0.0, 3.0 - along),
        };
        let drift = index as f64;
        let yaw = side as f64 * std::f64::consts::FRAC_PI_2 + 0.004 * drift;
        let rotation = tilt * SO3F64::exp(gravity_axis * -yaw).matrix();
        let center = Vec3F64::new(x + 0.01 * drift, y - 0.006 * drift, 0.002 * drift);
        let pose = Pose3d::new(rotation, -(rotation * center));
        map.insert_keyframe(Keyframe::from_frame(frame(index, pose, &[], Vec::new())))
            .unwrap();
    }
    let edge = |candidate_kf_idx, query_kf_idx, candidate_to_query| VerifiedLoopEdge {
        query_kf_idx,
        candidate_kf_idx,
        candidate_to_query,
        correspondences: 40,
        inliers: 35,
        inlier_ratio: 0.875,
        reprojection_rmse_px: 1.0,
        occupied_cells: 8,
    };
    let quarter = SO3F64::exp(gravity_axis * -std::f64::consts::FRAC_PI_2).matrix();
    let loops = vec![
        edge(1, 23, Pose3d::new(quarter, Vec3F64::new(0.1, -0.2, 0.05))),
        edge(0, 22, Pose3d::new(quarter, Vec3F64::new(-0.3, 0.4, 0.0))),
    ];
    (map, loops)
}

/// FNV-1a over every bit of the optimizer's output, so any change in arithmetic order shows.
fn pgo_result_fingerprint(result: &PgoResult) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut feed = |bits: u64| {
        for byte in bits.to_le_bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for pose in &result.optimized_poses {
        for column in 0..3 {
            let value = pose.rotation.col(column);
            feed(value.x.to_bits());
            feed(value.y.to_bits());
            feed(value.z.to_bits());
        }
        feed(pose.translation.x.to_bits());
        feed(pose.translation.y.to_bits());
        feed(pose.translation.z.to_bits());
    }
    feed(result.iterations as u64);
    feed(u64::from(result.usable));
    hash
}

#[test]
fn optimize_pose_graph_output_is_bit_identical_to_b1be250() {
    let gravity = Vec3F64::new(0.1, -9.7, 0.4);
    let (map, loops) = drifting_lap_map(gravity.normalize());
    let se3 = optimize_pose_graph(&map, &loops, &PgoConfig::default(), None).unwrap();
    let inertial = optimize_pose_graph(
        &map,
        &loops,
        &PgoConfig::default(),
        Some(InertialPgoContext {
            gravity_world: gravity,
        }),
    )
    .unwrap();
    // Captured at b1be250, before the solver took information-weighted and robust edges.
    assert_eq!((se3.iterations, se3.usable), (9, true));
    assert_eq!((inertial.iterations, inertial.usable), (6, true));
    assert_eq!(pgo_result_fingerprint(&se3), 0xc41d_4b73_a9e1_94a0);
    assert_eq!(pgo_result_fingerprint(&inertial), 0xd892_637a_5121_1385);
}

/// The closer's PoseGraph path with `w * I` information takes the snapshot optimizer's LM path,
/// on both manifolds.
#[test]
fn pose_graph_with_scalar_information_matches_optimize_pose_graph() {
    let gravity = Vec3F64::new(0.1, -9.7, 0.4);
    let (map, loops) = drifting_lap_map(gravity.normalize());
    for inertial in [
        None,
        Some(InertialPgoContext {
            gravity_world: gravity,
        }),
    ] {
        let expected = optimize_pose_graph(&map, &loops, &PgoConfig::default(), inertial).unwrap();
        let actual = optimize_through_graph(
            &map,
            &loops,
            &PgoConfig::default(),
            LoopGraphInformation::ScalarWeights,
            inertial,
        )
        .unwrap();
        assert_eq!(actual.keyframe_indices, expected.keyframe_indices);
        assert_eq!(actual.original_poses, expected.original_poses);
        assert_eq!(actual.usable, expected.usable);
        assert!(actual.usable);
        let largest = actual
            .optimized_poses
            .iter()
            .zip(&expected.optimized_poses)
            .map(|(a, e)| (a.inverse().translation - e.inverse().translation).length())
            .fold(0.0, f64::max);
        assert_eq!(actual.iterations, expected.iterations);
        assert!(largest < 1e-5, "largest centre difference {largest} m");
        let moved = expected
            .original_poses
            .iter()
            .zip(&expected.optimized_poses)
            .map(|(b, a)| (b.inverse().translation - a.inverse().translation).length())
            .fold(0.0, f64::max);
        assert!(moved > 0.05, "the loops must move the lap: {moved} m");
    }
}

/// Map rotations off orthonormal by 1e-6, as bundle adjustment leaves them: the graph must seed
/// its solve with the map's own poses, not with poses chained through `R^T` as an inverse.
#[test]
fn pose_graph_seeds_from_map_poses_that_are_not_quite_rotations() {
    let gravity = Vec3F64::new(0.1, -9.7, 0.4);
    let (clean, loops) = drifting_lap_map(gravity.normalize());
    let mut map = Map::new();
    for (index, keyframe) in clean.keyframes().iter().enumerate() {
        let mut pose = keyframe.frame.pose_world_to_cam;
        let wobble = 1e-6 * ((index * 7919) % 13) as f64 / 13.0;
        let mut columns = pose.rotation.to_cols_array();
        columns[..3]
            .iter_mut()
            .for_each(|value| *value *= 1.0 + wobble);
        columns[6..]
            .iter_mut()
            .for_each(|value| *value *= 1.0 - wobble);
        pose.rotation = Mat3F64::from_cols_array(&columns);
        map.insert_keyframe(Keyframe::from_frame(frame(
            keyframe.frame.idx,
            pose,
            &[],
            Vec::new(),
        )))
        .unwrap();
    }
    let inertial = Some(InertialPgoContext {
        gravity_world: gravity,
    });
    let expected = optimize_pose_graph(&map, &loops, &PgoConfig::default(), inertial).unwrap();
    let actual = optimize_through_graph(
        &map,
        &loops,
        &PgoConfig::default(),
        LoopGraphInformation::ScalarWeights,
        inertial,
    )
    .unwrap();
    assert_eq!(actual.iterations, expected.iterations);
    assert_eq!(actual.optimized_poses, expected.optimized_poses);
}

/// A 24-keyframe square lap, 0.5 m apart, drifting `drift` rad of yaw and a few mm per
/// keyframe about gravity; `drift = 0` is the truth.
fn square_lap(gravity_axis: Vec3F64, drift: f64) -> Vec<Pose3d> {
    (0..24)
        .map(|index| {
            let side = index / 6;
            let along = (index % 6) as f64 * 0.5;
            let (x, y) = match side {
                0 => (along, 0.0),
                1 => (3.0, along),
                2 => (3.0 - along, 3.0),
                _ => (0.0, 3.0 - along),
            };
            let walked = index as f64 * drift;
            let yaw = side as f64 * std::f64::consts::FRAC_PI_2 + walked;
            let rotation = SO3F64::exp(gravity_axis * -yaw).matrix();
            let centre = Vec3F64::new(x + 2.5 * walked, y - 1.5 * walked, 0.0);
            Pose3d::new(rotation, -(rotation * centre))
        })
        .collect()
}

/// Drift-model information with a loop measured from the truth: the query keyframe moves
/// towards the truth, but only a little. The keyframes are 0.5 m apart, so each odometry edge
/// sits on the 0.01 m / 0.002 rad floor and the lap chains to ~0.05 m / 0.01 rad, stiffer than
/// the loop's 0.10 m / 1 deg; the 5 deg correction starts near 6 sigma, where Cauchy at 3 keeps
/// a fifth of its weight. Measured: 0.268 m -> 0.255 m.
#[test]
fn drift_information_pulls_a_true_loop_towards_the_truth() {
    let gravity = Vec3F64::new(0.0, -9.81, 0.0);
    let truth = square_lap(gravity.normalize(), 0.0);
    let mut map = Map::new();
    for (index, pose) in square_lap(gravity.normalize(), 0.004)
        .into_iter()
        .enumerate()
    {
        map.insert_keyframe(Keyframe::from_frame(frame(index, pose, &[], Vec::new())))
            .unwrap();
    }
    let edge = VerifiedLoopEdge {
        candidate_to_query: Pose3d::between(&truth[0], &truth[23]),
        ..verified_edge(23, 0, 40, 0.875, 1.0, 8)
    };
    let result = optimize_through_graph(
        &map,
        std::slice::from_ref(&edge),
        &PgoConfig::default(),
        LoopGraphInformation::default(),
        Some(InertialPgoContext {
            gravity_world: gravity,
        }),
    )
    .unwrap();
    let error = |poses: &[Pose3d]| {
        (poses[23].inverse().translation - truth[23].inverse().translation).length()
    };
    let (before, after) = (
        error(&result.original_poses),
        error(&result.optimized_poses),
    );
    assert!(result.usable);
    assert!(after < before, "end error {before} m -> {after} m");
}

/// A loop keeps constraining the map after its candidate keyframe is culled: the persistent
/// graph still holds the node, where the snapshot optimizer loses the loop.
#[test]
fn persistent_graph_keeps_a_loop_whose_keyframe_was_culled() {
    let gravity = Vec3F64::new(0.0, -9.81, 0.0);
    let truth = square_lap(gravity.normalize(), 0.0);
    let mut map = Map::new();
    for (index, pose) in square_lap(gravity.normalize(), 0.004)
        .into_iter()
        .enumerate()
    {
        map.insert_keyframe(Keyframe::from_frame(frame(index, pose, &[], Vec::new())))
            .unwrap();
    }
    let loop_from = |candidate: usize, query: usize| VerifiedLoopEdge {
        candidate_to_query: Pose3d::between(&truth[candidate], &truth[query]),
        ..verified_edge(query, candidate, 40, 0.875, 1.0, 8)
    };
    let (first, second) = (loop_from(1, 23), loop_from(0, 22));
    let config = PgoConfig::default();
    let information = LoopGraphInformation::default();
    let inertial = Some(InertialPgoContext {
        gravity_world: gravity,
    });
    let mut graph = PoseGraph::new(snapshot_graph_config(&config, information, inertial));
    let apply = |map: &mut Map, result: &PgoResult| {
        map.apply_pose_graph_correction(
            &result.keyframe_indices,
            &result.original_poses,
            &result.optimized_poses,
        )
        .unwrap();
    };

    let (result, new_world) = solve_in_graph(
        &mut graph,
        &keyframe_poses(&map),
        std::slice::from_ref(&first),
        &config,
        information,
        inertial,
    )
    .unwrap();
    assert!(!new_world && result.usable);
    apply(&mut map, &result);
    map.remove_keyframe(1).unwrap();

    assert!(matches!(
        optimize_pose_graph(&map, &[first, second.clone()], &config, inertial),
        Err(PgoError::MissingLoopKeyframe)
    ));
    let (result, _) = solve_in_graph(
        &mut graph,
        &keyframe_poses(&map),
        std::slice::from_ref(&second),
        &config,
        information,
        inertial,
    )
    .unwrap();
    assert_eq!(result.keyframe_indices.len(), 23);
    assert_eq!(graph.nodes().len(), 24);
    assert_eq!(graph.node_of_keyframe(1), None);
    assert_eq!(
        graph
            .edges()
            .filter(|edge| edge.kind == EdgeKind::Loop)
            .count(),
        2
    );
    apply(&mut map, &result);
}

#[test]
fn loop_closer_gate_rejects_correction_beyond_odometry_drift() {
    let config = LoopClosingConfig {
        odometry_gate: LoopOdometryGate::default(),
        ..closing_config()
    };
    let (mut closer, mut map) = closing_fixture(config, false);
    let before = map.get_keyframe(10).unwrap().frame.pose_world_to_cam;
    let context = closing_context(&map);
    let outcome = closer.on_keyframe(&mut map, &camera(), 10, context);
    assert!(outcome.events.is_empty());
    assert!(outcome.tracking_correction.is_none());
    assert!(matches!(
        outcome.gate_messages.as_slice(),
        [line] if line.starts_with("[loop-gate] kf=10~0 ") && line.ends_with(" reject")
    ));
    assert_eq!(closer.verified_loop_count(), 0);
    assert_eq!(
        map.get_keyframe(10).unwrap().frame.pose_world_to_cam,
        before
    );
}

/// Out 12 m along x and 6 m back: the last keyframe sits 6 m from the first.
fn out_and_back_map() -> Map {
    let mut map = Map::new();
    let centres = (0..=12).chain((6..12).rev()).map(|x| x as f64);
    for (idx, x) in centres.enumerate() {
        let centre = Vec3F64::new(x, if idx > 12 { 0.5 } else { 0.0 }, 0.0);
        let pose = Pose3d::new(Mat3F64::IDENTITY, -centre);
        map.insert_keyframe(Keyframe::from_frame(frame(idx, pose, &[], Vec::new())))
            .unwrap();
    }
    map
}

fn edge_placing_query_at(map: &Map, query_kf_idx: usize, centre: Vec3F64) -> VerifiedLoopEdge {
    let candidate = map.get_keyframe(0).unwrap().frame.pose_world_to_cam;
    let query = Pose3d::new(Mat3F64::IDENTITY, -centre);
    VerifiedLoopEdge {
        candidate_to_query: Pose3d::between(&candidate, &query),
        ..verified_edge(query_kf_idx, 0, 62, 0.6, 0.73, 10)
    }
}

/// A repeated-structure match claims the last keyframe, 6 m from the first
/// along 18 m of path, is back at the start. Optimizing with that edge drags
/// the trajectory by metres; the gate refuses it and keeps a drift-sized loop.
#[test]
fn false_loop_edge_distorts_pose_graph_and_gate_rejects_it() {
    let map = out_and_back_map();
    let query_kf_idx = map.keyframes().last().unwrap().frame.idx;
    let query = map
        .get_keyframe(query_kf_idx)
        .unwrap()
        .frame
        .pose_world_to_cam;
    let candidate = map.get_keyframe(0).unwrap().frame.pose_world_to_cam;
    let path = || map.keyframes().iter().map(|kf| kf.frame.pose_world_to_cam);

    let false_edge = edge_placing_query_at(&map, query_kf_idx, Vec3F64::new(0.1, 0.0, 0.0));
    let config = PgoConfig {
        loop_edge_weight: 10.0,
        ..PgoConfig::default()
    };
    let distorted =
        optimize_pose_graph(&map, std::slice::from_ref(&false_edge), &config, None).unwrap();
    let largest_shift = distorted
        .original_poses
        .iter()
        .zip(&distorted.optimized_poses)
        .map(|(before, after)| {
            (before.inverse().translation - after.inverse().translation).length()
        })
        .fold(0.0, f64::max);
    assert!(largest_shift > 3.0, "shift {largest_shift}");

    let gate = LoopOdometryGate::default();
    let false_correction = LoopCorrection::measure(candidate, query, path(), &false_edge, None);
    assert!((false_correction.path_m - (17.0 + 1.25_f64.sqrt())).abs() < 1e-9);
    assert!((false_correction.translation_m() - 5.92).abs() < 0.01);
    assert!(!gate.judge(&false_correction).unwrap().accepted);

    let true_edge = edge_placing_query_at(&map, query_kf_idx, Vec3F64::new(6.2, 0.3, 0.0));
    let true_correction = LoopCorrection::measure(candidate, query, path(), &true_edge, None);
    assert!(gate.judge(&true_correction).unwrap().accepted);
}

/// The sim's false loop kf1398~345 in miniature: out 9.65 m along x and 3.65 m back, so the
/// path is 13.3 m and the last keyframe sits 6.0 m from the first, and a verified edge that
/// puts it back at the start.
fn false_loop_fixture() -> (Map, VerifiedLoopEdge) {
    let centres = (0..=28)
        .map(|step| 9.65 * step as f64 / 28.0)
        .chain((0..11).rev().map(|step| 6.0 + 3.65 * step as f64 / 11.0));
    let mut map = Map::new();
    for (index, x) in centres.enumerate() {
        let pose = Pose3d::new(Mat3F64::IDENTITY, -Vec3F64::new(x, 0.0, 0.0));
        map.insert_keyframe(Keyframe::from_frame(frame(index, pose, &[], Vec::new())))
            .unwrap();
    }
    let query_kf_idx = map.keyframes().last().unwrap().frame.idx;
    let edge = edge_placing_query_at(&map, query_kf_idx, Vec3F64::ZERO);
    (map, edge)
}

/// Theory: the gate's 3 sigma at 13.3 m is 3 * (0.019 * 13.3 + 0.10) = 1.06 m, far below 6 m.
/// With the gate off, the Cauchy loop edge starts 60 sigma out and keeps 1/400 of its weight;
/// against the odometry chain (39 edges at 0.01 m) it moves the far end by millimetres, where
/// least squares at `w * I` drags it most of the 6 m.
#[test]
fn false_loop_is_gated_and_cauchy_keeps_the_map_when_the_gate_is_off() {
    let (map, false_edge) = false_loop_fixture();
    let query = map.keyframes().last().unwrap().frame.pose_world_to_cam;
    let candidate = map.get_keyframe(0).unwrap().frame.pose_world_to_cam;
    let path = map.keyframes().iter().map(|kf| kf.frame.pose_world_to_cam);
    let correction = LoopCorrection::measure(candidate, query, path, &false_edge, None);
    assert!((correction.path_m - 13.3).abs() < 1e-9);
    assert!((correction.translation_m() - 6.0).abs() < 1e-9);
    let proposal = LoopProposal {
        edge: &false_edge,
        correction,
        query_order: map.keyframes().len() - 1,
        candidate_order: 0,
    };
    let mut gate = OdometryConsistencyGate(LoopOdometryGate::default());
    assert!(matches!(
        gate.decide(&proposal, &mut Vec::new()),
        GateDecision::Reject(_)
    ));

    let largest_shift = |information| {
        let result = optimize_through_graph(
            &map,
            std::slice::from_ref(&false_edge),
            &PgoConfig::default(),
            information,
            None,
        )
        .unwrap();
        result
            .original_poses
            .iter()
            .zip(&result.optimized_poses)
            .map(|(before, after)| {
                (before.inverse().translation - after.inverse().translation).length()
            })
            .fold(0.0, f64::max)
    };
    let cauchy = largest_shift(LoopGraphInformation::default());
    let least_squares = largest_shift(LoopGraphInformation::ScalarWeights);
    assert!(cauchy < 0.05, "Cauchy shift {cauchy} m");
    assert!(least_squares > 3.0, "least-squares shift {least_squares} m");
}

/// The fixture's loop, strong enough to correct the drift and fuse.
fn background_config() -> LoopClosingConfig {
    let mut config = closing_config();
    config.optimizer.loop_edge_weight = 100.0;
    config.information = LoopGraphInformation::ScalarWeights;
    config
}

/// The synchronous closure of the fixture's loop: the map and the tracking correction.
fn synchronous_closure() -> (Map, (Pose3d, Vec3F64)) {
    let (mut closer, mut map) = closing_fixture(background_config(), false);
    let context = closing_context(&map);
    let outcome = closer.on_keyframe(&mut map, &camera(), 10, context);
    (map, outcome.tracking_correction.unwrap())
}

/// Frame boundaries until a background outcome reports events, at most `limit` of them.
fn frames_until_reported(
    closer: &mut LoopCloser,
    map: &mut Map,
    context: LoopClosingContext,
    limit: usize,
) -> (usize, LoopClosingOutcome) {
    for frame in 1..=limit {
        let outcome = closer.on_frame(map, &camera(), context);
        if !outcome.events.is_empty() {
            return (frame, outcome);
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    panic!("no background outcome within {limit} frames");
}

/// Deferred by three frames, the loop is queued at the keyframe and written back at the third
/// frame boundary, onto an unchanged map exactly as the synchronous path writes it.
#[test]
fn deferred_writeback_matches_the_synchronous_closure() {
    let (expected_map, expected_tracking) = synchronous_closure();
    let (closer, mut map) = closing_fixture(background_config(), false);
    let mut closer = closer.with_solver(LoopSolver::deferred(3));
    let context = closing_context(&map);

    let queued = closer.on_keyframe(&mut map, &camera(), 10, context);
    assert!(queued.queued && queued.events.is_empty() && !queued.pgo_applied);
    let (frames, outcome) = frames_until_reported(&mut closer, &mut map, context, 10);

    assert_eq!(frames, 3);
    assert!(matches!(
        outcome.events.as_slice(),
        [LoopClosureEvent::Accepted { applied: true, .. }]
    ));
    assert_eq!(outcome.tracking_correction, Some(expected_tracking));
    assert_eq!(
        map.state_fingerprint_for_test(),
        expected_map.state_fingerprint_for_test()
    );
}

/// On the worker thread the same closure arrives within 2 s and writes the same map.
#[test]
fn asynchronous_writeback_matches_the_synchronous_closure() {
    let (expected_map, _) = synchronous_closure();
    let (closer, mut map) = closing_fixture(background_config(), false);
    let mut closer = closer.with_solver(LoopSolver::asynchronous().unwrap());
    let context = closing_context(&map);
    let started = std::time::Instant::now();

    assert!(closer.on_keyframe(&mut map, &camera(), 10, context).queued);
    let (_, outcome) = frames_until_reported(&mut closer, &mut map, context, 2000);

    assert!(started.elapsed() < std::time::Duration::from_secs(2));
    assert!(outcome.pgo_applied && outcome.written_back.is_some());
    assert_eq!(
        map.state_fingerprint_for_test(),
        expected_map.state_fingerprint_for_test()
    );
}

/// A world change during the solve discards the result and solves again; after three in a row
/// the loop is reported unapplied and the map keeps its pose.
#[test]
fn a_world_change_during_the_solve_discards_and_retries() {
    let (closer, mut map) = closing_fixture(background_config(), false);
    let mut closer = closer.with_solver(LoopSolver::deferred(1));
    let context = closing_context(&map);
    let drifted = map.get_keyframe(10).unwrap().frame.pose_world_to_cam;

    assert!(closer.on_keyframe(&mut map, &camera(), 10, context).queued);
    for _ in 0..2 {
        map.advance_world_epoch_for_test();
        let outcome = closer.on_frame(&mut map, &camera(), context);
        assert!(outcome.events.is_empty() && outcome.written_back.is_some());
    }
    map.advance_world_epoch_for_test();
    let outcome = closer.on_frame(&mut map, &camera(), context);

    assert!(matches!(outcome.events.as_slice(), [
        LoopClosureEvent::PgoFailed { reason, .. },
        LoopClosureEvent::Accepted { applied: false, .. }
    ] if reason.contains("changed world")));
    assert_eq!(
        map.get_keyframe(10).unwrap().frame.pose_world_to_cam,
        drifted
    );
    assert!(
        closer
            .on_frame(&mut map, &camera(), context)
            .events
            .is_empty()
    );
}

/// One discard is followed by a fresh solve that is applied.
#[test]
fn a_discarded_solve_is_resubmitted_and_applied() {
    let (closer, mut map) = closing_fixture(background_config(), false);
    let mut closer = closer.with_solver(LoopSolver::deferred(1));
    let context = closing_context(&map);

    closer.on_keyframe(&mut map, &camera(), 10, context);
    map.advance_world_epoch_for_test();
    assert!(
        closer
            .on_frame(&mut map, &camera(), context)
            .events
            .is_empty()
    );
    let outcome = closer.on_frame(&mut map, &camera(), context);

    assert!(outcome.pgo_applied);
    assert!(matches!(
        outcome.events.as_slice(),
        [LoopClosureEvent::Accepted { applied: true, .. }]
    ));
}

/// A worker that died turns loop correction off with an event; the map is left as it was.
#[test]
fn a_stopped_worker_turns_loop_correction_off() {
    let (closer, mut map) = closing_fixture(background_config(), false);
    let mut closer = closer.with_solver(LoopSolver::panicking_for_test());
    let context = closing_context(&map);
    let drifted = map.get_keyframe(10).unwrap().frame.pose_world_to_cam;

    assert!(closer.on_keyframe(&mut map, &camera(), 10, context).queued);
    let (_, outcome) = frames_until_reported(&mut closer, &mut map, context, 2000);

    assert!(matches!(outcome.events.as_slice(), [
        LoopClosureEvent::PgoFailed { reason, .. },
        LoopClosureEvent::Accepted { applied: false, .. }
    ] if reason.contains("worker stopped")));
    assert!(!outcome.pgo_applied && outcome.tracking_correction.is_none());
    assert_eq!(
        map.get_keyframe(10).unwrap().frame.pose_world_to_cam,
        drifted
    );
}

/// The correction epoch advances once per correction written, inline or at a writeback, and a
/// deferred solve is in flight until it is written back.
#[test]
fn correction_epoch_counts_written_corrections() {
    let (mut closer, mut map) = closing_fixture(background_config(), false);
    assert_eq!(closer.correction_epoch(), 0);
    let context = closing_context(&map);
    closer.on_keyframe(&mut map, &camera(), 10, context);
    assert_eq!(closer.correction_epoch(), 1);
    assert!(!closer.solve_in_flight());

    let (closer, mut map) = closing_fixture(background_config(), false);
    let mut closer = closer.with_solver(LoopSolver::deferred(2));
    closer.on_keyframe(&mut map, &camera(), 10, context);
    assert!(closer.solve_in_flight());
    closer.on_frame(&mut map, &camera(), context);
    assert_eq!(closer.correction_epoch(), 0);
    closer.on_frame(&mut map, &camera(), context);
    assert_eq!(closer.correction_epoch(), 1);
    assert!(!closer.solve_in_flight());
}

/// A solve cut off before it converges is not written back, and says why: an unusable result is
/// never a silent `applied: false`, inline or in the background.
#[test]
fn an_unusable_solve_reports_why() {
    let mut config = background_config();
    config.optimizer.max_iterations = 1;
    let unusable = |events: &[LoopClosureEvent]| {
        matches!(events, [
            LoopClosureEvent::PgoFailed { reason, .. },
            LoopClosureEvent::Accepted { applied: false, .. }
        ] if reason.starts_with("unusable solve") && reason.contains("not converged"))
    };

    let (mut closer, mut map) = closing_fixture(config.clone(), false);
    let context = closing_context(&map);
    let inline = closer.on_keyframe(&mut map, &camera(), 10, context);
    assert!(unusable(&inline.events), "{:?}", inline.events);

    let (closer, mut map) = closing_fixture(config, false);
    let mut closer = closer.with_solver(LoopSolver::deferred(1));
    let drifted = map.get_keyframe(10).unwrap().frame.pose_world_to_cam;
    assert!(closer.on_keyframe(&mut map, &camera(), 10, context).queued);
    let background = closer.on_frame(&mut map, &camera(), context);
    assert!(unusable(&background.events), "{:?}", background.events);
    assert!(!background.pgo_applied);
    assert_eq!(
        map.get_keyframe(10).unwrap().frame.pose_world_to_cam,
        drifted
    );
}

/// The fixture with the query keyframe holding its own drifted duplicates of the first 18
/// landmarks, so fusing the loop merges them as well as linking the rest.
fn duplicated_fixture(config: LoopClosingConfig) -> (LoopCloser, Map) {
    let (closer, mut map) = closing_fixture(config, false);
    let query = map.get_keyframe(10).unwrap().frame.pose_world_to_cam;
    let true_query = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.15, -0.05, 0.1));
    for feature in 0..18 {
        let landmark = map.get_keyframe(0).unwrap().map_point(feature).unwrap();
        let point = map.map_points()[landmark].position;
        map.insert_landmark(LandmarkSeed {
            position: query.untransform_point(&true_query.transform_point(&point)),
            color: [0; 3],
            reference: ObservationKey {
                keyframe_idx: 10,
                feature_idx: feature,
            },
        })
        .unwrap();
    }
    (closer, map)
}

fn merged_landmarks(map: &Map) -> usize {
    map.map_points().iter().filter(|point| point.culled).count()
}

/// Planned on the worker's snapshot and applied at the writeback, fusion of an unchanged map
/// links and merges exactly what the synchronous closure's `fuse_verified_loop` does.
#[test]
fn planned_fusion_matches_the_synchronous_fusion() {
    let (mut closer, mut expected) = duplicated_fixture(background_config());
    let context = closing_context(&expected);
    assert!(
        closer
            .on_keyframe(&mut expected, &camera(), 10, context)
            .pgo_applied
    );
    assert_eq!(merged_landmarks(&expected), 18);

    let (closer, mut map) = duplicated_fixture(background_config());
    let mut closer = closer.with_solver(LoopSolver::deferred(1));
    assert!(closer.on_keyframe(&mut map, &camera(), 10, context).queued);
    let outcome = closer.on_frame(&mut map, &camera(), context);

    assert!(outcome.pgo_applied);
    assert_eq!(outcome.timing.fusion_ops, 36);
    assert_eq!(outcome.timing.fusion_stale, 0);
    crate::mapping::map::tests::assert_map_consistent(&map);
    assert_eq!(
        map.state_fingerprint_for_test(),
        expected.state_fingerprint_for_test()
    );
}

/// The planned operations, and the map with the solve written back but not yet fused.
fn planned_fusion() -> (Map, Vec<FusionOp>, LoopFusionConfig) {
    let config = background_config();
    let (_, mut map) = duplicated_fixture(config.clone());
    let edge = verify_loop_candidate(&map, &camera(), 10, 0, &config.verification).unwrap();
    let solved = optimize_through_graph(
        &map,
        std::slice::from_ref(&edge),
        &config.optimizer,
        config.information,
        None,
    )
    .unwrap();
    let ops = FusionSnapshot::capture(&map, &camera(), &[edge], &config.fusion).plan(&solved);
    map.apply_graph_correction(
        map.world_epoch(),
        &solved.keyframe_indices,
        &solved.original_poses,
        &solved.optimized_poses,
    )
    .unwrap();
    (map, ops, config.fusion)
}

/// A landmark culled meanwhile, or a slot that changed hands, leaves its operation stale and
/// skipped; the others still apply and the map stays consistent.
#[test]
fn stale_fusion_operations_are_skipped() {
    let (mut map, ops, config) = planned_fusion();
    let link = ops
        .iter()
        .find_map(|op| match *op {
            FusionOp::Link { landmark, .. } => Some(landmark),
            FusionOp::Merge { .. } => None,
        })
        .unwrap();
    let (keyframe, feature) = ops
        .iter()
        .find_map(|op| match *op {
            FusionOp::Merge {
                target_keyframe,
                target_feature,
                ..
            } => Some((target_keyframe, target_feature)),
            FusionOp::Link { .. } => None,
        })
        .unwrap();
    map.remove_landmark(link).unwrap();
    map.unlink_observation(keyframe, feature).unwrap();

    let applied = apply_fusion(&mut map, &camera(), &ops, &config);

    assert_eq!(applied.stale, 2);
    assert_eq!(applied.linked + applied.merged, ops.len() - 2);
    crate::mapping::map::tests::assert_map_consistent(&map);
}

/// A keyframe bundle adjustment moved away after the snapshot no longer puts the planned
/// landmarks on its features, so nothing is fused into it.
#[test]
fn fusion_into_a_keyframe_that_moved_is_skipped() {
    let (mut map, ops, config) = planned_fusion();
    let mut moved = map.get_keyframe(10).unwrap().frame.pose_world_to_cam;
    moved.translation.x += 0.3;
    map.set_keyframe_pose_for_test(10, moved);

    let applied = apply_fusion(&mut map, &camera(), &ops, &config);

    assert_eq!((applied.linked, applied.merged), (0, 0));
    assert_eq!(applied.stale, ops.len());
}

/// The euroc3 Async defect: after an earlier loop's fusion the camera's local map is mostly
/// landmarks of the earlier visit. A later solve moves the reference keyframe but not them, so
/// tracking stays on them; the tracking correction and the map correction must follow them too,
/// not the reference keyframe, or the odometry `M^-1 * T` jumps by the reference's correction.
#[test]
fn a_background_correction_follows_the_landmarks_the_camera_sees() {
    let (closer, mut map) = closing_fixture(background_config(), false);
    let mut closer = closer.with_solver(LoopSolver::deferred(1));
    let context = closing_context(&map);
    let reference_before = map.get_keyframe(10).unwrap().frame.pose_world_to_cam;
    assert!(closer.on_keyframe(&mut map, &camera(), 10, context).queued);
    // Meanwhile local mapping fuses the query keyframe onto the earlier visit's landmarks.
    for feature in 0..36 {
        let landmark = map.get_keyframe(0).unwrap().map_point(feature).unwrap();
        map.link_observation(10, feature, landmark).unwrap();
    }

    let outcome = closer.on_frame(&mut map, &camera(), context);

    assert!(outcome.pgo_applied);
    let reference_after = map.get_keyframe(10).unwrap().frame.pose_world_to_cam;
    let reference_moved =
        (reference_after.inverse().translation - reference_before.inverse().translation).length();
    assert!(reference_moved > 0.3, "reference moved {reference_moved} m");
    let (tracked, _) = outcome.tracking_correction.unwrap();
    let camera_moved =
        (tracked.inverse().translation - context.pose_world_to_cam.inverse().translation).length();
    assert!(camera_moved < 1e-6, "camera moved {camera_moved} m");
    assert!(
        closer
            .map_correction()
            .map_from_odometry
            .translation
            .length()
            < 1e-6
    );
}
