use super::*;
use crate::Frame;
use crate::map::{Keyframe, Map, MapPoint};
use crate::pose_conversion::pose_to_se3;
use crate::sparse_pgo::{Se3Manifold, sparse_pose_graph_optimize};
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
    map.upsert_keyframe(candidate);
    map.upsert_keyframe(query);
    for (index, point) in world.into_iter().enumerate() {
        let map_point = map.push_map_point(MapPoint::new(point, descriptors[index], 0, [0; 3], 0));
        map.get_keyframe_mut(0)
            .unwrap()
            .associate_map_point(index, map_point);
    }
    (map, query_pose)
}

#[test]
fn correspondence_input_filters_unassociated_and_culled_points() {
    let (mut map, _) = synthetic_loop_map();
    map.get_keyframe_mut(0).unwrap().clear_map_point(0);
    let culled = map.get_keyframe(0).unwrap().map_point(1).unwrap();
    map.map_points_mut()[culled].mark_culled();
    let config = LoopVerificationConfig {
        min_correspondences: 1,
        ..LoopVerificationConfig::default()
    };
    let input = verification_input(&map, &camera(), 10, 0, &config).unwrap();
    assert_eq!(input.world_points.len(), 34);
    assert_eq!(input.image_points.len(), 34);
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
    map.upsert_keyframe(Keyframe::from_frame(frame(
        0,
        Pose3d::IDENTITY,
        &[[320.0, 240.0]],
        vec![descriptor],
    )));
    map.upsert_keyframe(Keyframe::from_frame(frame(
        10,
        Pose3d::IDENTITY,
        &[[320.0, 240.0]],
        vec![descriptor],
    )));
    let point = map.push_map_point(MapPoint::new(
        Vec3F64::new(0.0, 0.0, 5.0),
        descriptor,
        0,
        [0; 3],
        0,
    ));
    map.get_keyframe_mut(0)
        .unwrap()
        .associate_map_point(0, point);

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
    assert!(map.map_points()[point].observation_kf_indices.contains(&10));
}

#[test]
fn loop_fusion_merges_consistent_duplicate_points() {
    let mut map = Map::new();
    let descriptor = [9; 32];
    map.upsert_keyframe(Keyframe::from_frame(frame(
        0,
        Pose3d::IDENTITY,
        &[[320.0, 240.0]],
        vec![descriptor],
    )));
    map.upsert_keyframe(Keyframe::from_frame(frame(
        10,
        Pose3d::IDENTITY,
        &[[320.8, 240.0]],
        vec![descriptor],
    )));
    let candidate_point = map.push_map_point(MapPoint::new(
        Vec3F64::new(0.0, 0.0, 5.0),
        descriptor,
        0,
        [0; 3],
        0,
    ));
    let query_point = map.push_map_point(MapPoint::new(
        Vec3F64::new(0.01, 0.0, 5.0),
        descriptor,
        0,
        [0; 3],
        10,
    ));
    map.get_keyframe_mut(0)
        .unwrap()
        .associate_map_point(0, candidate_point);
    map.get_keyframe_mut(10)
        .unwrap()
        .associate_map_point(0, query_point);

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
    map.upsert_keyframe(Keyframe::from_frame(frame(
        0,
        Pose3d::IDENTITY,
        &[[320.0, 240.0]],
        vec![[0; 32]],
    )));
    map.upsert_keyframe(Keyframe::from_frame(frame(
        10,
        Pose3d::IDENTITY,
        &[[320.0, 240.0]],
        vec![[u8::MAX; 32]],
    )));
    let point = map.push_map_point(MapPoint::new(
        Vec3F64::new(0.0, 0.0, 5.0),
        [0; 32],
        0,
        [0; 3],
        0,
    ));
    map.get_keyframe_mut(0)
        .unwrap()
        .associate_map_point(0, point);

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
    map.upsert_keyframe(Keyframe::from_frame(frame(
        0,
        Pose3d::IDENTITY,
        &[[320.0, 240.0]],
        vec![descriptor],
    )));
    map.upsert_keyframe(Keyframe::from_frame(frame(
        10,
        Pose3d::IDENTITY,
        &[[320.0, 240.0]],
        vec![descriptor],
    )));
    let source_point = map.push_map_point(MapPoint::new(
        Vec3F64::new(0.0, 0.0, 5.0),
        descriptor,
        0,
        [0; 3],
        0,
    ));
    let inconsistent_target = map.push_map_point(MapPoint::new(
        Vec3F64::new(1.0, 0.0, 5.0),
        descriptor,
        0,
        [0; 3],
        10,
    ));
    map.get_keyframe_mut(0)
        .unwrap()
        .associate_map_point(0, source_point);
    map.get_keyframe_mut(10)
        .unwrap()
        .associate_map_point(0, inconsistent_target);

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
        map.upsert_keyframe(Keyframe::from_frame(frame(index, pose, &[], Vec::new())));
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
        map.upsert_keyframe(Keyframe::from_frame(frame(index, pose, &[], Vec::new())));
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
