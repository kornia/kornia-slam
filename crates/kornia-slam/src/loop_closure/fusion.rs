use std::collections::HashSet;

use kornia_3d::camera::PinholeCamera;
use kornia_imgproc::features::hamming_distance;

use crate::map::Map;

use super::VerifiedLoopEdge;

/// Bounded landmark search performed after an accepted loop correction.
#[derive(Debug, Clone)]
pub struct LoopFusionConfig {
    pub min_covisibility_weight: usize,
    pub max_neighbors_per_side: usize,
    pub search_radius_px: f32,
    pub max_hamming: u32,
    pub max_reprojection_error_px: f64,
}

impl Default for LoopFusionConfig {
    fn default() -> Self {
        Self {
            min_covisibility_weight: 15,
            max_neighbors_per_side: 5,
            search_radius_px: 7.0,
            max_hamming: 50,
            max_reprojection_error_px: 3.0,
        }
    }
}

/// Live-map changes made while fusing a verified loop.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LoopFusionStats {
    pub observations_added: usize,
    pub map_points_merged: usize,
}

/// Adds and merges landmark observations across a geometrically verified loop.
pub fn fuse_verified_loop(
    map: &mut Map,
    camera: &PinholeCamera,
    edge: &VerifiedLoopEdge,
    config: &LoopFusionConfig,
) -> LoopFusionStats {
    let query_side = loop_side_keyframes(map, edge.query_kf_idx, config);
    let candidate_side = loop_side_keyframes(map, edge.candidate_kf_idx, config);
    let mut stats = LoopFusionStats::default();
    fuse_loop_direction(
        map,
        camera,
        &candidate_side,
        &query_side,
        config,
        &mut stats,
    );
    fuse_loop_direction(
        map,
        camera,
        &query_side,
        &candidate_side,
        config,
        &mut stats,
    );
    stats
}

fn loop_side_keyframes(map: &Map, anchor: usize, config: &LoopFusionConfig) -> Vec<usize> {
    if map.get_keyframe(anchor).is_none() {
        return Vec::new();
    }
    let mut side = vec![anchor];
    side.extend(
        map.covisible_keyframes(anchor, config.min_covisibility_weight)
            .into_iter()
            .take(config.max_neighbors_per_side)
            .map(|(keyframe_idx, _)| keyframe_idx),
    );
    side
}

fn fuse_loop_direction(
    map: &mut Map,
    camera: &PinholeCamera,
    source_keyframes: &[usize],
    target_keyframes: &[usize],
    config: &LoopFusionConfig,
    stats: &mut LoopFusionStats,
) {
    let mut seen_points = HashSet::new();
    let mut source_points = Vec::new();
    for &source_kf_idx in source_keyframes {
        let Some(source) = map.get_keyframe(source_kf_idx) else {
            continue;
        };
        for (source_desc_idx, &map_point_idx) in source.map_point_by_desc_idx.iter().enumerate() {
            let Some(map_point_idx) = map_point_idx else {
                continue;
            };
            if seen_points.insert(map_point_idx) {
                source_points.push((map_point_idx, source_kf_idx, source_desc_idx));
            }
        }
    }

    let search_radius_sq = config.search_radius_px * config.search_radius_px;
    let reprojection_limit_sq = config.max_reprojection_error_px * config.max_reprojection_error_px;
    for &target_kf_idx in target_keyframes {
        for &(source_point_idx, source_kf_idx, source_desc_idx) in &source_points {
            if source_kf_idx == target_kf_idx {
                continue;
            }
            let proposal = {
                let Some(source_point) = map.map_points().get(source_point_idx) else {
                    continue;
                };
                if source_point.culled
                    || source_point.observation_kf_indices.contains(&target_kf_idx)
                {
                    continue;
                }
                let Some(target) = map.get_keyframe(target_kf_idx) else {
                    continue;
                };
                let point_target = target
                    .frame
                    .pose_world_to_cam
                    .transform_point(&source_point.position);
                if point_target.z <= 0.0 {
                    continue;
                }
                let Ok(projected) =
                    camera.project_to_image(&point_target, 0.0, target.frame.image_size)
                else {
                    continue;
                };
                let projected_x = projected.x as f32;
                let projected_y = projected.y as f32;
                let mut best: Option<(usize, u32, f32)> = None;
                for target_desc_idx in 0..target.frame.features.descriptors.len() {
                    let Some(keypoint) = target.frame.undistorted_xy(target_desc_idx, camera)
                    else {
                        continue;
                    };
                    let dx = keypoint[0] - projected_x;
                    let dy = keypoint[1] - projected_y;
                    let distance_sq = dx * dx + dy * dy;
                    if distance_sq > search_radius_sq || distance_sq as f64 > reprojection_limit_sq
                    {
                        continue;
                    }
                    let hamming = hamming_distance(
                        &source_point.descriptor,
                        &target.frame.features.descriptors[target_desc_idx],
                    );
                    if hamming > config.max_hamming {
                        continue;
                    }
                    if best.is_none_or(|(_, best_hamming, best_distance)| {
                        (hamming, distance_sq).lt(&(best_hamming, best_distance))
                    }) {
                        best = Some((target_desc_idx, hamming, distance_sq));
                    }
                }
                best.map(|(target_desc_idx, _, _)| {
                    (target_desc_idx, target.map_point(target_desc_idx))
                })
            };
            let Some((target_desc_idx, target_point_idx)) = proposal else {
                continue;
            };

            let target_point_idx = target_point_idx.filter(|&index| {
                map.map_points()
                    .get(index)
                    .is_some_and(|point| !point.culled)
            });
            let Some(target_point_idx) = target_point_idx else {
                map.register_observation_at(source_point_idx, target_kf_idx, target_desc_idx);
                if let Some(target) = map.get_keyframe_mut(target_kf_idx) {
                    target.associate_map_point(target_desc_idx, source_point_idx);
                }
                stats.observations_added += 1;
                continue;
            };
            if target_point_idx == source_point_idx {
                continue;
            }

            let reciprocal_is_consistent = {
                let Some(target_point) = map.map_points().get(target_point_idx) else {
                    continue;
                };
                let Some(source) = map.get_keyframe(source_kf_idx) else {
                    continue;
                };
                let point_source = source
                    .frame
                    .pose_world_to_cam
                    .transform_point(&target_point.position);
                if point_source.z <= 0.0 {
                    false
                } else if let Ok(projected) =
                    camera.project_to_image(&point_source, 0.0, source.frame.image_size)
                {
                    source
                        .frame
                        .undistorted_xy(source_desc_idx, camera)
                        .is_some_and(|keypoint| {
                            let dx = projected.x - keypoint[0] as f64;
                            let dy = projected.y - keypoint[1] as f64;
                            dx * dx + dy * dy <= reprojection_limit_sq
                        })
                } else {
                    false
                }
            };
            if reciprocal_is_consistent
                && map
                    .merge_map_points(source_point_idx, target_point_idx)
                    .is_some()
            {
                stats.map_points_merged += 1;
            }
        }
    }
}
