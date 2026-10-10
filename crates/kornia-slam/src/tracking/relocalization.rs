//! Relocalization: recovering the camera pose against the existing map after
//! tracking has failed, from place-recognition candidates.
//!
//! After ORB-SLAM3's `Tracking::Relocalization`: descriptor matches against
//! each candidate keyframe give 3D-2D correspondences, PnP RANSAC proposes a
//! pose, and the map-projection estimator refines it against the candidate's
//! local map. A pose is accepted only with as much support as normal tracking
//! would need from a cold start.

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pnp::{PnPMethod, RansacParams, solve_pnp_ransac};
use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3AF32, Vec2F32, Vec3AF32};
use kornia_imgproc::features::{OrbMatchConfig, match_orb_descriptors};

use crate::Frame;
use crate::mapping::{Keyframe, Map};
use crate::pose_conversion::{mat3_to_f64, vec3_to_f32, vec3_to_f64};
use crate::tracking::pose_estimation::{Estimate, MapProjectionEstimator};

/// Thresholds for relocalization (ORB-SLAM3's values).
#[derive(Debug, Clone, Copy)]
pub struct RelocalizationConfig {
    /// Descriptor matcher against candidate keyframes.
    pub match_config: OrbMatchConfig,
    /// Correspondences a candidate needs before PnP RANSAC is attempted.
    pub min_correspondences: usize,
    /// RANSAC inliers a candidate pose needs before refinement.
    pub min_ransac_inliers: usize,
    /// Refined inliers needed to accept the pose.
    pub min_inliers: usize,
    /// RANSAC reprojection threshold, in pixels.
    pub ransac_threshold_px: f32,
    /// RANSAC iteration budget per candidate.
    pub ransac_iterations: usize,
}

impl Default for RelocalizationConfig {
    fn default() -> Self {
        Self {
            match_config: OrbMatchConfig {
                nn_ratio: 0.75,
                th_low: 50,
                check_orientation: true,
                histo_length: 30,
            },
            min_correspondences: 15,
            min_ransac_inliers: 10,
            min_inliers: 50,
            ransac_threshold_px: 4.0,
            ransac_iterations: 300,
        }
    }
}

/// A recovered pose and the keyframe it was recovered against.
#[derive(Debug, Clone)]
pub struct Relocalization {
    pub estimate: Estimate,
    pub keyframe_idx: usize,
}

/// Tries each candidate keyframe in order and returns the first pose with
/// enough support.
pub fn relocalize(
    frame: &Frame,
    candidates: &[usize],
    map: &Map,
    camera: &PinholeCamera,
    estimator: &MapProjectionEstimator,
    config: &RelocalizationConfig,
) -> Option<Relocalization> {
    candidates.iter().find_map(|&keyframe_idx| {
        let keyframe = map.get_keyframe(keyframe_idx)?;
        let correspondences = keyframe_correspondences(frame, keyframe, map, config);
        if correspondences.len() < config.min_correspondences {
            return None;
        }
        let (pose, inliers) = ransac_pose(frame, keyframe, map, camera, &correspondences, config)?;
        let estimate = estimator
            .estimate_pose(
                frame,
                &pose,
                &pose,
                map,
                camera,
                Some(keyframe_idx),
                1.0,
                Some(inliers),
            )
            .ok()?;
        (estimate.inliers >= config.min_inliers).then_some(Relocalization {
            estimate,
            keyframe_idx,
        })
    })
}

/// `(map_point_idx, keypoint_idx)` pairs from descriptor matches against the
/// keyframe's associated landmarks.
fn keyframe_correspondences(
    frame: &Frame,
    keyframe: &Keyframe,
    map: &Map,
    config: &RelocalizationConfig,
) -> Vec<(usize, usize)> {
    let features = &keyframe.frame.features;
    match_orb_descriptors(
        &features.orientations,
        &features.descriptors,
        &frame.features.orientations,
        &frame.features.descriptors,
        config.match_config,
    )
    .into_iter()
    .filter_map(|(keyframe_desc, keypoint)| {
        let map_point = keyframe.map_point(keyframe_desc)?;
        let point = map.map_points().get(map_point)?;
        (!point.culled).then_some((map_point, keypoint))
    })
    .collect()
}

/// PnP RANSAC over the correspondences, returning the pose and its inlier
/// correspondences. Points are expressed relative to the keyframe's camera
/// centre and scaled to unit median depth for the f32 solver.
fn ransac_pose(
    frame: &Frame,
    keyframe: &Keyframe,
    map: &Map,
    camera: &PinholeCamera,
    correspondences: &[(usize, usize)],
    config: &RelocalizationConfig,
) -> Option<(Pose3d, Vec<(usize, usize)>)> {
    let keyframe_pose = keyframe.frame.pose_world_to_cam;
    let center = keyframe_pose.inverse().translation;
    let mut depths: Vec<f64> = correspondences
        .iter()
        .map(|&(mp, _)| {
            keyframe_pose
                .transform_point(&map.map_points()[mp].position)
                .z
                .abs()
        })
        .collect();
    let mid = depths.len() / 2;
    depths.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    let scale = 1.0 / depths[mid].max(1e-9);

    let world: Vec<Vec3AF32> = correspondences
        .iter()
        .map(|&(mp, _)| vec3_to_f32((map.map_points()[mp].position - center) * scale))
        .collect();
    let image: Vec<Vec2F32> = correspondences
        .iter()
        .map(|&(_, kp)| {
            let [u, v] = frame.keypoints_undist[kp];
            Vec2F32::new(u, v)
        })
        .collect();
    let k = Mat3AF32::from_cols(
        Vec3AF32::new(camera.fx as f32, 0.0, 0.0),
        Vec3AF32::new(0.0, camera.fy as f32, 0.0),
        Vec3AF32::new(camera.cx as f32, camera.cy as f32, 1.0),
    );
    let result = solve_pnp_ransac(
        &world,
        &image,
        &k,
        None,
        PnPMethod::EPnPDefault,
        &RansacParams {
            max_iterations: config.ransac_iterations,
            reproj_threshold_px: config.ransac_threshold_px,
            random_seed: Some(0),
            ..RansacParams::default()
        },
    )
    .ok()?;
    if result.inliers.len() < config.min_ransac_inliers {
        return None;
    }
    let rotation = mat3_to_f64(result.pose.rotation);
    let translation = vec3_to_f64(result.pose.translation);
    let pose = Pose3d::new(rotation, translation / scale - rotation * center);
    let inliers = result.inliers.iter().map(|&i| correspondences[i]).collect();
    Some((pose, inliers))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mapping::map::{LandmarkSeed, ObservationKey};
    use crate::tracking::pose_estimation::map_projection::MapProjectionConfig;
    use kornia_algebra::{Mat3F64, SO3F64, Vec3F64};
    use kornia_image::ImageSize;
    use kornia_imgproc::features::OrbFeatures;

    const LANDMARKS: usize = 80;

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

    fn descriptor(index: usize) -> [u8; 32] {
        std::array::from_fn(|byte| (index.wrapping_mul(37).wrapping_add(byte * 13)) as u8)
    }

    fn landmark(index: usize) -> Vec3F64 {
        Vec3F64::new(
            (index % 10) as f64 * 0.3 - 1.35,
            (index / 10) as f64 * 0.25 - 0.9,
            4.0 + (index % 3) as f64 * 0.3,
        )
    }

    fn frame(idx: usize, pose: Pose3d, descriptors: Vec<[u8; 32]>) -> Frame {
        let camera = camera();
        let pixels: Vec<[f32; 2]> = (0..LANDMARKS)
            .map(|index| {
                let p = pose.transform_point(&landmark(index));
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
                orientations: vec![0.0; LANDMARKS],
                descriptors,
                octaves: vec![0; LANDMARKS],
            },
            pose_world_to_cam: pose,
            image_size: ImageSize {
                width: 640,
                height: 480,
            },
            keypoint_colors: vec![[0; 3]; LANDMARKS],
            u_right: Vec::new(),
            depth: Vec::new(),
            keypoints_undist: pixels,
        }
    }

    fn mapped_keyframe() -> Map {
        let mut map = Map::new();
        let descriptors = (0..LANDMARKS).map(descriptor).collect();
        map.insert_keyframe(Keyframe::from_frame(frame(
            0,
            Pose3d::IDENTITY,
            descriptors,
        )))
        .unwrap();
        for index in 0..LANDMARKS {
            map.insert_landmark(LandmarkSeed {
                position: landmark(index),
                color: [0; 3],
                reference: ObservationKey {
                    keyframe_idx: 0,
                    feature_idx: index,
                },
            })
            .unwrap();
        }
        map
    }

    fn query_pose() -> Pose3d {
        Pose3d::new(
            SO3F64::exp(Vec3F64::new(0.02, -0.15, 0.03)).matrix(),
            Vec3F64::new(0.4, -0.1, 0.2),
        )
    }

    fn relocalize_query(query: &Frame, map: &Map, candidates: &[usize]) -> Option<Relocalization> {
        let estimator = MapProjectionEstimator::new(MapProjectionConfig::default());
        relocalize(
            query,
            candidates,
            map,
            &camera(),
            &estimator,
            &RelocalizationConfig::default(),
        )
    }

    #[test]
    fn recovers_the_pose_against_a_candidate_keyframe() {
        let map = mapped_keyframe();
        let truth = query_pose();
        let query = frame(50, truth, (0..LANDMARKS).map(descriptor).collect());

        let found = relocalize_query(&query, &map, &[0]).expect("the keyframe sees the query");

        assert_eq!(found.keyframe_idx, 0);
        assert!(found.estimate.inliers >= 50);
        let rotation_error = SO3F64::from_matrix(
            &(found.estimate.pose.rotation * Mat3F64::transpose(truth.rotation)),
        )
        .log()
        .length();
        assert!(rotation_error < 1e-3, "rotation error {rotation_error}");
        assert!((found.estimate.pose.translation - truth.translation).length() < 1e-2);
    }

    #[test]
    fn rejects_a_candidate_that_shares_no_appearance() {
        let map = mapped_keyframe();
        let unrelated = (0..LANDMARKS)
            .map(|index| descriptor(index + 1000))
            .collect();
        let query = frame(50, query_pose(), unrelated);

        assert!(relocalize_query(&query, &map, &[0]).is_none());
    }
}
