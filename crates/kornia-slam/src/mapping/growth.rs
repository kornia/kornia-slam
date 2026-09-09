//! Map growth: proposing new landmarks and observations from geometry.
//!
//! These functions read only. Each returns a request that the caller publishes
//! through the map's canonical mutation API, so a rejected batch leaves nothing
//! partially inserted and the geometry here never has to know how storage works.
//!
//! The map does not need to know whether a position came from stereo depth or
//! from pair triangulation — both paths build the same request.

use crate::frame::Frame;
use crate::map::{
    LandmarkSeed, LandmarkTarget, Map, MapInsertion, ORB_SCALE_FACTOR, ObservationKey,
    ObservationLink,
};
use crate::stereo::unproject_stereo;
use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::{Pose3d, TriangulationConfig, triangulate_matched_points};
use kornia_algebra::{Mat3F64, Vec2F64, Vec3F64};
use kornia_imgproc::features::{OrbMatchConfig, hamming_distance, match_orb_descriptors};
use std::collections::HashSet;

/// Minimum epipolar-consistent matches before a pair is worth triangulating.
const MIN_GROWTH_MATCHES: usize = 15;
/// 1-DOF chi-square gate at 95% for point-to-epipolar-line distance
/// (ORB-SLAM3's `CheckDistEpipolarLine`), scaled per octave.
const EPIPOLAR_CHI2: f64 = 3.84;

/// Seeds for `frame`'s unassociated close stereo keypoints (`z < mthdepth`).
///
/// Far points are left to multi-view triangulation. `claimed` lists the feature
/// slots already taken — tracked matches for a keyframe about to be published,
/// or the stored associations of one already in the map — so a seed can never
/// contend for a feature an earlier pass took.
pub fn stereo_seeds(
    frame: &Frame,
    camera: &PinholeCamera,
    mthdepth: f64,
    claimed: &[Option<usize>],
) -> Vec<LandmarkSeed> {
    let cam_points = unproject_stereo(frame, camera);
    if cam_points.is_empty() {
        return Vec::new();
    }
    let pose_inv = frame.pose_world_to_cam.inverse();

    let mut seeds = Vec::new();
    for (desc_idx, p_cam) in &cam_points {
        if p_cam.z > mthdepth {
            continue;
        }
        if claimed.get(*desc_idx).copied().flatten().is_some() {
            continue;
        }
        seeds.push(LandmarkSeed {
            position: pose_inv.transform_point(p_cam),
            color: frame
                .keypoint_colors
                .get(*desc_idx)
                .copied()
                .unwrap_or([128; 3]),
            reference: ObservationKey {
                keyframe_idx: frame.idx,
                feature_idx: *desc_idx,
            },
        });
    }
    seeds
}

/// Triangulates unassociated features shared by two stored keyframes.
///
/// Both keyframes must already be in the map. Each new landmark is referenced
/// to its `curr_kf_idx` feature and carries an explicit second observation on
/// `prev_kf_idx` — without that second observer the point would bias its own
/// scale and normal geometry and the cull would be over-aggressive.
pub fn pair_growth_request(
    map: &Map,
    prev_kf_idx: usize,
    curr_kf_idx: usize,
    match_config: OrbMatchConfig,
    triangulation_config: &TriangulationConfig,
    camera: &PinholeCamera,
) -> Option<MapInsertion> {
    let prev_kf = map.get_keyframe(prev_kf_idx)?;
    let curr_kf = map.get_keyframe(curr_kf_idx)?;

    // Only features without a landmark in either keyframe. Matching the full
    // arrays and filtering afterwards discards almost everything once the
    // keyframes are mature.
    let prev_unassoc: Vec<usize> = (0..prev_kf.frame.features.descriptors.len())
        .filter(|&i| prev_kf.map_point(i).is_none())
        .collect();
    let curr_unassoc: Vec<usize> = (0..curr_kf.frame.features.descriptors.len())
        .filter(|&i| curr_kf.map_point(i).is_none())
        .collect();
    if prev_unassoc.is_empty() || curr_unassoc.is_empty() {
        return None;
    }

    // Both poses are known, so the fundamental matrix is fully determined:
    // F = K^-T [t]x R K^-1 with (R, t) the prev->curr relative pose. Filtering
    // against it replaces the two-view estimator's F-matrix RANSAC (mirrors
    // ORB-SLAM3's SearchForTriangulation).
    let rel = Pose3d::between(
        &prev_kf.frame.pose_world_to_cam,
        &curr_kf.frame.pose_world_to_cam,
    );
    if rel.translation.length() <= 1e-8 {
        // No baseline: epipolar geometry degenerates and triangulation would
        // reject everything anyway.
        return None;
    }
    let t = rel.translation;
    let t_skew = Mat3F64::from_cols(
        Vec3F64::new(0.0, t.z, -t.y),
        Vec3F64::new(-t.z, 0.0, t.x),
        Vec3F64::new(t.y, -t.x, 0.0),
    );
    let k_inv = Mat3F64::from_cols(
        Vec3F64::new(1.0 / camera.fx, 0.0, 0.0),
        Vec3F64::new(0.0, 1.0 / camera.fy, 0.0),
        Vec3F64::new(-camera.cx / camera.fx, -camera.cy / camera.fy, 1.0),
    );
    let f_mat = k_inv.transpose() * (t_skew * rel.rotation) * k_inv;

    // Epipole of the prev camera in the curr image. Near it every keypoint is
    // close to every epipolar line, so the chi-square gate is uninformative
    // there: wrong matches survive and triangulate to depth-garbage points that
    // still reproject well in both views. Mirrors ORB-SLAM3's epipole-proximity
    // rejection; RANSAC consensus used to absorb these.
    let prev_center_world = prev_kf.frame.pose_world_to_cam.inverse().translation;
    let epipole_cam = curr_kf
        .frame
        .pose_world_to_cam
        .transform_point(&prev_center_world);
    let epipole_px = (epipole_cam.z.abs() > 1e-9).then(|| {
        Vec2F64::new(
            camera.fx * epipole_cam.x / epipole_cam.z + camera.cx,
            camera.fy * epipole_cam.y / epipole_cam.z + camera.cy,
        )
    });

    // Brute-force descriptor matching over the unassociated subsets; the global
    // second-best ratio test and orientation consistency live in the matcher and
    // are essential for match quality.
    let sub_matches = match_orb_descriptors(
        &prev_unassoc
            .iter()
            .map(|&i| prev_kf.frame.features.orientations[i])
            .collect::<Vec<_>>(),
        &prev_unassoc
            .iter()
            .map(|&i| prev_kf.frame.features.descriptors[i])
            .collect::<Vec<_>>(),
        &curr_unassoc
            .iter()
            .map(|&i| curr_kf.frame.features.orientations[i])
            .collect::<Vec<_>>(),
        &curr_unassoc
            .iter()
            .map(|&i| curr_kf.frame.features.descriptors[i])
            .collect::<Vec<_>>(),
        match_config,
    );

    let mut pair_indices: Vec<(usize, usize)> = Vec::new();
    let mut matched_prev: Vec<Vec2F64> = Vec::new();
    let mut matched_curr: Vec<Vec2F64> = Vec::new();
    for (prev_sub, curr_sub) in sub_matches {
        let (Some(&prev_idx), Some(&curr_idx)) =
            (prev_unassoc.get(prev_sub), curr_unassoc.get(curr_sub))
        else {
            continue;
        };
        let (Some(pu), Some(qu)) = (
            prev_kf.frame.undistorted_xy(prev_idx, camera),
            curr_kf.frame.undistorted_xy(curr_idx, camera),
        ) else {
            continue;
        };
        let p = Vec2F64::new(pu[0] as f64, pu[1] as f64);
        let q = Vec2F64::new(qu[0] as f64, qu[1] as f64);
        let octave = curr_kf
            .frame
            .features
            .octaves
            .get(curr_idx)
            .copied()
            .unwrap_or(0);

        // Reject curr keypoints near the epipole; radius grows with octave
        // (ORB-SLAM3 uses 100 * scaleFactor^octave px^2).
        if let Some(e) = epipole_px {
            let dx = q.x - e.x;
            let dy = q.y - e.y;
            if dx * dx + dy * dy < 100.0 * ORB_SCALE_FACTOR.powi(octave as i32) {
                continue;
            }
        }

        let l = f_mat * Vec3F64::new(p.x, p.y, 1.0);
        let line_norm_sq = l.x * l.x + l.y * l.y;
        if line_norm_sq <= 1e-12 {
            continue;
        }
        let d = l.x * q.x + l.y * q.y + l.z;
        let sigma_sq = ORB_SCALE_FACTOR.powi(2 * octave as i32);
        if d * d > EPIPOLAR_CHI2 * sigma_sq * line_norm_sq {
            continue;
        }

        pair_indices.push((prev_idx, curr_idx));
        matched_prev.push(p);
        matched_curr.push(q);
    }
    if pair_indices.len() < MIN_GROWTH_MATCHES {
        return None;
    }

    let triangulated = triangulate_matched_points(
        &matched_prev,
        &matched_curr,
        &prev_kf.frame.pose_world_to_cam,
        &curr_kf.frame.pose_world_to_cam,
        camera,
        triangulation_config,
    )
    .ok()?;

    let mut request = MapInsertion::default();
    for tp in &triangulated {
        let Some(&(prev_idx, curr_idx)) = pair_indices.get(tp.pair_index) else {
            continue;
        };
        // A landmark may only claim a free feature in each keyframe, and a
        // triangulated pair could name one twice; resolve here rather than
        // letting the batch fail as a whole.
        if curr_kf.map_point(curr_idx).is_some() || prev_kf.map_point(prev_idx).is_some() {
            continue;
        }
        if request
            .landmarks
            .iter()
            .any(|s| s.reference.feature_idx == curr_idx)
            || request
                .observations
                .iter()
                .any(|o| o.observation.feature_idx == prev_idx)
        {
            continue;
        }
        let new_index = request.landmarks.len();
        request.landmarks.push(LandmarkSeed {
            position: tp.position,
            color: curr_kf
                .frame
                .keypoint_colors
                .get(curr_idx)
                .copied()
                .unwrap_or([128; 3]),
            reference: ObservationKey {
                keyframe_idx: curr_kf_idx,
                feature_idx: curr_idx,
            },
        });
        request.observations.push(ObservationLink {
            observation: ObservationKey {
                keyframe_idx: prev_kf_idx,
                feature_idx: prev_idx,
            },
            landmark: LandmarkTarget::New(new_index),
        });
    }
    (!request.landmarks.is_empty()).then_some(request)
}

/// Search radius, in pixels, for a landmark's projection into a neighbour.
const FUSE_SEARCH_RADIUS_PX: f32 = 7.0;
/// Descriptor distance above which a projected landmark is not the keypoint.
const FUSE_MAX_HAMMING: u32 = 50;

/// Links the current keyframe's landmarks into neighbours that do not yet
/// observe them — the forward half of ORB-SLAM3's `SearchInNeighbors`.
///
/// Read-only: proposals are resolved deterministically here (a keypoint wanted
/// by two landmarks goes to the smaller Hamming distance) and the caller
/// publishes them. Duplicate merging and the second-hop covisible expansion are
/// still not implemented.
pub fn neighbor_fusion_links(
    map: &Map,
    curr_kf_idx: usize,
    neighbor_kf_indices: &[usize],
    camera: &PinholeCamera,
) -> Vec<ObservationLink> {
    let Some(curr_kf) = map.get_keyframe(curr_kf_idx) else {
        return Vec::new();
    };
    let curr_mp_indices: Vec<usize> = curr_kf
        .map_point_by_desc_idx
        .iter()
        .filter_map(|&mp| mp)
        .collect();
    if curr_mp_indices.is_empty() {
        return Vec::new();
    }

    let r2 = FUSE_SEARCH_RADIUS_PX * FUSE_SEARCH_RADIUS_PX;
    let mut links = Vec::new();

    for &nb_kf_idx in neighbor_kf_indices {
        if nb_kf_idx == curr_kf_idx {
            continue;
        }
        let Some(nb_kf) = map.get_keyframe(nb_kf_idx) else {
            continue;
        };

        // (keypoint in neighbour, landmark, hamming)
        let mut proposals: Vec<(usize, usize, u32)> = Vec::new();
        for &mp_idx in &curr_mp_indices {
            let mp = match map.map_points().get(mp_idx) {
                Some(mp) if !mp.culled => mp,
                _ => continue,
            };
            if mp.is_observed_by(nb_kf_idx) {
                continue;
            }
            let p_cam = nb_kf.frame.pose_world_to_cam.transform_point(&mp.position);
            if p_cam.z <= 0.0 {
                continue;
            }
            let Ok(pixel) = camera.project_to_image(&p_cam, 0.0, nb_kf.frame.image_size) else {
                continue;
            };
            let (u, v) = (pixel.x as f32, pixel.y as f32);

            let mut best_dist = u32::MAX;
            let mut best_kp = usize::MAX;
            for kp_idx in 0..nb_kf.frame.features.keypoints_xy.len() {
                if nb_kf.map_point(kp_idx).is_some() {
                    continue;
                }
                let Some(kp) = nb_kf.frame.undistorted_xy(kp_idx, camera) else {
                    continue;
                };
                let (dx, dy) = (kp[0] - u, kp[1] - v);
                if dx * dx + dy * dy > r2 {
                    continue;
                }
                let dist =
                    hamming_distance(&mp.descriptor, &nb_kf.frame.features.descriptors[kp_idx]);
                if dist < best_dist {
                    best_dist = dist;
                    best_kp = kp_idx;
                }
            }
            if best_dist <= FUSE_MAX_HAMMING && best_kp != usize::MAX {
                proposals.push((best_kp, mp_idx, best_dist));
            }
        }

        // Closest descriptor wins a contested keypoint; one claim per keypoint
        // and one per landmark, since a landmark may hold only one feature in
        // any keyframe.
        proposals.sort_by_key(|&(_, _, dist)| dist);
        let mut taken_kp: HashSet<usize> = HashSet::new();
        let mut taken_mp: HashSet<usize> = HashSet::new();
        for (kp_idx, mp_idx, _) in proposals {
            if !taken_kp.insert(kp_idx) || !taken_mp.insert(mp_idx) {
                continue;
            }
            links.push(ObservationLink {
                observation: ObservationKey {
                    keyframe_idx: nb_kf_idx,
                    feature_idx: kp_idx,
                },
                landmark: LandmarkTarget::Existing(mp_idx),
            });
        }
    }
    links
}

/// Which of `candidates`, given as `(reference feature, current feature)`, can
/// be published together.
///
/// A feature may hold at most one landmark in each view, so a candidate naming
/// a feature an earlier one took is dropped. Returns the accepted indices in
/// input order — first claim wins, matching the order the triangulator emits.
///
/// A rejected candidate must not reserve anything: reserving its *other*
/// feature would knock out a later candidate that was perfectly publishable.
pub fn accepted_pair_claims(candidates: &[(usize, usize)]) -> Vec<usize> {
    let mut claimed_reference: HashSet<usize> = HashSet::new();
    let mut claimed_current: HashSet<usize> = HashSet::new();
    let mut accepted = Vec::new();
    for (index, &(reference_feature, current_feature)) in candidates.iter().enumerate() {
        if claimed_reference.contains(&reference_feature)
            || claimed_current.contains(&current_feature)
        {
            continue;
        }
        claimed_reference.insert(reference_feature);
        claimed_current.insert(current_feature);
        accepted.push(index);
    }
    accepted
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::Keyframe;
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

    fn test_frame(idx: usize, descriptors: Vec<[u8; 32]>) -> Frame {
        let n = descriptors.len();
        Frame {
            idx,
            features: OrbFeatures {
                keypoints_xy: (0..n).map(|i| [i as f32, i as f32]).collect(),
                orientations: vec![0.0; n],
                descriptors,
                octaves: vec![0; n],
            },
            pose_world_to_cam: Pose3d::IDENTITY,
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
    fn stereo_seeds_filter_far_and_already_claimed_features() {
        let mut frame = test_frame(10, vec![[0; 32]; 4]);
        frame.features.keypoints_xy = vec![[320.0, 240.0]; 4];
        frame.depth = vec![2.0, 3.0, 20.0, -1.0];
        frame.pose_world_to_cam.translation = Vec3F64::new(-1.0, 0.0, 0.0);

        // Feature 0 is already taken by a tracked landmark; 2 is beyond
        // mthdepth; 3 has invalid depth.
        let claimed = vec![Some(7), None, None, None];
        let seeds = stereo_seeds(&frame, &camera(), 5.0, &claimed);

        assert_eq!(seeds.len(), 1);
        assert_eq!(seeds[0].reference.feature_idx, 1);
        assert_eq!(seeds[0].reference.keyframe_idx, 10);
    }

    #[test]
    fn proposal_generation_leaves_the_map_untouched() {
        let mut map = Map::new();
        map.insert_keyframe(Keyframe::from_frame(test_frame(10, vec![[0; 32]; 4])))
            .unwrap();
        let before_points = map.num_map_points();
        let before_kfs = map.keyframes().len();

        let _ = pair_growth_request(
            &map,
            10,
            10,
            OrbMatchConfig::default(),
            &TriangulationConfig::default(),
            &camera(),
        );
        let _ = neighbor_fusion_links(&map, 10, &[10], &camera());

        assert_eq!(map.num_map_points(), before_points);
        assert_eq!(map.keyframes().len(), before_kfs);
    }

    /// A pair request names both observers, so publishing it gives each new
    /// landmark two observations; repeating it proposes nothing, because every
    /// feature is now claimed.
    #[test]
    fn pair_growth_yields_two_observers_and_does_not_duplicate() {
        let camera = camera();
        let descriptors: Vec<[u8; 32]> = (0..20).map(|i| [i as u8; 32]).collect();
        let mut previous = test_frame(10, descriptors.clone());
        let mut current = test_frame(20, descriptors);
        current.pose_world_to_cam.translation = Vec3F64::new(-1.0, 0.0, 0.0);
        let points: Vec<Vec3F64> = (0..20)
            .map(|i| Vec3F64::new((i % 5) as f64 * 0.3 - 0.6, (i / 5) as f64 * 0.3 - 0.45, 5.0))
            .collect();
        for frame in [&mut previous, &mut current] {
            frame.features.keypoints_xy = points
                .iter()
                .map(|p| {
                    let p = frame.pose_world_to_cam.transform_point(p);
                    [
                        (camera.fx * p.x / p.z + camera.cx) as f32,
                        (camera.fy * p.y / p.z + camera.cy) as f32,
                    ]
                })
                .collect();
        }

        let mut map = Map::new();
        map.insert_keyframe(Keyframe::from_frame(previous)).unwrap();
        map.insert_keyframe(Keyframe::from_frame(current)).unwrap();

        let request = pair_growth_request(
            &map,
            10,
            20,
            OrbMatchConfig::default(),
            &TriangulationConfig::default(),
            &camera,
        )
        .expect("a well-conditioned pair should propose landmarks");
        assert_eq!(request.landmarks.len(), points.len());
        let result = map.apply_insertion(request).expect("valid batch");

        for (i, expected) in points.iter().enumerate() {
            let mp = map.get_keyframe(20).unwrap().map_point(i).unwrap();
            assert_eq!(map.get_keyframe(10).unwrap().map_point(i), Some(mp));
            let point = &map.map_points()[mp];
            assert!((point.position - *expected).length() < 1e-4);
            assert_eq!(point.observations().len(), 2);
            assert!(point.is_observed_by(10));
            assert!(point.is_observed_by(20));
        }
        assert_eq!(result.landmark_ids.len(), points.len());

        assert!(
            pair_growth_request(
                &map,
                10,
                20,
                OrbMatchConfig::default(),
                &TriangulationConfig::default(),
                &camera,
            )
            .is_none(),
            "every feature is claimed, so nothing is left to propose"
        );
    }

    /// Two landmarks projecting onto one neighbour keypoint: the closer
    /// descriptor wins and the keypoint is claimed once.
    #[test]
    fn fusion_gives_a_contested_keypoint_to_the_best_descriptor_once() {
        let mut map = Map::new();
        map.insert_keyframe(Keyframe::from_frame(test_frame(10, vec![[1; 32], [0; 32]])))
            .unwrap();
        let mut neighbor = test_frame(20, vec![[0; 32]]);
        neighbor.features.keypoints_xy = vec![[320.0, 240.0]];
        map.insert_keyframe(Keyframe::from_frame(neighbor)).unwrap();

        let worse = map
            .insert_landmark(LandmarkSeed {
                position: Vec3F64::new(0.0, 0.0, 5.0),
                color: [0; 3],
                reference: ObservationKey {
                    keyframe_idx: 10,
                    feature_idx: 0,
                },
            })
            .unwrap();
        let best = map
            .insert_landmark(LandmarkSeed {
                position: Vec3F64::new(0.0, 0.0, 5.0),
                color: [0; 3],
                reference: ObservationKey {
                    keyframe_idx: 10,
                    feature_idx: 1,
                },
            })
            .unwrap();

        let links = neighbor_fusion_links(&map, 10, &[20], &camera());
        assert_eq!(links.len(), 1, "the keypoint is claimed once");
        assert_eq!(links[0].landmark, LandmarkTarget::Existing(best));
        assert_eq!(links[0].observation.feature_idx, 0);
        assert_ne!(links[0].landmark, LandmarkTarget::Existing(worse));
    }

    /// A rejected candidate must reserve nothing. `(0, 1)` is refused because
    /// reference 0 is taken; if it still reserved current 1, the publishable
    /// `(1, 1)` would be lost with it — enough dropped support to fail the
    /// bootstrap acceptance threshold.
    #[test]
    fn a_rejected_candidate_does_not_reserve_its_other_feature() {
        assert_eq!(
            accepted_pair_claims(&[(0, 0), (0, 1), (1, 1)]),
            vec![0, 2],
            "the third candidate is publishable and must survive"
        );
    }

    #[test]
    fn first_claim_wins_in_either_view() {
        // Repeated current feature.
        assert_eq!(accepted_pair_claims(&[(0, 0), (1, 0)]), vec![0]);
        // Repeated reference feature.
        assert_eq!(accepted_pair_claims(&[(0, 0), (0, 1)]), vec![0]);
        // Disjoint in both views.
        assert_eq!(
            accepted_pair_claims(&[(0, 0), (1, 1), (2, 2)]),
            vec![0, 1, 2]
        );
        assert!(accepted_pair_claims(&[]).is_empty());
    }
}
