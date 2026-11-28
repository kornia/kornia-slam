//! Map growth: proposing new landmarks and observations from geometry.
//!
//! These functions read only. Each returns a request that the caller publishes
//! through the map's canonical mutation API, so a rejected batch leaves nothing
//! partially inserted and the geometry here never has to know how storage works.
//!
//! The map does not need to know whether a position came from stereo depth or
//! from pair triangulation — both paths build the same request.

use crate::features::Descriptor;
use crate::frame::{Frame, undistorted_camera};
use crate::guided_match::{EpipolarGeometry, GuidedView, guided_matches};
use crate::mapping::map::{
    Keyframe, LandmarkSeed, LandmarkTarget, Map, MapInsertion, ObservationKey, ObservationLink,
};
use crate::rig::project_in_view;
use crate::stereo::unproject_stereo;
use kornia_3d::camera::{Camera, PinholeCamera};
use kornia_3d::pose::{Pose3d, TriangulationConfig, triangulate_matched_points};
use kornia_algebra::Vec2F64;
use kornia_image::ImageSize;
use std::cmp::Ordering;
use std::collections::HashSet;

/// Minimum epipolar-consistent matches before a pair is worth triangulating.
const MIN_GROWTH_MATCHES: usize = 15;

/// Seeds for `frame`'s unassociated close stereo keypoints (`z < mthdepth`).
///
/// Far points are left to multi-view triangulation. `claimed` lists the feature
/// slots already taken — tracked matches for a keyframe about to be published,
/// or the stored associations of one already in the map — so a seed can never
/// contend for a feature an earlier pass took.
pub fn stereo_seeds<D>(
    frame: &Frame<D>,
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

/// Seeds and second observations for a rig frame's overlap-triangulated
/// points nearer than `max_depth` (metres, primary camera), whose primary and
/// camera-1 features are both unclaimed: each landmark is referenced to its
/// primary feature and also observed by the camera-1 match (ORB-SLAM3's
/// `StereoInitialization` and `CreateNewKeyFrame` for non-rectified stereo).
/// Link targets index the returned seeds.
pub fn rig_seeds<D>(
    frame: &Frame<D>,
    max_depth: f64,
    claimed: &[Option<usize>],
) -> (Vec<LandmarkSeed>, Vec<ObservationLink>) {
    let pose_inv = frame.pose_world_to_cam.inverse();
    let is_claimed = |i: usize| claimed.get(i).copied().flatten().is_some();
    let mut seeds = Vec::new();
    let mut links = Vec::new();
    for (i, (point, right)) in frame
        .rig
        .stereo_points
        .iter()
        .zip(&frame.rig.left_to_right)
        .enumerate()
    {
        let (Some(point), Some(right)) = (point, right) else {
            continue;
        };
        let finite = point.x.is_finite() && point.y.is_finite() && point.z.is_finite();
        if !finite || point.z > max_depth || is_claimed(i) || is_claimed(*right) {
            continue;
        }
        links.push(ObservationLink {
            observation: ObservationKey {
                keyframe_idx: frame.idx,
                feature_idx: *right,
            },
            landmark: LandmarkTarget::New(seeds.len()),
        });
        seeds.push(LandmarkSeed {
            position: pose_inv.transform_point(point),
            color: frame.keypoint_colors.get(i).copied().unwrap_or([128; 3]),
            reference: ObservationKey {
                keyframe_idx: frame.idx,
                feature_idx: i,
            },
        });
    }
    (seeds, links)
}

/// Triangulates unassociated features shared by two stored keyframes.
///
/// Both keyframes must already be in the map. Each new landmark is referenced
/// to its `curr_kf_idx` feature and carries an explicit second observation on
/// `prev_kf_idx` — without that second observer the point would bias its own
/// scale and normal geometry and the cull would be over-aggressive.
///
/// A rig keyframe pair grows in each camera separately: camera `c`'s features
/// of both keyframes, triangulated from camera `c`'s poses, so a landmark
/// may be seen by a secondary camera only.
pub fn pair_growth_request<D: Descriptor>(
    map: &Map<D>,
    prev_kf_idx: usize,
    curr_kf_idx: usize,
    match_config: &D::MatchConfig,
    triangulation_config: &TriangulationConfig,
    camera: &PinholeCamera,
) -> Option<MapInsertion<D>> {
    let prev_kf = map.get_keyframe(prev_kf_idx)?;
    let curr_kf = map.get_keyframe(curr_kf_idx)?;
    let scale_factor = map.scale_pyramid().factor;
    let geometry = curr_kf
        .frame
        .rig
        .geometry
        .as_ref()
        .filter(|_| curr_kf.frame.is_rig() && prev_kf.frame.is_rig());
    let Some(geometry) = geometry else {
        return pair_growth_in_camera(
            (prev_kf, curr_kf),
            0,
            camera,
            (
                &prev_kf.frame.pose_world_to_cam,
                &curr_kf.frame.pose_world_to_cam,
            ),
            match_config,
            triangulation_config,
            scale_factor,
        );
    };
    let mut merged = MapInsertion::default();
    for (c, view) in geometry.views().enumerate() {
        let (view_camera, prev_pose, curr_pose) = if c == 0 {
            (
                camera,
                prev_kf.frame.pose_world_to_cam,
                curr_kf.frame.pose_world_to_cam,
            )
        } else {
            (
                view.pinhole,
                view.pose(&prev_kf.frame.pose_world_to_cam),
                view.pose(&curr_kf.frame.pose_world_to_cam),
            )
        };
        let Some(request) = pair_growth_in_camera(
            (prev_kf, curr_kf),
            c,
            view_camera,
            (&prev_pose, &curr_pose),
            match_config,
            triangulation_config,
            scale_factor,
        ) else {
            continue;
        };
        let base = merged.landmarks.len();
        merged.landmarks.extend(request.landmarks);
        merged.observations.extend(
            request
                .observations
                .into_iter()
                .map(|link| ObservationLink {
                    landmark: match link.landmark {
                        LandmarkTarget::New(i) => LandmarkTarget::New(base + i),
                        existing => existing,
                    },
                    ..link
                }),
        );
    }
    (!merged.landmarks.is_empty()).then_some(merged)
}

/// [`pair_growth_request`] over the features of rig camera `camera_idx`,
/// with that camera's model and world-to-camera poses of both keyframes.
fn pair_growth_in_camera<D: Descriptor>(
    (prev_kf, curr_kf): (&Keyframe<D>, &Keyframe<D>),
    camera_idx: usize,
    camera: &PinholeCamera,
    (prev_pose, curr_pose): (&Pose3d, &Pose3d),
    match_config: &D::MatchConfig,
    triangulation_config: &TriangulationConfig,
    scale_factor: f64,
) -> Option<MapInsertion<D>> {
    let prev_kf_idx = prev_kf.frame.idx;
    let curr_kf_idx = curr_kf.frame.idx;

    // Only features without a landmark in either keyframe. Matching the full
    // arrays and filtering afterwards discards almost everything once the
    // keyframes are mature.
    let prev_unassoc: Vec<usize> = (0..prev_kf.frame.features.descriptors.len())
        .filter(|&i| prev_kf.frame.camera_of(i) == camera_idx && prev_kf.map_point(i).is_none())
        .collect();
    let curr_unassoc: Vec<usize> = (0..curr_kf.frame.features.descriptors.len())
        .filter(|&i| curr_kf.frame.camera_of(i) == camera_idx && curr_kf.map_point(i).is_none())
        .collect();
    if prev_unassoc.is_empty() || curr_unassoc.is_empty() {
        return None;
    }

    let geometry = EpipolarGeometry::between(camera, prev_pose, camera, curr_pose)?;
    let prev_pixel = |i: usize| prev_kf.frame.undistorted_xy(i, camera);
    let curr_pixel = |i: usize| curr_kf.frame.undistorted_xy(i, camera);
    let pair_indices = guided_matches(
        &GuidedView {
            features: &prev_kf.frame.features,
            candidates: &prev_unassoc,
            pixel: prev_pixel,
        },
        &GuidedView {
            features: &curr_kf.frame.features,
            candidates: &curr_unassoc,
            pixel: curr_pixel,
        },
        &geometry,
        scale_factor,
        match_config,
        1.0,
    );
    if pair_indices.len() < MIN_GROWTH_MATCHES {
        return None;
    }

    // `guided_matches` keeps only pairs with both pixels, so the two lists stay
    // parallel to `pair_indices`.
    let to_vec = |xy: [f32; 2]| Vec2F64::new(xy[0] as f64, xy[1] as f64);
    let (matched_prev, matched_curr): (Vec<Vec2F64>, Vec<Vec2F64>) = pair_indices
        .iter()
        .filter_map(|&(prev_idx, curr_idx)| {
            Some((to_vec(prev_pixel(prev_idx)?), to_vec(curr_pixel(curr_idx)?)))
        })
        .unzip();
    let triangulated = triangulate_matched_points(
        &matched_prev,
        &matched_curr,
        prev_pose,
        curr_pose,
        &undistorted_camera(camera),
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

/// Links the current keyframe's landmarks into neighbours that do not yet
/// observe them — the forward half of ORB-SLAM3's `SearchInNeighbors`.
///
/// Read-only: proposals are resolved deterministically here (a keypoint wanted
/// by two landmarks goes to the smaller descriptor distance) and the caller
/// publishes them. Duplicate merging and the second-hop covisible expansion are
/// still not implemented.
pub fn neighbor_fusion_links<D: Descriptor>(
    map: &Map<D>,
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

        // (keypoint in neighbour, landmark, camera, descriptor distance)
        let mut proposals: Vec<(usize, usize, usize, D::Distance)> = Vec::new();
        for (view, view_camera, raw, pose, size) in neighbour_views(nb_kf, camera) {
            for &mp_idx in &curr_mp_indices {
                let mp = match map.map_points().get(mp_idx) {
                    Some(mp) if !mp.culled => mp,
                    _ => continue,
                };
                if mp.is_observed_in(nb_kf_idx, view as u8) {
                    continue;
                }
                let p_cam = pose.transform_point(&mp.position);
                let Some(pixel) = project_in_view(view_camera, raw, size, &p_cam, 0.0) else {
                    continue;
                };
                let (u, v) = (pixel.x as f32, pixel.y as f32);

                let mut best: Option<(D::Distance, usize)> = None;
                for kp_idx in nb_kf.frame.camera_range(view) {
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
                    let dist = mp
                        .descriptor
                        .distance(&nb_kf.frame.features.descriptors[kp_idx]);
                    if best.is_none_or(|(best_dist, _)| dist < best_dist) {
                        best = Some((dist, kp_idx));
                    }
                }
                if let Some((best_dist, best_kp)) = best
                    && best_dist <= D::TH_LOW
                {
                    proposals.push((best_kp, mp_idx, view, best_dist));
                }
            }
        }

        // Closest descriptor wins a contested keypoint; one claim per keypoint
        // and one per landmark, since a landmark may hold only one feature in
        // any keyframe.
        proposals.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap_or(Ordering::Equal));
        let mut taken_kp: HashSet<usize> = HashSet::new();
        let mut taken_mp: HashSet<(usize, usize)> = HashSet::new();
        for (kp_idx, mp_idx, view, _) in proposals {
            if !taken_kp.insert(kp_idx) || !taken_mp.insert((mp_idx, view)) {
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

/// `(camera index, model, raw fisheye model, world-to-camera pose, image
/// size)` of each camera of a keyframe: every rig camera, or the one camera
/// outside a rig.
fn neighbour_views<'a, D>(
    keyframe: &'a Keyframe<D>,
    camera: &'a PinholeCamera,
) -> Vec<(
    usize,
    &'a PinholeCamera,
    Option<&'a Camera>,
    Pose3d,
    ImageSize,
)> {
    let frame = &keyframe.frame;
    let Some(rig) = frame.rig.geometry.as_ref().filter(|_| frame.is_rig()) else {
        return vec![(0, camera, None, frame.pose_world_to_cam, frame.image_size)];
    };
    rig.views()
        .enumerate()
        .map(|(c, view)| {
            if c == 0 {
                (
                    0,
                    camera,
                    view.fisheye_model(),
                    frame.pose_world_to_cam,
                    frame.image_size,
                )
            } else {
                (
                    c,
                    view.pinhole,
                    view.fisheye_model(),
                    view.pose(&frame.pose_world_to_cam),
                    view.camera.size,
                )
            }
        })
        .collect()
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
    use crate::features::{Features, RBrief};
    use crate::mapping::map::Keyframe;
    use kornia_3d::camera::CameraModel;
    use kornia_algebra::Vec3F64;
    use kornia_image::ImageSize;

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
            features: Features {
                keypoints_xy: (0..n).map(|i| [i as f32, i as f32]).collect(),
                orientations: Some(vec![0.0; n]),
                descriptors: descriptors.into_iter().map(RBrief).collect(),
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
            rig: Default::default(),
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
    fn rig_seeds_link_both_cameras_of_close_unclaimed_overlap_points() {
        // Features 0..3 are the primary camera's, 3..6 camera 1's.
        let mut frame = test_frame(4, vec![[0; 32]; 6]);
        frame.rig.camera_ranges = vec![0..3, 3..6];
        frame.rig.left_to_right = vec![Some(3), Some(4), Some(5)];
        frame.rig.stereo_points = vec![
            Some(Vec3F64::new(0.0, 0.0, 2.0)),
            Some(Vec3F64::new(0.0, 0.0, 40.0)),
            Some(Vec3F64::new(0.0, 0.0, 3.0)),
        ];
        frame.pose_world_to_cam.translation = Vec3F64::new(-1.0, 0.0, 0.0);
        // Camera 1's match of feature 2 is already tracked.
        let claimed = vec![None, None, None, None, None, Some(9)];

        let (seeds, links) = rig_seeds(&frame, 10.0, &claimed);

        assert_eq!(seeds.len(), 1);
        assert_eq!(seeds[0].reference.feature_idx, 0);
        assert_eq!(seeds[0].position, Vec3F64::new(1.0, 0.0, 2.0));
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].observation.feature_idx, 3);
        assert_eq!(links[0].landmark, LandmarkTarget::New(0));
    }

    /// A NaN overlap point fails `z > max_depth`; it must still not become a
    /// seed, which would make the whole insertion batch refused.
    #[test]
    fn a_non_finite_stereo_point_is_not_a_seed() {
        let mut frame = test_frame(4, vec![[0; 32]; 8]);
        frame.rig.camera_ranges = vec![0..4, 4..8];
        frame.rig.left_to_right = vec![Some(4), Some(5), Some(6), Some(7)];
        frame.rig.stereo_points = vec![
            Some(Vec3F64::new(0.0, 0.0, f64::NAN)),
            Some(Vec3F64::new(f64::NAN, 0.0, 2.0)),
            Some(Vec3F64::new(0.0, f64::INFINITY, 2.0)),
            Some(Vec3F64::new(0.0, 0.0, 2.0)),
        ];

        let (seeds, links) = rig_seeds(&frame, f64::INFINITY, &[]);

        assert_eq!(seeds.len(), 1);
        assert_eq!(seeds[0].reference.feature_idx, 3);
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].observation.feature_idx, 7);
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
            &kornia_imgproc::features::OrbMatchConfig::default(),
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
            &kornia_imgproc::features::OrbMatchConfig::default(),
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
                &kornia_imgproc::features::OrbMatchConfig::default(),
                &TriangulationConfig::default(),
                &camera,
            )
            .is_none(),
            "every feature is claimed, so nothing is left to propose"
        );
    }

    /// Keypoints are raw pixels of a distorted camera; triangulation must
    /// undistort them exactly once to recover the landmarks.
    #[test]
    fn pair_growth_triangulates_through_a_distorted_camera() {
        let camera = PinholeCamera {
            k1: -0.3,
            k2: 0.1,
            p1: 0.001,
            p2: -0.0005,
            ..camera()
        };
        let descriptors: Vec<[u8; 32]> = (0..20).map(|i| [i as u8; 32]).collect();
        let mut previous = test_frame(10, descriptors.clone());
        let mut current = test_frame(20, descriptors);
        current.pose_world_to_cam.translation = Vec3F64::new(-1.0, 0.0, 0.0);
        let points: Vec<Vec3F64> = (0..20)
            .map(|i| Vec3F64::new((i % 5) as f64 * 0.8 - 1.6, (i / 5) as f64 * 0.5 - 0.75, 5.0))
            .collect();
        for frame in [&mut previous, &mut current] {
            frame.features.keypoints_xy = points
                .iter()
                .map(|p| {
                    let pixel = camera
                        .project(&frame.pose_world_to_cam.transform_point(p))
                        .unwrap();
                    [pixel.x as f32, pixel.y as f32]
                })
                .collect();
            frame.ensure_undistorted(&camera);
        }

        let mut map = Map::new();
        map.insert_keyframe(Keyframe::from_frame(previous)).unwrap();
        map.insert_keyframe(Keyframe::from_frame(current)).unwrap();

        let request = pair_growth_request(
            &map,
            10,
            20,
            &kornia_imgproc::features::OrbMatchConfig::default(),
            &TriangulationConfig::default(),
            &camera,
        )
        .expect("a well-conditioned pair should propose landmarks");
        assert_eq!(request.landmarks.len(), points.len());
        for seed in &request.landmarks {
            let expected = points[seed.reference.feature_idx];
            assert!((seed.position - expected).length() < 1e-3);
        }
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

    /// A landmark 70° off a raw fisheye neighbour's axis lies outside its
    /// virtual-pinhole window yet inside the raw image: fusion links it, as
    /// tracking's projection search would match it.
    #[test]
    fn fusion_reaches_wide_landmarks_of_a_raw_fisheye_neighbour() {
        let rig = std::sync::Arc::new(crate::rig::tests::raw_two_camera_rig(75.0));
        let pinhole = rig.view(0).map(|view| view.pinhole.clone()).unwrap();
        let (s, c) = 70f64.to_radians().sin_cos();
        let position = Vec3F64::new(s, 0.0, c) * 4.0;
        let virtual_px = pinhole.project_to_pixel(&position, 0.0).unwrap();
        assert!(virtual_px.x > 640.0);

        let mut map = Map::new();
        map.insert_keyframe(Keyframe::from_frame(test_frame(10, vec![[0; 32]])))
            .unwrap();
        let mut neighbor = test_frame(20, vec![[0; 32]]);
        neighbor.keypoints_undist = vec![[virtual_px.x as f32, virtual_px.y as f32]];
        neighbor.rig = crate::frame::RigFeatures {
            geometry: Some(rig),
            camera_ranges: vec![0..1, 1..1],
            ..Default::default()
        };
        map.insert_keyframe(Keyframe::from_frame(neighbor)).unwrap();
        let landmark = map
            .insert_landmark(LandmarkSeed {
                position,
                color: [0; 3],
                reference: ObservationKey {
                    keyframe_idx: 10,
                    feature_idx: 0,
                },
            })
            .unwrap();

        let links = neighbor_fusion_links(&map, 10, &[20], &pinhole);
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].landmark, LandmarkTarget::Existing(landmark));
        assert_eq!(links[0].observation.feature_idx, 0);
    }

    #[test]
    fn fusion_runs_on_float_descriptors_without_orientations() {
        use crate::features::{ScalePyramid, scalar::Scalar};

        let frame = |idx: usize, descriptors: Vec<Scalar>| {
            let n = descriptors.len();
            Frame {
                idx,
                features: Features {
                    keypoints_xy: vec![[320.0, 240.0]; n],
                    orientations: None,
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
                rig: Default::default(),
            }
        };
        let mut map = Map::with_scale_pyramid(ScalePyramid {
            factor: 1.0,
            n_levels: 1,
        });
        map.insert_keyframe(Keyframe::from_frame(frame(
            10,
            vec![Scalar(0.9), Scalar(0.1), Scalar(9.0)],
        )))
        .unwrap();
        map.insert_keyframe(Keyframe::from_frame(frame(20, vec![Scalar(0.0)])))
            .unwrap();
        let landmark = |map: &mut Map<Scalar>, feature_idx| {
            map.insert_landmark(LandmarkSeed {
                position: Vec3F64::new(0.0, 0.0, 5.0),
                color: [0; 3],
                reference: ObservationKey {
                    keyframe_idx: 10,
                    feature_idx,
                },
            })
            .unwrap()
        };
        let _worse = landmark(&mut map, 0);
        let best = landmark(&mut map, 1);
        let _beyond_threshold = landmark(&mut map, 2);

        let links = neighbor_fusion_links(&map, 10, &[20], &camera());
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].landmark, LandmarkTarget::Existing(best));
    }

    /// A keyframe pair seen from `current_pose`: `points` projected into both
    /// views with a deterministic sub-pixel offset in the current one, every
    /// feature at `octave`.
    fn synthetic_pair(points: &[Vec3F64], current_pose: Pose3d, octave: u8) -> Map<RBrief> {
        let camera = camera();
        let descriptors: Vec<[u8; 32]> = (0..points.len())
            .map(|i| {
                let mut d = [0u8; 32];
                for (k, byte) in d.iter_mut().enumerate() {
                    *byte = (i as u8)
                        .wrapping_mul(37)
                        .wrapping_add((k as u8).wrapping_mul(11));
                }
                d
            })
            .collect();
        let mut previous = test_frame(10, descriptors.clone());
        let mut current = test_frame(20, descriptors);
        current.pose_world_to_cam = current_pose;
        for (f, frame) in [&mut previous, &mut current].into_iter().enumerate() {
            frame.features.octaves = vec![octave; points.len()];
            frame.features.keypoints_xy = points
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let p = frame.pose_world_to_cam.transform_point(p);
                    let jitter = f as f32 * ((i % 7) as f32 - 3.0) * 0.4;
                    [
                        (camera.fx * p.x / p.z + camera.cx) as f32 + jitter,
                        (camera.fy * p.y / p.z + camera.cy) as f32 - jitter,
                    ]
                })
                .collect();
        }
        let mut map = Map::new();
        map.insert_keyframe(Keyframe::from_frame(previous)).unwrap();
        map.insert_keyframe(Keyframe::from_frame(current)).unwrap();
        map
    }

    /// The shared guided matcher reproduces the pre-move growth body bit for
    /// bit: a sideways pair, a forward pair whose epipole lies among the
    /// keypoints, and a pair detected at octave 2.
    #[test]
    fn pair_growth_request_matches_main_bit_for_bit() {
        let grid = |z: f64, step: f64| -> Vec<Vec3F64> {
            (0..40)
                .map(|i| {
                    // Offsets include (0, 0): on the forward pair that point sits on
                    // the epipole in both views.
                    let (x, y) = ((i % 8) as f64 - 3.0, (i / 8) as f64 - 2.0);
                    Vec3F64::new(x * step, y * step, z + 0.1 * (i % 3) as f64)
                })
                .collect()
        };
        let sideways = Pose3d::new(
            kornia_algebra::Mat3F64::IDENTITY,
            Vec3F64::new(-0.6, 0.05, 0.0),
        );
        let forward = Pose3d::new(
            kornia_algebra::Mat3F64::IDENTITY,
            Vec3F64::new(0.02, 0.0, -0.8),
        );
        let cases = [
            synthetic_pair(&grid(5.0, 0.35), sideways, 0),
            synthetic_pair(&grid(6.0, 0.4), forward, 0),
            synthetic_pair(&grid(5.0, 0.35), sideways, 2),
        ];
        let camera = camera();
        let match_config = kornia_imgproc::features::OrbMatchConfig::default();
        let triangulation_config = TriangulationConfig::default();
        for (case, map) in cases.iter().enumerate() {
            let (prev, curr) = (map.get_keyframe(10).unwrap(), map.get_keyframe(20).unwrap());
            let poses = (&prev.frame.pose_world_to_cam, &curr.frame.pose_world_to_cam);
            let scale_factor = map.scale_pyramid().factor;
            let expected = crate::mapping::growth_reference_tests::reference_pair_growth(
                (prev, curr),
                0,
                &camera,
                poses,
                &match_config,
                &triangulation_config,
                scale_factor,
            );
            let got =
                pair_growth_request(map, 10, 20, &match_config, &triangulation_config, &camera);
            let (Some(expected), Some(got)) = (expected, got) else {
                panic!("case {case}: both bodies must propose landmarks");
            };
            assert_eq!(got.landmarks.len(), expected.landmarks.len(), "case {case}");
            for (g, e) in got.landmarks.iter().zip(&expected.landmarks) {
                let bits = |p: &Vec3F64| [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()];
                assert_eq!(bits(&g.position), bits(&e.position), "case {case}");
                assert_eq!(g.reference, e.reference, "case {case}");
            }
            assert_eq!(got.observations, expected.observations, "case {case}");
        }
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
