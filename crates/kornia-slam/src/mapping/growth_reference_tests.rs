//! A copy of the pair-growth body as it was before the guided matcher moved
//! into [`crate::guided_match`], kept to prove the move changes no bit.

use crate::features::{Descriptor, DescriptorSet, Features};
use crate::frame::undistorted_camera;
use crate::mapping::map::{
    Keyframe, LandmarkSeed, LandmarkTarget, MapInsertion, ObservationKey, ObservationLink,
};
use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::{Pose3d, TriangulationConfig, triangulate_matched_points};
use kornia_algebra::{Mat3F64, SO3F64, Vec2F64, Vec3F64};

const MIN_GROWTH_MATCHES: usize = 15;
const EPIPOLAR_CHI2: f64 = 3.84;

pub(super) fn reference_pair_growth<D: Descriptor>(
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

    // Both poses are known, so the fundamental matrix is fully determined:
    // F = K^-T [t]x R K^-1 with (R, t) the prev->curr relative pose. Filtering
    // against it replaces the two-view estimator's F-matrix RANSAC (mirrors
    // ORB-SLAM3's SearchForTriangulation).
    let rel = Pose3d::between(prev_pose, curr_pose);
    if rel.translation.length() <= 1e-8 {
        // No baseline: epipolar geometry degenerates and triangulation would
        // reject everything anyway.
        return None;
    }
    let t_skew = SO3F64::hat(rel.translation);
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
    let prev_center_world = prev_pose.camera_centre();
    let epipole_cam = curr_pose.transform_point(&prev_center_world);
    let epipole_px = (epipole_cam.z.abs() > 1e-9).then(|| {
        Vec2F64::new(
            camera.fx * epipole_cam.x / epipole_cam.z + camera.cx,
            camera.fy * epipole_cam.y / epipole_cam.z + camera.cy,
        )
    });

    // Brute-force descriptor matching over the unassociated subsets; the global
    // second-best ratio test and orientation consistency live in the matcher and
    // are essential for match quality.
    let (prev_orientations, prev_descriptors) = subset(&prev_kf.frame.features, &prev_unassoc);
    let (curr_orientations, curr_descriptors) = subset(&curr_kf.frame.features, &curr_unassoc);
    let sub_matches = D::match_sets(
        DescriptorSet {
            orientations: prev_orientations.as_deref(),
            descriptors: &prev_descriptors,
        },
        DescriptorSet {
            orientations: curr_orientations.as_deref(),
            descriptors: &curr_descriptors,
        },
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
            if dx * dx + dy * dy < 100.0 * scale_factor.powi(octave as i32) {
                continue;
            }
        }

        let l = f_mat * Vec3F64::new(p.x, p.y, 1.0);
        let line_norm_sq = l.x * l.x + l.y * l.y;
        if line_norm_sq <= 1e-12 {
            continue;
        }
        let d = l.x * q.x + l.y * q.y + l.z;
        let sigma_sq = scale_factor.powi(2 * octave as i32);
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

/// Orientations (when the detector has them) and descriptors of `indices`.
fn subset<D: Descriptor>(features: &Features<D>, indices: &[usize]) -> (Option<Vec<f32>>, Vec<D>) {
    let orientations = features
        .orientations
        .as_ref()
        .map(|orientations| indices.iter().map(|&i| orientations[i]).collect());
    let descriptors = indices.iter().map(|&i| features.descriptors[i]).collect();
    (orientations, descriptors)
}
