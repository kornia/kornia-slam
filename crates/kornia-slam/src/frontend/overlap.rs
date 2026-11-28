//! Cross-camera matching inside the overlap of two rig cameras, and
//! triangulation of the matches (ORB-SLAM3's non-rectified stereo: matches
//! searched along the epipolar line the extrinsics give, as in
//! `ORBmatcher::SearchForTriangulation`). The overlap comes from the
//! extrinsics rather than ORB-SLAM3's fixed x-intervals (idea from visloc-rs
//! PR #247).

use std::ops::Range;

use kornia_3d::pose::{Pose3d, triangulate_n_view_linear};
use kornia_algebra::{Vec2F64, Vec3F64};

use crate::features::{Descriptor, Features, RBrief};
use crate::frame::is_finite_xy;
use crate::rig::RigGeometry;

/// Search band around the epipolar line at octave 0, in pixels (ORB-SLAM3:
/// chi² 3.84 at one-pixel sigma, so about two pixels).
const EPIPOLAR_BAND_PX: f64 = 2.0;
/// Highest accepted Hamming distance: ORB-SLAM3's `TH_HIGH`. The two views
/// differ by the rig's 75 deg and meet at the stretched edges of wide
/// pinhole images, where `TH_LOW` (50) leaves almost no match; ORB-SLAM3's
/// fisheye stereo matching has no absolute cap at all. The epipolar band
/// and the triangulation gates reject what this lets through.
const MAX_HAMMING: u32 = 100;
/// Best-to-second Hamming ratio along the line.
const NN_RATIO: f64 = 0.9;
/// Largest reprojection error of a triangulated match at octave 0, in pixels.
const MAX_REPROJECTION_PX: f64 = 2.0;
/// Smallest ray parallax a triangulated point must have; below it the depth
/// is undetermined (0.25 deg is about 25 m with Aria's 11 cm baseline).
const MIN_PARALLAX_COS: f64 = 0.999_990_5;
const ORB_SCALE_FACTOR: f64 = 1.2;

/// Matches between the primary camera's features and the second camera's,
/// per primary feature (ORB-SLAM3's `mvLeftToRightMatch`), with the point each
/// one triangulates to in the primary camera frame (metres).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OverlapMatches {
    pub left_to_right: Vec<Option<usize>>,
    pub points: Vec<Option<Vec3F64>>,
}

impl OverlapMatches {
    pub fn num_matched(&self) -> usize {
        self.points.iter().filter(|p| p.is_some()).count()
    }
}

/// Matches primary-camera features (`ranges[0]`) with those of camera 1
/// (`ranges[1]`) where both cameras see the same scene. Feature indices are
/// those of the concatenated `features`. The overlap boxes test raw
/// `keypoints_xy`; the epipolar search and triangulation read `pixels`, the
/// pinhole coordinates parallel to them (non-finite entries are skipped).
/// `right_from_left` is the rig's `T_10` ([`kornia_sensors::SensorRig::b_from_a`]).
pub fn match_overlap(
    features: &Features<RBrief>,
    pixels: &[[f32; 2]],
    ranges: &[Range<usize>],
    rig: &RigGeometry,
    right_from_left: &Pose3d,
) -> OverlapMatches {
    let n_primary = ranges.first().map_or(0, |r| r.len());
    let mut out = OverlapMatches {
        left_to_right: vec![None; n_primary],
        points: vec![None; n_primary],
    };
    let (Some(left), Some(right), Some(view0), Some(view1)) =
        (ranges.first(), ranges.get(1), rig.view(0), rig.view(1))
    else {
        return out;
    };
    let (Some(box0), Some(box1)) = (view0.overlap, view1.overlap) else {
        return out;
    };
    let kps = &features.keypoints_xy;
    let right_candidates: Vec<usize> = right
        .clone()
        .filter(|&j| box1.contains(kps[j]) && is_finite_xy(&pixels[j]))
        .collect();
    if right_candidates.is_empty() {
        return out;
    }

    let (r, t) = right_from_left.to_rt();
    let k0_inv = view0.pinhole.intrinsic_matrix().inverse();
    let k1_inv_t = view1.pinhole.intrinsic_matrix().inverse().transpose();
    let t_cross = kornia_algebra::Mat3F64::from_cols(
        Vec3F64::new(0.0, t.z, -t.y),
        Vec3F64::new(-t.z, 0.0, t.x),
        Vec3F64::new(t.y, -t.x, 0.0),
    );
    let fundamental = k1_inv_t * (t_cross * r) * k0_inv;
    let projection0 = projection_matrix(&view0.pinhole.intrinsic_matrix(), &Pose3d::IDENTITY);
    let projection1 = projection_matrix(&view1.pinhole.intrinsic_matrix(), right_from_left);

    // Best left feature per right feature, so each right feature matches once.
    let mut claimed: Vec<Option<(u32, usize)>> = vec![None; right.len()];
    let mut proposals: Vec<(usize, usize, u32)> = Vec::new();
    for i in left.clone() {
        if !box0.contains(kps[i]) || !is_finite_xy(&pixels[i]) {
            continue;
        }
        let line = fundamental * Vec3F64::new(pixels[i][0] as f64, pixels[i][1] as f64, 1.0);
        let norm = (line.x * line.x + line.y * line.y).sqrt();
        if norm < 1e-12 {
            continue;
        }
        let octave_i = features.octaves[i] as i32;
        let mut best = (u32::MAX, usize::MAX);
        let mut second = u32::MAX;
        for &j in &right_candidates {
            let octave_j = features.octaves[j] as i32;
            if (octave_i - octave_j).abs() > 1 {
                continue;
            }
            let distance_px =
                (line.x * pixels[j][0] as f64 + line.y * pixels[j][1] as f64 + line.z).abs() / norm;
            if distance_px > EPIPOLAR_BAND_PX * ORB_SCALE_FACTOR.powi(octave_j) {
                continue;
            }
            let dist = features.descriptors[i].distance(&features.descriptors[j]);
            if dist < best.0 {
                second = best.0;
                best = (dist, j);
            } else if dist < second {
                second = dist;
            }
        }
        let (dist, j) = best;
        if dist > MAX_HAMMING || (second != u32::MAX && dist as f64 > NN_RATIO * second as f64) {
            continue;
        }
        let slot = j - right.start;
        if claimed[slot].is_none_or(|(d, _)| dist < d) {
            claimed[slot] = Some((dist, i));
        }
        proposals.push((i, j, dist));
    }

    for (i, j, dist) in proposals {
        if claimed[j - right.start] != Some((dist, i)) {
            continue;
        }
        let Some(point) = triangulate(
            pixels[i],
            pixels[j],
            &projection0,
            &projection1,
            right_from_left,
            [features.octaves[i], features.octaves[j]],
            rig,
        ) else {
            continue;
        };
        let slot = i - left.start;
        out.left_to_right[slot] = Some(j);
        out.points[slot] = Some(point);
    }
    out
}

fn projection_matrix(k: &kornia_algebra::Mat3F64, pose: &Pose3d) -> [f64; 12] {
    let (r, t) = pose.to_rt();
    let kr = *k * r;
    let kt = *k * t;
    [
        kr.x_axis.x,
        kr.y_axis.x,
        kr.z_axis.x,
        kt.x, //
        kr.x_axis.y,
        kr.y_axis.y,
        kr.z_axis.y,
        kt.y, //
        kr.x_axis.z,
        kr.y_axis.z,
        kr.z_axis.z,
        kt.z,
    ]
}

/// DLT point in the primary camera frame, kept only when in front of both
/// cameras, with enough parallax and a small reprojection error in both.
fn triangulate(
    left: [f32; 2],
    right: [f32; 2],
    projection0: &[f64; 12],
    projection1: &[f64; 12],
    right_from_left: &Pose3d,
    octaves: [u8; 2],
    rig: &RigGeometry,
) -> Option<Vec3F64> {
    let pixels = [
        Vec2F64::new(left[0] as f64, left[1] as f64),
        Vec2F64::new(right[0] as f64, right[1] as f64),
    ];
    let p0 = triangulate_n_view_linear(&pixels, &[*projection0, *projection1])?;
    let p1 = right_from_left.transform_point(&p0);
    if p0.z <= 0.0 || p1.z <= 0.0 {
        return None;
    }
    let centre1 = right_from_left.camera_centre();
    let ray0 = p0.normalize();
    let ray1 = (p0 - centre1).normalize();
    if ray0.dot(ray1) > MIN_PARALLAX_COS {
        return None;
    }
    for ((p_cam, pixel), (view, octave)) in [(p0, pixels[0]), (p1, pixels[1])]
        .into_iter()
        .zip(rig.views().zip(octaves))
    {
        let max_px = MAX_REPROJECTION_PX * ORB_SCALE_FACTOR.powi(octave as i32);
        let error_sq = view
            .pinhole
            .reprojection_error_sq_cam(&p_cam, pixel.x, pixel.y)?;
        if error_sq > max_px * max_px {
            return None;
        }
    }
    Some(p0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_3d::camera::{CameraModel, PinholeCamera};
    use kornia_algebra::Mat3F64;
    use kornia_image::ImageSize;
    use kornia_sensors::{RigCamera, SensorRig};

    fn camera() -> PinholeCamera {
        PinholeCamera {
            fx: 240.0,
            fy: 240.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

    fn sensor_rig() -> SensorRig {
        let (s, c) = 40f64.to_radians().sin_cos();
        let yaw = Mat3F64::from_cols(
            Vec3F64::new(c, 0.0, -s),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(s, 0.0, c),
        );
        let size = ImageSize {
            width: 640,
            height: 480,
        };
        SensorRig::from_cameras(vec![
            RigCamera::primary(camera(), size),
            RigCamera::new(
                camera(),
                size,
                Pose3d::new(yaw, Vec3F64::new(0.11, 0.0, 0.0)),
            ),
        ])
        .unwrap()
    }

    fn descriptor(seed: u8) -> RBrief {
        let mut d = [0u8; 32];
        for (k, byte) in d.iter_mut().enumerate() {
            *byte = seed
                .wrapping_mul(37)
                .wrapping_add((k as u8).wrapping_mul(11))
                ^ (seed >> 1);
        }
        RBrief(d)
    }

    fn features(keypoints_xy: Vec<[f32; 2]>, descriptors: Vec<RBrief>) -> Features<RBrief> {
        let n = keypoints_xy.len();
        Features {
            keypoints_xy,
            orientations: Some(vec![0.0; n]),
            descriptors,
            octaves: vec![0; n],
        }
    }

    #[test]
    fn overlap_matches_triangulate_the_scene_points() {
        let sensor_rig = sensor_rig();
        let rig = RigGeometry::from_rig(&sensor_rig).unwrap();
        let right_from_left = sensor_rig.b_from_a(0, 1).unwrap();
        let scene = [
            Vec3F64::new(1.5, -0.3, 4.0),
            Vec3F64::new(2.0, 0.4, 3.0),
            Vec3F64::new(1.2, 0.1, 6.0),
        ];
        let mut keypoints_xy = Vec::new();
        let mut right_xy = Vec::new();
        for p in &scene {
            let l = camera().project_to_pixel(p, 0.0).unwrap();
            let r = camera()
                .project_to_pixel(&right_from_left.transform_point(p), 0.0)
                .unwrap();
            keypoints_xy.push([l.x as f32, l.y as f32]);
            right_xy.push([r.x as f32, r.y as f32]);
        }
        let n = scene.len();
        keypoints_xy.extend(right_xy);
        let descriptors = (0..n as u8).chain(0..n as u8).map(descriptor).collect();
        let features = features(keypoints_xy, descriptors);
        let matches = match_overlap(
            &features,
            &features.keypoints_xy,
            &[0..n, n..2 * n],
            &rig,
            &right_from_left,
        );
        assert_eq!(matches.num_matched(), n);
        for (i, p) in scene.iter().enumerate() {
            assert_eq!(matches.left_to_right[i], Some(n + i));
            let got = matches.points[i].unwrap();
            assert!((got - *p).length() < 0.05 * p.z, "{got:?} vs {p:?}");
        }
    }

    /// Raw fisheye pair: the overlap triangulates virtual-pinhole pixels, and
    /// reading the raw pixels as pinhole ones recovers none of the points.
    #[test]
    fn raw_fisheye_overlap_triangulates_virtual_pixels() {
        let raw = crate::rig::tests::raw_model();
        let sensor_rig = SensorRig::from_cameras(vec![
            RigCamera::primary(
                raw,
                ImageSize {
                    width: 640,
                    height: 480,
                },
            ),
            RigCamera::new(
                raw,
                ImageSize {
                    width: 640,
                    height: 480,
                },
                Pose3d::new(crate::rig::tests::yaw(40.0), Vec3F64::new(0.1, 0.0, 0.0)),
            ),
        ])
        .unwrap();
        let rig = RigGeometry::from_rig(&sensor_rig).unwrap();
        let (view0, view1) = (rig.view(0).unwrap(), rig.view(1).unwrap());
        let right_from_left = sensor_rig.b_from_a(0, 1).unwrap();
        let scene = [
            Vec3F64::new(3.0, -0.3, 2.0),
            Vec3F64::new(2.0, 0.4, 3.0),
            Vec3F64::new(1.2, 0.1, 6.0),
            Vec3F64::new(2.5, 0.8, 1.5),
        ];
        let raw_pixel = |p: &Vec3F64| {
            let px = raw.project(p).unwrap();
            [px.x as f32, px.y as f32]
        };
        let mut raw_xy: Vec<[f32; 2]> = scene.iter().map(raw_pixel).collect();
        raw_xy.extend(
            scene
                .iter()
                .map(|p| raw_pixel(&right_from_left.transform_point(p))),
        );
        let n = scene.len();
        let virtual_xy: Vec<[f32; 2]> = raw_xy
            .iter()
            .enumerate()
            .map(|(k, &px)| {
                let view = if k < n { view0 } else { view1 };
                view.raw_to_virtual(px).unwrap()
            })
            .collect();
        let descriptors = (0..n as u8).chain(0..n as u8).map(descriptor).collect();
        let features = features(raw_xy, descriptors);
        let ranges = [0..n, n..2 * n];
        let recovered = |pixels: &[[f32; 2]]| {
            let matches = match_overlap(&features, pixels, &ranges, &rig, &right_from_left);
            scene
                .iter()
                .zip(&matches.points)
                .filter(|(p, got)| got.is_some_and(|got| (got - **p).length() < 1e-3 * p.length()))
                .count()
        };
        assert_eq!(recovered(&virtual_xy), n);
        assert_eq!(recovered(&features.keypoints_xy), 0);
    }

    #[test]
    fn a_descriptor_mismatch_is_not_matched() {
        let sensor_rig = sensor_rig();
        let rig = RigGeometry::from_rig(&sensor_rig).unwrap();
        let right_from_left = sensor_rig.b_from_a(0, 1).unwrap();
        let p = Vec3F64::new(1.5, -0.3, 4.0);
        let l = camera().project_to_pixel(&p, 0.0).unwrap();
        let r = camera()
            .project_to_pixel(&right_from_left.transform_point(&p), 0.0)
            .unwrap();
        let features = features(
            vec![[l.x as f32, l.y as f32], [r.x as f32, r.y as f32]],
            vec![RBrief([0u8; 32]), RBrief([0xFFu8; 32])],
        );
        let matches = match_overlap(
            &features,
            &features.keypoints_xy,
            &[0..1, 1..2],
            &rig,
            &right_from_left,
        );
        assert_eq!(matches.num_matched(), 0);
    }

    /// EuRoC-like radtan pair (k1 -0.283, 0.11 m parallel baseline): the
    /// overlap triangulates corner points from undistorted pixels, and reading
    /// the raw pixels as pinhole ones matches none of them.
    #[test]
    fn radtan_overlap_needs_undistorted_pixels() {
        let camera = PinholeCamera {
            fx: 458.0,
            fy: 458.0,
            cx: 376.0,
            cy: 240.0,
            k1: -0.2834,
            k2: 0.0740,
            p1: 0.0,
            p2: 0.0,
        };
        let size = ImageSize {
            width: 752,
            height: 480,
        };
        let sensor_rig = SensorRig::from_cameras(vec![
            RigCamera::primary(camera.clone(), size),
            RigCamera::new(
                camera.clone(),
                size,
                Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.11, 0.0, 0.0)),
            ),
        ])
        .unwrap();
        let rig = RigGeometry::from_rig(&sensor_rig).unwrap();
        let right_from_left = sensor_rig.b_from_a(0, 1).unwrap();
        // Near the four image corners, 1.6-1.9 m away: the raw pair is 3.0-3.5 px
        // off its epipolar line there, outside the 2 px band.
        let scene = [
            Vec3F64::new(1.2, 0.75, 1.7),
            Vec3F64::new(-1.1, 0.75, 1.7),
            Vec3F64::new(1.35, -0.85, 1.9),
            Vec3F64::new(-1.0, -0.72, 1.6),
        ];
        let mut raw_xy = Vec::new();
        let mut right_xy = Vec::new();
        for p in &scene {
            let l = camera.project(p).unwrap();
            let r = camera.project(&right_from_left.transform_point(p)).unwrap();
            raw_xy.push([l.x as f32, l.y as f32]);
            right_xy.push([r.x as f32, r.y as f32]);
        }
        let n = scene.len();
        raw_xy.extend(right_xy);
        let undist_xy: Vec<[f32; 2]> = raw_xy
            .iter()
            .map(|kp| crate::frame::undistort_keypoint(&camera, kp))
            .collect();
        let descriptors = (0..n as u8).chain(0..n as u8).map(descriptor).collect();
        let features = features(raw_xy, descriptors);
        let ranges = [0..n, n..2 * n];
        let matches = match_overlap(&features, &undist_xy, &ranges, &rig, &right_from_left);
        assert_eq!(matches.num_matched(), n);
        for (i, p) in scene.iter().enumerate() {
            let got = matches.points[i].unwrap();
            assert!((got - *p).length() < 0.02, "{got:?} vs {p:?}");
        }
        let raw = match_overlap(
            &features,
            &features.keypoints_xy,
            &ranges,
            &rig,
            &right_from_left,
        );
        assert_eq!(raw.num_matched(), 0);
    }
}
