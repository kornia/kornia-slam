//! Cross-camera matching inside the overlap of two rig cameras, and
//! triangulation of the matches (ORB-SLAM3's non-rectified stereo: matches
//! searched along the epipolar line the extrinsics give, as in
//! `ORBmatcher::SearchForTriangulation`). The overlap comes from the
//! extrinsics rather than ORB-SLAM3's fixed x-intervals (idea from visloc-rs
//! PR #247).

use std::ops::Range;
use std::sync::Arc;

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::{Pose3d, TriangulationConfig, triangulate_matched_points_two_cameras};
use kornia_algebra::{Vec2F64, Vec3F64};
use kornia_imgproc::features::OrbMatchConfig;

use crate::features::{Descriptor, Features, RBrief};
use crate::frame::{is_finite_xy, undistorted_camera};
use crate::guided_match::{EpipolarGeometry, GuidedView, guided_matches};
use crate::mapping::growth::accepted_pair_claims;
use crate::rig::RigGeometry;

/// Largest reprojection error of a triangulated match at octave 0, in pixels.
const MAX_REPROJECTION_PX: f64 = 2.0;

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

/// Matches the overlap of rig cameras 0 and 1 frame after frame, with the
/// geometry fixed by the extrinsics computed once.
pub(crate) struct OverlapMatcher {
    rig: Arc<RigGeometry>,
    /// `T_10`, the pose of camera 1 relative to the primary.
    right_from_left: Pose3d,
    /// Undistorted pinholes of cameras 0 and 1: the overlap reads pinhole pixels.
    cameras: [PinholeCamera; 2],
    epipolar: EpipolarGeometry,
    scale_factor: f64,
    match_config: OrbMatchConfig,
    triangulation: TriangulationConfig,
    scratch: Scratch,
}

/// Per-frame working buffers, kept between frames so they are not reallocated.
#[derive(Default)]
struct Scratch {
    left: Vec<usize>,
    right: Vec<usize>,
    pts0: Vec<Vec2F64>,
    pts1: Vec<Vec2F64>,
    thresholds: Vec<[f64; 2]>,
    /// `(Hamming distance, index into the triangulated points)`.
    ranked: Vec<(u32, usize)>,
    triangulated: Vec<(usize, usize)>,
}

impl OverlapMatcher {
    /// `right_from_left` is the rig's `T_10` ([`kornia_sensors::SensorRig::b_from_a`]);
    /// `scale_factor` the detector's octave scale. `None` without a second
    /// camera or a baseline.
    pub(crate) fn new(
        rig: Arc<RigGeometry>,
        right_from_left: Pose3d,
        scale_factor: f64,
    ) -> Option<Self> {
        let (view0, view1) = (rig.view(0)?, rig.view(1)?);
        let epipolar = EpipolarGeometry::between(
            view0.pinhole,
            &Pose3d::IDENTITY,
            view1.pinhole,
            &right_from_left,
        )?;
        let cameras = [
            undistorted_camera(view0.pinhole),
            undistorted_camera(view1.pinhole),
        ];
        Some(Self {
            right_from_left,
            cameras,
            epipolar,
            scale_factor,
            // `TH_HIGH`: the two views differ by the rig's rotation and meet at
            // the stretched edges of wide pinhole images, where `TH_LOW` (50)
            // leaves almost no match; the epipolar band and the triangulation
            // gates reject what this lets through.
            match_config: OrbMatchConfig {
                th_low: 100,
                nn_ratio: 0.9,
                ..Default::default()
            },
            // 0.25 deg is about 25 m with Aria's 11 cm baseline; below it the
            // depth is undetermined.
            triangulation: TriangulationConfig {
                min_parallax_deg: 0.25,
                max_reprojection_error: MAX_REPROJECTION_PX,
                ..Default::default()
            },
            scratch: Scratch::default(),
            rig,
        })
    }

    pub(crate) fn rig(&self) -> &Arc<RigGeometry> {
        &self.rig
    }

    /// Matches primary-camera features (`ranges[0]`) with those of camera 1
    /// (`ranges[1]`) where both cameras see the same scene. Feature indices are
    /// those of the concatenated `features`. The overlap boxes test raw
    /// `keypoints_xy`; the epipolar search and triangulation read `pixels`, the
    /// pinhole coordinates parallel to them (non-finite entries are skipped).
    pub(crate) fn run(
        &mut self,
        features: &Features<RBrief>,
        pixels: &[[f32; 2]],
        ranges: &[Range<usize>],
    ) -> OverlapMatches {
        let n_primary = ranges.first().map_or(0, |r| r.len());
        let mut out = OverlapMatches {
            left_to_right: vec![None; n_primary],
            points: vec![None; n_primary],
        };
        let (Some(left), Some(right), Some(view0), Some(view1)) = (
            ranges.first(),
            ranges.get(1),
            self.rig.view(0),
            self.rig.view(1),
        ) else {
            return out;
        };
        let (Some(box0), Some(box1)) = (view0.overlap, view1.overlap) else {
            return out;
        };
        let kps = &features.keypoints_xy;
        let in_box = |b: crate::rig::PixelBox| {
            move |&i: &usize| b.contains(kps[i]) && is_finite_xy(&pixels[i])
        };
        let scratch = &mut self.scratch;
        scratch.left.clear();
        scratch.left.extend(left.clone().filter(in_box(box0)));
        scratch.right.clear();
        scratch.right.extend(right.clone().filter(in_box(box1)));
        if scratch.left.is_empty() || scratch.right.is_empty() {
            return out;
        }

        let pixel = |i: usize| pixels.get(i).copied().filter(is_finite_xy);
        let pairs = guided_matches(
            &GuidedView {
                features,
                candidates: &scratch.left,
                pixel,
            },
            &GuidedView {
                features,
                candidates: &scratch.right,
                pixel,
            },
            &self.epipolar,
            self.scale_factor,
            &self.match_config,
            1.0,
        );

        let to_vec = |xy: [f32; 2]| Vec2F64::new(xy[0] as f64, xy[1] as f64);
        let max_px = |octave: u8| MAX_REPROJECTION_PX * self.scale_factor.powi(octave as i32);
        scratch.pts0.clear();
        scratch.pts1.clear();
        scratch.thresholds.clear();
        for &(i, j) in &pairs {
            scratch.pts0.push(to_vec(pixels[i]));
            scratch.pts1.push(to_vec(pixels[j]));
            scratch
                .thresholds
                .push([max_px(features.octaves[i]), max_px(features.octaves[j])]);
        }
        let Ok(triangulated) = triangulate_matched_points_two_cameras(
            &scratch.pts0,
            &scratch.pts1,
            &Pose3d::IDENTITY,
            &self.right_from_left,
            &self.cameras[0],
            &self.cameras[1],
            &self.triangulation,
            Some(&scratch.thresholds),
        ) else {
            return out;
        };

        // A feature holds one match in each camera. The matcher lets two
        // primary features pick the same camera-1 feature, so the closest
        // descriptor claims it first; a stable sort keeps feature order on ties.
        scratch.ranked.clear();
        scratch
            .ranked
            .extend(triangulated.iter().enumerate().map(|(k, tp)| {
                let (i, j) = pairs[tp.pair_index];
                (
                    features.descriptors[i].distance(&features.descriptors[j]),
                    k,
                )
            }));
        scratch.ranked.sort_by_key(|&(distance, _)| distance);
        scratch.triangulated.clear();
        scratch.triangulated.extend(
            scratch
                .ranked
                .iter()
                .map(|&(_, k)| pairs[triangulated[k].pair_index]),
        );
        for accepted in accepted_pair_claims(&scratch.triangulated) {
            let (i, j) = scratch.triangulated[accepted];
            let k = scratch.ranked[accepted].1;
            let slot = i - left.start;
            out.left_to_right[slot] = Some(j);
            out.points[slot] = Some(triangulated[k].position);
        }
        out
    }
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

    /// One frame through a fresh matcher at ORB's 1.2 octave scale.
    fn match_overlap(
        features: &Features<RBrief>,
        pixels: &[[f32; 2]],
        ranges: &[Range<usize>],
        rig: &RigGeometry,
        right_from_left: &Pose3d,
    ) -> OverlapMatches {
        OverlapMatcher::new(Arc::new(rig.clone()), *right_from_left, 1.2)
            .unwrap()
            .run(features, pixels, ranges)
    }

    fn sensor_rig() -> SensorRig {
        let yaw = crate::rig::tests::yaw(40.0);
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
        let raw = crate::rig::tests::aria_lens();
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

    /// Camera 1 sits 0.3 m ahead of the primary, so the epipole is the
    /// principal point. A point 1.8 px from it lies on its epipolar line,
    /// which is not degenerate, and only the epipole guard refuses it; one far
    /// off the axis still matches.
    #[test]
    fn overlap_rejects_a_match_at_the_epipole() {
        let size = ImageSize {
            width: 640,
            height: 480,
        };
        let sensor_rig = SensorRig::from_cameras(vec![
            RigCamera::primary(camera(), size),
            RigCamera::new(
                camera(),
                size,
                Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.0, 0.0, 0.3)),
            ),
        ])
        .unwrap();
        let rig = RigGeometry::from_rig(&sensor_rig).unwrap();
        let right_from_left = sensor_rig.b_from_a(0, 1).unwrap();
        let scene = [Vec3F64::new(0.02, 0.0, 3.0), Vec3F64::new(1.5, 0.5, 3.0)];
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
        keypoints_xy.extend(right_xy);
        let descriptors = (0..2).chain(0..2).map(descriptor).collect();
        let features = features(keypoints_xy, descriptors);
        let mut matcher = OverlapMatcher::new(Arc::new(rig), right_from_left, 1.2).unwrap();
        let candidates = [0, 1, 2, 3];
        let view = |candidates| GuidedView {
            features: &features,
            candidates,
            pixel: |i: usize| features.keypoints_xy.get(i).copied(),
        };
        let pairs = guided_matches(
            &view(&candidates[..2]),
            &view(&candidates[2..]),
            &matcher.epipolar,
            1.2,
            &matcher.match_config,
            1.0,
        );
        assert_eq!(pairs, vec![(1, 3)]);
        let matches = matcher.run(&features, &features.keypoints_xy, &[0..2, 2..4]);
        assert_eq!(matches.left_to_right, vec![None, Some(3)]);
    }

    /// The matcher's geometry is built in `new` only, and a frame of the same
    /// size reuses the previous frame's buffers.
    #[test]
    fn overlap_matcher_reuses_its_buffers() {
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
        let mut matcher = OverlapMatcher::new(Arc::new(rig), right_from_left, 1.2).unwrap();
        let ranges = [0..n, n..2 * n];
        let first = matcher.run(&features, &features.keypoints_xy, &ranges);
        let scratch = |m: &OverlapMatcher| {
            [
                m.scratch.left.as_ptr() as usize,
                m.scratch.right.as_ptr() as usize,
                m.scratch.pts0.as_ptr() as usize,
                m.scratch.pts1.as_ptr() as usize,
                m.scratch.thresholds.as_ptr() as usize,
                m.scratch.triangulated.as_ptr() as usize,
            ]
        };
        let pointers = scratch(&matcher);
        let fundamental = matcher.epipolar;
        let second = matcher.run(&features, &features.keypoints_xy, &ranges);
        assert_eq!(first, second);
        assert_eq!(first.num_matched(), n);
        assert_eq!(scratch(&matcher), pointers);
        assert_eq!(
            format!("{fundamental:?}"),
            format!("{:?}", matcher.epipolar)
        );
    }

    /// Two primary features on camera 1's ray through one of its features
    /// both match it and both triangulate, at different depths: the exact
    /// descriptor claims it, not the earlier feature 40 bits away.
    #[test]
    fn the_closest_descriptor_claims_a_shared_feature() {
        let sensor_rig = sensor_rig();
        let rig = RigGeometry::from_rig(&sensor_rig).unwrap();
        let right_from_left = sensor_rig.b_from_a(0, 1).unwrap();
        let p = Vec3F64::new(1.5, -0.3, 4.0);
        let centre1 = right_from_left.camera_centre();
        let q = centre1 + (p - centre1) * 0.6;
        let pixel = |p: &Vec3F64| {
            let px = camera().project_to_pixel(p, 0.0).unwrap();
            [px.x as f32, px.y as f32]
        };
        let r = pixel(&right_from_left.transform_point(&p));
        let exact = descriptor(0);
        let mut near = exact;
        for byte in &mut near.0[..5] {
            *byte = !*byte;
        }
        assert_eq!(near.distance(&exact), 40);
        let features = features(vec![pixel(&q), pixel(&p), r], vec![near, exact, exact]);
        let matches = match_overlap(
            &features,
            &features.keypoints_xy,
            &[0..2, 2..3],
            &rig,
            &right_from_left,
        );
        assert_eq!(matches.left_to_right, vec![None, Some(2)]);
        let got = matches.points[1].unwrap();
        assert!((got - p).length() < 0.05 * p.z, "{got:?} vs {p:?}");
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

    #[test]
    fn two_exact_candidates_on_the_line_are_ambiguous() {
        let sensor_rig = sensor_rig();
        let rig = RigGeometry::from_rig(&sensor_rig).unwrap();
        let right_from_left = sensor_rig.b_from_a(0, 1).unwrap();
        let p = Vec3F64::new(1.5, -0.3, 4.0);
        let l = camera().project_to_pixel(&p, 0.0).unwrap();
        let r = camera()
            .project_to_pixel(&right_from_left.transform_point(&p), 0.0)
            .unwrap();
        let (l, r) = ([l.x as f32, l.y as f32], [r.x as f32, r.y as f32]);
        let features = features(vec![l, r, r], vec![descriptor(3); 3]);
        let matches = match_overlap(
            &features,
            &features.keypoints_xy,
            &[0..1, 1..3],
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
