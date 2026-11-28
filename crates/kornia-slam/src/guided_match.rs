//! Descriptor matching guided by a known two-view geometry, shared by keyframe
//! pair growth and the rig overlap (ORB-SLAM3's `SearchForTriangulation`).

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, SO3F64, Vec2F64, Vec3F64};

use crate::features::{Descriptor, DescriptorSet, Features};

/// 1-DOF chi-square gate at 95% for point-to-epipolar-line distance
/// (ORB-SLAM3's `CheckDistEpipolarLine`), scaled per octave.
const EPIPOLAR_CHI2: f64 = 3.84;

/// Epipolar geometry of view `b` relative to view `a`, both with known poses.
#[derive(Debug, Clone, Copy)]
pub(crate) struct EpipolarGeometry {
    /// Maps a pixel of `a` to its epipolar line in `b`.
    fundamental: Mat3F64,
    /// Epipole of view `a` in view `b`'s pixels; `None` at infinity.
    epipole_px: Option<Vec2F64>,
}

impl EpipolarGeometry {
    /// Geometry between two world-to-camera poses; `None` under a 1e-8 m
    /// baseline, where the epipolar geometry degenerates and triangulation
    /// would reject everything anyway.
    pub(crate) fn between(
        camera_a: &PinholeCamera,
        pose_a: &Pose3d,
        camera_b: &PinholeCamera,
        pose_b: &Pose3d,
    ) -> Option<Self> {
        // Both poses are known, so the fundamental matrix is fully determined:
        // F = K_b^-T [t]x R K_a^-1 with (R, t) the a->b relative pose. Filtering
        // against it replaces the two-view estimator's F-matrix RANSAC.
        let rel = Pose3d::between(pose_a, pose_b);
        if rel.translation.length() <= 1e-8 {
            return None;
        }
        let t_skew = SO3F64::hat(rel.translation);
        let fundamental = k_inv(camera_b).transpose() * (t_skew * rel.rotation) * k_inv(camera_a);

        // Near the epipole every keypoint is close to every epipolar line, so
        // the chi-square gate is uninformative there: wrong matches survive and
        // triangulate to depth-garbage points that still reproject well in both
        // views. Mirrors ORB-SLAM3's epipole-proximity rejection.
        let centre_a_world = pose_a.camera_centre();
        let epipole_cam = pose_b.transform_point(&centre_a_world);
        let epipole_px = (epipole_cam.z.abs() > 1e-9).then(|| {
            Vec2F64::new(
                camera_b.fx * epipole_cam.x / epipole_cam.z + camera_b.cx,
                camera_b.fy * epipole_cam.y / epipole_cam.z + camera_b.cy,
            )
        });
        Some(Self {
            fundamental,
            epipole_px,
        })
    }
}

fn k_inv(camera: &PinholeCamera) -> Mat3F64 {
    Mat3F64::from_cols(
        Vec3F64::new(1.0 / camera.fx, 0.0, 0.0),
        Vec3F64::new(0.0, 1.0 / camera.fy, 0.0),
        Vec3F64::new(-camera.cx / camera.fx, -camera.cy / camera.fy, 1.0),
    )
}

/// The features of one view a guided match may pair: `candidates` index
/// `features`, and `pixel(i)` is feature `i`'s pinhole pixel, `None` when it
/// has none.
pub(crate) struct GuidedView<'a, D, P> {
    pub features: &'a Features<D>,
    pub candidates: &'a [usize],
    pub pixel: P,
}

/// `(a, b)` feature pairs that match by descriptor among the candidates and
/// agree with `geometry`: `b` away from the epipole and within the octave-scaled
/// chi-square band of `a`'s epipolar line, the band's variance multiplied by
/// `band_scale`. Pairs come in `a`-candidate order.
pub(crate) fn guided_matches<D, PA, PB>(
    a: &GuidedView<'_, D, PA>,
    b: &GuidedView<'_, D, PB>,
    geometry: &EpipolarGeometry,
    scale_factor: f64,
    match_config: &D::MatchConfig,
    band_scale: f64,
) -> Vec<(usize, usize)>
where
    D: Descriptor,
    PA: Fn(usize) -> Option<[f32; 2]>,
    PB: Fn(usize) -> Option<[f32; 2]>,
{
    // Brute-force descriptor matching over the candidate subsets; the global
    // second-best ratio test and orientation consistency live in the matcher and
    // are essential for match quality.
    let (a_orientations, a_descriptors) = subset(a.features, a.candidates);
    let (b_orientations, b_descriptors) = subset(b.features, b.candidates);
    let sub_matches = D::match_sets(
        DescriptorSet {
            orientations: a_orientations.as_deref(),
            descriptors: &a_descriptors,
        },
        DescriptorSet {
            orientations: b_orientations.as_deref(),
            descriptors: &b_descriptors,
        },
        match_config,
    );

    let mut pairs = Vec::new();
    for (a_sub, b_sub) in sub_matches {
        let (Some(&a_idx), Some(&b_idx)) = (a.candidates.get(a_sub), b.candidates.get(b_sub))
        else {
            continue;
        };
        let (Some(pu), Some(qu)) = ((a.pixel)(a_idx), (b.pixel)(b_idx)) else {
            continue;
        };
        let p = Vec2F64::new(pu[0] as f64, pu[1] as f64);
        let q = Vec2F64::new(qu[0] as f64, qu[1] as f64);
        let octave = b.features.octaves.get(b_idx).copied().unwrap_or(0);

        // Reject keypoints near the epipole; radius grows with octave
        // (ORB-SLAM3 uses 100 * scaleFactor^octave px^2).
        if let Some(e) = geometry.epipole_px {
            let dx = q.x - e.x;
            let dy = q.y - e.y;
            if dx * dx + dy * dy < 100.0 * scale_factor.powi(octave as i32) {
                continue;
            }
        }

        let l = geometry.fundamental * Vec3F64::new(p.x, p.y, 1.0);
        let line_norm_sq = l.x * l.x + l.y * l.y;
        if line_norm_sq <= 1e-12 {
            continue;
        }
        let d = l.x * q.x + l.y * q.y + l.z;
        let sigma_sq = scale_factor.powi(2 * octave as i32) * band_scale;
        if d * d > EPIPOLAR_CHI2 * sigma_sq * line_norm_sq {
            continue;
        }
        pairs.push((a_idx, b_idx));
    }
    pairs
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
