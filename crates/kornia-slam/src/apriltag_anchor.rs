//! AprilTag-based metric anchor.
//!
//! Observes a known physical tag from streamed keyframes and accumulates
//! camera-frame snapshots of the tag's pose. Once the tag has been seen from at
//! least two distinct keyframes, [`AprilTagAnchor::solve`] fits a Sim3 (scale,
//! rotation, translation) mapping the SLAM map frame onto the metric tag frame,
//! which can then be applied globally with
//! [`crate::mapping::map::Map::apply_world_similarity`].
//!
//! Units are metres and the tag frame is the one `kornia_apriltag`'s planar
//! pose uses: origin at the tag centre, `+X`/`+Y` in the printed plane, `+Z`
//! out of it.

use std::collections::HashSet;

use thiserror::Error;

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;
pub use kornia_apriltag::family::TagFamilyKind;
use kornia_apriltag::{AprilTagDecoder, DecodeTagsConfig};
use kornia_image::{Image, ImageSize};

use crate::sim3::{
    HuberIrlsConfig, Sim3Alignment, align_sim3, align_sim3_gravity_preserving, align_sim3_robust,
    median_of,
};

/// Configuration for the AprilTag anchor.
#[derive(Debug, Clone)]
pub struct AprilTagAnchorConfig {
    /// Decode only this tag id.
    pub tag_id: u16,
    /// Physical tag size in metres (square tag edge length).
    pub tag_size_m: f64,
    /// Tag family used by the decoder.
    pub family: TagFamilyKind,
    /// Minimum acceptable decision margin (tag finder confidence score).
    pub min_decision_margin: f32,
    /// Maximum tolerable Hamming distance from the canonical tag code.
    pub max_hamming: u8,
    /// Robust-weights configuration used by [`AprilTagAnchor::solve`].
    pub robust: HuberIrlsConfig,
}

impl Default for AprilTagAnchorConfig {
    fn default() -> Self {
        Self {
            tag_id: 0,
            tag_size_m: 0.2,
            family: TagFamilyKind::Tag36H11,
            min_decision_margin: 30.0,
            max_hamming: 0,
            robust: HuberIrlsConfig::default(),
        }
    }
}

/// One camera-frame snapshot of the anchor tag's pose.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TagObservation {
    /// Index of the keyframe in which the tag was observed.
    pub kf_idx: usize,
    /// Tag pose in the *camera* frame (`T_cam<-tag`), computed by
    /// [`kornia_apriltag`]'s planar-pose solver from the image quad, intrinsics
    /// and tag size. This is independent of the SLAM keyframe pose, so it stays
    /// valid even after local BA refines the keyframe; the map-frame corners
    /// are resolved from it at solve time using the keyframe's final pose.
    pub tag_to_cam: Pose3d,
}

/// Errors produced by [`AprilTagAnchor::solve`].
#[derive(Debug, Error, PartialEq)]
pub enum AprilTagAnchorError {
    /// Too few distinct keyframes have observed the tag to constrain the full
    /// similarity transform.
    #[error(
        "insufficient tag observations: {keyframes} distinct keyframe(s) observed the tag (need at least 2)"
    )]
    InsufficientObservations {
        /// Number of distinct keyframes that observed the tag.
        keyframes: usize,
    },
    /// The pipeline has no AprilTag anchor configured.
    #[error("no AprilTag anchor configured")]
    AnchorNotConfigured,
    /// The correspondence fit was rejected: a defensive path, since the
    /// observations are always flattened into equal-length point sets.
    #[error(transparent)]
    Sim3(#[from] crate::sim3::Sim3Error),
    /// The robust Sim3 solve produced a degenerate fit (non-finite or
    /// implausible scale, or a residual far larger than the tag size). The map
    /// is left untouched rather than being silently corrupted.
    #[error(
        "degenerate Sim3 fit: scale={scale:.6} median_residual={med:.6} (tag size {tag_size_m:.3} m)"
    )]
    DegenerateFit {
        /// Fitted scale factor.
        scale: f64,
        /// Median residual of the fitted corners in metres.
        med: f64,
        /// Physical tag size in metres.
        tag_size_m: f64,
    },
    /// The IMU-only gravity-preserving mode needs the tag plane roughly
    /// perpendicular to gravity: a near-vertical tag cannot be represented by
    /// a yaw-about-gravity correction.
    #[error(
        "tag plane is not gravity-aligned: |tag normal . gravity| = {cosine:.3} < {min:.3} \
         (mount the tag level, or the IMU map needs full-Sim3 mode)"
    )]
    TagNotGravityAligned {
        /// |tag-plane normal · gravity unit vector|.
        cosine: f64,
        /// Rejection threshold.
        min: f64,
    },
}

/// Accumulates anchor-tag observations across keyframes.
pub struct AprilTagAnchor {
    config: AprilTagAnchorConfig,
    /// Lazily constructed on the first keyframe, once the image size is known.
    decoder: Option<AprilTagDecoder>,
    /// The image size `decoder` was built for; a change rebuilds it.
    decoder_size: Option<(usize, usize)>,
    observations: Vec<TagObservation>,
}

/// Canonical tag-corner positions in the tag frame, ordered top-right,
/// bottom-right, bottom-left, top-left. This winding matches
/// `Detection::estimate_pose`'s object points (`(+s, -s)`, `(+s, +s)`,
/// `(-s, +s)`, `(-s, -s)`, z = 0), so each observed corner lands on the same
/// canonical coordinate across keyframes.
fn canonical_corners(tag_size_m: f64) -> [Vec3F64; 4] {
    let s = tag_size_m / 2.0;
    [
        Vec3F64::new(s, -s, 0.0),  // TR
        Vec3F64::new(s, s, 0.0),   // BR
        Vec3F64::new(-s, s, 0.0),  // BL
        Vec3F64::new(-s, -s, 0.0), // TL
    ]
}

impl AprilTagAnchor {
    /// Creates an anchor with the given configuration.
    pub fn new(config: AprilTagAnchorConfig) -> Self {
        Self {
            config,
            decoder: None,
            decoder_size: None,
            observations: Vec::new(),
        }
    }

    /// Returns the accumulated observations.
    pub fn observations(&self) -> &[TagObservation] {
        &self.observations
    }

    /// Total tag detections accumulated across keyframes.
    pub fn num_observations(&self) -> usize {
        self.observations.len()
    }

    /// Number of distinct keyframes that observed the tag.
    pub fn num_distinct_keyframes(&self) -> usize {
        self.observations
            .iter()
            .map(|o| o.kf_idx)
            .collect::<std::collections::HashSet<_>>()
            .len()
    }

    /// Returns the anchor configuration.
    pub fn config(&self) -> &AprilTagAnchorConfig {
        &self.config
    }

    /// Test-only: seeds an observation directly, bypassing image decoding.
    #[cfg(test)]
    pub(crate) fn push_observation_for_test(&mut self, obs: TagObservation) {
        self.observations.push(obs);
    }

    /// Ensures a decoder exists for a `width`x`height` image, rebuilding it if
    /// the size changed. The decoder caches working buffers for the size it was
    /// constructed with, so a resolution change must not reuse a stale one.
    /// Returns `false` when a decoder for this size cannot be built.
    fn ensure_decoder(&mut self, width: usize, height: usize) -> bool {
        let size_key = (width, height);
        if self.decoder_size == Some(size_key) && self.decoder.is_some() {
            return true;
        }
        let Ok(decode_config) = DecodeTagsConfig::new(vec![self.config.family.clone()]) else {
            self.decoder = None;
            self.decoder_size = None;
            return false;
        };
        let size = ImageSize { width, height };
        match AprilTagDecoder::new(decode_config, size) {
            Ok(decoder) => {
                self.decoder = Some(decoder);
                self.decoder_size = Some(size_key);
                true
            }
            Err(_) => {
                self.decoder = None;
                self.decoder_size = None;
                false
            }
        }
    }

    /// Ingests one keyframe image.
    ///
    /// The decoder is built lazily on the first call (or rebuilt when the image
    /// size changes). The image is decoded, detections for the configured tag
    /// are filtered (`id`, `hamming`, `decision_margin`), and the best match's
    /// camera-frame tag pose (`T_cam<-tag`) is stored. The pose is independent
    /// of the SLAM keyframe pose, so local-BA refinement later does not
    /// invalidate it; [`AprilTagAnchor::solve`] resolves it into the map frame
    /// using the keyframe's final pose.
    /// Decoding or pose errors are ignored: a failed frame must never poison
    /// the SLAM chain.
    pub fn on_keyframe(&mut self, kf_idx: usize, image: &Image<u8, 1>, camera: &PinholeCamera) {
        if !self.ensure_decoder(image.width(), image.height()) {
            return;
        }

        let detections = match self.decoder.as_mut() {
            Some(decoder) => match decoder.decode(image) {
                Ok(decoded) => decoded,
                Err(_) => return,
            },
            None => return,
        };

        let Some(detection) = detections
            .into_iter()
            .filter(|det| {
                det.id == self.config.tag_id
                    && det.hamming <= self.config.max_hamming
                    && det.decision_margin >= self.config.min_decision_margin
            })
            .max_by(|a, b| a.decision_margin.total_cmp(&b.decision_margin))
        else {
            return;
        };

        // T_cam<-tag from the planar-pose solver, in the camera frame and
        // independent of the SLAM keyframe pose.
        let pose_pair = match detection.estimate_pose(camera, self.config.tag_size_m, 50) {
            Ok(pair) => pair,
            Err(_) => return,
        };

        self.observations.push(TagObservation {
            kf_idx,
            tag_to_cam: pose_pair.best.pose,
        });
    }

    /// Resolves a stored observation into map-frame tag corners using the
    /// keyframe's (final) world-to-camera pose. Returns `None` if the pose is
    /// unavailable (e.g. the keyframe was culled).
    fn resolve_observation(
        &self,
        obs: &TagObservation,
        pose_world_to_cam: &Pose3d,
    ) -> [Vec3F64; 4] {
        // Compose into map frame: tag -> cam -> world.
        let tag_to_map = pose_world_to_cam.inverse().compose(&obs.tag_to_cam);
        canonical_corners(self.config.tag_size_m).map(|p| tag_to_map.transform_point(&p))
    }

    /// Fits the map -> tag-frame Sim3 from the accumulated observations.
    ///
    /// `kf_pose` must return the keyframe's *final* world-to-camera pose (e.g.
    /// read from the live map under the map lock). Resolving the stored
    /// camera-frame tag poses against these final poses — rather than against
    /// the provisional pose seen at ingest — makes the fit independent of when
    /// each keyframe was observed and of any later local-BA refinement.
    ///
    /// Observations are screened first (by implied scale), then the largest
    /// mutually consistent group is found by RANSAC and refined with the
    /// robust weighted fit. At least two distinct keyframes are required.
    ///
    /// The Huber threshold is floored at half the tag size, so the robustifier
    /// is never a no-op linear region for a small tag; a caller-supplied larger
    /// `delta` still wins.
    pub fn solve(
        &self,
        kf_pose: impl Fn(usize) -> Option<Pose3d>,
    ) -> Result<Sim3Alignment, AprilTagAnchorError> {
        self.solve_mode(kf_pose, None)
    }

    /// Gravity-preserving variant for IMU-initialized maps: translation + a yaw
    /// rotation about `gravity_axis`, with scale fixed at 1. An IMU-initialized
    /// world frame is already gravity-aligned and metric, so a full Sim3 would
    /// tilt it off gravity; this keeps the gravity direction an exact fixed
    /// point (ORB-SLAM3's four-degree-of-freedom inertial correction).
    pub fn solve_gravity_preserving(
        &self,
        kf_pose: impl Fn(usize) -> Option<Pose3d>,
        gravity_axis: Vec3F64,
    ) -> Result<Sim3Alignment, AprilTagAnchorError> {
        self.solve_mode(kf_pose, Some(gravity_axis))
    }

    /// Shared solve path: resolve observations, keep the largest consistent
    /// group by RANSAC, refine it, and gate the result.
    ///
    /// A misdecoded tag, or a far/oblique detection whose planar-pose solve is
    /// garbage, relocates the tag's rigid square elsewhere in the map — it
    /// agrees with the population on shape and scale but not on pose, so it
    /// cannot be removed by per-observation screening alone, and enough of them
    /// collapse a global fit toward zero scale. RANSAC finds the majority-
    /// consistent group first, which the robust fit then refines.
    fn solve_mode(
        &self,
        kf_pose: impl Fn(usize) -> Option<Pose3d>,
        gravity: Option<Vec3F64>,
    ) -> Result<Sim3Alignment, AprilTagAnchorError> {
        let canonical = canonical_corners(self.config.tag_size_m);
        let quads = self.resolved_quads(kf_pose, &canonical)?;
        let (est, gt) = largest_consistent_group(&quads, &canonical, self.config.tag_size_m)?;

        let fit = match gravity {
            Some(axis) => {
                ensure_tag_gravity_aligned(&est, axis)?;
                align_sim3_gravity_preserving(&est, &gt, axis)?
            }
            None => {
                // A tag-scaled Huber threshold: residuals are in metres and a
                // wrong corner is off by a good fraction of the tag, so the
                // threshold must sit around the tag scale. `.min` keeps it
                // there (an explicit smaller override still wins); `.max`
                // would take the 1.0 m generic default and silently disable
                // the weighting for any real tag.
                let robust = HuberIrlsConfig {
                    delta: self.config.robust.delta.min(0.5 * self.config.tag_size_m),
                    ..self.config.robust
                };
                align_sim3_robust(&est, &gt, robust)?
            }
        };
        self.gate_fit(fit, &est, &gt)
    }

    /// Resolves every stored observation against the keyframe's final pose and
    /// drops observations implying a wildly different tag scale. At least two
    /// distinct keyframes are required.
    ///
    /// A pose-estimation error corrupts all four corners of an observation
    /// together, so an observation's *shape* — the scale its corner geometry
    /// implies relative to the canonical tag ([`implied_scale`]) — is a
    /// unit-free first filter: a misdecoded tag implies a scale far from the
    /// population's, whatever the map's own units are.
    fn resolved_quads(
        &self,
        kf_pose: impl Fn(usize) -> Option<Pose3d>,
        canonical: &[Vec3F64; 4],
    ) -> Result<Vec<[Vec3F64; 4]>, AprilTagAnchorError> {
        // Observations whose keyframe has no final pose (e.g. culled) are
        // dropped.
        let resolved: Vec<(usize, [Vec3F64; 4])> = self
            .observations
            .iter()
            .filter_map(|obs| {
                kf_pose(obs.kf_idx).map(|pose| (obs.kf_idx, self.resolve_observation(obs, &pose)))
            })
            .collect();

        let distinct: HashSet<usize> = resolved.iter().map(|(kf_idx, _)| *kf_idx).collect();
        if distinct.len() < 2 {
            return Err(AprilTagAnchorError::InsufficientObservations {
                keyframes: distinct.len(),
            });
        }

        let scales: Vec<f64> = resolved
            .iter()
            .map(|(_, corners)| implied_scale(corners, canonical))
            .collect();
        let med_scale = median_of(&scales);
        let screened: Vec<&(usize, [Vec3F64; 4])> = if med_scale > 0.0 {
            resolved
                .iter()
                .zip(&scales)
                .filter(|(_, s)| {
                    let s = **s;
                    s > med_scale / 4.0 && s < med_scale * 4.0
                })
                .map(|(obs, _)| obs)
                .collect()
        } else {
            resolved.iter().collect()
        };

        let distinct_survivors: HashSet<usize> =
            screened.iter().map(|(kf_idx, _)| *kf_idx).collect();
        if distinct_survivors.len() < 2 {
            return Err(AprilTagAnchorError::InsufficientObservations {
                keyframes: distinct_survivors.len(),
            });
        }

        Ok(screened.iter().map(|(_, corners)| *corners).collect())
    }

    /// Rejects a degenerate fit (implausible scale, or a residual far larger
    /// than the tag) so a bad solve never corrupts the map.
    ///
    /// A fit that collapses every corner onto the tag center yields residuals
    /// all near the tag half-diagonal (~0.71 * tag size); a genuine fit on
    /// noisy hand-held data can scatter corners by a good fraction of the tag
    /// size (uncalibrated intrinsics dominate). The scale gate is the reliable
    /// discriminator for the observed collapse (scale -> 0); the median
    /// residual is a coarse sanity check that catches genuinely broken fits.
    fn gate_fit(
        &self,
        fit: Sim3Alignment,
        est: &[Vec3F64],
        gt: &[Vec3F64],
    ) -> Result<Sim3Alignment, AprilTagAnchorError> {
        let residuals: Vec<f64> = est
            .iter()
            .zip(gt)
            .map(|(e, g)| {
                let d = fit.apply(*e) - *g;
                (d.x * d.x + d.y * d.y + d.z * d.z).sqrt()
            })
            .collect();
        let med = median_of(&residuals);

        let scale = fit.scale;
        if !scale.is_finite()
            || !scale.is_sign_positive()
            || scale <= 1e-6
            || scale >= 1e6
            || med > 1.5 * self.config.tag_size_m
        {
            return Err(AprilTagAnchorError::DegenerateFit {
                scale,
                med,
                tag_size_m: self.config.tag_size_m,
            });
        }

        Ok(fit)
    }
}

/// The smallest |tag normal · gravity| the gravity-preserving mode accepts,
/// i.e. the tag may be tilted up to 60 degrees from level. A wall/vertical tag
/// (cosine near 0) is refused: the yaw-only correction cannot represent it and
/// the tilt residual is too small to trip the fit gate.
const MIN_TAG_GRAVITY_COSINE: f64 = 0.5;

/// Rejects a tag whose plane is not roughly perpendicular to gravity, for the
/// IMU-only gravity-preserving mode. Skips the check when the plane normal is
/// degenerate (a collinear quad), leaving that to the fit gate.
fn ensure_tag_gravity_aligned(
    corners: &[Vec3F64],
    gravity_axis: Vec3F64,
) -> Result<(), AprilTagAnchorError> {
    let Some(normal) = tag_plane_normal(corners) else {
        return Ok(());
    };
    let g_len = gravity_axis.length();
    let cosine = if g_len > f64::EPSILON {
        (normal.dot(gravity_axis / g_len)).abs()
    } else {
        0.0
    };
    if cosine < MIN_TAG_GRAVITY_COSINE {
        return Err(AprilTagAnchorError::TagNotGravityAligned {
            cosine,
            min: MIN_TAG_GRAVITY_COSINE,
        });
    }
    Ok(())
}

/// Unit normal of the plane through a solved corner set, or `None` when the
/// corners are collinear/degenerate.
fn tag_plane_normal(corners: &[Vec3F64]) -> Option<Vec3F64> {
    let base = *corners.first()?;
    let a = *corners.get(1)? - base;
    let b = *corners.get(2)? - base;
    let n = a.cross(b);
    let len = n.length();
    (len > f64::EPSILON).then(|| n / len)
}

/// Largest group of observations that a single similarity reproduces.
///
/// Every observation is a rigid, scaled copy of the canonical tag; a
/// hypothesis fitted from any pair (8 coplanar corners, which fix the plane
/// normal and so the tilt) is scored by how many observations' corners it
/// reproduces within `tol`. The winning group's correspondences are returned
/// for refinement. A relocated tag — a misdecode, or a far/oblique detection
/// whose planar-pose solve is garbage — agrees on shape but not on pose and is
/// left out of the consensus.
fn largest_consistent_group(
    quads: &[[Vec3F64; 4]],
    canonical: &[Vec3F64; 4],
    tag_size_m: f64,
) -> Result<(Vec<Vec3F64>, Vec<Vec3F64>), AprilTagAnchorError> {
    let n = quads.len();

    let mut best_inliers: Vec<usize> = Vec::new();
    let mut best_median = f64::INFINITY;
    for i in 0..n {
        for j in (i + 1)..n {
            let mut est = Vec::with_capacity(8);
            let mut gt = Vec::with_capacity(8);
            est.extend(quads[i]);
            gt.extend(canonical);
            est.extend(quads[j]);
            gt.extend(canonical);
            // A hypothesis whose fit is refused cannot score inliers; skip it.
            let Ok(hypothesis) = align_sim3(&est, &gt) else {
                continue;
            };

            // The hypothesis maps map coordinates onto the canonical tag frame,
            // so `worst` is in metres — and half a tag is a fixed metric
            // tolerance, independent of the map's arbitrary scale. A collapsed
            // hypothesis (scale -> 0, every corner near the tag centre) has a
            // worst-corner residual of at least `tag_size / sqrt(2)` (~0.707 of
            // a tag), so it can never pass this tolerance; scaling the
            // tolerance by the map scale would let it pass on large maps.
            let tol = 0.5 * tag_size_m;
            let mut residuals = Vec::with_capacity(n);
            let mut inliers = Vec::new();
            for (k, quad) in quads.iter().enumerate() {
                let worst = quad
                    .iter()
                    .zip(canonical)
                    .map(|(c, g)| (hypothesis.apply(*c) - *g).length())
                    .fold(0.0_f64, f64::max);
                residuals.push(worst);
                if worst <= tol {
                    inliers.push(k);
                }
            }
            let median = median_of(&residuals);

            // More inliers wins; ties break on the lower median residual.
            if inliers.len() > best_inliers.len()
                || (inliers.len() == best_inliers.len() && median < best_median)
            {
                best_inliers = inliers;
                best_median = median;
            }
        }
    }

    if best_inliers.len() < 2 {
        return Err(AprilTagAnchorError::InsufficientObservations {
            keyframes: best_inliers.len(),
        });
    }

    let mut est = Vec::with_capacity(best_inliers.len() * 4);
    let mut gt = Vec::with_capacity(best_inliers.len() * 4);
    for &k in &best_inliers {
        est.extend(quads[k]);
        gt.extend(canonical);
    }
    Ok((est, gt))
}

/// Scale an observation's corner geometry implies: the median pairwise
/// distance ratio of its corners to the canonical tag corners. A rigid tag
/// observed in map frame implies one consistent ratio; a misdecoded or
/// degenerate quad implies a ratio far from the population's. Zero when the
/// canonical geometry is degenerate.
fn implied_scale(corners: &[Vec3F64; 4], canonical: &[Vec3F64; 4]) -> f64 {
    const EDGES: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    let mut ratios = [0.0_f64; 6];
    for (k, (i, j)) in EDGES.iter().enumerate() {
        let canon = (canonical[*i] - canonical[*j]).length();
        if canon <= f64::EPSILON {
            return 0.0;
        }
        ratios[k] = (corners[*i] - corners[*j]).length() / canon;
    }
    ratios.sort_by(|a, b| a.total_cmp(b));
    0.5 * (ratios[2] + ratios[3])
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_algebra::Mat3F64;

    fn vec3_norm(v: Vec3F64) -> f64 {
        (v.x * v.x + v.y * v.y + v.z * v.z).sqrt()
    }

    fn mat3_frobenius_diff(a: &Mat3F64, b: &Mat3F64) -> f64 {
        let d = *a - *b;
        let cols = [d.x_axis, d.y_axis, d.z_axis];
        cols.iter()
            .map(|c| c.x * c.x + c.y * c.y + c.z * c.z)
            .sum::<f64>()
            .sqrt()
    }

    /// Builds a keyframe-pose lookup closure that returns the given pose for
    /// every keyframe index.
    fn identity_kf_pose() -> impl Fn(usize) -> Option<Pose3d> {
        |_| Some(Pose3d::IDENTITY)
    }

    #[test]
    fn winding_convention_yields_identity_when_tag_aligned_with_map() {
        // Tag at the map origin, axes aligned: the solved anchor must be the
        // identity Sim3. If the canonical winding order ever diverges from the
        // quad corner order, the fits would disagree and this dies loudly.
        let mut anchor = AprilTagAnchor::new(AprilTagAnchorConfig::default());
        for kf_idx in [0, 1] {
            anchor.observations.push(TagObservation {
                kf_idx,
                tag_to_cam: Pose3d::IDENTITY,
            });
        }

        let fit = anchor.solve(identity_kf_pose()).unwrap();
        assert!((fit.scale - 1.0).abs() < 1e-9);
        assert!(mat3_frobenius_diff(&fit.rotation, &Mat3F64::IDENTITY) < 1e-9);
        assert!(vec3_norm(fit.translation) < 1e-9);
    }

    #[test]
    fn solve_resolves_observations_against_final_keyframe_poses() {
        // The fix: observations are stored in the camera frame and resolved
        // against the keyframe's *final* pose supplied to solve(). If solve
        // ignored the closure (e.g. assumed identity), the recovered Sim3
        // would be identity; here it must invert the supplied final pose.
        let tag_size_m = 0.2;
        let r = rotation_from_axis_angle(Vec3F64::new(0.0, 0.0, 1.0), 0.5);
        let t = Vec3F64::new(0.3, -0.2, 1.0);
        let final_pose = Pose3d::from_rt(r, t);

        let mut anchor = AprilTagAnchor::new(AprilTagAnchorConfig {
            tag_size_m,
            ..Default::default()
        });
        for kf_idx in [0, 1, 2] {
            anchor.observations.push(TagObservation {
                kf_idx,
                tag_to_cam: Pose3d::IDENTITY,
            });
        }

        let fit = anchor
            .solve(|_| Some(final_pose))
            .expect("solve must succeed with a valid final pose");
        // Resolved corners = final_pose^-1(canonical); the map->tag Sim3 must
        // invert that, recovering `final_pose` with unit scale.
        assert!((fit.scale - 1.0).abs() < 1e-6, "scale error: {}", fit.scale);
        assert!(
            mat3_frobenius_diff(&fit.rotation, &r) < 1e-6,
            "rotation error"
        );
        assert!(vec3_norm(fit.translation - t) < 1e-6, "translation error");
    }

    #[test]
    fn robust_solve_ignores_corrupted_observations() {
        let tag_size_m = 0.2;
        let r = rotation_from_axis_angle(Vec3F64::new(0.0, 0.0, 1.0), 1.1);
        let t = Vec3F64::new(0.5, -0.5, 1.0);
        let final_pose = Pose3d::from_rt(r, t);

        let mut anchor = AprilTagAnchor::new(AprilTagAnchorConfig {
            tag_size_m,
            ..Default::default()
        });
        for kf_idx in 0..10 {
            // A consistent observation: the tag aligned with the camera frame.
            let tag_to_cam = Pose3d::IDENTITY;
            // Grossly corrupt two of the ten observations: a far-away,
            // rotated tag that the screening must reject.
            let tag_to_cam = if kf_idx == 3 || kf_idx == 6 {
                Pose3d::from_rt(
                    rotation_from_axis_angle(Vec3F64::new(1.0, 0.0, 0.0), 2.0),
                    Vec3F64::new(8.0, -6.0, 4.0),
                )
            } else {
                tag_to_cam
            };
            anchor
                .observations
                .push(TagObservation { kf_idx, tag_to_cam });
        }

        let fit = anchor
            .solve(|_| Some(final_pose))
            .expect("solve must succeed with mostly-good observations");
        // Resolved corners = final_pose^-1(canonical); the map->tag Sim3
        // recovers `final_pose` (unit scale), matching the pose test above.
        assert!((fit.scale - 1.0).abs() < 1e-3, "scale error: {}", fit.scale);
        assert!(
            mat3_frobenius_diff(&fit.rotation, &r) < 5e-3,
            "rotation error"
        );
        assert!(vec3_norm(fit.translation - t) < 1e-2, "translation error");
    }

    #[test]
    fn solve_rejects_single_keyframe() {
        let mut anchor = AprilTagAnchor::new(AprilTagAnchorConfig::default());
        anchor.observations.push(TagObservation {
            kf_idx: 0,
            tag_to_cam: Pose3d::IDENTITY,
        });

        let err = match anchor.solve(identity_kf_pose()) {
            Err(err) => err,
            Ok(_) => panic!("expected solve to fail with a single keyframe"),
        };
        assert_eq!(
            err,
            AprilTagAnchorError::InsufficientObservations { keyframes: 1 }
        );
    }

    #[test]
    fn duplicate_observations_of_one_keyframe_do_not_satisfy_two_keyframes() {
        let mut anchor = AprilTagAnchor::new(AprilTagAnchorConfig::default());
        for _ in 0..5 {
            anchor.observations.push(TagObservation {
                kf_idx: 0,
                tag_to_cam: Pose3d::IDENTITY,
            });
        }
        let err = match anchor.solve(identity_kf_pose()) {
            Err(err) => err,
            Ok(_) => panic!("expected insufficient distinct keyframes"),
        };
        assert_eq!(
            err,
            AprilTagAnchorError::InsufficientObservations { keyframes: 1 }
        );

        // A second, distinct keyframe is enough.
        anchor.observations.push(TagObservation {
            kf_idx: 1,
            tag_to_cam: Pose3d::IDENTITY,
        });
        assert!(anchor.solve(identity_kf_pose()).is_ok());
    }

    #[test]
    fn gravity_mode_rejects_a_vertical_tag() {
        // A wall-mounted tag (plane normal perpendicular to gravity) cannot be
        // represented by a yaw-about-gravity correction; it must be refused
        // rather than silently mis-corrected.
        let mut anchor = AprilTagAnchor::new(AprilTagAnchorConfig {
            tag_size_m: 0.1,
            ..Default::default()
        });
        let quarter_turn =
            rotation_from_axis_angle(Vec3F64::new(1.0, 0.0, 0.0), std::f64::consts::FRAC_PI_2);
        for kf_idx in [0, 1] {
            anchor.push_observation_for_test(TagObservation {
                kf_idx,
                tag_to_cam: Pose3d::from_rt(quarter_turn, Vec3F64::new(0.0, 0.0, 1.0)),
            });
        }

        let err = anchor
            .solve_gravity_preserving(|_| Some(Pose3d::IDENTITY), Vec3F64::new(0.0, 0.0, -9.81))
            .expect_err("a vertical tag must be refused");
        assert!(matches!(
            err,
            AprilTagAnchorError::TagNotGravityAligned { .. }
        ));
    }

    #[test]
    fn decoder_is_rebuilt_when_image_size_changes() {
        let mut anchor = AprilTagAnchor::new(AprilTagAnchorConfig::default());
        assert!(anchor.ensure_decoder(640, 480));
        assert_eq!(anchor.decoder_size, Some((640, 480)));
        // Same size: the cached decoder is reused.
        assert!(anchor.ensure_decoder(640, 480));
        // Different size: rebuild for the new size.
        assert!(anchor.ensure_decoder(800, 600));
        assert_eq!(anchor.decoder_size, Some((800, 600)));
    }

    #[test]
    fn gravity_preserving_solve_keeps_scale_one_and_gravity_fixed() {
        let tag_size_m = 0.2;
        // A final keyframe pose with a tilt: the gravity-preserving fit must
        // not adopt that tilt.
        let tilt = rotation_from_axis_angle(Vec3F64::new(1.0, 0.4, 0.0), 0.6);
        let final_pose = Pose3d::from_rt(tilt, Vec3F64::new(0.4, -0.3, 1.2));
        let gravity = Vec3F64::new(0.1, -0.2, -9.81);
        let g_hat = gravity / gravity.length();

        let mut anchor = AprilTagAnchor::new(AprilTagAnchorConfig {
            tag_size_m,
            ..Default::default()
        });
        for kf_idx in [0, 1, 2] {
            anchor.observations.push(TagObservation {
                kf_idx,
                tag_to_cam: Pose3d::IDENTITY,
            });
        }

        let fit = anchor
            .solve_gravity_preserving(|_| Some(final_pose), gravity)
            .expect("gravity-preserving solve must succeed");
        assert!((fit.scale - 1.0).abs() < 1e-12, "scale must stay 1");
        let g_after = fit.rotation * g_hat;
        assert!(
            vec3_norm(g_after - g_hat) < 1e-9,
            "gravity direction moved: {}",
            vec3_norm(g_after - g_hat)
        );
    }

    fn rotation_from_axis_angle(axis: Vec3F64, angle: f64) -> Mat3F64 {
        let len = axis.length();
        let (nx, ny, nz) = (axis.x / len, axis.y / len, axis.z / len);
        let (c, s, ti) = (angle.cos(), angle.sin(), 1.0 - angle.cos());

        let col0 = Vec3F64::new(
            ti * nx * nx + c,
            ti * nx * ny + s * nz,
            ti * nx * nz - s * ny,
        );
        let col1 = Vec3F64::new(
            ti * nx * ny - s * nz,
            ti * ny * ny + c,
            ti * ny * nz + s * nx,
        );
        let col2 = Vec3F64::new(
            ti * nx * nz + s * ny,
            ti * ny * nz - s * nx,
            ti * nz * nz + c,
        );
        Mat3F64::from_cols(col0, col1, col2)
    }
}
