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
use kornia_algebra::{Vec2F32, Vec3AF32};
use kornia_imgproc::features::{OrbMatchConfig, match_orb_descriptors};

use crate::Frame;
use crate::frame::is_finite_xy;
use crate::mapping::{Keyframe, Map};
use crate::pose_conversion::{mat3_to_f32, mat3_to_f64, vec3_to_f32, vec3_to_f64};
use crate::tracking::pose_estimation::pnp::PnpConfig;
use crate::tracking::pose_estimation::rig_pnp::{RigObservation, solve_rig_pnp};
use crate::tracking::pose_estimation::{Estimate, MapProjectionEstimator};
use crate::upstream::kornia_3d::gpnp::{RayMatch, ransac_lifted_ap3p};

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
    pub solver: RelocalizationSolver,
}

/// How a relocalisation hypothesis is drawn.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RelocalizationSolver {
    /// EPnP RANSAC on primary-camera matches.
    #[default]
    PrimaryEpnp,
    /// AP3P RANSAC on primary-camera matches: a minimal set of 3, not 5.
    PrimaryAp3p,
    /// AP3P on the camera with the most matches, lifted to the rig and scored
    /// on every camera's rays, then refined by rig PnP.
    RigAp3p,
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
            solver: RelocalizationSolver::PrimaryEpnp,
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
/// enough support that `accept` also takes.
pub fn relocalize(
    frame: &Frame,
    candidates: &[usize],
    map: &Map,
    camera: &PinholeCamera,
    estimator: &MapProjectionEstimator,
    config: &RelocalizationConfig,
    mut accept: impl FnMut(&Relocalization) -> bool,
) -> Option<Relocalization> {
    candidates.iter().find_map(|&keyframe_idx| {
        let keyframe = map.get_keyframe(keyframe_idx)?;
        let correspondences = keyframe_correspondences(frame, keyframe, map, config);
        if correspondences.len() < config.min_correspondences {
            return None;
        }
        let (pose, inliers) = hypothesis(
            frame,
            keyframe,
            map,
            camera,
            &correspondences,
            &estimator.config().pnp,
            config,
            0,
        )?;
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
        (estimate.inliers >= config.min_inliers)
            .then_some(Relocalization {
                estimate,
                keyframe_idx,
            })
            .filter(|found| accept(found))
    })
}

// TODO: adapt for global descriptors (e.g. NetVLAD/DINO) — retrieve candidate
// keyframes by global embedding instead of ORB/BoW, and make local matching
// descriptor-agnostic.
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
    // Only the rig solver models cameras other than the primary.
    .filter(|&(_, keypoint)| {
        config.solver == RelocalizationSolver::RigAp3p || frame.camera_of(keypoint) == 0
    })
    .filter_map(|(keyframe_desc, keypoint)| {
        let map_point = keyframe.map_point(keyframe_desc)?;
        let point = map.map_points().get(map_point)?;
        let undistorted = frame
            .keypoints_undist
            .get(keypoint)
            .is_some_and(is_finite_xy);
        (undistorted && !point.culled).then_some((map_point, keypoint))
    })
    .collect()
}

/// A pose hypothesis from the correspondences with `config.solver`, and its
/// inlier correspondences; `seed` seeds the RANSAC sampler.
#[allow(clippy::too_many_arguments)]
fn hypothesis(
    frame: &Frame,
    keyframe: &Keyframe,
    map: &Map,
    camera: &PinholeCamera,
    correspondences: &[(usize, usize)],
    pnp_config: &PnpConfig,
    config: &RelocalizationConfig,
    seed: u64,
) -> Option<(Pose3d, Vec<(usize, usize)>)> {
    let pnp = |method| {
        ransac_pose(
            frame,
            keyframe,
            map,
            camera,
            correspondences,
            config,
            method,
            seed,
        )
    };
    match config.solver {
        RelocalizationSolver::PrimaryEpnp => pnp(PnPMethod::EPnPDefault),
        RelocalizationSolver::PrimaryAp3p => pnp(PnPMethod::AP3PDefault),
        RelocalizationSolver::RigAp3p => rig_ransac_pose(
            frame,
            map,
            camera,
            correspondences,
            pnp_config,
            config,
            seed,
        ),
    }
}

/// PnP RANSAC over the correspondences, returning the pose and its inlier
/// correspondences. Points are expressed relative to the keyframe's camera
/// centre and scaled to unit median depth for the f32 solver.
#[allow(clippy::too_many_arguments)]
fn ransac_pose(
    frame: &Frame,
    keyframe: &Keyframe,
    map: &Map,
    camera: &PinholeCamera,
    correspondences: &[(usize, usize)],
    config: &RelocalizationConfig,
    method: PnPMethod,
    seed: u64,
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
    let k = mat3_to_f32(camera.intrinsic_matrix());
    let result = solve_pnp_ransac(
        &world,
        &image,
        &k,
        None,
        method,
        &RansacParams {
            max_iterations: config.ransac_iterations,
            reproj_threshold_px: config.ransac_threshold_px,
            random_seed: Some(seed),
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

/// `RigAp3p`: AP3P on three rays of the camera with the most matches (ties to
/// the lower index), lifted by its `T_PC` and scored on every camera's rays,
/// then refined by rig PnP from that hypothesis. ORB-SLAM3's MLPnP draws and
/// scores camera 0 only (`MLPnPsolver.cpp:55-78, 262-290`).
fn rig_ransac_pose(
    frame: &Frame,
    map: &Map,
    camera: &PinholeCamera,
    correspondences: &[(usize, usize)],
    pnp_config: &PnpConfig,
    config: &RelocalizationConfig,
    seed: u64,
) -> Option<(Pose3d, Vec<(usize, usize)>)> {
    let camera_of: Vec<usize> = correspondences
        .iter()
        .map(|&(_, kp)| frame.secondary_view(kp).map_or(0, |_| frame.camera_of(kp)))
        .collect();
    let mut counts = vec![0usize; camera_of.iter().max().map_or(0, |&c| c + 1)];
    for &c in &camera_of {
        counts[c] += 1;
    }
    let hypothesis_camera =
        (0..counts.len()).max_by(|&a, &b| counts[a].cmp(&counts[b]).then(b.cmp(&a)))?;
    let mut order: Vec<usize> = (0..correspondences.len()).collect();
    order.sort_by_key(|&i| camera_of[i] != hypothesis_camera);

    let pixel_of = |kp: usize| {
        let [u, v] = frame.keypoints_undist[kp];
        [f64::from(u), f64::from(v)]
    };
    let rays: Vec<RayMatch> = order
        .iter()
        .map(|&i| {
            let (mp, kp) = correspondences[i];
            let pixel = pixel_of(kp);
            let ((origin, direction), pixel_scale) = match frame.secondary_view(kp) {
                Some(view) => (view.ray(pixel), view.camera.fx),
                None => (
                    kornia_sensors::camera_ray(camera, &Pose3d::IDENTITY, pixel),
                    camera.fx,
                ),
            };
            RayMatch {
                origin,
                direction,
                point_world: map.map_points()[mp].position,
                pixel_scale,
            }
        })
        .collect();
    let (pose, mask) = ransac_lifted_ap3p(
        &rays,
        counts[hypothesis_camera],
        f64::from(config.ransac_threshold_px),
        u32::try_from(config.ransac_iterations).unwrap_or(u32::MAX),
        seed,
    )?;
    let inliers: Vec<(usize, usize)> = order
        .iter()
        .zip(&mask)
        .filter(|&(_, &inlier)| inlier)
        .map(|(&i, _)| correspondences[i])
        .collect();
    if inliers.len() < config.min_ransac_inliers {
        return None;
    }

    let information = pnp_config.visual_information;
    let observations: Vec<RigObservation> = correspondences
        .iter()
        .map(|&(mp, kp)| {
            let view = frame.secondary_view(kp);
            let octave = frame.features.octaves.get(kp).copied().unwrap_or(0);
            RigObservation {
                point_world: map.map_points()[mp].position,
                pixel: pixel_of(kp),
                camera: view.map_or(camera, |view| &view.camera),
                camera_from_primary: view.map(|view| &view.camera_from_primary),
                sigma_px: f64::from(information.sigma_px(octave)),
            }
        })
        .collect();
    // The gate is centred on the hypothesis; a refused refinement leaves it.
    let refined = solve_rig_pnp(&observations, &pose, pnp_config).map_or(pose, |(pose, _)| pose);
    Some((refined, inliers))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::RigFeatures;
    use crate::mapping::map::{LandmarkSeed, ObservationKey};
    use crate::rig::RigGeometry;
    use crate::tracking::pose_estimation::map_projection::MapProjectionConfig;
    use kornia_algebra::{Mat3F64, SO3F64, Vec3F64};
    use kornia_image::ImageSize;
    use kornia_imgproc::features::OrbFeatures;
    use kornia_sensors::{RigCamera, SensorRig};
    use std::sync::Arc;

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
            rig: Default::default(),
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

    fn relocalize_query(
        query: &Frame,
        map: &Map,
        candidates: &[usize],
        accept: impl FnMut(&Relocalization) -> bool,
    ) -> Option<Relocalization> {
        let estimator = MapProjectionEstimator::new(MapProjectionConfig::default());
        relocalize(
            query,
            candidates,
            map,
            &camera(),
            &estimator,
            &RelocalizationConfig::default(),
            accept,
        )
    }

    #[test]
    fn recovers_the_pose_against_a_candidate_keyframe() {
        let map = mapped_keyframe();
        let truth = query_pose();
        let query = frame(50, truth, (0..LANDMARKS).map(descriptor).collect());

        let found =
            relocalize_query(&query, &map, &[0], |_| true).expect("the keyframe sees the query");

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
    fn keypoints_without_an_undistorted_pixel_are_not_matched() {
        let map = mapped_keyframe();
        let mut query = frame(50, query_pose(), (0..LANDMARKS).map(descriptor).collect());
        let unusable = 10;
        for xy in &mut query.keypoints_undist[..unusable] {
            *xy = [f32::NAN; 2];
        }

        let found =
            relocalize_query(&query, &map, &[0], |_| true).expect("the keyframe sees the query");

        assert!(found.estimate.matches.iter().all(|&(_, kp)| kp >= unusable));
    }

    #[test]
    fn rejects_a_candidate_that_shares_no_appearance() {
        let map = mapped_keyframe();
        let unrelated = (0..LANDMARKS)
            .map(|index| descriptor(index + 1000))
            .collect();
        let query = frame(50, query_pose(), unrelated);

        assert!(relocalize_query(&query, &map, &[0], |_| true).is_none());
    }

    #[test]
    fn a_refused_pose_moves_on_to_the_next_candidate() {
        let map = mapped_keyframe();
        let query = frame(50, query_pose(), (0..LANDMARKS).map(descriptor).collect());
        let mut offered = 0;

        let found = relocalize_query(&query, &map, &[0, 0], |_| {
            offered += 1;
            offered > 1
        });

        assert!(found.is_some());
        assert_eq!(offered, 2);
    }

    #[test]
    fn fails_when_every_supported_pose_is_refused() {
        let map = mapped_keyframe();
        let query = frame(50, query_pose(), (0..LANDMARKS).map(descriptor).collect());

        assert!(relocalize_query(&query, &map, &[0, 0], |_| false).is_none());
    }

    struct Lcg(u64);

    impl Lcg {
        /// Uniform in `[0, 1)`.
        fn unit(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn pixel_of(camera: &PinholeCamera, camera_from_world: &Pose3d, point: &Vec3F64) -> [f32; 2] {
        let p = camera_from_world.transform_point(point);
        [
            (camera.fx * p.x / p.z + camera.cx) as f32,
            (camera.fy * p.y / p.z + camera.cy) as f32,
        ]
    }

    fn frame_with(idx: usize, pose: Pose3d, pixels: Vec<[f32; 2]>, rig: RigFeatures) -> Frame {
        let n = pixels.len();
        Frame {
            idx,
            features: OrbFeatures {
                keypoints_xy: pixels.clone(),
                orientations: vec![0.0; n],
                descriptors: (0..n).map(descriptor).collect(),
                octaves: vec![0; n],
            },
            pose_world_to_cam: pose,
            image_size: ImageSize {
                width: 640,
                height: 480,
            },
            keypoint_colors: vec![[0; 3]; n],
            u_right: Vec::new(),
            depth: Vec::new(),
            keypoints_undist: pixels,
            rig,
        }
    }

    /// A map whose keyframe 0, at the identity, owns landmark `i` at `points[i]`
    /// through its feature `i`.
    fn map_of(points: &[Vec3F64]) -> Map {
        let mut map = Map::new();
        let pixels = vec![[320.0, 240.0]; points.len()];
        map.insert_keyframe(Keyframe::from_frame(frame_with(
            0,
            Pose3d::IDENTITY,
            pixels,
            RigFeatures::default(),
        )))
        .unwrap();
        for (index, &position) in points.iter().enumerate() {
            map.insert_landmark(LandmarkSeed {
                position,
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

    fn solver_config(solver: RelocalizationSolver) -> RelocalizationConfig {
        RelocalizationConfig {
            solver,
            ..RelocalizationConfig::default()
        }
    }

    /// The hypothesis from feature `i` matched to landmark `i`, for every `i`.
    fn hypothesis_with(
        solver: RelocalizationSolver,
        frame: &Frame,
        map: &Map,
        seed: u64,
    ) -> Option<(Pose3d, Vec<(usize, usize)>)> {
        let correspondences: Vec<(usize, usize)> = (0..frame.features.keypoints_xy.len())
            .map(|i| (i, i))
            .collect();
        hypothesis(
            frame,
            map.get_keyframe(0).unwrap(),
            map,
            &camera(),
            &correspondences,
            &PnpConfig::default(),
            &solver_config(solver),
            seed,
        )
    }

    fn rotation_error(truth: &Pose3d, pose: &Pose3d) -> f64 {
        SO3F64::from_matrix(&(pose.rotation * truth.rotation.transpose()))
            .log()
            .length()
    }

    /// B3-P2 of `experiments/pr180-B-card.md`: one match in five is an inlier.
    /// A clean 3-point sample in 300 draws: 1 - (1 - 0.2^3)^300 = 0.91; for 5
    /// points 0.09.
    #[test]
    fn three_point_hypotheses_survive_four_outliers_in_five() {
        let truth = query_pose();
        let points: Vec<Vec3F64> = (0..100)
            .map(|i| {
                Vec3F64::new(
                    (i % 10) as f64 * 0.3 - 1.35,
                    (i / 10) as f64 * 0.22 - 1.0,
                    4.0 + (i % 7) as f64 * 0.2,
                )
            })
            .collect();
        let mut noise = Lcg(11);
        let pixels = points
            .iter()
            .enumerate()
            .map(|(i, p)| {
                if i % 5 == 0 {
                    let [u, v] = pixel_of(&camera(), &truth, p);
                    [
                        u + (noise.unit() - 0.5) as f32,
                        v + (noise.unit() - 0.5) as f32,
                    ]
                } else {
                    [(noise.unit() * 640.0) as f32, (noise.unit() * 480.0) as f32]
                }
            })
            .collect();
        let frame = frame_with(50, truth, pixels, RigFeatures::default());
        let map = map_of(&points);
        let successes = |solver| {
            (0..100u64)
                .filter(|&seed| {
                    hypothesis_with(solver, &frame, &map, seed).is_some_and(|(pose, inliers)| {
                        rotation_error(&truth, &pose) < 5f64.to_radians() && inliers.len() >= 15
                    })
                })
                .count()
        };
        let epnp = successes(RelocalizationSolver::PrimaryEpnp);
        let ap3p = successes(RelocalizationSolver::PrimaryAp3p);
        let rig = successes(RelocalizationSolver::RigAp3p);
        assert!((3..=20).contains(&epnp), "PrimaryEpnp {epnp} of 100");
        assert!((80..=98).contains(&ap3p), "PrimaryAp3p {ap3p} of 100");
        assert!((80..=98).contains(&rig), "RigAp3p {rig} of 100");
    }

    /// Camera 1 looks 70 deg to the side of the primary, 11 cm away.
    fn side_rig() -> Arc<RigGeometry> {
        let size = ImageSize {
            width: 640,
            height: 480,
        };
        let yaw = SO3F64::exp(Vec3F64::new(0.0, 70f64.to_radians(), 0.0)).matrix();
        let rig = SensorRig::new(camera()).with_rig_cameras(vec![
            RigCamera {
                camera: camera(),
                size,
                primary_from_camera: Pose3d::IDENTITY,
            },
            RigCamera {
                camera: camera(),
                size,
                primary_from_camera: Pose3d::new(yaw, Vec3F64::new(0.11, 0.0, 0.0)),
            },
        ]);
        Arc::new(RigGeometry::from_rig(&rig).unwrap())
    }

    /// B3-P3: 10 matches on the primary, 80 on camera 1, one in four of each
    /// 20 px off.
    fn starved_primary() -> (Frame, Map, Pose3d) {
        let truth = query_pose();
        let rig = side_rig();
        let side = rig.view(1).unwrap().clone();
        let mut points = Vec::new();
        let mut pixels = Vec::new();
        for i in 0..90 {
            let (x, y) = ((i % 9) as f64 * 0.35 - 1.4, (i / 9 % 5) as f64 * 0.3 - 0.6);
            let depth = 3.5 + (i % 4) as f64 * 0.6;
            let (point_world, mut pixel) = if i < 10 {
                let p_primary = Vec3F64::new(x, y, depth);
                let world = truth.inverse().transform_point(&p_primary);
                (world, pixel_of(&camera(), &truth, &world))
            } else {
                let p_side = Vec3F64::new(x, y, depth);
                let p_primary = side.primary_from_camera.transform_point(&p_side);
                let world = truth.inverse().transform_point(&p_primary);
                (world, pixel_of(&side.camera, &side.pose(&truth), &world))
            };
            if i % 4 == 0 {
                pixel[0] += 20.0;
            }
            points.push(point_world);
            pixels.push(pixel);
        }
        let features = RigFeatures {
            geometry: Some(rig),
            camera_ranges: vec![0..10, 10..90],
            ..RigFeatures::default()
        };
        (
            frame_with(50, truth, pixels, features),
            map_of(&points),
            truth,
        )
    }

    #[test]
    fn a_starved_primary_relocalizes_only_on_the_rig() {
        let (query, map, truth) = starved_primary();
        let estimator = MapProjectionEstimator::new(MapProjectionConfig::default());
        for solver in [
            RelocalizationSolver::PrimaryEpnp,
            RelocalizationSolver::PrimaryAp3p,
        ] {
            let found = relocalize(
                &query,
                &[0],
                &map,
                &camera(),
                &estimator,
                &solver_config(solver),
                |_| true,
            );
            assert!(found.is_none(), "{solver:?} has 10 correspondences");
        }

        let (pose, inliers) =
            hypothesis_with(RelocalizationSolver::RigAp3p, &query, &map, 0).unwrap();
        assert!(inliers.len() >= 50, "inliers {}", inliers.len());
        assert!((pose.translation - truth.translation).length() < 0.01);
        assert!(rotation_error(&truth, &pose) < 1e-3);
    }

    /// B3-P4: on a single camera the rig solver is AP3P with another sampler.
    #[test]
    fn on_one_camera_the_rig_solver_agrees_with_primary_ap3p() {
        let map = mapped_keyframe();
        let query = frame(50, query_pose(), (0..LANDMARKS).map(descriptor).collect());
        let estimator = MapProjectionEstimator::new(MapProjectionConfig::default());
        let found = |solver| {
            relocalize(
                &query,
                &[0],
                &map,
                &camera(),
                &estimator,
                &solver_config(solver),
                |_| true,
            )
            .expect("the keyframe sees the query")
        };
        let ap3p = found(RelocalizationSolver::PrimaryAp3p);
        let rig = found(RelocalizationSolver::RigAp3p);
        assert_eq!(rig.estimate.inliers, ap3p.estimate.inliers);
        let gap = (rig.estimate.pose.translation - ap3p.estimate.pose.translation).length();
        assert!(gap < 1e-6, "{gap} m apart");
    }
}
