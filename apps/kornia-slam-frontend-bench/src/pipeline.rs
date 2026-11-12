//! The frontend designs the bench times, built from kornia-slam's public
//! pieces: ORB, stereo matching, KLT, snapping and PnP are the library's own
//! calls; projection matching repeats the library's gates (radius, octave
//! window, Hamming) over the trace's local map.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::{Vec2F32, Vec3F64};
use kornia_image::{Image, ImageSize, InterpolationMode};
use kornia_imgproc::features::{OrbDetector, OrbFeatures, hamming_distance};
use kornia_imgproc::resize::resize_fast_mono;
use kornia_slam::stereo::{StereoMatchConfig, compute_stereo_matches};
use kornia_slam::tracking::TrackingTimings;
use kornia_slam::tracking::optical_flow::{
    FlowSurvivor, KltPyramidCache, KltTracker, MapKeypointMatch, TrackSet, snap_unique,
};
use kornia_slam::tracking::pose_estimation::pnp::{PnpConfig, solve_pnp};

use crate::trace::{LocalPoint, Trace, TraceFrame, TraceTrack};

/// Fewest flow survivors with a map point that KLT-first trusts alone.
const KLT_MIN_TRACKS: usize = 80;
const ORB_SCALE_FACTOR: f64 = 1.2;
const ORB_N_LEVELS: i32 = 8;
const OCTAVE_WINDOW_HALF_WIDTH: i32 = 2;
const MIN_CORRESPONDENCES: usize = 4;
const MIN_INLIERS_EARLY: usize = 10;
const MIN_INLIERS: usize = 30;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// ORB on every image every frame, LK seeds, projection search, PnP and
    /// a local refine: the design the system runs at 91e279d.
    Today,
    /// ORB only on keyframes (from the trace's schedule) or when too few
    /// tracks survive; otherwise PnP on the flow survivors alone.
    KltFirst,
}

impl Mode {
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "today" => Some(Self::Today),
            "klt-first" => Some(Self::KltFirst),
            _ => None,
        }
    }
}

#[derive(Default)]
pub struct Stages {
    orb: Duration,
    stereo: Duration,
    klt_build: Duration,
    klt_track: Duration,
    snap: Duration,
    projection: Duration,
    pnp: Duration,
    refine: Duration,
}

impl Stages {
    pub const HEADER: &'static str =
        "orb_ms,stereo_ms,klt_build_ms,klt_track_ms,snap_ms,projection_ms,pnp_ms,refine_ms";

    pub fn row(&self) -> String {
        let ms = |d: Duration| d.as_secs_f64() * 1000.0;
        format!(
            "{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
            ms(self.orb),
            ms(self.stereo),
            ms(self.klt_build),
            ms(self.klt_track),
            ms(self.snap),
            ms(self.projection),
            ms(self.pnp),
            ms(self.refine)
        )
    }
}

fn timed<T>(slot: &mut Duration, f: impl FnOnce() -> T) -> T {
    let start = Instant::now();
    let out = f();
    *slot += start.elapsed();
    out
}

pub struct StepOutput {
    pub stages: Stages,
    pub n_kp: usize,
    pub n_stereo: usize,
    pub n_seeded: usize,
    pub n_proj: usize,
    /// `(local-map index, undistorted pixel)` of the accepted correspondences.
    pub correspondences: Vec<(usize, [f32; 2])>,
    pub inliers: usize,
    pub orb_ran: bool,
}

pub struct Pipeline {
    mode: Mode,
    detector: OrbDetector,
    stereo: Option<StereoMatchConfig>,
    camera: PinholeCamera,
    size: ImageSize,
    klt: KltTracker,
    klt_cache: KltPyramidCache,
    tracks: TrackSet,
    previous: Option<Image<u8, 1>>,
    pnp: PnpConfig,
}

/// Keypoints sorted by x for radius queries.
struct KeypointIndex {
    order: Vec<usize>,
    xs: Vec<f32>,
}

impl KeypointIndex {
    fn new(keypoints: &[[f32; 2]]) -> Self {
        let mut order: Vec<usize> = (0..keypoints.len()).collect();
        order.sort_by(|&a, &b| keypoints[a][0].total_cmp(&keypoints[b][0]));
        let xs = order.iter().map(|&i| keypoints[i][0]).collect();
        Self { order, xs }
    }

    fn within<'a>(
        &'a self,
        keypoints: &'a [[f32; 2]],
        u: f32,
        v: f32,
        radius: f32,
    ) -> impl Iterator<Item = usize> + 'a {
        let lo = self.xs.partition_point(|&x| x < u - radius);
        let hi = self.xs.partition_point(|&x| x <= u + radius);
        let r2 = radius * radius;
        self.order[lo..hi].iter().copied().filter(move |&i| {
            let dx = keypoints[i][0] - u;
            let dy = keypoints[i][1] - v;
            dx * dx + dy * dy <= r2
        })
    }
}

struct Features {
    keypoints: Vec<[f32; 2]>,
    undistorted: Vec<[f32; 2]>,
    descriptors: Vec<[u8; 32]>,
    octaves: Vec<u8>,
}

impl Pipeline {
    pub fn new(trace: &Trace, mode: Mode, first_tracks: Option<&Vec<TraceTrack>>) -> Self {
        let primary = &trace.cameras[0];
        let detector = OrbDetector::default();
        let stereo = (trace.n_images() == 2 && trace.cameras.len() == 1).then(|| {
            StereoMatchConfig::new(
                primary.stereo_baseline_m as f32,
                primary.camera.fx as f32,
                detector.downscale,
                detector.n_scales,
            )
        });
        let mut tracks = TrackSet::new();
        if let Some(first) = first_tracks {
            let pixels: Vec<[f32; 2]> = first.iter().map(|(p, _)| *p).collect();
            let matches: Vec<MapKeypointMatch> = first
                .iter()
                .enumerate()
                .filter_map(|(i, (_, mp))| {
                    mp.map(|map_point_idx| MapKeypointMatch {
                        map_point_idx,
                        keypoint_idx: i,
                    })
                })
                .collect();
            if tracks.reconcile_from_matches(&matches, &pixels).is_err() {
                tracks = TrackSet::new();
            }
        }
        Self {
            mode,
            detector,
            stereo,
            camera: primary.camera.clone(),
            size: primary.size,
            klt: KltTracker::default(),
            klt_cache: KltPyramidCache::new(),
            tracks,
            previous: None,
            pnp: PnpConfig::default(),
        }
    }

    pub fn step(
        &mut self,
        frame: &TraceFrame,
        images: &[Image<u8, 1>],
        local: &[LocalPoint],
    ) -> Result<StepOutput, Box<dyn std::error::Error>> {
        let mut stages = Stages::default();
        let primary = images.first().ok_or("no image")?;
        let survivors = self.flow(primary, &mut stages);
        let index_of: HashMap<usize, usize> = local
            .iter()
            .enumerate()
            .map(|(i, p)| (p.map_point_idx, i))
            .collect();

        let klt_only = self.mode == Mode::KltFirst && !frame.keyframe && {
            let with_map_point = self
                .tracks
                .tracks()
                .iter()
                .filter(|t| t.map_point_idx().is_some())
                .count();
            survivors.as_ref().map_or(0, Vec::len) >= KLT_MIN_TRACKS
                && with_map_point >= KLT_MIN_TRACKS
        };
        let out = if klt_only {
            self.klt_only_step(
                frame,
                survivors.unwrap_or_default(),
                local,
                &index_of,
                stages,
            )
        } else {
            self.full_step(frame, images, survivors, local, &index_of, stages)?
        };
        self.previous = Some(primary.clone());
        Ok(out)
    }

    fn flow(&mut self, current: &Image<u8, 1>, stages: &mut Stages) -> Option<Vec<FlowSurvivor>> {
        let previous = self.previous.as_ref()?;
        if self.tracks.is_empty() {
            return None;
        }
        let mut timings = TrackingTimings::default();
        let survivors = self
            .klt
            .track_cached(
                &mut self.klt_cache,
                self.tracks.tracks(),
                previous,
                current,
                &mut timings,
            )
            .ok();
        stages.klt_build += timings.klt_build;
        stages.klt_track += timings.klt_track;
        survivors
    }

    fn extract(
        &self,
        images: &[Image<u8, 1>],
        stages: &mut Stages,
    ) -> Result<(Features, usize), Box<dyn std::error::Error>> {
        let per_image: Vec<OrbFeatures> = timed(&mut stages.orb, || {
            images
                .iter()
                .map(|image| self.detector.detect_and_extract_u8(image))
                .collect::<Result<_, _>>()
        })?;
        let n_stereo = match (&self.stereo, per_image.as_slice()) {
            (Some(config), [left, right, ..]) => timed(&mut stages.stereo, || {
                let left_pyr = self.pyramid(&images[0])?;
                let right_pyr = self.pyramid(&images[1])?;
                Ok::<_, kornia_image::ImageError>(
                    compute_stereo_matches(&left_pyr, &right_pyr, left, right, config)
                        .num_matched(),
                )
            })?,
            _ => 0,
        };
        let primary = per_image.into_iter().next().ok_or("no features")?;
        let undistorted = primary
            .keypoints_xy
            .iter()
            .map(|kp| self.undistort(*kp))
            .collect();
        Ok((
            Features {
                keypoints: primary.keypoints_xy,
                undistorted,
                descriptors: primary.descriptors,
                octaves: primary.octaves,
            },
            n_stereo,
        ))
    }

    /// The frontend's stereo pyramid: each level resized from the original.
    fn pyramid(&self, image: &Image<u8, 1>) -> Result<Vec<Image<u8, 1>>, kornia_image::ImageError> {
        let mut levels = vec![image.clone()];
        let (w0, h0) = (image.width() as f32, image.height() as f32);
        for level in 1..self.detector.n_scales {
            let inv = 1.0 / self.detector.downscale.powi(level as i32);
            let size = ImageSize {
                width: ((w0 * inv).round() as usize).max(1),
                height: ((h0 * inv).round() as usize).max(1),
            };
            let mut dst = Image::from_size_val(size, 0u8)?;
            resize_fast_mono(image, &mut dst, InterpolationMode::Bilinear)?;
            levels.push(dst);
        }
        Ok(levels)
    }

    fn undistort(&self, kp: [f32; 2]) -> [f32; 2] {
        let p = self.camera.undistort(kp[0] as f64, kp[1] as f64);
        [p.x as f32, p.y as f32]
    }

    fn full_step(
        &mut self,
        frame: &TraceFrame,
        images: &[Image<u8, 1>],
        survivors: Option<Vec<FlowSurvivor>>,
        local: &[LocalPoint],
        index_of: &HashMap<usize, usize>,
        mut stages: Stages,
    ) -> Result<StepOutput, Box<dyn std::error::Error>> {
        let (features, n_stereo) = self.extract(images, &mut stages)?;
        let seeded: Vec<(usize, usize)> = timed(&mut stages.snap, || {
            survivors
                .as_ref()
                .and_then(|s| snap_unique(&self.tracks, s, &features.keypoints, 3.0).ok())
                .map(|matches| {
                    matches
                        .into_iter()
                        .filter_map(|m| Some((*index_of.get(&m.map_point_idx)?, m.keypoint_idx)))
                        .collect()
                })
                .unwrap_or_default()
        });
        let index = KeypointIndex::new(&features.undistorted);
        let projection = timed(&mut stages.projection, || {
            let narrow =
                self.match_projection(local, &features, &index, &frame.candidate_pose, 15.0, 50);
            if narrow.len() < 20 {
                self.match_projection(local, &features, &index, &frame.candidate_pose, 30.0, 50)
            } else {
                narrow
            }
        });
        let n_seeded = seeded.len();
        let n_proj = projection.len();
        let mut result = None;
        for candidate in [seeded, projection] {
            if candidate.len() < MIN_CORRESPONDENCES {
                continue;
            }
            let solved = timed(&mut stages.pnp, || {
                self.solve(
                    local,
                    &features.undistorted,
                    &candidate,
                    &frame.candidate_pose,
                )
            });
            if let Some((pose, inliers)) = solved.filter(|&(_, n)| n >= MIN_INLIERS_EARLY) {
                result = Some((pose, inliers, candidate));
                break;
            }
        }
        let (mut correspondences, mut inliers) = (Vec::new(), 0);
        if let Some((pose, n, matches)) = result {
            let (matches, n) = timed(&mut stages.refine, || {
                let refined = self.match_projection(local, &features, &index, &pose, 30.0, 60);
                match self.solve(local, &features.undistorted, &refined, &pose) {
                    Some((_, n2)) if refined.len() >= MIN_CORRESPONDENCES && n2 >= MIN_INLIERS => {
                        (refined, n2)
                    }
                    _ => (matches, n),
                }
            });
            inliers = n;
            self.reconcile(local, &matches, &features.keypoints);
            correspondences = matches
                .iter()
                .map(|&(li, kp)| (li, features.undistorted[kp]))
                .collect();
        } else {
            self.tracks = TrackSet::new();
        }
        Ok(StepOutput {
            stages,
            n_kp: features.keypoints.len(),
            n_stereo,
            n_seeded,
            n_proj,
            correspondences,
            inliers,
            orb_ran: true,
        })
    }

    fn klt_only_step(
        &mut self,
        frame: &TraceFrame,
        survivors: Vec<FlowSurvivor>,
        local: &[LocalPoint],
        index_of: &HashMap<usize, usize>,
        mut stages: Stages,
    ) -> StepOutput {
        let by_id: HashMap<_, _> = self
            .tracks
            .tracks()
            .iter()
            .map(|t| (t.id(), t.map_point_idx()))
            .collect();
        let mut pixels = Vec::with_capacity(survivors.len());
        let mut matches = Vec::with_capacity(survivors.len());
        for survivor in &survivors {
            let Some(Some(mp)) = by_id.get(&survivor.track_id) else {
                continue;
            };
            let Some(&li) = index_of.get(mp) else {
                continue;
            };
            matches.push((li, pixels.len()));
            pixels.push(survivor.pixel);
        }
        let undistorted: Vec<[f32; 2]> = pixels.iter().map(|p| self.undistort(*p)).collect();
        let solved = timed(&mut stages.pnp, || {
            self.solve(local, &undistorted, &matches, &frame.candidate_pose)
        });
        let n_seeded = matches.len();
        let (correspondences, inliers) = match solved {
            Some((_, n)) if n >= MIN_INLIERS_EARLY => {
                self.reconcile(local, &matches, &pixels);
                (
                    matches
                        .iter()
                        .map(|&(li, k)| (li, undistorted[k]))
                        .collect(),
                    n,
                )
            }
            _ => {
                self.tracks = TrackSet::new();
                (Vec::new(), 0)
            }
        };
        StepOutput {
            stages,
            n_kp: 0,
            n_stereo: 0,
            n_seeded,
            n_proj: 0,
            correspondences,
            inliers,
            orb_ran: false,
        }
    }

    fn reconcile(
        &mut self,
        local: &[LocalPoint],
        matches: &[(usize, usize)],
        keypoints: &[[f32; 2]],
    ) {
        let track_matches: Vec<MapKeypointMatch> = matches
            .iter()
            .map(|&(li, kp)| MapKeypointMatch {
                map_point_idx: local[li].map_point_idx,
                keypoint_idx: kp,
            })
            .collect();
        if self
            .tracks
            .reconcile_from_matches(&track_matches, keypoints)
            .is_err()
        {
            self.tracks = TrackSet::new();
        }
    }

    fn solve(
        &self,
        local: &[LocalPoint],
        undistorted: &[[f32; 2]],
        matches: &[(usize, usize)],
        pose_init: &Pose3d,
    ) -> Option<(Pose3d, usize)> {
        let world: Vec<Vec3F64> = matches.iter().map(|&(li, _)| local[li].position).collect();
        let image: Vec<Vec2F32> = matches
            .iter()
            .map(|&(_, kp)| Vec2F32::new(undistorted[kp][0], undistorted[kp][1]))
            .collect();
        solve_pnp(&world, &image, &self.camera, pose_init, &self.pnp)
    }

    /// The library's projection search: each local point, in order, takes
    /// the best unused keypoint within `radius` inside its octave window.
    fn match_projection(
        &self,
        local: &[LocalPoint],
        features: &Features,
        index: &KeypointIndex,
        pose: &Pose3d,
        radius: f32,
        max_hamming: u32,
    ) -> Vec<(usize, usize)> {
        let center = pose.inverse().translation;
        let mut used = vec![false; features.undistorted.len()];
        let mut matches = Vec::new();
        for (li, point) in local.iter().enumerate() {
            let p_cam = pose.transform_point(&point.position);
            let Ok(pixel) = self.camera.project_to_image(&p_cam, 0.0, self.size) else {
                continue;
            };
            let window = if point.max_distance > 0.0 {
                let dist = (point.position - center).length();
                if dist < 0.8 * point.min_distance || dist > 1.2 * point.max_distance {
                    continue;
                }
                let level = ((point.max_distance / dist).ln() / ORB_SCALE_FACTOR.ln()).ceil();
                let level = if level.is_nan() || level < 0.0 {
                    0
                } else {
                    (level as i32).min(ORB_N_LEVELS - 1)
                };
                Some((
                    (level - OCTAVE_WINDOW_HALF_WIDTH).max(0) as u8,
                    (level + OCTAVE_WINDOW_HALF_WIDTH).min(ORB_N_LEVELS - 1) as u8,
                ))
            } else {
                None
            };
            let (mut best, mut best_kp) = (u32::MAX, usize::MAX);
            for kp in index.within(
                &features.undistorted,
                pixel.x as f32,
                pixel.y as f32,
                radius,
            ) {
                if used[kp] {
                    continue;
                }
                if let Some((lo, hi)) = window
                    && !(lo..=hi).contains(&features.octaves[kp])
                {
                    continue;
                }
                let d = hamming_distance(&point.descriptor, &features.descriptors[kp]);
                if d < best {
                    best = d;
                    best_kp = kp;
                }
            }
            if best <= max_hamming && best_kp != usize::MAX {
                used[best_kp] = true;
                matches.push((li, best_kp));
            }
        }
        matches
    }

    /// Reprojection RMSE of `out`'s correspondences at the trace's tracked
    /// pose, each residual clipped at 10 px.
    pub fn rmse_at(&self, pose: &Pose3d, local: &[LocalPoint], out: &StepOutput) -> f64 {
        let mut sum = 0.0;
        let mut n = 0usize;
        for &(li, px) in &out.correspondences {
            let p_cam = pose.transform_point(&local[li].position);
            if p_cam.z <= 0.0 {
                continue;
            }
            let u = self.camera.fx * p_cam.x / p_cam.z + self.camera.cx;
            let v = self.camera.fy * p_cam.y / p_cam.z + self.camera.cy;
            let e = ((u - px[0] as f64).powi(2) + (v - px[1] as f64).powi(2))
                .sqrt()
                .min(10.0);
            sum += e * e;
            n += 1;
        }
        if n == 0 {
            f64::NAN
        } else {
            (sum / n as f64).sqrt()
        }
    }
}
