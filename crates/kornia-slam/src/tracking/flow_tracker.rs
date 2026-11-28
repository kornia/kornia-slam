//! One flow shard: the tracks of one worker, from start pixel to survivor,
//! with every buffer reused from frame to frame, and [`RigFlow`], the
//! `WindowedU8` flow step of every camera that runs them.

use std::sync::Arc;
use std::time::Instant;

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, Vec3F64};
use kornia_image::Image;
use kornia_imgproc::features::{
    GoodFeaturesParams, GoodFeaturesScratch, corner_sub_pix, good_features_to_track,
};
use kornia_imgproc::optical_flow_pyr_lk::{
    LkFrame, LkTemplate, LkTrack, extract_template, track_template,
};

use crate::Frame;
use crate::mapping::Map;
use crate::system::{ForwardBackward, GyroGuess, KltRetry, TrackRefill};
use crate::tracking::flow_pool::{FlowFrames, FlowPool, FlowPoolError};
use crate::tracking::optical_flow::{
    FlowSurvivor, KltTracker, SurvivorFilterConfig, Track, TrackId, TrackSet,
};
use crate::tracking::timings::{MAX_FLOW_SHARDS, ShardTimings, TrackingTimings};
use crate::tracking::tracker::project_raw;

/// Which flow shard a new track runs on, fixed at its spawn.
#[derive(Debug, Clone, Default)]
pub(crate) struct ShardLayout {
    threads: usize,
    cameras: usize,
    /// Tracks spawned so far per camera, for the `cameras + 1` split.
    spawned: Vec<u64>,
}

impl ShardLayout {
    /// Shards for `threads` flow threads on `cameras` cameras; validation
    /// allows 1, `cameras` or `cameras + 1` threads.
    pub(crate) fn new(threads: usize, cameras: usize) -> Self {
        Self {
            threads,
            cameras,
            spawned: vec![0; cameras],
        }
    }

    fn shards(&self) -> usize {
        self.threads.max(1)
    }

    /// The shard of the next track spawned in `camera`. With one thread per
    /// camera a camera is its shard; with one thread more, worker `c + 1`
    /// takes two thirds of camera `c` and the tracking core the rest.
    pub(crate) fn assign(&mut self, camera: usize) -> usize {
        if self.shards() == 1 {
            return 0;
        }
        let shard = if self.threads == self.cameras + 1 {
            match self.spawned.get_mut(camera) {
                Some(k) => {
                    let shard = if *k % 3 == 0 { 0 } else { camera + 1 };
                    *k += 1;
                    shard
                }
                None => 0,
            }
        } else {
            camera
        };
        shard.min(self.shards() - 1)
    }

    /// The shard that refills `camera`: the camera's own worker.
    pub(crate) fn refill_shard(&self, camera: usize) -> usize {
        let shard = match self.shards() {
            1 => 0,
            threads if threads == self.cameras + 1 => camera + 1,
            _ => camera,
        };
        shard.min(self.shards() - 1)
    }
}

/// One camera's model for a flow step, refreshed in place each frame.
#[derive(Debug, Clone)]
pub(crate) struct FlowCamera {
    pub camera: PinholeCamera,
    /// `T_CP`; `None` for the primary camera.
    pub camera_from_primary: Option<Pose3d>,
    /// `R_cur_prev` of this camera from the gyro (`GyroGuess::RotatedRay`).
    pub gyro_rotation: Option<Mat3F64>,
}

/// What every shard needs to start its tracks; filled by the tracking core.
#[derive(Debug, Default)]
pub(crate) struct ShardGuess {
    pub frame_idx: u64,
    /// World to primary camera at the IMU prediction, when trusted.
    pub pose_world_to_primary: Option<Pose3d>,
    pub cameras: Vec<FlowCamera>,
    pub max_level: usize,
    pub max_iter: usize,
    pub epsilon: f32,
    pub window: usize,
    pub forward_backward: ForwardBackward,
    pub refill: TrackRefill,
    /// Camera refilled this frame, if any.
    pub refill_camera: Option<usize>,
    /// The shard that runs the refill of `refill_camera`.
    pub refill_shard: usize,
    /// Start-of-frame pixels of every track of `refill_camera`, all shards.
    pub refill_occupied: Vec<[f32; 2]>,
    /// Retry pass: only `retry` tracks, from `max_level`.
    pub retry: bool,
}

impl Clone for ShardGuess {
    fn clone(&self) -> Self {
        let mut guess = Self::default();
        guess.clone_from(self);
        guess
    }

    /// Reuses `self`'s vectors: the pool copies the guess into every job
    /// each frame, and a derived `clone_from` would allocate.
    fn clone_from(&mut self, source: &Self) {
        self.frame_idx = source.frame_idx;
        self.pose_world_to_primary = source.pose_world_to_primary;
        self.cameras.clone_from(&source.cameras);
        self.max_level = source.max_level;
        self.max_iter = source.max_iter;
        self.epsilon = source.epsilon;
        self.window = source.window;
        self.forward_backward = source.forward_backward;
        self.refill = source.refill;
        self.refill_camera = source.refill_camera;
        self.refill_shard = source.refill_shard;
        self.refill_occupied.clone_from(&source.refill_occupied);
        self.retry = source.retry;
    }
}

/// Frame id of a cache slot with no template.
const NO_TEMPLATE: u64 = u64::MAX;
/// Frame id of a slot whose template moved to the aligned list.
const TAKEN: u64 = u64::MAX - 1;

/// Templates keyed by track, the frame they were cut from and the pixel they
/// were cut at, since ORB can move a track's pixel between frames. Templates
/// of dropped tracks go to a spare list and are reused, so once the track
/// count has peaked nothing allocates.
#[derive(Debug, Default)]
struct TemplateCache {
    /// Sorted by track id between frames; aligned with the shard's tracks
    /// during one.
    entries: Vec<(TrackId, u64, [f32; 2], LkTemplate)>,
    scratch: Vec<(TrackId, u64, [f32; 2], LkTemplate)>,
    spare: Vec<LkTemplate>,
}

impl TemplateCache {
    /// Reorders the entries to follow `tracks`, a slot per track; a new
    /// track gets a spare template marked [`NO_TEMPLATE`].
    fn align(&mut self, tracks: &[Track]) {
        self.scratch.clear();
        for track in tracks {
            let (frame_id, source, template) = match self
                .entries
                .binary_search_by_key(&track.id(), |entry| entry.0)
            {
                Ok(k) => {
                    let entry = &mut self.entries[k];
                    let frame_id = std::mem::replace(&mut entry.1, TAKEN);
                    (frame_id, entry.2, std::mem::take(&mut entry.3))
                }
                Err(_) => (
                    NO_TEMPLATE,
                    [f32::NAN; 2],
                    self.spare.pop().unwrap_or_default(),
                ),
            };
            self.scratch.push((track.id(), frame_id, source, template));
        }
        for (_, frame_id, _, template) in self.entries.drain(..) {
            if frame_id != TAKEN {
                self.spare.push(template);
            }
        }
        std::mem::swap(&mut self.entries, &mut self.scratch);
    }

    /// Restores the id order lookups need.
    fn sort(&mut self) {
        self.entries.sort_unstable_by_key(|entry| entry.0);
    }

    fn find(&self, id: TrackId) -> Option<usize> {
        self.entries.binary_search_by_key(&id, |entry| entry.0).ok()
    }
}

/// One shard's tracks and every buffer it reuses; moved to a worker and back.
#[derive(Default)]
pub(crate) struct FlowShard {
    pub index: usize,
    /// Input, in (camera, track id) order.
    pub tracks: Vec<Track>,
    /// Output, in the order of `tracks`.
    pub survivors: Vec<FlowSurvivor>,
    /// Each survivor's pixel undistorted in its camera, NaN where the
    /// inversion fails; in the order of `survivors`.
    pub undistorted: Vec<[f32; 2]>,
    /// Tracks that failed at the first pass's level, for `klt_retry`.
    pub retry: Vec<TrackId>,
    pub spawned: Vec<(usize, [f32; 2])>,
    pub fb_rejects: u32,
    pub template_rebuilds: u32,
    /// Each track's template, cut where it ended in the last frame.
    templates: TemplateCache,
    /// Coarse-level template of the track being solved.
    coarse: LkTemplate,
    features: GoodFeaturesScratch,
    corners: Vec<[f32; 2]>,
}

/// Refill's Shi-Tomasi quality: VINS-Fusion's `goodFeaturesToTrack` level
/// (`feature_tracker.cpp:184`).
const REFILL_QUALITY: f32 = 0.01;
/// `cornerSubPix` window half side, iterations and step, as OpenCV's LK
/// sample (`samples/cpp/lkdemo.cpp`) refines new corners, at half its window.
const SUBPIX_HALF_WIN: usize = 5;
const SUBPIX_MAX_ITER: usize = 20;
const SUBPIX_EPSILON: f32 = 0.03;

/// Whether a check that runs every `every_n_frames` frames runs on this one.
fn due(frame_idx: u64, every_n_frames: usize) -> bool {
    frame_idx.is_multiple_of(every_n_frames.max(1) as u64)
}

fn distance(a: [f32; 2], b: [f32; 2]) -> f32 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

impl FlowShard {
    /// Tracks `self.tracks` from `previous[c]` into `current[c]`.
    pub(crate) fn run(
        &mut self,
        previous: &[Arc<LkFrame>],
        current: &[Arc<LkFrame>],
        guess: &ShardGuess,
        timings: &mut ShardTimings,
    ) {
        *timings = ShardTimings::default();
        self.survivors.clear();
        self.undistorted.clear();
        self.spawned.clear();
        self.fb_rejects = 0;
        self.template_rebuilds = 0;
        if !guess.retry {
            self.retry.clear();
            self.templates.align(&self.tracks);
        }
        for i in 0..self.tracks.len() {
            let track = self.tracks[i];
            let camera = track.camera();
            let (Some(previous), Some(current), Some(model)) = (
                previous.get(camera),
                current.get(camera),
                guess.cameras.get(camera),
            ) else {
                continue;
            };
            let slot = if guess.retry {
                self.templates.find(track.id())
            } else {
                Some(i)
            };
            let Some(slot) = slot else {
                continue;
            };
            let Some(result) = self.forward(slot, &track, previous, current, model, guess, timings)
            else {
                self.retry.push(track.id());
                continue;
            };
            self.keep(
                slot, &track, result, previous, current, model, guess, timings,
            );
        }
        if !guess.retry {
            self.templates.sort();
            self.refill(current, guess, timings);
        }
    }

    /// The forward solve of `track` from its cached template, recut when
    /// it came from another frame or another pixel; `None` when it fails the
    /// survivor filter.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &mut self,
        slot: usize,
        track: &Track,
        previous: &LkFrame,
        current: &LkFrame,
        model: &FlowCamera,
        guess: &ShardGuess,
        timings: &mut ShardTimings,
    ) -> Option<LkTrack> {
        let started = Instant::now();
        let start = start_pixel(track, model, guess);
        let (_, frame_id, source, template) = &mut self.templates.entries[slot];
        if *frame_id != previous.frame_id() || *source != track.pixel() {
            self.template_rebuilds += u32::from(*frame_id != NO_TEMPLATE);
            *source = track.pixel();
            *frame_id = if extract_template(previous, track.pixel(), 0, guess.window, template) {
                previous.frame_id()
            } else {
                NO_TEMPLATE
            };
        }
        let cut = *frame_id != NO_TEMPLATE;
        timings.flow_template += started.elapsed();
        if !cut {
            return None;
        }

        let started = Instant::now();
        let mut start = start;
        let levels = guess
            .max_level
            .min(previous.levels().min(current.levels()).saturating_sub(1));
        for level in (1..=levels).rev() {
            if !extract_template(
                previous,
                track.pixel(),
                level,
                guess.window,
                &mut self.coarse,
            ) {
                continue;
            }
            let coarse =
                track_template(&self.coarse, current, start, guess.max_iter, guess.epsilon);
            if coarse.status {
                start = coarse.pixel;
            }
        }
        let template = &self.templates.entries[slot].3;
        let result = track_template(template, current, start, guess.max_iter, guess.epsilon);
        timings.flow_forward += started.elapsed();
        let size = current.plane().size();
        KltTracker::accepts(
            track,
            u8::from(result.status),
            result.error,
            result.pixel,
            [size.width as f32, size.height as f32],
            &SurvivorFilterConfig::default(),
        )
        .then_some(result)
    }
    /// Records a forward result as a survivor, with its next template cut
    /// at the new pixel and its undistorted pixel; drops it instead when its
    /// round trip back into `previous` misses (`ForwardBackward`).
    #[allow(clippy::too_many_arguments)]
    fn keep(
        &mut self,
        slot: usize,
        track: &Track,
        result: LkTrack,
        previous: &LkFrame,
        current: &LkFrame,
        model: &FlowCamera,
        guess: &ShardGuess,
        timings: &mut ShardTimings,
    ) {
        let started = Instant::now();
        // The template at the forward result is the next frame's forward
        // template, and the backward solve's, so a track costs one cut per
        // frame.
        let (_, frame_id, source, template) = &mut self.templates.entries[slot];
        let cut = extract_template(current, result.pixel, 0, guess.window, template);
        *frame_id = if cut { current.frame_id() } else { NO_TEMPLATE };
        *source = result.pixel;
        timings.flow_template += started.elapsed();

        if let ForwardBackward::Enabled {
            max_round_trip_px,
            max_iter,
            every_n_frames,
        } = guess.forward_backward
            && cut
            && due(guess.frame_idx, every_n_frames)
        {
            let started = Instant::now();
            let template = &self.templates.entries[slot].3;
            let back = track_template(template, previous, track.pixel(), max_iter, guess.epsilon);
            timings.flow_backward += started.elapsed();
            if !back.status || distance(back.pixel, track.pixel()) > max_round_trip_px {
                // A re-matched track must not start from the rejected patch.
                self.templates.entries[slot].1 = NO_TEMPLATE;
                self.fb_rejects += 1;
                return;
            }
        }

        let started = Instant::now();
        let undistorted = model
            .camera
            .undistort(result.pixel[0] as f64, result.pixel[1] as f64)
            .map_or([f32::NAN; 2], |uv| [uv.x as f32, uv.y as f32]);
        self.survivors.push(FlowSurvivor {
            track_id: track.id(),
            pixel: result.pixel,
            error: result.error,
        });
        self.undistorted.push(undistorted);
        timings.flow_project += started.elapsed();
    }

    /// New corners for the camera this shard refills this frame: Shi-Tomasi
    /// in the cells no track of that camera occupies, then sub-pixel.
    fn refill(&mut self, current: &[Arc<LkFrame>], guess: &ShardGuess, timings: &mut ShardTimings) {
        let (
            TrackRefill::Enabled {
                per_camera,
                min_spacing_px,
                border_px,
                ..
            },
            Some(camera),
        ) = (guess.refill, guess.refill_camera)
        else {
            return;
        };
        let Some(frame) = current
            .get(camera)
            .filter(|_| guess.refill_shard == self.index)
        else {
            return;
        };
        let missing = per_camera.saturating_sub(guess.refill_occupied.len());
        if missing == 0 {
            return;
        }
        let started = Instant::now();
        let params = GoodFeaturesParams {
            max_corners: missing,
            quality: REFILL_QUALITY,
            min_spacing_px,
            border_px,
            half_resolution: true,
        };
        self.corners.clear();
        good_features_to_track(
            frame.plane(),
            &guess.refill_occupied,
            &params,
            &mut self.features,
            &mut self.corners,
        );
        corner_sub_pix(
            frame.plane(),
            &mut self.corners,
            SUBPIX_HALF_WIN,
            SUBPIX_MAX_ITER,
            SUBPIX_EPSILON,
        );
        // Sub-pixel can move a corner up to its half window, past the
        // survivor filter's border, where its first step would drop it.
        let margin = SurvivorFilterConfig::default().border_margin_px;
        let size = frame.plane().size();
        let (max_x, max_y) = (size.width as f32 - margin, size.height as f32 - margin);
        self.corners
            .retain(|&[x, y]| (margin..=max_x).contains(&x) && (margin..=max_y).contains(&y));
        self.spawned
            .extend(self.corners.iter().map(|&corner| (camera, corner)));
        timings.flow_refill += started.elapsed();
    }
}

/// Where `track` starts in the current frame: its map point at the trusted
/// prediction, carried into raw pixels by the distortion offset at its last
/// pixel; else its last ray rotated by the gyro (`GyroGuess::RotatedRay`);
/// else its last pixel.
fn start_pixel(track: &Track, model: &FlowCamera, guess: &ShardGuess) -> [f32; 2] {
    let raw = track.pixel();
    if let (Some(pose), Some(position)) = (guess.pose_world_to_primary, track.position()) {
        let p_primary = pose.transform_point(&position);
        let p_camera = model
            .camera_from_primary
            .map_or(p_primary, |t| t.transform_point(&p_primary));
        if let Some(pixel) = project_raw(&model.camera, &p_camera, raw) {
            return pixel;
        }
    }
    model
        .gyro_rotation
        .and_then(|rotation| rotated_ray_pixel(&model.camera, &rotation, raw))
        .unwrap_or(raw)
}

/// The ray through `raw` rotated by `R_cur_prev`, back in raw pixels.
pub(crate) fn rotated_ray_pixel(
    camera: &PinholeCamera,
    rotation: &Mat3F64,
    raw: [f32; 2],
) -> Option<[f32; 2]> {
    let undistorted = camera.undistort(raw[0] as f64, raw[1] as f64)?;
    let ray = kornia_algebra::Vec3F64::new(
        (undistorted.x - camera.cx) / camera.fx,
        (undistorted.y - camera.cy) / camera.fy,
        1.0,
    );
    project_raw(camera, &(*rotation * ray), raw)
}

/// A camera of the rig [`RigFlow`] tracks.
#[derive(Debug, Clone)]
pub struct RigFlowCamera {
    pub camera: PinholeCamera,
    /// `T_CP`; `None` for the primary camera.
    pub camera_from_primary: Option<Pose3d>,
}

/// The frontend knobs a [`RigFlow`] step reads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RigFlowSettings {
    /// Flow threads, the calling thread included.
    pub threads: usize,
    /// LK window side, odd.
    pub window: usize,
    /// `(max_level, max_iter)` from a trusted prediction.
    pub predicted: (usize, usize),
    /// `(max_level, max_iter)` without one.
    pub unpredicted: (usize, usize),
    /// LK convergence step (pixels).
    pub epsilon: f32,
    pub retry: KltRetry,
    pub forward_backward: ForwardBackward,
    pub refill: TrackRefill,
    pub gyro_guess: GyroGuess,
}

/// What the tracking core knows about a frame before its flow step.
#[derive(Debug, Clone, Copy, Default)]
pub struct FlowStepInput {
    pub frame_idx: u64,
    /// World to primary camera at the IMU prediction, when trusted.
    pub pose_world_to_primary: Option<Pose3d>,
    /// `R_cur_prev` of the primary camera from the gyro.
    pub primary_rotation: Option<Mat3F64>,
}

/// Tallies of one flow step.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FlowStepCounts {
    pub tracks_tried: u64,
    pub fb_rejects: u64,
    pub retries: u64,
    pub template_rebuilds: u64,
}

/// The survivors of one flow step.
#[derive(Debug, Default)]
pub struct FlowOutput {
    /// In (camera, track id) order, so the output is the same for any
    /// thread count.
    pub survivors: Vec<FlowSurvivor>,
    /// Camera of each survivor.
    pub cameras: Vec<usize>,
    /// Each survivor's pixel undistorted in its camera (NaN where that
    /// fails).
    pub undistorted: Vec<[f32; 2]>,
    /// `(camera, pixel)` of each refill corner, current frame, in order.
    pub spawned: Vec<(usize, [f32; 2])>,
    pub counts: FlowStepCounts,
    /// Tracks each shard ran in the first pass (shard imbalance).
    pub shard_tracks: [usize; MAX_FLOW_SHARDS],
}

impl FlowOutput {
    fn clear(&mut self) {
        self.survivors.clear();
        self.cameras.clear();
        self.undistorted.clear();
        self.spawned.clear();
        self.counts = FlowStepCounts::default();
        self.shard_tracks = [0; MAX_FLOW_SHARDS];
    }
}

/// The `WindowedU8` optical flow of every rig camera: a padded level-0 frame
/// pair per camera, one shard per flow thread, and the pool that runs them.
pub struct RigFlow {
    settings: RigFlowSettings,
    pool: FlowPool,
    frames: Option<FlowFrames>,
    shards: Vec<FlowShard>,
    guess: ShardGuess,
    /// `(camera, track id, survivor, undistorted)` of a step, before sorting.
    merged: Vec<(usize, TrackId, FlowSurvivor, [f32; 2])>,
    output: FlowOutput,
    /// Copies of the latest keyframe's frames, for `adopt_landmarks`.
    keyframe: Vec<LkFrame>,
    /// Template a landmark is adopted with.
    adopted: LkTemplate,
}

impl RigFlow {
    /// Starts `settings.threads - 1` flow workers for `cameras`, the primary
    /// first.
    pub fn new(
        settings: RigFlowSettings,
        cameras: Vec<RigFlowCamera>,
    ) -> Result<Self, FlowPoolError> {
        let pool = FlowPool::new(settings.threads)?;
        let shards = (0..pool.threads())
            .map(|index| FlowShard {
                index,
                ..FlowShard::default()
            })
            .collect();
        let guess = ShardGuess {
            cameras: cameras
                .into_iter()
                .map(|camera| FlowCamera {
                    camera: camera.camera,
                    camera_from_primary: camera.camera_from_primary,
                    gyro_rotation: None,
                })
                .collect(),
            window: settings.window,
            epsilon: settings.epsilon,
            forward_backward: settings.forward_backward,
            refill: settings.refill,
            ..ShardGuess::default()
        };
        Ok(Self {
            settings,
            pool,
            frames: None,
            shards,
            guess,
            merged: Vec::new(),
            output: FlowOutput::default(),
            keyframe: Vec::new(),
            adopted: LkTemplate::default(),
        })
    }

    /// Keeps a copy of the current frames as the latest keyframe's, which
    /// its new landmarks are adopted from once mapping returns.
    pub(crate) fn keep_keyframe(&mut self) -> Result<(), FlowPoolError> {
        let Some(frames) = self.frames.as_ref() else {
            return Ok(());
        };
        self.keyframe.truncate(frames.current.len());
        for (camera, frame) in frames.current.iter().enumerate() {
            match self.keyframe.get_mut(camera) {
                Some(copy) => copy.rebuild(frame.plane(), frame.frame_id())?,
                None => self.keyframe.push(LkFrame::new(
                    frame.plane(),
                    self.settings.window,
                    frame.frame_id(),
                )?),
            }
        }
        Ok(())
    }

    /// Where a landmark seen at `pixel` in camera `camera` of the kept
    /// keyframe `keyframe_idx` is in the previous frame, which the tracks'
    /// pixels refer to: its keyframe template tracked from `start`. `None`
    /// without that keyframe or when the solve fails the survivor filter.
    pub(crate) fn adopt(
        &mut self,
        keyframe_idx: u64,
        camera: usize,
        pixel: [f32; 2],
        start: [f32; 2],
    ) -> Option<[f32; 2]> {
        let keyframe = self
            .keyframe
            .get(camera)
            .filter(|frame| frame.frame_id() == keyframe_idx)?;
        let previous = self.frames.as_ref()?.previous.get(camera)?;
        if !extract_template(keyframe, pixel, 0, self.settings.window, &mut self.adopted) {
            return None;
        }
        let (_, max_iter) = self.settings.predicted;
        let result = track_template(
            &self.adopted,
            previous,
            start,
            max_iter,
            self.settings.epsilon,
        );
        let size = previous.plane().size();
        KltTracker::accepts(
            &Track::at(pixel),
            u8::from(result.status),
            result.error,
            result.pixel,
            [size.width as f32, size.height as f32],
            &SurvivorFilterConfig::default(),
        )
        .then_some(result.pixel)
    }

    pub fn threads(&self) -> usize {
        self.pool.threads()
    }

    pub fn cameras(&self) -> usize {
        self.guess.cameras.len()
    }

    /// Whether a frame has been prepared, so the next one has a previous.
    pub fn has_frames(&self) -> bool {
        self.frames.is_some()
    }

    /// Makes `images` (one per camera, primary first) the current frames and
    /// the last ones the previous: one padded level-0 copy per camera, in
    /// place. Images past the rig's cameras are ignored.
    pub fn prepare(
        &mut self,
        images: &[&Image<u8, 1>],
        frame_idx: u64,
        timings: &mut TrackingTimings,
    ) -> Result<(), FlowPoolError> {
        let started = Instant::now();
        let prepared = self.rebuild(images, frame_idx);
        if prepared.is_err() {
            self.frames = None;
        }
        timings.flow_prepare += started.elapsed();
        prepared
    }

    fn rebuild(&mut self, images: &[&Image<u8, 1>], frame_idx: u64) -> Result<(), FlowPoolError> {
        let cameras = self.cameras();
        let images = &images[..images.len().min(cameras)];
        let Some(frames) = self.frames.as_mut() else {
            // Two copies of the first frame, so each side can be rebuilt in place.
            let mut frames = FlowFrames {
                previous: Vec::with_capacity(cameras),
                current: Vec::with_capacity(cameras),
            };
            for image in images {
                let window = self.settings.window;
                frames
                    .previous
                    .push(Arc::new(LkFrame::new(image, window, frame_idx)?));
                frames
                    .current
                    .push(Arc::new(LkFrame::new(image, window, frame_idx)?));
            }
            self.frames = Some(frames);
            return Ok(());
        };
        std::mem::swap(&mut frames.previous, &mut frames.current);
        for (frame, image) in frames.current.iter_mut().zip(images) {
            Arc::get_mut(frame)
                .ok_or(FlowPoolError::FrameShared)?
                .rebuild(image, frame_idx)?;
        }
        Ok(())
    }

    /// The frame of `camera` the track set's pixels refer to after
    /// [`Self::prepare`]: the previous one.
    pub(crate) fn previous_frame(&self, camera: usize) -> Option<&LkFrame> {
        self.frames
            .as_ref()?
            .previous
            .get(camera)
            .map(|frame| frame.as_ref())
    }

    /// The output of the latest [`Self::step`].
    pub fn output(&self) -> &FlowOutput {
        &self.output
    }

    /// Tracks `tracks` from the previous frames into the current ones.
    pub fn step(
        &mut self,
        tracks: &TrackSet,
        input: &FlowStepInput,
        timings: &mut TrackingTimings,
    ) -> Result<&FlowOutput, FlowPoolError> {
        self.output.clear();
        self.merged.clear();
        if self.frames.is_none() || tracks.is_empty() {
            return Ok(&self.output);
        }
        let started = Instant::now();
        let (max_level, max_iter) = match input.pose_world_to_primary {
            Some(_) => self.settings.predicted,
            None => self.settings.unpredicted,
        };
        self.fill_guess(input, max_level, max_iter);
        self.partition(tracks);
        self.plan_refill(tracks, input.frame_idx);
        self.build_coarse(max_level)?;
        timings.flow_prepare += started.elapsed();

        self.run_pass(timings)?;
        if let KltRetry::CoarseLevel {
            max_level: retry_level,
        } = self.settings.retry
            && retry_level > max_level
            && self.shards.iter().any(|shard| !shard.retry.is_empty())
        {
            let started = Instant::now();
            self.keep_retries();
            self.guess.retry = true;
            self.guess.max_level = retry_level;
            self.guess.refill_camera = None;
            self.build_coarse(retry_level)?;
            timings.flow_prepare += started.elapsed();
            self.run_pass(timings)?;
        }
        self.finish();
        Ok(&self.output)
    }

    /// The per-frame part of the guess, in place.
    fn fill_guess(&mut self, input: &FlowStepInput, max_level: usize, max_iter: usize) {
        let guess = &mut self.guess;
        guess.frame_idx = input.frame_idx;
        guess.pose_world_to_primary = input.pose_world_to_primary;
        guess.max_level = max_level;
        guess.max_iter = max_iter;
        guess.retry = false;
        guess.refill_camera = None;
        guess.refill_occupied.clear();
        let primary_rotation = input
            .primary_rotation
            .filter(|_| self.settings.gyro_guess == GyroGuess::RotatedRay);
        for camera in &mut guess.cameras {
            // R_cur_prev(c) = R_cp R_cur_prev(0) R_pc, through the extrinsic.
            camera.gyro_rotation = primary_rotation.map(|rotation| {
                camera.camera_from_primary.map_or(rotation, |t_cp| {
                    t_cp.rotation * rotation * t_cp.rotation.transpose()
                })
            });
        }
    }

    /// Picks this frame's refill camera, round robin, when it holds fewer
    /// than `refill_below` of its tracks; its occupied pixels are every one of
    /// its tracks' start pixels in the current frame, which refill detects
    /// on, across shards, so the corners do not depend on the split.
    fn plan_refill(&mut self, tracks: &TrackSet, frame_idx: u64) {
        let TrackRefill::Enabled {
            per_camera,
            refill_below,
            ..
        } = self.settings.refill
        else {
            return;
        };
        let camera = (frame_idx % self.cameras().max(1) as u64) as usize;
        let Some(model) = self.guess.cameras.get(camera) else {
            return;
        };
        let mut occupied = std::mem::take(&mut self.guess.refill_occupied);
        occupied.extend(
            tracks
                .tracks()
                .iter()
                .filter(|track| track.camera() == camera)
                .map(|track| start_pixel(track, model, &self.guess)),
        );
        let guess = &mut self.guess;
        guess.refill_occupied = occupied;
        if (guess.refill_occupied.len() as f32) < refill_below * per_camera as f32 {
            guess.refill_camera = Some(camera);
            guess.refill_shard = tracks.refill_shard(camera);
        } else {
            guess.refill_occupied.clear();
        }
    }

    /// Hands every track to the shard fixed at its spawn, in (camera, track
    /// id) order.
    fn partition(&mut self, tracks: &TrackSet) {
        let last = self.shards.len() - 1;
        for shard in &mut self.shards {
            shard.tracks.clear();
        }
        for track in tracks.tracks() {
            self.shards[track.shard().min(last)].tracks.push(*track);
        }
        for shard in &mut self.shards {
            shard
                .tracks
                .sort_unstable_by_key(|track| (track.camera(), track.id()));
        }
    }

    /// Levels `1..=max_level` of both frame sides, which the workers no
    /// longer share after a join.
    fn build_coarse(&mut self, max_level: usize) -> Result<(), FlowPoolError> {
        if max_level == 0 {
            return Ok(());
        }
        let frames = self.frames.as_mut().ok_or(FlowPoolError::FrameShared)?;
        for frame in frames.previous.iter_mut().chain(frames.current.iter_mut()) {
            Arc::get_mut(frame)
                .ok_or(FlowPoolError::FrameShared)?
                .build_coarse(max_level)?;
        }
        Ok(())
    }

    /// One pool step over every shard, its survivors merged in.
    fn run_pass(&mut self, timings: &mut TrackingTimings) -> Result<(), FlowPoolError> {
        let frames = self.frames.as_ref().ok_or(FlowPoolError::FrameShared)?;
        if !self.guess.retry {
            for (shard, slot) in self.shards.iter().zip(&mut self.output.shard_tracks) {
                self.output.counts.tracks_tried += shard.tracks.len() as u64;
                *slot = shard.tracks.len();
            }
        }
        self.pool.step(&mut self.shards, frames, &self.guess)?;
        self.pool.add_timings(timings);
        for shard in &self.shards {
            let counts = &mut self.output.counts;
            counts.fb_rejects += u64::from(shard.fb_rejects);
            counts.template_rebuilds += u64::from(shard.template_rebuilds);
            if self.guess.retry {
                counts.retries += shard.tracks.len() as u64;
            }
            self.output.spawned.extend_from_slice(&shard.spawned);
            // Survivors are a subsequence of the shard's tracks.
            let mut tracks = shard.tracks.iter();
            for (survivor, undistorted) in shard.survivors.iter().zip(&shard.undistorted) {
                if let Some(track) = tracks.find(|track| track.id() == survivor.track_id) {
                    self.merged
                        .push((track.camera(), track.id(), *survivor, *undistorted));
                }
            }
        }
        Ok(())
    }

    /// Keeps in each shard only the tracks that failed the first pass.
    fn keep_retries(&mut self) {
        for shard in &mut self.shards {
            let FlowShard { tracks, retry, .. } = shard;
            let mut failed = retry.iter().peekable();
            tracks.retain(|track| failed.next_if(|&&id| id == track.id()).is_some());
        }
    }

    /// Sorts the merged survivors into the output.
    fn finish(&mut self) {
        self.merged
            .sort_unstable_by_key(|&(camera, id, _, _)| (camera, id));
        for &(camera, _, survivor, undistorted) in &self.merged {
            self.output.survivors.push(survivor);
            self.output.cameras.push(camera);
            self.output.undistorted.push(undistorted);
        }
    }
}

/// The seeds of a frame tracked from flow alone, without a frame: per
/// survivor with a live map point, its raw and undistorted pixel, rig camera,
/// map point and world point, in camera order.
#[derive(Debug, Default)]
pub struct FlowSeeds {
    pub pixels: Vec<[f32; 2]>,
    /// NaN where the camera cannot undistort the pixel.
    pub undistorted: Vec<[f32; 2]>,
    pub cameras: Vec<usize>,
    pub map_points: Vec<usize>,
    /// World frame.
    pub points: Vec<Vec3F64>,
    /// The feature each seed stands for in a pose's matches: its keypoint
    /// in the frame it came from, else its own index.
    pub keypoints: Vec<usize>,
    /// `(track, map point, world point)` of every associated track, by
    /// track, reused so assembling allocates nothing once warm.
    associated: Vec<(TrackId, usize, Vec3F64)>,
}

impl FlowSeeds {
    pub(crate) fn len(&self) -> usize {
        self.map_points.len()
    }

    fn clear(&mut self) {
        self.pixels.clear();
        self.undistorted.clear();
        self.cameras.clear();
        self.map_points.clear();
        self.points.clear();
        self.keypoints.clear();
    }

    fn push(
        &mut self,
        pixel: [f32; 2],
        undistorted: [f32; 2],
        camera: usize,
        map_point: usize,
        point: Vec3F64,
        keypoint: usize,
    ) {
        self.keypoints.push(keypoint);
        self.pixels.push(pixel);
        self.undistorted.push(undistorted);
        self.cameras.push(camera);
        self.map_points.push(map_point);
        self.points.push(point);
    }

    /// The seeds of a `WindowedU8` step's `output` from each track's stored
    /// position, without the map.
    pub fn assemble(&mut self, tracks: &TrackSet, output: &FlowOutput) {
        self.clear();
        let mut associated = std::mem::take(&mut self.associated);
        associated.clear();
        associated.extend(
            tracks
                .tracks()
                .iter()
                .filter_map(|track| Some((track.id(), track.map_point_idx()?, track.position()?))),
        );
        associated.sort_unstable_by_key(|&(id, _, _)| id);
        for ((survivor, &camera), &undistorted) in output
            .survivors
            .iter()
            .zip(&output.cameras)
            .zip(&output.undistorted)
        {
            let Ok(slot) = associated.binary_search_by_key(&survivor.track_id, |&(id, _, _)| id)
            else {
                continue;
            };
            let (_, map_point, point) = associated[slot];
            let seed = self.len();
            self.push(survivor.pixel, undistorted, camera, map_point, point, seed);
        }
        self.associated = associated;
    }

    /// The seeds of a flow frame: `seeded[k]` is `(map point, keypoint)` of
    /// `frame`, its world point from `map`.
    pub(crate) fn assemble_from_frame(
        &mut self,
        frame: &Frame,
        seeded: &[(usize, usize)],
        map: &Map,
    ) {
        self.clear();
        for &(map_point, keypoint) in seeded {
            let (Some(&pixel), Some(&undistorted), Some(mp)) = (
                frame.features.keypoints_xy.get(keypoint),
                frame.keypoints_undist.get(keypoint),
                map.map_points().get(map_point),
            ) else {
                continue;
            };
            let camera = frame.camera_of(keypoint);
            self.push(pixel, undistorted, camera, map_point, mp.position, keypoint);
        }
    }
}

#[cfg(test)]
mod tests {
    use kornia_image::ImageSize;

    use super::*;
    use crate::tracking::optical_flow::TrackSet;

    const WIDTH: usize = 200;
    const HEIGHT: usize = 160;

    /// Smooth texture, so LK converges anywhere, moved right by `shift` px.
    fn textured(shift: f32) -> Image<u8, 1> {
        let data = (0..WIDTH * HEIGHT)
            .map(|i| {
                let x = (i % WIDTH) as f32 - shift;
                let y = (i / WIDTH) as f32;
                let v = 128.0
                    + 50.0 * (x / 5.0).sin() * (y / 7.0).cos()
                    + 40.0 * ((x + 2.0 * y) / 9.0).sin();
                v.round() as u8
            })
            .collect();
        Image::new(
            ImageSize {
                width: WIDTH,
                height: HEIGHT,
            },
            data,
        )
        .unwrap()
    }

    fn frame(shift: f32, frame_id: u64) -> Arc<LkFrame> {
        Arc::new(LkFrame::new(&textured(shift), 11, frame_id).unwrap())
    }

    fn pinhole() -> PinholeCamera {
        PinholeCamera {
            fx: 200.0,
            fy: 200.0,
            cx: 100.0,
            cy: 80.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

    /// One camera, level 0 only, round trip at the preset's 0.6 px.
    fn guess() -> ShardGuess {
        ShardGuess {
            cameras: vec![FlowCamera {
                camera: pinhole(),
                camera_from_primary: None,
                gyro_rotation: None,
            }],
            max_level: 0,
            max_iter: 20,
            epsilon: 0.01,
            window: 11,
            forward_backward: ForwardBackward::Enabled {
                max_round_trip_px: 0.6,
                max_iter: 10,
                every_n_frames: 1,
            },
            ..ShardGuess::default()
        }
    }

    /// ORB rewrites a track's pixel on a keyframe frame; the next frame's
    /// template must be cut there, not at the old flow pixel: flow at
    /// (100.4, 80.7), keypoint at (101, 80), 0.92 px apart.
    #[test]
    fn a_track_rewritten_between_frames_survives_its_round_trip() {
        let frames = [frame(0.0, 0), frame(2.0, 1), frame(4.0, 2)];
        let guess = guess();
        let mut timings = ShardTimings::default();
        let mut tracks = TrackSet::new();
        tracks.spawn([98.4, 80.7]);
        let mut shard = FlowShard {
            tracks: tracks.tracks().to_vec(),
            ..FlowShard::default()
        };
        shard.run(&frames[0..1], &frames[1..2], &guess, &mut timings);
        let flow = shard.survivors[0];
        assert!(distance(flow.pixel, [100.4, 80.7]) < 0.05, "{flow:?}");

        let keypoint = [101.0, 80.0];
        assert!(distance(flow.pixel, keypoint) > 0.9);
        tracks
            .advance(vec![FlowSurvivor {
                pixel: keypoint,
                ..flow
            }])
            .unwrap();
        shard.tracks = tracks.tracks().to_vec();
        shard.run(&frames[1..2], &frames[2..3], &guess, &mut timings);
        assert_eq!(shard.fb_rejects, 0);
        assert_eq!(shard.survivors.len(), 1);
        assert!(distance(shard.survivors[0].pixel, [103.0, 80.0]) < 0.05);
    }

    fn refill() -> TrackRefill {
        TrackRefill::Enabled {
            per_camera: 48,
            min_spacing_px: 23.0,
            refill_below: 0.8,
            border_px: 9,
        }
    }

    fn rig_flow(gyro_guess: GyroGuess) -> RigFlow {
        let settings = RigFlowSettings {
            threads: 1,
            window: 11,
            predicted: (0, 8),
            unpredicted: (0, 8),
            epsilon: 0.01,
            retry: KltRetry::Off,
            forward_backward: ForwardBackward::Off,
            refill: refill(),
            gyro_guess,
        };
        let camera = RigFlowCamera {
            camera: pinhole(),
            camera_from_primary: None,
        };
        RigFlow::new(settings, vec![camera]).unwrap()
    }

    /// Refill detects on the current frame, so the occupied pixels are
    /// where the tracks start there, not where they were.
    #[test]
    fn refill_occupancy_is_where_the_tracks_start() {
        let mut flow = rig_flow(GyroGuess::RotatedRay);
        let mut tracks = TrackSet::new();
        for pixel in [[60.0, 50.0], [120.0, 90.0]] {
            tracks.spawn(pixel);
        }
        // 7 degrees of yaw: about 25 px of flow at f = 200 px.
        let yaw = 7f64.to_radians();
        let rotation = Mat3F64::from_cols(
            kornia_algebra::Vec3F64::new(yaw.cos(), 0.0, -yaw.sin()),
            kornia_algebra::Vec3F64::new(0.0, 1.0, 0.0),
            kornia_algebra::Vec3F64::new(yaw.sin(), 0.0, yaw.cos()),
        );
        let input = FlowStepInput {
            frame_idx: 0,
            pose_world_to_primary: None,
            primary_rotation: Some(rotation),
        };

        flow.fill_guess(&input, 0, 8);
        flow.plan_refill(&tracks, 0);

        assert_eq!(flow.guess.refill_camera, Some(0));
        assert_eq!(flow.guess.refill_occupied.len(), 2);
        for (track, occupied) in tracks.tracks().iter().zip(&flow.guess.refill_occupied) {
            let start = rotated_ray_pixel(&pinhole(), &rotation, track.pixel()).unwrap();
            assert!(distance(start, track.pixel()) > 20.0);
            assert_eq!(*occupied, start);
        }
    }

    /// A camera yawing by +5 degrees about its y axis (towards +x) sees
    /// a point ahead move to `cx - f tan 5°`, whatever the formula inside.
    #[test]
    fn a_yaw_moves_the_ray_by_f_tan_yaw_against_the_turn() {
        let yaw = 5f64.to_radians();
        let (sin, cos) = yaw.sin_cos();
        // R_cur_prev = R_y(yaw)^T for R_wc_cur = R_wc_prev R_y(yaw).
        let cur_from_prev = Mat3F64::from_cols(
            kornia_algebra::Vec3F64::new(cos, 0.0, sin),
            kornia_algebra::Vec3F64::new(0.0, 1.0, 0.0),
            kornia_algebra::Vec3F64::new(-sin, 0.0, cos),
        );
        let camera = pinhole();
        let centre = [camera.cx as f32, camera.cy as f32];

        let moved = rotated_ray_pixel(&camera, &cur_from_prev, centre).unwrap();

        let want = camera.cx - camera.fx * yaw.tan();
        assert!((f64::from(moved[0]) - want).abs() < 1e-3, "{moved:?}");
        assert!((f64::from(moved[1]) - camera.cy).abs() < 1e-3, "{moved:?}");
    }

    /// An adopted landmark is tracked from the kept keyframe into the
    /// previous frame, which the tracks' pixels refer to, and only from the
    /// keyframe it was seen in.
    #[test]
    fn adoption_tracks_a_landmark_from_its_keyframe_into_the_previous_frame() {
        let mut flow = rig_flow(GyroGuess::Off);
        let mut timings = TrackingTimings::default();
        flow.prepare(&[&textured(0.0)], 0, &mut timings).unwrap();
        flow.keep_keyframe().unwrap();
        flow.prepare(&[&textured(2.0)], 1, &mut timings).unwrap();
        flow.prepare(&[&textured(4.0)], 2, &mut timings).unwrap();
        let pixel = [90.0, 70.0];

        let adopted = flow.adopt(0, 0, pixel, [93.0, 70.5]).unwrap();

        assert!(distance(adopted, [92.0, 70.0]) < 0.05, "{adopted:?}");
        assert_eq!(flow.adopt(1, 0, pixel, pixel), None);
    }

    /// Bright squares whose left or top edge lies between pixels 9 and 10,
    /// and one well inside.
    fn squares_at_the_border() -> Image<u8, 1> {
        let inside = |x: usize, y: usize| {
            let left = (10..30).contains(&x) && (y % 40) >= 20 && (y % 40) < 36 && y < 140;
            let top = (10..26).contains(&y) && (x % 40) >= 20 && (x % 40) < 36 && x > 40;
            let centre = (100..130).contains(&x) && (70..100).contains(&y);
            left || top || centre
        };
        let data = (0..WIDTH * HEIGHT)
            .map(|i| {
                if inside(i % WIDTH, i / WIDTH) {
                    220
                } else {
                    30
                }
            })
            .collect();
        Image::new(
            ImageSize {
                width: WIDTH,
                height: HEIGHT,
            },
            data,
        )
        .unwrap()
    }

    /// A refill corner `corner_sub_pix` moved inside the survivor
    /// filter's border would die on its first step, so it is not spawned.
    #[test]
    fn refill_spawns_no_corner_the_survivor_filter_would_drop() {
        let current = [Arc::new(
            LkFrame::new(&squares_at_the_border(), 11, 0).unwrap(),
        )];
        let guess = ShardGuess {
            refill: refill(),
            refill_camera: Some(0),
            ..guess()
        };
        let mut shard = FlowShard::default();
        let mut timings = ShardTimings::default();

        shard.refill(&current, &guess, &mut timings);

        let margin = SurvivorFilterConfig::default().border_margin_px;
        let (w, h) = (WIDTH as f32, HEIGHT as f32);
        assert!(
            shard
                .spawned
                .iter()
                .filter(|(_, [x, y])| *x > 90.0 && *y > 60.0)
                .count()
                >= 4,
            "{:?}",
            shard.spawned
        );
        for &(_, [x, y]) in &shard.spawned {
            assert!(
                x >= margin && y >= margin && x <= w - margin && y <= h - margin,
                "({x}, {y}) of {:?}",
                shard.spawned
            );
        }
    }
}
