//! SLAM runtime: its configuration, construction and processing flow.
//!
//! The processing flow is kept in this file so it can be read from top to
//! bottom in the same order frames move through the system. With the `serde`
//! feature, configurations load from versioned RON files.

mod build;
mod config;
#[cfg(feature = "serde")]
mod file;
mod flow_counters;
mod input;
mod state;
mod validation;

pub use build::BuildError;
pub use config::{
    CameraSelection, FlowPose, ForwardBackward, FrontendConfig, GyroGuess, KltFirst, KltPrediction,
    KltRetry, KltRig, LoopClosingMode, MAX_FLOW_THREADS, MAX_KLT_WINDOW, MappingConfig,
    OrbFrontendConfig, OrbSlamPipeline, OrbTuning, PipelineDefinition, SYSTEM_CONFIG_VERSION,
    SensorSelection, StereoCloseDepth, SystemConfig, TrackRefill,
};
#[cfg(feature = "serde")]
pub use file::LoadError;
pub use flow_counters::FlowCounters;
pub use input::ProcessError;
pub use state::{TrackingResult, TrackingStatus};
pub use validation::ConfigError;

use crate::frontend::{FrontendObservation, OrbFrontend};
use state::{SystemMode, SystemState};

use crate::tracking::{
    KeyframeEvidence, KeyframePolicy, TrackingLossRecoveryPolicy, median_parallax_px,
};

use crate::tracking::flow_tracker::{FlowStepInput, RigFlow, RigFlowCamera, RigFlowSettings};
use crate::tracking::local_map::{LocalMapSelectionConfig, select_local_map_points};
use crate::tracking::motion::{InertialPrediction, predict_pose};
use crate::tracking::optical_flow::{FlowSurvivor, LkKernel};
use crate::tracking::parallax::median_parallax_px_with;
use crate::tracking::pose_estimation::Estimate;
use crate::tracking::timings::TrackingTimings;
use crate::tracking::tracker::{AnalyticSeedSource, FlowGuess, FrameInput, Tracker};

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::Frame;
use crate::inertial::{
    AppliedInitialization, ImuInitResult, InertialState, viba0_accel_bias_prior,
};
use crate::initialization::bootstrap::{
    self, BootstrapDecision, MIN_KEYPOINTS_FOR_BOOTSTRAP, MIN_STEREO_POINTS, MIN_VALID_POINTS,
    TooFewStereoPoints, evaluate_bootstrap,
};
use crate::initialization::two_view::TwoViewInitConfig;
use crate::loop_closure::{LoopCloser, LoopClosingContext, LoopClosureEvent};
use crate::mapping::keyframe_mapping::{self, KeyframeGrowthResult, KeyframeInsertion};
use crate::mapping::map::{
    InertialAlignmentError, Keyframe, Map, MapInsertion, MapMutationError, MapPoint,
};
use crate::mapping::{KeyframeJob, LocalMapping};
use crate::pose_conversion::apply_reference_pose_correction;
use build::{FlowSettings, SystemSettings};
use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, Vec3F64};
use kornia_image::Image;
use kornia_sensors::SensorRig;
use kornia_sensors::imu::ImuMeasurement;

/// The images of one capture, kept until its features are extracted.
#[derive(Clone, Copy)]
enum Capture<'a> {
    Mono(&'a Image<u8, 1>),
    Stereo(&'a Image<u8, 1>, &'a Image<u8, 1>),
    Rig(&'a [&'a Image<u8, 1>]),
}

/// A capture numbered as frame `idx`, not yet extracted.
#[derive(Clone, Copy)]
struct PendingFrame<'a> {
    idx: usize,
    capture: Capture<'a>,
}

/// A frame's features: extracted, or deferred until tracking needs them.
enum FrameSource<'a> {
    Ready(Box<Frame>),
    Deferred(PendingFrame<'a>),
}

impl FrameSource<'_> {
    fn idx(&self) -> usize {
        match self {
            Self::Ready(frame) => frame.idx,
            Self::Deferred(pending) => pending.idx,
        }
    }
}

/// The frame tracking settled on: an ORB frame, or the source of one that
/// optical flow tracked without ORB.
enum TrackedFrame<'a> {
    Orb(Box<Frame>),
    Flow(FrameSource<'a>),
}

/// Per-frame estimation inputs from the motion model and loss state.
struct Estimation {
    candidate_pose: Pose3d,
    pose_before: Pose3d,
    lost_for_sec: f64,
    trusted_prediction: bool,
}

/// What tracking a deferred frame from its flow survivors alone gave.
enum FlowOnly {
    /// Too few survivors with a map point, or no flow at all.
    Unavailable,
    /// The flow-only pose was not accepted.
    Refused,
    /// The flow-only pose was accepted, but the frame is due as a keyframe,
    /// which needs ORB; the pose seeds the full estimate.
    KeyframeDue(Pose3d),
    /// Tracked without ORB, on the tracker's flow stand-in frame.
    Tracked(Estimate),
}

/// Top-level ORB-SLAM system: orchestrates tracking, mapping, and state transitions.
pub struct SlamSystem {
    rig: SensorRig,
    /// The primary camera's measurement pinhole, cached once.
    camera: PinholeCamera,
    /// Baseline of the selected rectified pair; `None` unless stereo is selected.
    stereo_baseline_m: Option<f64>,
    /// Every rig camera is tracked; only under `CameraSelection::Rig`.
    multi_camera: bool,
    tracker: Tracker,
    two_view_init_config: TwoViewInitConfig,
    keyframe_policy: KeyframePolicy,
    tracking_loss_recovery: TrackingLossRecoveryPolicy,
    // mThDepth (metres): back-project close stereo points at each keyframe when set
    stereo_close_depth: Option<f64>,
    debug: bool,
    debug_messages: Vec<String>,
    map: Arc<Mutex<Map>>,
    // Serializes compound map publication and short local-BA snapshot/merge phases.
    map_publication_gate: Option<Arc<Mutex<()>>>,
    // IMU estimates, sample buffer and the inertial-initialization schedule
    inertial: InertialState,
    local_mapping: LocalMapping,
    // Place recognition for every keyframe, and loop closing when configured.
    loop_closer: Option<LoopCloser>,
    loop_closure_events: Vec<LoopClosureEvent>,
    state: SystemState,
    frontend: OrbFrontend,
    // The last processed image, for optical-flow tracks into the next frame.
    previous_image: Option<Image<u8, 1>>,
    tracking_duration: Duration,
    tracking_timings: TrackingTimings,
    /// Extraction time of a deferred frame, kept out of `tracking_duration`.
    deferred_extraction: Duration,
    /// Survivors with a map point a frame needs to be tracked by optical flow
    /// alone (`klt_first`); `None` extracts ORB on every frame.
    klt_min_tracks: Option<usize>,
    flow_counters: FlowCounters,
    /// `(max_level, max_iter)` of the optical-flow search from a trusted
    /// inertial prediction; `None` searches the full pyramid.
    klt_prediction: Option<(usize, usize)>,
    flow: FlowSettings,
    /// `WindowedU8` flow: level-0 frames, shards and workers; `None` on the
    /// full-pyramid kernels.
    rig_flow: Option<RigFlow>,
    /// The last images of rig cameras 1 and up, when the full-pyramid
    /// kernels track every camera.
    previous_secondary: Vec<Image<u8, 1>>,
    /// The primary camera's `R_cur_prev` from this frame's preintegrated
    /// gyro (`GyroGuess::RotatedRay`).
    gyro_rotation: Option<Mat3F64>,
    /// The keyframe whose landmarks were last adopted (`adopt_landmarks`).
    adopted_keyframe: Option<usize>,
    // Number given to the next frame; it becomes the keyframe's id in the map.
    next_frame_idx: usize,
}

impl SlamSystem {
    /// Builds the system `config` describes from the source's calibrated rig,
    /// restricted to the selected sensors. Resources such as the vocabulary
    /// are loaded before any mapping worker starts.
    pub fn build(config: SystemConfig, rig: SensorRig) -> Result<Self, BuildError> {
        config.validate()?;
        let rig = config.sensors.select_rig(rig)?;
        config.validate_flow_threads(&rig)?;
        config.validate_raw_fisheye(&rig)?;
        let settings = config.settings(&rig);
        let frontend = OrbFrontend::new(
            config.orb_detector(),
            &rig,
            settings.stereo_baseline_m,
            settings.multi_camera,
        );
        let loop_closer = config.loop_closer(&rig)?;
        let rig_flow = match settings.klt_kernel {
            LkKernel::WindowedU8 => Some(RigFlow::new(
                rig_flow_settings(&settings),
                rig_flow_cameras(&rig, &frontend, settings.flow.rig),
            )?),
            LkKernel::F32 | LkKernel::FixedPointU8 => None,
        };
        Ok(Self::assemble(
            rig,
            settings,
            frontend,
            loop_closer,
            rig_flow,
        ))
    }

    fn assemble(
        rig: SensorRig,
        settings: SystemSettings,
        frontend: OrbFrontend,
        loop_closer: Option<LoopCloser>,
        rig_flow: Option<RigFlow>,
    ) -> Self {
        let camera = rig.primary().pinhole();
        let flow = settings.flow;
        let mut tracker = Tracker::new(settings.map_projection)
            .with_lk_kernel(settings.klt_kernel)
            .with_lk_window(flow.window)
            .with_track_refill(flow.refill);
        if flow.rig == KltRig::AllCameras
            && let Some(geometry) = frontend.rig_geometry()
        {
            tracker = tracker.with_rig_flow(Arc::clone(geometry), flow.threads);
        }
        let map = Arc::new(Mutex::new(Map::new()));
        let local_mapping =
            LocalMapping::new(settings.local_mapping, Arc::clone(&map), camera.clone());
        let map_publication_gate = local_mapping.publication_gate();
        Self {
            rig,
            camera,
            stereo_baseline_m: settings.stereo_baseline_m,
            multi_camera: settings.multi_camera,
            tracker,
            two_view_init_config: settings.two_view_init,
            keyframe_policy: settings.keyframe_policy,
            tracking_loss_recovery: settings.tracking_loss_recovery,
            stereo_close_depth: settings.stereo_close_depth_m,
            debug: false,
            debug_messages: Vec::new(),
            map,
            map_publication_gate,
            local_mapping,
            state: SystemState::new(),
            inertial: InertialState::new(),
            loop_closer,
            loop_closure_events: Vec::new(),
            frontend,
            previous_image: None,
            tracking_duration: Duration::ZERO,
            tracking_timings: TrackingTimings::default(),
            deferred_extraction: Duration::ZERO,
            klt_min_tracks: settings.klt_min_tracks,
            flow_counters: FlowCounters::default(),
            klt_prediction: settings.klt_prediction,
            flow,
            rig_flow,
            previous_secondary: Vec::new(),
            gyro_rotation: None,
            adopted_keyframe: None,
            next_frame_idx: 0,
        }
    }

    pub fn drain_loop_closure_events(&mut self) -> Vec<LoopClosureEvent> {
        std::mem::take(&mut self.loop_closure_events)
    }

    /// The system's sensor calibration, restricted to the selected sensors.
    pub fn rig(&self) -> &SensorRig {
        &self.rig
    }

    /// Buffers IMU samples for the frames to come, in the body frame. Call it
    /// as samples arrive; each frame uses the samples up to its own
    /// timestamp. The caller must feed every sample up to an image's
    /// timestamp before passing that image: a frame processed ahead of its
    /// samples predicts from stale ones, and its keyframe's IMU edge stays
    /// short for good. Samples with a non-finite timestamp or more than 30 s
    /// ahead of the latest image are ignored, and so are all samples when the
    /// rig has no IMU.
    pub fn process_imu(&mut self, samples: &[ImuMeasurement]) {
        if self.rig.imu.is_none() {
            return;
        }
        self.inertial.insert_buffer_samples(samples);
    }

    /// Extracts features from one image captured at `timestamp_sec` (on the
    /// IMU samples' clock) and tracks it. The system numbers its frames
    /// itself, in call order. For a rig with a fisheye model the image is the
    /// raw fisheye image (see [`SensorRig`]).
    ///
    /// # Errors
    ///
    /// Invalid input, a stereo rig (use [`Self::process_stereo`]) or a
    /// feature-extraction failure; the system state, including image history,
    /// is then unchanged. IMU samples already buffered are kept.
    pub fn process_image(
        &mut self,
        timestamp_sec: f64,
        image: &Image<u8, 1>,
    ) -> Result<TrackingResult, ProcessError> {
        input::validate_image(timestamp_sec, image)?;
        if self.stereo_baseline_m.is_some() {
            return Err(ProcessError::StereoRequired);
        }
        if self.multi_camera {
            return Err(ProcessError::RigRequired);
        }
        self.track_capture(Capture::Mono(image), image, timestamp_sec)
    }

    /// Like [`Self::process_image`] for a rectified stereo pair; a mono
    /// pipeline rejects it (use [`Self::process_image`]).
    pub fn process_stereo(
        &mut self,
        timestamp_sec: f64,
        left: &Image<u8, 1>,
        right: &Image<u8, 1>,
    ) -> Result<TrackingResult, ProcessError> {
        input::validate_stereo_pair(timestamp_sec, left, right)?;
        if self.multi_camera {
            return Err(ProcessError::RigRequired);
        }
        if self.stereo_baseline_m.is_none() {
            return Err(ProcessError::MonoPipeline);
        }
        self.track_capture(Capture::Stereo(left, right), left, timestamp_sec)
    }

    /// Like [`Self::process_image`] for one capture of a multi-camera rig:
    /// `images[c]` comes from rig camera `c` (see [`SensorRig::cameras`]),
    /// all taken at `timestamp_sec`. Other pipelines reject it.
    pub fn process_rig(
        &mut self,
        timestamp_sec: f64,
        images: &[&Image<u8, 1>],
    ) -> Result<TrackingResult, ProcessError> {
        if !self.multi_camera {
            return Err(ProcessError::NotARig);
        }
        input::validate_rig_images(timestamp_sec, images, self.rig.cameras())?;
        let Some(&primary) = images.first() else {
            return Err(ProcessError::NotARig);
        };
        self.track_capture(Capture::Rig(images), primary, timestamp_sec)
    }

    /// Extracts the capture's features now, or defers extraction to tracking
    /// when optical flow may carry the frame alone.
    fn track_capture(
        &mut self,
        capture: Capture<'_>,
        image: &Image<u8, 1>,
        timestamp_sec: f64,
    ) -> Result<TrackingResult, ProcessError> {
        let pending = PendingFrame {
            idx: self.next_frame_idx,
            capture,
        };
        let primary_only = [image];
        let images: &[&Image<u8, 1>] = match capture {
            Capture::Rig(images) => images,
            Capture::Mono(_) | Capture::Stereo(..) => &primary_only,
        };
        let deferrable = self.klt_min_tracks.is_some()
            && self.state.mode != SystemMode::Bootstrap
            && (self.previous_image.is_some()
                || self.rig_flow.as_ref().is_some_and(RigFlow::has_frames));
        let source = if deferrable {
            FrameSource::Deferred(pending)
        } else {
            FrameSource::Ready(Box::new(self.extract(pending)?))
        };
        self.track_frame(source, image, images, timestamp_sec)
    }

    /// ORB features of a capture.
    fn extract(&mut self, pending: PendingFrame<'_>) -> Result<Frame, ProcessError> {
        match pending.capture {
            Capture::Mono(image) => self.frontend.prepare_image(pending.idx, image),
            Capture::Stereo(left, right) => {
                self.frontend.prepare_stereo_pair(pending.idx, left, right)
            }
            Capture::Rig(images) => self.frontend.prepare_rig(pending.idx, images),
        }
        .map_err(ProcessError::Frontend)
    }

    /// The frame `source` describes, extracting it if it was deferred; the
    /// extraction time is kept out of `tracking_duration`.
    fn resolve(&mut self, source: FrameSource<'_>) -> Result<Box<Frame>, ProcessError> {
        match source {
            FrameSource::Ready(frame) => Ok(frame),
            FrameSource::Deferred(pending) => {
                let mut frame = self.extract(pending)?;
                self.deferred_extraction += self.frontend.observation().duration;
                self.report_stereo(&frame);
                frame.ensure_undistorted(&self.camera);
                Ok(Box::new(frame))
            }
        }
    }

    /// The ORB frame of `tracked`, extracting it when optical flow tracked
    /// the frame without one.
    fn orb_frame(&mut self, tracked: TrackedFrame<'_>) -> Result<Frame, ProcessError> {
        match tracked {
            TrackedFrame::Orb(frame) => Ok(*frame),
            TrackedFrame::Flow(source) => self.resolve(source).map(|frame| *frame),
        }
    }

    /// Takes a prepared or deferred frame as the next one and tracks it. On
    /// an error, from a deferred extraction, the system is left as it was.
    fn track_frame(
        &mut self,
        source: FrameSource<'_>,
        image: &Image<u8, 1>,
        images: &[&Image<u8, 1>],
        timestamp_sec: f64,
    ) -> Result<TrackingResult, ProcessError> {
        let next_frame_idx = self.next_frame_idx;
        let latest_image_timestamp_sec = self.inertial.latest_image_timestamp_sec;
        let tracking_timings = self.tracking_timings;
        self.next_frame_idx += 1;
        self.inertial.latest_image_timestamp_sec = Some(timestamp_sec);
        if let FrameSource::Ready(frame) = &source {
            self.report_stereo(frame);
        }
        // Taken out for the call so tracking can borrow it alongside `self`.
        let previous_image = self.previous_image.take();
        self.tracking_timings = TrackingTimings::default();
        self.deferred_extraction = Duration::ZERO;
        let tracking_start = Instant::now();
        self.prepare_flow(images, next_frame_idx as u64);
        match self.process_prepared(
            source,
            previous_image.as_ref(),
            image,
            images,
            timestamp_sec,
        ) {
            Ok(result) => {
                self.tracking_duration = tracking_start
                    .elapsed()
                    .saturating_sub(self.deferred_extraction);
                // `WindowedU8` keeps its own level-0 copies.
                if self.rig_flow.is_none() {
                    self.retain_previous_image(image, previous_image);
                }
                if self.tracker.flows_rig() {
                    self.retain_secondary_images(images.get(1..).unwrap_or_default());
                }
                Ok(result)
            }
            Err(error) => {
                self.next_frame_idx = next_frame_idx;
                self.inertial.latest_image_timestamp_sec = latest_image_timestamp_sec;
                self.tracking_timings = tracking_timings;
                self.previous_image = previous_image;
                Err(error)
            }
        }
    }

    /// Prepares the `WindowedU8` flow frames of capture `frame_idx`. The
    /// tracks' pixels refer to the previous capture: when prepare fails, or
    /// the previous frame is not capture `frame_idx - 1` (a failed prepare
    /// left a fresh pair of this capture, or a failed extraction this
    /// capture), they are cleared, so no template is cut at stale pixels in
    /// another image.
    fn prepare_flow(&mut self, images: &[&Image<u8, 1>], frame_idx: u64) {
        let Some(rig_flow) = self.rig_flow.as_mut() else {
            return;
        };
        match rig_flow.prepare(images, frame_idx, &mut self.tracking_timings) {
            Ok(()) => {
                let follows = rig_flow
                    .previous_frame(0)
                    .is_some_and(|previous| previous.frame_id() + 1 == frame_idx);
                if !follows {
                    self.tracker.reset_tracks();
                }
            }
            Err(error) => {
                self.tracker.reset_tracks();
                if self.debug {
                    self.dbg(format!(
                        "[flow] frame={frame_idx} frames not prepared: {error}"
                    ));
                }
            }
        }
    }

    /// Time the latest successfully processed frame spent in tracking and
    /// mapping, excluding feature extraction (see [`Self::frontend_observation`]).
    pub fn tracking_duration(&self) -> Duration {
        self.tracking_duration
    }

    /// Per-stage times of the latest successfully processed frame's
    /// tracking; zero for stages it did not run (all of them in bootstrap).
    pub fn tracking_timings(&self) -> &TrackingTimings {
        &self.tracking_timings
    }

    /// Tallies of optical-flow-first tracking (`klt_first`) so far.
    pub fn flow_counters(&self) -> FlowCounters {
        let (klt_errors, snap_errors) = self.tracker.flow_errors();
        FlowCounters {
            klt_errors,
            snap_errors,
            ..self.flow_counters
        }
    }

    /// The frontend output of the latest successfully processed frame.
    pub fn frontend_observation(&self) -> &FrontendObservation {
        self.frontend.observation()
    }

    /// Copies `image` into the previous-image buffer, reusing its allocation.
    fn retain_previous_image(&mut self, image: &Image<u8, 1>, buffer: Option<Image<u8, 1>>) {
        self.previous_image = match buffer {
            Some(mut buffer) if buffer.size() == image.size() => {
                buffer.as_slice_mut().copy_from_slice(image.as_slice());
                Some(buffer)
            }
            _ => Some(image.clone()),
        };
    }

    /// Copies rig cameras 1 and up into their previous-image buffers.
    fn retain_secondary_images(&mut self, images: &[&Image<u8, 1>]) {
        self.previous_secondary.truncate(images.len());
        for (camera, image) in images.iter().enumerate() {
            match self.previous_secondary.get_mut(camera) {
                Some(buffer) if buffer.size() == image.size() => {
                    buffer.as_slice_mut().copy_from_slice(image.as_slice());
                }
                Some(buffer) => *buffer = (*image).clone(),
                None => self.previous_secondary.push((*image).clone()),
            }
        }
    }

    fn report_stereo(&mut self, frame: &Frame) {
        if !self.debug {
            return;
        }
        if frame.is_rig() {
            let mut depths: Vec<f64> = frame
                .rig
                .stereo_points
                .iter()
                .flatten()
                .map(|p| p.z)
                .collect();
            depths.sort_by(f64::total_cmp);
            let median = depths.get(depths.len() / 2).copied().unwrap_or(0.0);
            let per_camera: Vec<usize> = frame.rig.camera_ranges.iter().map(|r| r.len()).collect();
            self.dbg(format!(
                "[overlap] frame={} features={per_camera:?} triangulated={} median_depth={median:.3}m",
                frame.idx,
                depths.len()
            ));
            return;
        }
        let Some(matched) = self.frontend.observation().stereo_matched else {
            return;
        };
        let mut depths: Vec<f32> = frame.depth.iter().copied().filter(|&d| d > 0.0).collect();
        let median = if depths.is_empty() {
            0.0
        } else {
            depths.sort_by(|a, b| a.total_cmp(b));
            depths[depths.len() / 2]
        };
        self.dbg(format!(
            "[stereo] frame={} matched={matched} median_depth={median:.3}m",
            frame.idx
        ));
    }

    /// Tracks one frame of prepared or deferred features.
    fn process_prepared(
        &mut self,
        source: FrameSource<'_>,
        previous_image: Option<&Image<u8, 1>>,
        current_image: &Image<u8, 1>,
        images: &[&Image<u8, 1>],
        timestamp_sec: f64,
    ) -> Result<TrackingResult, ProcessError> {
        // Local-BA snapshots, merges, and their correction messages can only
        // cross this boundary between complete tracking frames.
        let publication_gate = self.map_publication_gate.clone();
        let _publication = publication_gate
            .as_ref()
            .map(|gate| gate.lock().unwrap_or_else(|poisoned| poisoned.into_inner()));
        self.apply_local_mapping_results();
        self.adopt_landmarks();
        // Fill the per-frame undistortion cache once; tracking, BA gathering,
        // growth, and fuse all read from it. A deferred frame fills it when
        // it is extracted.
        let source = match source {
            FrameSource::Ready(mut frame) => {
                frame.ensure_undistorted(&self.camera);
                FrameSource::Ready(frame)
            }
            deferred => deferred,
        };

        match self.state.mode {
            SystemMode::Bootstrap => {
                let frame = self.resolve(source)?;
                Ok(self.bootstrap_step(*frame, timestamp_sec))
            }
            SystemMode::ImuInit => self.inertial_init_step(
                source,
                previous_image,
                current_image,
                images,
                timestamp_sec,
            ),
            SystemMode::Tracking => {
                self.tracking_step(source, previous_image, current_image, images, timestamp_sec)
            }
        }
    }

    /// Runs `f` against the live map points, holding the map lock only for the
    /// duration of the call. Avoids cloning the whole point list (descriptors
    /// included) for read-only consumers such as viz logging and summaries.
    pub fn with_map_points<R>(&self, f: impl FnOnce(&[MapPoint]) -> R) -> R {
        f(self.map.lock().unwrap().map_points())
    }

    /// Returns the index of the current reference keyframe, if tracking has one.
    pub fn current_keyframe_idx(&self) -> Option<usize> {
        self.state.current_keyframe_idx.and_then(|ki| {
            self.map
                .lock()
                .unwrap()
                .get_keyframe(ki)
                .map(|kf| kf.frame.idx)
        })
    }

    /// Returns the number of active (non-culled) map points.
    pub fn num_active_map_points(&self) -> usize {
        self.map.lock().unwrap().num_active_map_points()
    }

    /// Drain any debug messages accumulated since the last call.
    pub fn drain_debug_messages(&mut self) -> Vec<String> {
        std::mem::take(&mut self.debug_messages)
    }

    /// Toggles per-frame diagnostics: bootstrap skip and reject reasons,
    /// tracking reject reasons, keyframe growth and fuse counters. Off after
    /// [`SlamSystem::build`].
    pub fn set_debug(&mut self, on: bool) {
        self.debug = on;
        if !on {
            self.debug_messages.clear();
        }
    }

    fn apply_local_mapping_results(&mut self) {
        let Some(reference_idx) = self.state.current_keyframe_idx else {
            // Still drain results so a completed worker cannot build a result backlog.
            let _ = self.local_mapping.drain_results();
            return;
        };

        let results = self.local_mapping.drain_results();
        if !results.is_empty() {
            self.refresh_track_positions();
        }
        for result in results {
            let Some(correction) = result
                .keyframe_corrections
                .iter()
                .find(|correction| correction.kf_idx == reference_idx)
            else {
                continue;
            };

            self.state.pose_world_to_cam = apply_reference_pose_correction(
                self.state.pose_world_to_cam,
                correction.pose_before,
                correction.pose_after,
            );
            self.state.velocity_world = correction.velocity_world;
            self.inertial.bias = correction.imu_bias;
        }
    }

    /// Rewrites the tracks' stored map-point positions after a map change,
    /// under one map lock; only `WindowedU8` shards read them.
    fn refresh_track_positions(&mut self) {
        if self.rig_flow.is_none() {
            return;
        }
        let map = self
            .map
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.tracker.refresh_positions(&map);
    }

    /// Spawns tracks on the landmarks the latest keyframe created, on the
    /// frame after its insertion (`adopt_landmarks`). Tracks refer to the
    /// previous frame here: a landmark of that very frame starts at its
    /// keypoint; an older keyframe's is tracked into it from the kept
    /// keyframe frames, starting at its projection.
    fn adopt_landmarks(&mut self) {
        let (true, Some(kf_idx)) = (self.flow.adopt_landmarks, self.state.last_keyframe_idx) else {
            return;
        };
        if self.adopted_keyframe == Some(kf_idx) {
            return;
        }
        self.adopted_keyframe = Some(kf_idx);
        let previous_idx = self.next_frame_idx.checked_sub(2);
        let pose = self.state.pose_world_to_cam;
        let cameras = self.rig_flow.as_ref().map_or(1, RigFlow::cameras);
        let landmarks: Vec<([f32; 2], usize, usize, Vec3F64)> = {
            let map = self
                .map
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let Some(kf) = map.get_keyframe(kf_idx) else {
                return;
            };
            kf.map_point_by_desc_idx
                .iter()
                .enumerate()
                .filter_map(|(desc_idx, mp_idx)| {
                    let mp_idx = (*mp_idx)?;
                    let mp = map
                        .map_points()
                        .get(mp_idx)
                        .filter(|mp| !mp.culled && mp.keyframe_idx == kf_idx)?;
                    let camera = kf.frame.camera_of(desc_idx);
                    let pixel = *kf.frame.features.keypoints_xy.get(desc_idx)?;
                    (camera < cameras).then_some((pixel, camera, mp_idx, mp.position))
                })
                .collect()
        };
        for (pixel, camera, mp_idx, position) in landmarks {
            let pixel = if previous_idx == Some(kf_idx) {
                Some(pixel)
            } else {
                self.carry_landmark(kf_idx, camera, pixel, &pose, &position)
            };
            if let Some(pixel) = pixel {
                self.tracker.adopt(pixel, camera, mp_idx, position);
            }
        }
        self.tracker.finish_adoption();
    }

    /// A kept keyframe landmark's pixel in the previous frame (`WindowedU8`
    /// only), starting at its projection at that frame's pose.
    fn carry_landmark(
        &mut self,
        kf_idx: usize,
        camera: usize,
        pixel: [f32; 2],
        pose_world_to_primary: &Pose3d,
        position: &Vec3F64,
    ) -> Option<[f32; 2]> {
        let (model, pose) = match camera {
            0 => (&self.camera, *pose_world_to_primary),
            c => {
                let view = self.frontend.rig_geometry()?.view(c)?;
                (view.pinhole, view.pose(pose_world_to_primary))
            }
        };
        let start =
            crate::tracking::tracker::project_raw(model, &pose.transform_point(position), pixel)
                .unwrap_or(pixel);
        self.rig_flow
            .as_mut()?
            .adopt(kf_idx as u64, camera, pixel, start)
    }

    fn dbg(&mut self, msg: String) {
        if self.debug {
            self.debug_messages.push(msg);
        }
    }

    fn bootstrap_step(&mut self, curr_frame: Frame, timestamp_sec: f64) -> TrackingResult {
        // Stereo frames carry metric per-keypoint depth, so we can build a
        // metric map from a single keyframe (ORB-SLAM3's StereoInitialization)
        // instead of waiting for two-view parallax.
        if curr_frame.is_metric() {
            return self.bootstrap_stereo(curr_frame, timestamp_sec);
        }
        self.bootstrap_mono(curr_frame, timestamp_sec)
    }

    /// Single-frame metric initialization from stereo depth.
    fn bootstrap_stereo(&mut self, mut curr_frame: Frame, timestamp_sec: f64) -> TrackingResult {
        // The map will start at this frame or a later one, so older samples
        // can never enter an edge; a failed attempt (too few stereo points,
        // as in the dark) must not keep them, or the buffer grows for as
        // long as bootstrap keeps failing.
        self.prune_imu_before(timestamp_sec);
        // Build the new map in the current odometry frame (identity at start,
        // or the recovery pose after a tracking loss).
        curr_frame.pose_world_to_cam = self.state.pose_world_to_cam;
        let curr_idx = curr_frame.idx;

        let insertion =
            match bootstrap::stereo_initial_map(Keyframe::from_frame(curr_frame), &self.camera) {
                Ok(insertion) => insertion,
                Err(TooFewStereoPoints { found }) => {
                    self.dbg(format!(
                        "[bootstrap_stereo] frame={curr_idx} skip: only {found} stereo points \
                     (need >= {MIN_STEREO_POINTS})"
                    ));
                    return self.frame_result(TrackingStatus::Skipped);
                }
            };

        // The keyframe and its seeds are published together; tracker state is
        // adopted below only once that succeeded.
        let published = self.map.lock().unwrap().apply_insertion(insertion);
        let added = match published {
            Ok(result) => result.landmark_ids.len(),
            Err(error) => {
                self.dbg(format!(
                    "[bootstrap_stereo] frame={curr_idx} publication rejected: {error}"
                ));
                return self.frame_result(TrackingStatus::Skipped);
            }
        };

        self.dbg(format!(
            "[bootstrap_stereo] frame={curr_idx} metric map created with {added} points",
        ));

        self.state.velocity = None;
        // The map is already metric (stereo baseline), but gravity, velocities,
        // and the gyro bias still need the inertial init before IMU prediction
        // can run; the solve there keeps scale fixed at 1.
        self.finish_bootstrap(curr_idx, timestamp_sec);
        self.prune_imu_before(timestamp_sec);

        self.frame_result(TrackingStatus::KeyframeAccepted)
    }

    fn bootstrap_mono(&mut self, mut curr_frame: Frame, timestamp_sec: f64) -> TrackingResult {
        // Stamp frames with current odometry pose so bootstrap builds
        // the new map in the existing coordinate frame.
        curr_frame.pose_world_to_cam = self.state.pose_world_to_cam;

        let decision = evaluate_bootstrap(
            self.state.bootstrap_frame.as_ref(),
            &curr_frame,
            &self.camera,
            &self.two_view_init_config,
        );
        let two_view_estimate = match decision {
            // A stale reference is dropped with the frame: neither is viable.
            BootstrapDecision::Unusable { keypoints } => {
                self.dbg(format!(
                    "[bootstrap] frame={} skip: too few keypoints ({}, need > {})",
                    curr_frame.idx, keypoints, MIN_KEYPOINTS_FOR_BOOTSTRAP,
                ));
                self.state.bootstrap_frame = None;
                // No reference is held: the next one is a later frame.
                self.prune_imu_before(timestamp_sec);
                return self.frame_result(TrackingStatus::Skipped);
            }
            BootstrapDecision::StoreAsReference => {
                self.dbg(format!(
                    "[bootstrap] frame={} stored as reference (awaiting second frame)",
                    curr_frame.idx,
                ));
                self.store_bootstrap_reference(curr_frame, timestamp_sec);
                return self.frame_result(TrackingStatus::Skipped);
            }
            // The reference is kept: only the second frame was unsuitable.
            BootstrapDecision::Rejected {
                reference_idx,
                reason,
            } => {
                self.dbg(format!(
                    "[bootstrap] frame={} (ref={}) reject: {:?}",
                    curr_frame.idx, reference_idx, reason,
                ));
                // A reference that keeps failing to match has lost the scene
                // (ORB-SLAM3 replaces it on the first such frame; waiting a
                // little keeps the better-conditioned early pairs).
                if self
                    .state
                    .bootstrap_reference_lost(&reason, MAX_LOW_MATCH_REJECTIONS)
                {
                    self.dbg(format!(
                        "[bootstrap] frame={} replaces reference {reference_idx}",
                        curr_frame.idx
                    ));
                    self.store_bootstrap_reference(curr_frame, timestamp_sec);
                }
                return self.frame_result(TrackingStatus::Skipped);
            }
            BootstrapDecision::Initialized(estimate) => *estimate,
        };
        let prev_bootstrap_frame = self
            .state
            .bootstrap_frame
            .take()
            .expect("an accepted bootstrap consumes the reference it matched against");

        self.dbg(format!(
            "[bootstrap] frame={} accept: model={} triangulated={} inliers={}",
            curr_frame.idx,
            two_view_estimate.model_kind,
            two_view_estimate.points3d.len(),
            two_view_estimate.estimate.inliers,
        ));

        let estimated_pose = two_view_estimate.estimate.pose;
        let prev_pose_world_to_cam = curr_frame.pose_world_to_cam;
        self.state.pose_world_to_cam = estimated_pose;
        curr_frame.pose_world_to_cam = estimated_pose;

        // Promote to Keyframes
        let prev_idx = prev_bootstrap_frame.idx;
        let reference_kf = Keyframe::from_frame(prev_bootstrap_frame);
        let current_kf = Keyframe::from_frame(curr_frame);
        let curr_idx = current_kf.frame.idx;

        // A rejected publication writes nothing, so there is no map to
        // evaluate; treat it like a failed health gate.
        let insertion =
            bootstrap::two_view_initial_map(reference_kf, current_kf, &two_view_estimate);
        if let Err(error) = self.publish_initial_map(insertion, [prev_idx, curr_idx]) {
            self.dbg(format!(
                "[bootstrap] frame={curr_idx} publication rejected: {error}"
            ));
            self.state.reset();
            return self.frame_result(TrackingStatus::Skipped);
        }

        // The gate runs after initial BA, which can collapse a badly
        // conditioned pair.
        let health = crate::initialization::initial_map_health(&self.map.lock().unwrap());
        if health.is_degenerate() {
            self.dbg(format!(
                "[init_gate] reject: valid_in_both={} median_depth={:.3} (need >= {} and > 0)",
                health.valid_in_both, health.median_depth_older_kf, MIN_VALID_POINTS,
            ));
            self.map.lock().unwrap().clear_active();
            self.state.reset();
            return self.frame_result(TrackingStatus::Skipped);
        }

        // Initial BA in publish_initial_map may have refined KF1's pose; sync state
        // and recompute velocity from the post-BA pose.
        if let Some(kf) = self.map.lock().unwrap().get_keyframe(curr_idx) {
            self.state.pose_world_to_cam = kf.frame.pose_world_to_cam;
        }

        if let Some(prev_ts) = self.inertial.bootstrap_timestamp_sec {
            if let Some(edge) = self.inertial.keyframe_edge(
                self.rig.imu_noise(),
                prev_idx,
                curr_idx,
                prev_ts,
                timestamp_sec,
            ) {
                // Through the validated path: endpoints must exist, the edge
                // must be new, and the interval finite and ordered.
                let published = self.map.lock().unwrap().apply_insertion(MapInsertion {
                    imu_factors: vec![edge],
                    ..Default::default()
                });
                if let Err(error) = published {
                    self.dbg(format!(
                        "[bootstrap] frame={curr_idx} imu edge rejected: {error}"
                    ));
                }
            }
            self.prune_imu_before(timestamp_sec);
        }

        self.state.velocity = Some(Pose3d::between(
            &prev_pose_world_to_cam,
            &self.state.pose_world_to_cam,
        ));

        self.finish_bootstrap(curr_idx, timestamp_sec);

        self.frame_result(TrackingStatus::KeyframeAccepted)
    }

    fn store_bootstrap_reference(&mut self, frame: Frame, timestamp_sec: f64) {
        self.state.bootstrap_frame = Some(frame);
        self.state.bootstrap_low_match_rejections = 0;
        self.inertial.bootstrap_timestamp_sec = Some(timestamp_sec);
        // Samples before the reference frame can never enter an edge.
        self.prune_imu_before(timestamp_sec);
    }

    /// Publishes the two-view map, refines it with initial BA, and offers both
    /// keyframes to place recognition so a later revisit of the start can
    /// match them.
    fn publish_initial_map(
        &mut self,
        insertion: MapInsertion,
        keyframes: [usize; 2],
    ) -> Result<(), MapMutationError> {
        self.map.lock().unwrap().apply_insertion(insertion)?;
        let refined =
            crate::mapping::ba::run_initial_ba(&mut self.map.lock().unwrap(), &self.camera);
        if let Err(error) = refined {
            self.dbg(format!("[init_ba] {error}; keeping the unrefined map"));
        }
        for kf_idx in keyframes {
            self.register_place_recognition(kf_idx);
        }
        Ok(())
    }

    /// Leaves bootstrap with `kf_idx` as the reference keyframe. Inertial
    /// initialization follows when the camera-to-body extrinsic is known, since
    /// it relates IMU deltas to camera poses; otherwise tracking is visual-only.
    fn finish_bootstrap(&mut self, kf_idx: usize, timestamp_sec: f64) {
        self.state.current_keyframe_idx = Some(kf_idx);
        self.state.last_keyframe_idx = Some(kf_idx);
        self.state.mode = if self.rig.camera_to_body().is_some() {
            self.inertial.schedule.start(kf_idx, timestamp_sec);
            SystemMode::ImuInit
        } else {
            SystemMode::Tracking
        };
        self.inertial.last_keyframe_timestamp_sec = Some(timestamp_sec);
    }

    fn frame_result(&self, status: TrackingStatus) -> TrackingResult {
        TrackingResult {
            pose_world_to_cam: self.state.pose_world_to_cam,
            status,
        }
    }

    /// Drops buffered IMU samples strictly older than `t` (typically the last
    /// keyframe timestamp: the next edge and all per-frame windows start there).
    fn prune_imu_before(&mut self, t: f64) {
        self.inertial.prune_before(t);
    }

    /// Tracks the frame, then attempts VIBA0 on accepted keyframes once the
    /// window is ready and a retry is due.
    fn inertial_init_step(
        &mut self,
        source: FrameSource<'_>,
        previous_image: Option<&Image<u8, 1>>,
        current_image: &Image<u8, 1>,
        images: &[&Image<u8, 1>],
        timestamp_sec: f64,
    ) -> Result<TrackingResult, ProcessError> {
        let result =
            self.tracking_step(source, previous_image, current_image, images, timestamp_sec)?;

        if result.status == TrackingStatus::KeyframeAccepted
            && let Some(start_idx) = self.inertial.schedule.start_kf_idx()
        {
            // Snapshot the fields needed after releasing the map lock.
            let kfs: Vec<usize> = self
                .map
                .lock()
                .unwrap()
                .keyframes()
                .iter()
                .filter(|kf| kf.frame.idx >= start_idx)
                .map(|kf| kf.frame.idx)
                .collect();
            let imu_time: f64 = self
                .map
                .lock()
                .unwrap()
                .imu_factors()
                .iter()
                .filter(|f| f.curr_kf_idx >= start_idx)
                .map(|f| f.preintegrated.dt)
                .sum();
            let gate_msg = format_imu_init_gate(
                start_idx,
                kfs.first().copied(),
                kfs.last().copied(),
                kfs.len(),
                self.inertial.initializer.config.min_keyframes,
                imu_time,
                self.inertial.initializer.config.min_time_sec,
            );
            self.dbg(gate_msg);
        }

        let due_for_retry = self.inertial.schedule.retry_due(timestamp_sec);
        let imu_init_ready = self.inertial.initializer.ready(
            &self.map.lock().unwrap(),
            self.inertial.schedule.start_kf_idx(),
        );

        if result.status == TrackingStatus::KeyframeAccepted && due_for_retry && imu_init_ready {
            let Some(start_idx) = self.inertial.schedule.start_kf_idx() else {
                return Ok(result);
            };
            self.inertial.schedule.record_attempt(timestamp_sec);
            let is_mono = !self
                .map
                .lock()
                .unwrap()
                .keyframes()
                .iter()
                .find(|kf| kf.frame.idx >= start_idx)
                .map(|kf| kf.frame.is_metric())
                .unwrap_or(false);
            let prior_a0 = viba0_accel_bias_prior(is_mono);
            // Drop the solve's map lock before applying its result with a new lock.
            let init_result = self.inertial.initializer.try_initialize(
                &self.map.lock().unwrap(),
                self.rig.camera_to_body(),
                self.inertial.bias,
                start_idx,
                1e2,
                prior_a0,
                false,
            );
            match init_result {
                Some(init) => {
                    let applied = self.apply_inertial_initialization(init, start_idx);
                    match applied {
                        Ok(applied) => self.resume_tracking_after_viba0(timestamp_sec, applied),
                        // Staying in ImuInit lets the retry throttle re-solve later.
                        Err(error) => {
                            self.dbg(format!("[imu_init] VIBA0 alignment refused: {error}"));
                        }
                    }
                }
                None => {
                    self.dbg("[imu_init] VIBA0 rejected: solve failed or invalid scale".into());
                }
            }
        }

        Ok(result)
    }

    /// Applies an inertial initialization to the map and resumes tracking from
    /// the aligned state; a refused alignment changes nothing.
    fn apply_inertial_initialization(
        &mut self,
        init: ImuInitResult,
        start_kf_idx: usize,
    ) -> Result<AppliedInitialization, InertialAlignmentError> {
        let applied = self.inertial.apply_initialization(
            &mut self.map.lock().unwrap(),
            init,
            start_kf_idx,
        )?;
        self.state.adopt_inertial_initialization(applied.aligned);
        self.refresh_track_positions();
        Ok(applied)
    }

    fn resume_tracking_after_viba0(&mut self, timestamp_sec: f64, applied: AppliedInitialization) {
        // Mirrors ORB-SLAM3: IMU is marked initialized (and tracking resumes)
        // immediately after VIBA0 succeeds — VIBA1/VIBA2 refine
        // bg/ba/scale/gravity further in the background (see
        // try_insert_keyframe), they don't gate resuming tracking.
        self.state.mode = SystemMode::Tracking;
        self.state.imu_init_timestamp_sec = Some(timestamp_sec);
        if !self.local_mapping.submit(KeyframeJob {
            imu_initialized: true,
            imu_t_bc: self.rig.camera_to_body(),
            gravity_world: self.inertial.gravity_world,
        }) {
            self.dbg("[local_mapping] worker is unavailable".into());
        }
        self.apply_local_mapping_results();
        let AppliedInitialization {
            scale,
            gravity_world: gravity,
            gyro_bias: bg,
            ..
        } = applied;
        self.dbg(format!(
            "[imu_init] VIBA0 accepted: scale={scale:.4} gravity=({:.3},{:.3},{:.3}) \
             gyro_bias=({:.4},{:.4},{:.4})",
            gravity.x, gravity.y, gravity.z, bg.x, bg.y, bg.z
        ));
    }

    fn tracking_step(
        &mut self,
        source: FrameSource<'_>,
        previous_image: Option<&Image<u8, 1>>,
        current_image: &Image<u8, 1>,
        images: &[&Image<u8, 1>],
        timestamp_sec: f64,
    ) -> Result<TrackingResult, ProcessError> {
        let frame_idx = source.idx();
        let pose_before = self.state.pose_world_to_cam;
        let prev_timestamp = self.state.last_frame_timestamp_sec;
        // Restored when a deferred extraction fails, with the rest of the
        // frame's state (see `track_frame`).
        let (velocity_world_before, bias_before) = (self.state.velocity_world, self.inertial.bias);

        // Local BA updates keyframe state asynchronously, so refresh cached IMU state.
        if let Some(kf_idx) = self.state.current_keyframe_idx
            && let Some(kf) = self.map.lock().unwrap().get_keyframe(kf_idx)
        {
            self.state.velocity_world = kf.velocity_world;
            self.inertial.bias = kf.imu_bias;
        }

        // Preintegration is prepared here: the system owns bias and the sample
        // buffer. The motion model chooses between it and the visual model.
        let preintegrated = (self.state.imu_initialized && prev_timestamp > 0.0).then(|| {
            self.inertial
                .preintegrate_window(self.rig.imu_noise(), prev_timestamp, timestamp_sec)
                .0
        });
        let (candidate_pose, predicted_velocity) = predict_pose(
            pose_before,
            self.state.velocity,
            self.state.velocity_world,
            preintegrated
                .as_ref()
                .map(|preintegrated| InertialPrediction {
                    preintegrated,
                    camera_to_body: self.rig.camera_to_body(),
                    gravity_world: self.inertial.gravity_world,
                }),
        );
        if let Some(velocity_world) = predicted_velocity {
            self.state.velocity_world = velocity_world; // propagate for next frame
        }
        // A ray seen in the previous camera, in the current one: R_cur_prev
        // over this frame only.
        self.gyro_rotation = (self.flow.gyro_guess == GyroGuess::RotatedRay
            && predicted_velocity.is_some())
        .then(|| candidate_pose.rotation * pose_before.rotation.transpose());

        let currently_lost_for = self
            .state
            .lost_since_sec
            .map_or(0.0, |t0| timestamp_sec - t0);
        let trusted_prediction = self.imu_confident(timestamp_sec);
        let search = self.predicted_flow_search(trusted_prediction, predicted_velocity.is_some());
        let mut survivors = self.flow(previous_image, images, &candidate_pose, search);
        let estimation = Estimation {
            candidate_pose,
            pose_before,
            lost_for_sec: currently_lost_for,
            trusted_prediction,
        };
        let deferred = matches!(source, FrameSource::Deferred(_));
        let flow_only = if deferred {
            self.track_flow_only(
                frame_idx,
                current_image,
                survivors.as_deref(),
                &estimation,
                timestamp_sec,
            )
        } else {
            FlowOnly::Unavailable
        };
        let flow_due = matches!(flow_only, FlowOnly::KeyframeDue(_));
        let (tracked, result) = match flow_only {
            FlowOnly::Tracked(estimate) => {
                self.tracker.recycle_survivors(survivors.take());
                (TrackedFrame::Flow(source), Ok(estimate))
            }
            outcome => {
                let frame = match self.resolve(source) {
                    Ok(frame) => frame,
                    Err(error) => {
                        self.state.velocity_world = velocity_world_before;
                        self.inertial.bias = bias_before;
                        return Err(error);
                    }
                };
                if deferred {
                    self.count_flow_decision(&outcome);
                }
                let candidate_pose = match outcome {
                    FlowOnly::KeyframeDue(pose) => pose,
                    _ => candidate_pose,
                };
                let result = {
                    let map = self.map.lock().unwrap();
                    self.tracker.estimate(
                        FrameInput {
                            frame: &frame,
                            candidate_pose,
                            pose_before,
                            current_keyframe_idx: self.state.current_keyframe_idx,
                            lost_for_sec: currently_lost_for,
                            trusted_prediction,
                        },
                        survivors.take(),
                        &map,
                        &self.camera,
                    )
                };
                (TrackedFrame::Orb(frame), result)
            }
        };
        self.tracker.spawn_refills();
        self.flow_counters.flow_due_orb_rejected += u64::from(flow_due && result.is_err());

        let (mut status, matches, tracked_inliers, reject_reason) = match result {
            Ok(estimate) => {
                self.state.pose_world_to_cam = estimate.pose;

                // When IMU-initialized, velocity_world was already updated by IMU
                // preintegration (predict_pose_imu → pred_vel) before PnP ran; don't
                // overwrite it with a visual finite-difference, since at 30 fps the
                // inter-frame displacement is noise-dominated during low-translation
                // segments, which would collapse velocity to zero and permanently
                // freeze the IMU pose prediction.
                if !self.state.imu_initialized {
                    self.state.velocity = Some(Pose3d::between(&pose_before, &estimate.pose));
                }

                (
                    TrackingStatus::Tracked,
                    estimate.matches,
                    estimate.inliers,
                    None,
                )
            }
            Err(reason) => {
                // Carry the predicted pose forward instead of freezing at
                // pose_before. state.velocity_world was already advanced by
                // predict_pose_imu above regardless of visual outcome, so
                // anchoring the next frame's prediction on a stale position
                // would desync position/rotation from velocity: every
                // subsequent frame's candidate pose would drift further from
                // reality, making the projection search miss again and
                // compounding a single bad frame into a full tracking loss.
                self.state.pose_world_to_cam = candidate_pose;
                (TrackingStatus::Skipped, Vec::new(), 0, Some(reason))
            }
        };
        if self.debug {
            let msg = match reject_reason {
                Some(reason) => format!("[track] frame={} reject: {:?}", frame_idx, reason),
                None => format!(
                    "[track] frame={} ok: matches={} inliers={}",
                    frame_idx,
                    matches.len(),
                    tracked_inliers,
                ),
            };
            self.debug_messages.push(msg);
        }

        // A flow-tracked frame counts no visibility, as ORB-SLAM3 counts none
        // on a frame without a projection search, and its keyframe verdict was
        // given in `track_flow_only`.
        if status == TrackingStatus::Tracked
            && let TrackedFrame::Orb(frame) = &tracked
        {
            self.count_visibility(frame, &matches, &candidate_pose);
            // A flow frame found due is inserted on its first verdict.
            let pose = self.state.pose_world_to_cam;
            let inserted = if flow_due {
                self.insert_keyframe(frame, timestamp_sec, &matches)
            } else {
                self.try_insert_keyframe(frame, &pose, timestamp_sec, tracked_inliers, &matches)
            };
            if inserted {
                status = TrackingStatus::KeyframeAccepted;
            }
        }

        if status == TrackingStatus::Skipped {
            // Only a tracked flow frame is left unextracted, so a rejected
            // frame has its ORB frame already; this extracts only if a later
            // check rejects a flow-tracked pose.
            let frame = self.orb_frame(tracked)?;
            if let Some(result) = self.recover_or_coast(&mut status, frame, timestamp_sec) {
                return Ok(result);
            }
        }
        if status != TrackingStatus::Skipped {
            self.state.lost_since_sec = None;
        }
        self.state.last_frame_timestamp_sec = timestamp_sec;
        // Samples older than the last keyframe can't enter any future window
        // (the next edge and all per-frame predictions start at or after it).
        if let Some(kf_ts) = self.inertial.last_keyframe_timestamp_sec {
            self.prune_imu_before(kf_ts.min(timestamp_sec));
        }
        Ok(self.frame_result(status))
    }

    /// A rejected frame: relocalization, which sets `status` to tracked, the
    /// coast on the IMU with its keyframe while lost, or giving up, which
    /// returns the bootstrap result.
    fn recover_or_coast(
        &mut self,
        status: &mut TrackingStatus,
        frame: Frame,
        timestamp_sec: f64,
    ) -> Option<TrackingResult> {
        // With a settled IMU the system coasts instead (below): a relocalized
        // pose taken from an older part of a drifted map would jump away from
        // the inertial state, which ORB-SLAM3 also avoids in inertial mode.
        if !self.imu_confident(timestamp_sec) && self.relocalize(&frame) {
            *status = TrackingStatus::Tracked;
            return None;
        }
        let policy = &self.tracking_loss_recovery;

        let lost_since = *self.state.lost_since_sec.get_or_insert(timestamp_sec);
        let recently_lost_for = timestamp_sec - lost_since;

        let imu_confident = self.imu_confident(timestamp_sec);
        let grace_period_sec = policy.grace_period_sec(imu_confident);
        let map_established =
            self.map.lock().unwrap().keyframes().len() > policy.min_keyframes_for_grace;

        if !map_established || recently_lost_for >= grace_period_sec {
            self.dbg(format!(
                "[lost] frame={} giving up after {:.2}s (map_established={}): resetting",
                frame.idx, recently_lost_for, map_established,
            ));
            self.tracker.reset_tracks();
            self.state.reset();
            return Some(self.bootstrap_step(frame, timestamp_sec));
        }
        if imu_confident
            && self.keyframe_due_while_lost(timestamp_sec)
            && self.insert_keyframe(&frame, timestamp_sec, &[])
        {
            self.dbg(format!(
                "[lost] frame={} keyframe inserted at the inertial prediction",
                frame.idx
            ));
        }
        None
    }

    /// Visibility bookkeeping over the local map only (mirrors ORB-SLAM3,
    /// which counts mnVisible on local-map points): full-map scans here would
    /// grow with trajectory length.
    fn count_visibility(&self, frame: &Frame, matches: &[(usize, usize)], candidate_pose: &Pose3d) {
        let mut map_guard = self.map.lock().unwrap();
        let current_kf = self
            .state
            .current_keyframe_idx
            .and_then(|ki| map_guard.get_keyframe(ki));
        let local_indices = select_local_map_points(
            &map_guard,
            matches,
            current_kf,
            LocalMapSelectionConfig::default(),
        );
        let visible = crate::tracking::local_map::landmarks_in_frame_frustum(
            &map_guard,
            &local_indices,
            &self.camera,
            candidate_pose,
            frame,
        );
        map_guard.update_observation_counts(&visible, matches);
    }

    /// Tracks a deferred frame from its flow survivors alone when they carry
    /// it and no keyframe is due; ORB then never runs for this frame.
    fn track_flow_only(
        &mut self,
        idx: usize,
        current_image: &Image<u8, 1>,
        survivors: Option<&[FlowSurvivor]>,
        estimation: &Estimation,
        timestamp_sec: f64,
    ) -> FlowOnly {
        let (Some(min_tracks), Some(survivors)) = (self.klt_min_tracks, survivors) else {
            return FlowOnly::Unavailable;
        };
        if let FlowPose::AnalyticGn { max_iter } = self.flow.pose
            && let Some(outcome) = self.track_flow_analytic(
                idx,
                current_image,
                survivors,
                estimation,
                timestamp_sec,
                (min_tracks, max_iter),
            )
        {
            return outcome;
        }
        let map = self
            .map
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(seeded) = self.tracker.flow_frame(
            idx,
            survivors,
            &map,
            current_image.size(),
            min_tracks,
            &self.camera,
        ) else {
            return FlowOnly::Unavailable;
        };
        let n_seeded = seeded.len();
        let result = self.tracker.estimate_flow_only(
            flow_input(
                self.tracker.flow_stand_in(),
                estimation,
                self.state.current_keyframe_idx,
            ),
            seeded,
            &map,
            &self.camera,
            &mut self.tracking_timings,
        );
        let secondary = match &result {
            Ok(estimate) if self.debug && self.tracker.flow_stand_in().is_rig() => {
                let threshold_px = self
                    .tracker
                    .map_projection_config()
                    .pnp
                    .final_reproj_threshold_px;
                secondary_inliers(
                    self.tracker.flow_stand_in(),
                    &estimate.matches,
                    &map,
                    &estimate.pose,
                    threshold_px,
                )
            }
            _ => 0,
        };
        drop(map);
        let Ok(estimate) = result else {
            if self.debug {
                self.dbg(format!("[klt-first] frame={idx} refused seeded={n_seeded}"));
            }
            return FlowOnly::Refused;
        };

        let evidence = self.flow_keyframe_evidence(
            self.tracker.flow_stand_in(),
            &estimate.pose,
            estimate.inliers,
            &estimate.matches,
        );
        if self.keyframe_due(&evidence, timestamp_sec) {
            if self.debug {
                self.dbg(format!(
                    "[klt-first] frame={idx} keyframe due seeded={n_seeded} inliers={}",
                    estimate.inliers
                ));
            }
            return FlowOnly::KeyframeDue(estimate.pose);
        }
        self.tracker.accept_flow_only(&estimate, survivors);
        self.frontend
            .observe_flow(&self.tracker.flow_stand_in().features.keypoints_xy);
        self.count_flow_frame(idx, n_seeded, &estimate, secondary, false);
        FlowOnly::Tracked(estimate)
    }

    /// [`Self::track_flow_only`] by the analytic flow pose
    /// (`FlowPose::AnalyticGn`); `None` when it falls back to RANSAC PnP. On
    /// `WindowedU8` the seeds are built from the tracks' stored positions,
    /// without the map lock.
    fn track_flow_analytic(
        &mut self,
        idx: usize,
        current_image: &Image<u8, 1>,
        survivors: &[FlowSurvivor],
        estimation: &Estimation,
        timestamp_sec: f64,
        (min_tracks, max_iter): (usize, usize),
    ) -> Option<FlowOnly> {
        let windowed = self.rig_flow.is_some();
        let seeded = match self.rig_flow.as_ref() {
            Some(rig_flow) => self
                .tracker
                .analytic_seeds(AnalyticSeedSource::Windowed(rig_flow), min_tracks)
                .is_some(),
            None => {
                let map = self
                    .map
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                self.tracker
                    .flow_frame(
                        idx,
                        survivors,
                        &map,
                        current_image.size(),
                        min_tracks,
                        &self.camera,
                    )
                    .and_then(|seeded| {
                        self.tracker.analytic_seeds(
                            AnalyticSeedSource::Frame {
                                seeded: &seeded,
                                map: &map,
                            },
                            min_tracks,
                        )
                    })
                    .is_some()
            }
        };
        if !seeded {
            return Some(FlowOnly::Unavailable);
        }
        let n_seeded = self.tracker.flow_seeds().len();
        let started = Instant::now();
        let solved = self.tracker.estimate_flow_analytic(
            &estimation.candidate_pose,
            estimation.trusted_prediction,
            &self.camera,
            max_iter,
        );
        self.tracking_timings.flow_pose += started.elapsed();
        let (estimate, secondary) = match solved? {
            Ok(solved) => solved,
            Err(_) => {
                if self.debug {
                    self.dbg(format!("[klt-first] frame={idx} refused seeded={n_seeded}"));
                }
                return Some(FlowOnly::Refused);
            }
        };
        let evidence = if windowed {
            self.seeds_keyframe_evidence(idx, &estimate)
        } else {
            self.flow_keyframe_evidence(
                self.tracker.flow_stand_in(),
                &estimate.pose,
                estimate.inliers,
                &estimate.matches,
            )
        };
        if self.keyframe_due(&evidence, timestamp_sec) {
            if self.debug {
                self.dbg(format!(
                    "[klt-first] frame={idx} keyframe due seeded={n_seeded} inliers={}",
                    estimate.inliers
                ));
            }
            return Some(FlowOnly::KeyframeDue(estimate.pose));
        }
        self.tracker.accept_flow_only(&estimate, survivors);
        let pixels = if windowed {
            &self.tracker.flow_seeds().pixels
        } else {
            &self.tracker.flow_stand_in().features.keypoints_xy
        };
        self.frontend.observe_flow(pixels);
        self.count_flow_frame(idx, n_seeded, &estimate, secondary, true);
        Some(FlowOnly::Tracked(estimate))
    }

    /// Counts a frame tracked from flow alone.
    fn count_flow_frame(
        &mut self,
        idx: usize,
        n_seeded: usize,
        estimate: &Estimate,
        secondary: usize,
        analytic: bool,
    ) {
        self.flow_counters.flow_frames += 1;
        self.flow_counters.flow_seeds += n_seeded as u64;
        self.flow_counters.flow_inliers += estimate.inliers as u64;
        if self.debug && (self.flow.rig == KltRig::AllCameras || self.rig_flow.is_some()) {
            self.dbg(format!(
                "[klt-first] frame={idx} tracked seeded={n_seeded} inliers={} secondary={secondary} analytic={}",
                estimate.inliers,
                u8::from(analytic),
            ));
        }
    }

    /// Counts what a deferred frame's flow-only attempt gave; `flow_frames`
    /// counts the tracked ones where they are accepted.
    fn count_flow_decision(&mut self, flow_only: &FlowOnly) {
        let counter = match flow_only {
            FlowOnly::Unavailable => &mut self.flow_counters.flow_unavailable,
            FlowOnly::Refused => &mut self.flow_counters.flow_refused,
            FlowOnly::KeyframeDue(_) => &mut self.flow_counters.flow_keyframe_due,
            FlowOnly::Tracked(_) => return,
        };
        *counter += 1;
    }

    /// The `(max_level, max_iter)` of `klt_prediction` when this frame's
    /// pose comes from a trusted inertial prediction; a trusted IMU whose
    /// window held no samples falls back to the visual model, too weak a
    /// start for the shallow search.
    fn predicted_flow_search(
        &self,
        trusted_prediction: bool,
        inertial_prediction: bool,
    ) -> Option<(usize, usize)> {
        self.klt_prediction
            .filter(|_| trusted_prediction && inertial_prediction)
    }

    /// The optical-flow survivors of the current frame, searched from the
    /// predicted pose with `search` levels and iterations when given.
    fn flow(
        &mut self,
        previous_image: Option<&Image<u8, 1>>,
        images: &[&Image<u8, 1>],
        candidate_pose: &Pose3d,
        search: Option<(usize, usize)>,
    ) -> Option<Vec<FlowSurvivor>> {
        if self.rig_flow.is_some() {
            return self.flow_windowed(candidate_pose, search.is_some());
        }
        if self.tracker.flows_rig() {
            return self.flow_cameras(previous_image, images, candidate_pose, search);
        }
        let current_image = images.first()?;
        let Some((max_level, max_iter)) = search else {
            return self.tracker.flow(
                previous_image,
                current_image,
                None,
                &mut self.tracking_timings,
            );
        };
        let map = self
            .map
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let guess = FlowGuess {
            map: &map,
            pose: candidate_pose,
            camera: &self.camera,
            max_level,
            max_iter,
            gyro_rotation: self.gyro_rotation,
        };
        self.tracker.flow(
            previous_image,
            current_image,
            Some(guess),
            &mut self.tracking_timings,
        )
    }

    /// [`Self::flow`] of every rig camera on the full-pyramid kernels, each
    /// from its own previous image.
    fn flow_cameras(
        &mut self,
        previous_image: Option<&Image<u8, 1>>,
        images: &[&Image<u8, 1>],
        candidate_pose: &Pose3d,
        prediction: Option<(usize, usize)>,
    ) -> Option<Vec<FlowSurvivor>> {
        let mut previous: Vec<&Image<u8, 1>> = Vec::with_capacity(images.len());
        previous.extend(previous_image);
        previous.extend(self.previous_secondary.iter());
        let map = self
            .map
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let guess = prediction.map(|(max_level, max_iter)| FlowGuess {
            map: &map,
            pose: candidate_pose,
            camera: &self.camera,
            max_level,
            max_iter,
            gyro_rotation: self.gyro_rotation,
        });
        self.tracker
            .flow_cameras(&previous, images, guess, &mut self.tracking_timings)
    }

    /// [`Self::flow`] on the `WindowedU8` shards, from the predicted pose
    /// when `predicted` (the shallow predicted search applies).
    fn flow_windowed(
        &mut self,
        candidate_pose: &Pose3d,
        predicted: bool,
    ) -> Option<Vec<FlowSurvivor>> {
        let rig_flow = self.rig_flow.as_mut()?;
        let frame_idx = self.next_frame_idx - 1;
        let input = FlowStepInput {
            frame_idx: frame_idx as u64,
            pose_world_to_primary: predicted.then_some(*candidate_pose),
            primary_rotation: self.gyro_rotation,
        };
        let survivors = self
            .tracker
            .flow_windowed(rig_flow, &input, &mut self.tracking_timings);
        if self.debug {
            let output = rig_flow.output();
            let counts = output.counts;
            for (shard, tracks) in output
                .shard_tracks
                .iter()
                .take(rig_flow.threads())
                .enumerate()
            {
                self.debug_messages.push(format!(
                    "[flow] frame={frame_idx} shard={shard} tracks={tracks}"
                ));
            }
            self.debug_messages.push(format!(
                "[flow] frame={frame_idx} tried={} fb_rejects={} retries={} template_rebuilds={} refill={}",
                counts.tracks_tried,
                counts.fb_rejects,
                counts.retries,
                counts.template_rebuilds,
                output.spawned.len(),
            ));
        }
        survivors
    }

    /// What the keyframe policy is shown for `frame`: the parallax is taken
    /// against the last keyframe at `pose`, the pose the frame was tracked
    /// at, not at the frame's own (pre-tracking) pose.
    fn keyframe_evidence(
        &self,
        frame: &Frame,
        pose: &Pose3d,
        tracked_inliers: usize,
        matches: &[(usize, usize)],
    ) -> KeyframeEvidence {
        let map = self.map.lock().unwrap();
        let n_ref_map_points = self
            .state
            .current_keyframe_idx
            .and_then(|ki| map.get_keyframe(ki))
            .map(|kf| kf.num_associated_points())
            .unwrap_or(0);
        // Measured only when the policy reads it: it walks every match.
        let median_parallax_px = self
            .keyframe_policy
            .min_parallax_px
            .zip(self.state.last_keyframe_idx)
            .and_then(|(_, last)| {
                median_parallax_px(&map, last, frame, pose, matches, &self.camera)
            });
        KeyframeEvidence {
            curr_idx: frame.idx,
            last_keyframe_idx: self.state.last_keyframe_idx,
            tracked_inliers,
            n_ref_map_points,
            median_parallax_px,
        }
    }

    /// [`Self::keyframe_evidence`] for a flow frame: its inliers can only be
    /// landmarks its tracks follow, so they are compared with the tracks that
    /// followed one after the last ORB frame, not with the reference
    /// keyframe's landmarks.
    fn flow_keyframe_evidence(
        &self,
        frame: &Frame,
        pose: &Pose3d,
        tracked_inliers: usize,
        matches: &[(usize, usize)],
    ) -> KeyframeEvidence {
        KeyframeEvidence {
            n_ref_map_points: self.tracker.followed_baseline(),
            ..self.keyframe_evidence(frame, pose, tracked_inliers, matches)
        }
    }

    /// [`Self::flow_keyframe_evidence`] of an analytic `WindowedU8` flow
    /// frame, whose matches are `(map point, seed)` of the tracker's seeds.
    fn seeds_keyframe_evidence(&self, idx: usize, estimate: &Estimate) -> KeyframeEvidence {
        let seeds = self.tracker.flow_seeds();
        let map = self
            .map
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let primary_xy = |seed: usize| {
            let pixel = *seeds.undistorted.get(seed)?;
            (seeds.cameras.get(seed) == Some(&0) && pixel[0].is_finite() && pixel[1].is_finite())
                .then_some(pixel)
        };
        let median_parallax_px = self
            .keyframe_policy
            .min_parallax_px
            .zip(self.state.last_keyframe_idx)
            .and_then(|(_, last)| {
                median_parallax_px_with(
                    &map,
                    last,
                    &estimate.pose,
                    &estimate.matches,
                    &self.camera,
                    primary_xy,
                )
            });
        KeyframeEvidence {
            curr_idx: idx,
            last_keyframe_idx: self.state.last_keyframe_idx,
            tracked_inliers: estimate.inliers,
            n_ref_map_points: self.tracker.followed_baseline(),
            median_parallax_px,
        }
    }

    /// The policy's decision, or `max_interval_sec` elapsed since the last
    /// keyframe's timestamp.
    fn keyframe_due(&self, evidence: &KeyframeEvidence, timestamp_sec: f64) -> bool {
        let interval_elapsed = evidence
            .last_keyframe_idx
            .zip(self.inertial.last_keyframe_timestamp_sec)
            .is_some_and(|(last_idx, last_ts)| {
                self.keyframe_policy.interval_elapsed(
                    evidence.curr_idx.saturating_sub(last_idx),
                    timestamp_sec - last_ts,
                )
            });
        interval_elapsed || self.keyframe_policy.should_insert_with(evidence)
    }

    /// Whether the IMU has been initialized long enough to coast on.
    fn imu_confident(&self, timestamp_sec: f64) -> bool {
        let min_confidence_sec = self.tracking_loss_recovery.min_imu_confidence_sec;
        self.state.imu_initialized
            && self
                .state
                .imu_init_timestamp_sec
                .is_some_and(|t0| timestamp_sec - t0 >= min_confidence_sec)
    }

    /// Recovers the pose against the map from place-recognition candidates
    /// and resumes tracking from it. Needs the loop closer's vocabulary and
    /// keyframe database.
    fn relocalize(&mut self, frame: &Frame) -> bool {
        const MAX_CANDIDATES: usize = 5;
        let Some(loop_closer) = self.loop_closer.as_ref() else {
            return false;
        };
        // A pose far from where the motion model put the camera belongs to a
        // part of the map that has drifted from here; adopting it would
        // teleport the trajectory.
        let predicted_position = self.state.pose_world_to_cam.camera_centre();
        let mut refused = Vec::new();
        let found = {
            let map = self.map.lock().unwrap();
            let candidates = loop_closer.relocalization_candidates(
                &frame.features.descriptors,
                &map,
                MAX_CANDIDATES,
            );
            self.tracker
                .relocalize(frame, &candidates, &map, &self.camera, |found| {
                    let jump = (found.estimate.pose.camera_centre() - predicted_position).length();
                    let near = jump <= RELOCALIZATION_MAX_JUMP_M;
                    if !near {
                        refused.push((found.keyframe_idx, jump));
                    }
                    near
                })
        };
        for (keyframe_idx, jump) in refused {
            self.dbg(format!(
                "[reloc] frame={} refused kf={keyframe_idx} jump={jump:.2}m",
                frame.idx
            ));
        }
        let Some(found) = found else {
            return false;
        };
        self.dbg(format!(
            "[reloc] frame={} against kf={} inliers={}",
            frame.idx, found.keyframe_idx, found.estimate.inliers
        ));
        self.state.pose_world_to_cam = found.estimate.pose;
        self.state.current_keyframe_idx = Some(found.keyframe_idx);
        self.state.velocity = None;
        true
    }

    /// Whether to extend the map from this frame while coasting on the IMU:
    /// the policy says the last keyframe is old enough. A stereo keyframe
    /// creates landmarks from its depth; a monocular one through triangulation
    /// against its neighbours at the inertial poses.
    fn keyframe_due_while_lost(&self, timestamp_sec: f64) -> bool {
        self.inertial
            .last_keyframe_timestamp_sec
            .is_some_and(|t| self.tracking_loss_recovery.keyframe_due(timestamp_sec - t))
    }

    fn try_insert_keyframe(
        &mut self,
        frame: &Frame,
        pose: &Pose3d,
        timestamp_sec: f64,
        tracked_inliers: usize,
        matches: &[(usize, usize)],
    ) -> bool {
        let evidence = self.keyframe_evidence(frame, pose, tracked_inliers, matches);
        if !self.keyframe_due(&evidence, timestamp_sec) {
            return false;
        }
        self.insert_keyframe(frame, timestamp_sec, matches)
    }

    /// Publishes `frame` as a keyframe at the current tracking pose and runs
    /// the mapping work it earns.
    fn insert_keyframe(
        &mut self,
        frame: &Frame,
        timestamp_sec: f64,
        matches: &[(usize, usize)],
    ) -> bool {
        // Guard: reference KF must exist before we can triangulate.
        if let Some(ki) = self.state.current_keyframe_idx {
            let map = self.map.lock().unwrap();
            if map.get_keyframe(ki).is_none() {
                return false;
            }
        } else {
            return false;
        }

        let mut curr_kf = Keyframe::from_frame(Frame {
            pose_world_to_cam: self.state.pose_world_to_cam,
            ..frame.clone()
        });
        // Seed the new keyframe with the IMU-propagated velocity and current bias
        // estimate so that VI-BA starts from a reasonable linearisation point rather
        // than zero, which would produce huge residuals on the newest IMU edge.
        if self.state.imu_initialized {
            curr_kf.velocity_world = self.state.velocity_world;
            curr_kf.imu_bias = self.inertial.bias;
        }
        let imu_initialized = self.state.imu_initialized;
        let is_stereo = curr_kf.frame.is_stereo();

        // Neighbours are captured BEFORE publication: growing against a list
        // that already contains the current keyframe would triangulate it
        // against itself and drop the oldest real neighbour. Selection policy
        // belongs to mapping.
        let growth_plan =
            keyframe_mapping::prepare_keyframe_growth(&self.map.lock().unwrap(), frame.idx);

        let imu_edge = match (
            self.state.last_keyframe_idx,
            self.inertial.last_keyframe_timestamp_sec,
        ) {
            (Some(prev_kf_idx), Some(prev_ts)) => self.inertial.keyframe_edge(
                self.rig.imu_noise(),
                prev_kf_idx,
                frame.idx,
                prev_ts,
                timestamp_sec,
            ),
            _ => None,
        };
        let KeyframeInsertion {
            insertion,
            close_points: n_close,
        } = keyframe_mapping::keyframe_insertion(
            curr_kf,
            matches,
            self.stereo_close_depth,
            &self.camera,
            imu_edge,
        );

        let published = self.map.lock().unwrap().apply_insertion(insertion);
        if let Err(error) = published {
            // Nothing was written, so no tracker or IMU reference may advance to
            // a keyframe the map does not hold.
            self.dbg(format!(
                "[kf] frame={} publication rejected: {error}",
                frame.idx
            ));
            return false;
        }
        if self.stereo_close_depth.is_some() && is_stereo {
            self.dbg(format!(
                "[kf_stereo] frame={} close_points={}",
                frame.idx, n_close
            ));
        }

        self.inertial.last_keyframe_timestamp_sec = Some(timestamp_sec);

        self.state.current_keyframe_idx = Some(frame.idx);
        self.state.last_keyframe_idx = Some(frame.idx);
        if self.flow.adopt_landmarks
            && let Some(rig_flow) = self.rig_flow.as_mut()
            && let Err(error) = rig_flow.keep_keyframe()
            && self.debug
        {
            self.dbg(format!(
                "[flow] frame={} keyframe frames not kept: {error}",
                frame.idx
            ));
        }

        // Optional mapping work the accepted keyframe earns: grow against the
        // captured neighbours, then forward SearchInNeighbors / Fuse, before
        // local BA so BA sees the extra constraints. Mapping owns the sequence
        // and its lock boundaries; the messages stay here.
        let growth = keyframe_mapping::grow_keyframe(
            &self.map,
            growth_plan,
            &self.camera,
            &self.two_view_init_config.match_config,
            &self.two_view_init_config.triangulation_config,
        );
        self.report_growth(frame.idx, &growth);

        // Refinement can rotate/scale the world and update gravity. Do it before
        // constructing the BA request so the job and its future snapshot agree.
        if imu_initialized {
            self.refine_inertial_init(timestamp_sec);
        }

        if !self.local_mapping.submit(KeyframeJob {
            imu_initialized,
            imu_t_bc: self.rig.camera_to_body(),
            gravity_world: self.inertial.gravity_world,
        }) {
            self.dbg("[local_mapping] worker is unavailable".into());
        }
        // Synchronous mode has a completed correction available immediately;
        // asynchronous mode will deliver it at a later frame boundary.
        self.apply_local_mapping_results();

        self.register_place_recognition(frame.idx);
        // Growth, fusion and an inertial refinement moved landmarks; an
        // asynchronous result refreshes when it arrives.
        self.refresh_track_positions();

        true
    }

    fn report_growth(&mut self, frame_idx: usize, growth: &KeyframeGrowthResult) {
        if !self.debug {
            return;
        }
        for failure in &growth.pair_failures {
            self.dbg(format!(
                "[kf] frame={frame_idx} pair growth against {} rejected: {}",
                failure.neighbor_keyframe_idx, failure.error
            ));
        }
        self.dbg(format!(
            "[kf] frame={frame_idx} grown={} from {} neighbor kfs",
            growth.landmarks_added, growth.neighbor_count
        ));
        self.dbg(format!(
            "[fuse] frame={frame_idx} fused={}",
            growth.observations_added
        ));
        for error in &growth.fusion_failures {
            self.dbg(format!("[fuse] frame={frame_idx} link refused: {error}"));
        }
    }

    /// VIBA1/VIBA2: progressive re-solves with relaxed priors over the same
    /// (now-growing) window VIBA0 used, mirroring LocalMapping.cc:200-228.
    /// Tracking already runs on VIBA0's result, so a rejection here only means
    /// that stage never runs, not a tracking failure.
    fn refine_inertial_init(&mut self, timestamp_sec: f64) {
        let (Some(start_idx), Some(refinement)) = (
            self.inertial.schedule.start_kf_idx(),
            self.inertial.schedule.due_refinement(timestamp_sec),
        ) else {
            return;
        };
        let stage = refinement.name();

        // Drop the solve's map lock before applying its result with a new lock.
        let init_result = self.inertial.initializer.try_initialize(
            &self.map.lock().unwrap(),
            self.rig.camera_to_body(),
            self.inertial.bias,
            start_idx,
            refinement.prior_g(),
            refinement.prior_a(),
            true,
        );
        match init_result {
            Some(init) => {
                let applied = self.apply_inertial_initialization(init, start_idx);
                match applied {
                    Ok(AppliedInitialization {
                        scale,
                        gyro_bias: bg,
                        ..
                    }) => self.dbg(format!(
                        "[imu_init] {stage} accepted: scale_correction={scale:.4} gyro_bias=({:.4},{:.4},{:.4})",
                        bg.x, bg.y, bg.z
                    )),
                    Err(error) => {
                        self.dbg(format!("[imu_init] {stage} alignment refused: {error}"));
                    }
                }
            }
            None => {
                self.dbg(format!(
                    "[imu_init] {stage} rejected: solve failed or invalid scale"
                ));
            }
        }

        self.inertial.schedule.complete(refinement);
    }

    /// Offers a freshly inserted keyframe to loop closing, when enabled, and
    /// applies whatever the closer reports back.
    fn register_place_recognition(&mut self, kf_idx: usize) {
        let Some(loop_closer) = self.loop_closer.as_mut() else {
            return;
        };
        let context = LoopClosingContext {
            pose_world_to_cam: self.state.pose_world_to_cam,
            velocity_world: self.state.velocity_world,
            current_keyframe_idx: self.state.current_keyframe_idx,
            imu_initialized: self.state.imu_initialized,
            gravity_world: self.inertial.gravity_world,
        };
        let outcome = {
            let mut map = self.map.lock().unwrap();
            loop_closer.on_keyframe(&mut map, &self.camera, kf_idx, context)
        };
        if let Some(message) = outcome.debug_message {
            self.dbg(message);
        }
        if let Some((corrected_tracking_pose, corrected_velocity)) = outcome.tracking_correction {
            self.state.pose_world_to_cam = corrected_tracking_pose;
            self.state.velocity_world = corrected_velocity;
            self.refresh_track_positions();
        }
        if outcome.pgo_applied {
            self.tracker.reset_tracks();
            if self.state.imu_initialized {
                if !self.local_mapping.submit(KeyframeJob {
                    imu_initialized: true,
                    imu_t_bc: self.rig.camera_to_body(),
                    gravity_world: self.inertial.gravity_world,
                }) {
                    self.dbg("[local_mapping] worker is unavailable after PGO".into());
                }
                self.apply_local_mapping_results();
            }
        }
        self.loop_closure_events.extend(outcome.events);
    }
}

/// Largest distance, in metres, between the motion-model pose and a
/// relocalized one that is adopted.
const RELOCALIZATION_MAX_JUMP_M: f64 = 0.5;

/// Low-match two-view rejections a monocular bootstrap reference survives.
const MAX_LOW_MATCH_REJECTIONS: usize = 10;

/// How many of `matches` are rig cameras 1 and up within `threshold_px` of
/// their map point at `pose`.
fn secondary_inliers(
    frame: &Frame,
    matches: &[(usize, usize)],
    map: &Map,
    pose: &Pose3d,
    threshold_px: f64,
) -> usize {
    let threshold_sq = threshold_px * threshold_px;
    matches
        .iter()
        .filter(|&&(mp_idx, kp_idx)| {
            let (Some(view), Some(mp), Some(pixel)) = (
                frame.secondary_view(kp_idx),
                map.map_points().get(mp_idx),
                frame.keypoints_undist.get(kp_idx),
            ) else {
                return false;
            };
            view.pinhole
                .reprojection_error_sq_world(
                    &view.pose(pose),
                    &mp.position,
                    pixel[0] as f64,
                    pixel[1] as f64,
                )
                .is_some_and(|error_sq| error_sq <= threshold_sq)
        })
        .count()
}

/// The `WindowedU8` flow knobs of `settings`; without a prediction search LK
/// runs kornia-imgproc's default pyramid, as the full-pyramid kernels do.
fn rig_flow_settings(settings: &SystemSettings) -> RigFlowSettings {
    let flow = settings.flow;
    let default = kornia_imgproc::optical_flow_pyr_lk::PyrLKParams::default();
    let unpredicted = (default.max_level, default.max_iter);
    RigFlowSettings {
        threads: flow.threads,
        window: flow.window,
        predicted: settings.klt_prediction.unwrap_or(unpredicted),
        unpredicted,
        epsilon: default.epsilon,
        retry: flow.retry,
        forward_backward: flow.forward_backward,
        refill: flow.refill,
        gyro_guess: flow.gyro_guess,
    }
}

/// The cameras flow tracks: every rig camera under `klt_rig: AllCameras`,
/// the primary otherwise.
fn rig_flow_cameras(
    rig: &SensorRig,
    frontend: &OrbFrontend,
    klt_rig: KltRig,
) -> Vec<RigFlowCamera> {
    match (klt_rig, frontend.rig_geometry()) {
        (KltRig::AllCameras, Some(geometry)) => geometry
            .views()
            .enumerate()
            .map(|(camera, view)| RigFlowCamera {
                camera: view.pinhole.clone(),
                camera_from_primary: (camera > 0).then_some(*view.camera_from_primary),
            })
            .collect(),
        _ => vec![RigFlowCamera {
            camera: rig.primary().pinhole(),
            camera_from_primary: None,
        }],
    }
}

/// The estimation inputs of `frame` from this frame's `estimation`.
fn flow_input<'a>(
    frame: &'a Frame,
    estimation: &Estimation,
    current_keyframe_idx: Option<usize>,
) -> FrameInput<'a> {
    FrameInput {
        frame,
        candidate_pose: estimation.candidate_pose,
        pose_before: estimation.pose_before,
        current_keyframe_idx,
        lost_for_sec: estimation.lost_for_sec,
        trusted_prediction: estimation.trusted_prediction,
    }
}

fn format_imu_init_gate(
    start_idx: usize,
    first_idx: Option<usize>,
    last_idx: Option<usize>,
    keyframes: usize,
    min_keyframes: usize,
    imu_time: f64,
    min_time_sec: f64,
) -> String {
    format!(
        "[imu_init_gate] start_idx={start_idx} first_idx={first_idx:?} last_idx={last_idx:?} kfs={keyframes}/{min_keyframes} imu_time={imu_time:.2}/{min_time_sec:.1}s"
    )
}

#[cfg(test)]
mod tests;
