//! Two-camera KLT over the rectified pair for the window: left temporal and
//! right temporal tracks with a 0.2 px round trip, new left corners linked
//! into the right image, and the stereo row gate (Basalt
//! `frame_to_frame_optical_flow.h:161-386`).

use std::collections::BTreeMap;

use kornia_algebra::Mat3F64;
use kornia_image::Image;

use super::OdometryError;
use super::config::OdometryTuning;
use super::window::{Observation, StereoView};
use crate::system::{AnchorGuard, ForwardBackward, GyroGuess, KltRetry, TrackRefill};
use crate::tracking::TrackingTimings;
use crate::tracking::flow_pool::FlowPoolError;
use crate::tracking::flow_tracker::{
    FlowStepInput, RefillCameras, RigFlow, RigFlowCamera, RigFlowSettings,
};
use crate::tracking::optical_flow::{FlowSurvivor, TrackSet};

/// Tallies of the frontend since it started.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct FrontendCounts {
    pub stereo_links: u64,
    pub stereo_link_failures: u64,
    pub fb_rejects: u64,
    pub row_gate_rejects: u64,
    /// Tracks the forward solve or the survivor filter dropped (the 60 px
    /// movement cap among them).
    pub flow_failures: u64,
}

pub(crate) struct FlowFrontend {
    flow: RigFlow,
    tracks: TrackSet,
    /// Left track of each right track.
    partner: BTreeMap<u64, u64>,
    row_gate_px: f64,
    frame_idx: u64,
    timings: TrackingTimings,
    observations: Vec<Observation>,
    counts: FrontendCounts,
}

/// The flow settings of the design card section 6.
fn settings(tuning: &OdometryTuning) -> RigFlowSettings {
    let default = kornia_imgproc::optical_flow_pyr_lk::PyrLKParams::default();
    let search = (tuning.klt_levels, tuning.klt_iterations);
    RigFlowSettings {
        threads: 1,
        first_core: None,
        window: tuning.klt_window,
        predicted: search,
        unpredicted: search,
        epsilon: default.epsilon,
        retry: KltRetry::Off,
        forward_backward: ForwardBackward::Enabled {
            max_round_trip_px: tuning.max_round_trip_px,
            max_iter: tuning.klt_iterations,
            every_n_frames: 1,
        },
        refill: TrackRefill::Enabled {
            per_camera: tuning.corners_per_frame,
            min_spacing_px: tuning.corner_spacing_px,
            refill_below: 1.0,
            border_px: 10,
        },
        anchor_guard: AnchorGuard::Off,
        gyro_guess: GyroGuess::RotatedRay,
        refill_cameras: RefillCameras::PrimaryOnly,
    }
}

impl FlowFrontend {
    pub(crate) fn new(
        tuning: &OdometryTuning,
        cameras: &[StereoView; 2],
    ) -> Result<Self, FlowPoolError> {
        let flow = RigFlow::new(
            settings(tuning),
            vec![
                RigFlowCamera {
                    camera: cameras[0].camera.clone(),
                    camera_from_primary: None,
                },
                RigFlowCamera {
                    camera: cameras[1].camera.clone(),
                    camera_from_primary: Some(cameras[1].camera_from_primary),
                },
            ],
        )?;
        Ok(Self {
            flow,
            tracks: TrackSet::with_flow_threads(1, 2).with_monotonic_ids(),
            partner: BTreeMap::new(),
            row_gate_px: tuning.row_gate_fy_fraction * cameras[0].camera.fy,
            frame_idx: 0,
            timings: TrackingTimings::default(),
            observations: Vec::new(),
            counts: FrontendCounts::default(),
        })
    }

    pub(crate) fn counts(&self) -> &FrontendCounts {
        &self.counts
    }

    /// Tracks the pair; `rotation` is the left camera's `R_cur_prev` from the
    /// gyro. Returns the frame's observations sorted by (camera, landmark),
    /// the landmark being the left track's id.
    pub(crate) fn process(
        &mut self,
        left: &Image<u8, 1>,
        right: &Image<u8, 1>,
        rotation: Option<Mat3F64>,
    ) -> Result<&[Observation], OdometryError> {
        let frame_idx = self.frame_idx;
        self.frame_idx += 1;
        self.flow
            .prepare(&[left, right], frame_idx, &mut self.timings)?;
        let mut spawned: Vec<[f32; 2]> = Vec::new();
        if !self.tracks.is_empty() {
            let input = FlowStepInput {
                frame_idx,
                pose_world_to_primary: None,
                primary_rotation: rotation,
            };
            let output = self.flow.step(&self.tracks, &input, &mut self.timings)?;
            self.counts.fb_rejects += output.counts.fb_rejects;
            self.counts.flow_failures += output
                .counts
                .tracks_tried
                .saturating_sub(output.survivors.len() as u64 + output.counts.fb_rejects);
            spawned.extend(
                output
                    .spawned
                    .iter()
                    .filter(|(c, _)| *c == 0)
                    .map(|(_, p)| *p),
            );
            let left: BTreeMap<u64, [f32; 2]> = output
                .survivors
                .iter()
                .zip(&output.cameras)
                .filter(|(_, c)| **c == 0)
                .map(|(s, _)| (s.track_id.as_u64(), s.pixel))
                .collect();
            let mut kept: Vec<FlowSurvivor> = Vec::with_capacity(output.survivors.len());
            for (survivor, camera) in output.survivors.iter().zip(&output.cameras) {
                if *camera == 0 {
                    kept.push(*survivor);
                    continue;
                }
                let partner = self.partner.get(&survivor.track_id.as_u64());
                let Some(left_pixel) = partner.and_then(|p| left.get(p)) else {
                    continue;
                };
                if f64::from((left_pixel[1] - survivor.pixel[1]).abs()) > self.row_gate_px {
                    self.counts.row_gate_rejects += 1;
                    continue;
                }
                kept.push(*survivor);
            }
            self.tracks.advance(kept)?;
            let alive: std::collections::BTreeSet<u64> = self
                .tracks
                .tracks()
                .iter()
                .map(|t| t.id().as_u64())
                .collect();
            self.partner.retain(|right, _| alive.contains(right));
        } else {
            spawned.extend(
                self.flow
                    .seed(0)?
                    .iter()
                    .filter(|(c, _)| *c == 0)
                    .map(|(_, p)| *p),
            );
        }
        for pixel in spawned {
            let left_id = self.tracks.spawn_on(pixel, 0).as_u64();
            match self.flow.track_across(0, 1, pixel, pixel) {
                Some(right) if f64::from((right[1] - pixel[1]).abs()) <= self.row_gate_px => {
                    let right_id = self.tracks.spawn_on(right, 1).as_u64();
                    self.partner.insert(right_id, left_id);
                    self.counts.stereo_links += 1;
                }
                Some(_) => self.counts.row_gate_rejects += 1,
                None => self.counts.stereo_link_failures += 1,
            }
        }
        self.observations.clear();
        for track in self.tracks.tracks() {
            let id = track.id().as_u64();
            let landmark = match track.camera() {
                0 => id,
                _ => match self.partner.get(&id) {
                    Some(left) => *left,
                    None => continue,
                },
            };
            let [x, y] = track.pixel();
            self.observations.push(Observation {
                landmark,
                camera: track.camera() as u8,
                pixel: [f64::from(x), f64::from(y)],
            });
        }
        self.observations
            .sort_unstable_by_key(|o| (o.camera, o.landmark));
        Ok(&self.observations)
    }
}
