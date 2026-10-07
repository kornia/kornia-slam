use std::fmt;

use super::config::{
    CameraSelection, FrontendConfig, LoopClosingMode, OrbSlamPipeline, PipelineConfig,
    PipelineDefinition, StereoCloseDepth,
};
use crate::mapping::LocalMappingMode;

/// A stage role in the resolved pipeline graph.
///
/// The ORB family's connections are fixed: the frontend feeds tracking (with
/// stereo depth and IMU integration when enabled), tracking feeds keyframe
/// selection, keyframes feed local mapping and place recognition, and loop
/// correction feeds back into the map and tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Stage {
    OrbFrontend,
    StereoDepth,
    ImuIntegration,
    Tracking,
    KeyframeSelection,
    LocalMapping,
    PlaceRecognition,
    LoopCorrection,
}

impl PipelineConfig {
    /// Stages the definition enables, in data-flow order.
    pub(crate) fn stages(&self) -> Vec<Stage> {
        match &self.pipeline {
            PipelineDefinition::OrbSlam(orb) => self.orb_stages(orb),
        }
    }

    fn orb_stages(&self, orb: &OrbSlamPipeline) -> Vec<Stage> {
        let mut stages = vec![match orb.frontend {
            FrontendConfig::Orb(_) => Stage::OrbFrontend,
        }];
        if self.sensors.cameras == CameraSelection::Stereo {
            stages.push(Stage::StereoDepth);
        }
        if self.sensors.imu {
            stages.push(Stage::ImuIntegration);
        }
        stages.extend([
            Stage::Tracking,
            Stage::KeyframeSelection,
            Stage::LocalMapping,
        ]);
        match orb.loop_closing {
            LoopClosingMode::Disabled => {}
            LoopClosingMode::DetectOnly { .. } => stages.push(Stage::PlaceRecognition),
            LoopClosingMode::DetectAndCorrect { .. } => {
                stages.extend([Stage::PlaceRecognition, Stage::LoopCorrection]);
            }
        }
        stages
    }
}

/// One line per enabled stage with its resolved settings.
impl fmt::Display for PipelineConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let PipelineDefinition::OrbSlam(orb) = &self.pipeline;
        writeln!(f, "OrbSlam pipeline (config version {})", self.version)?;
        for stage in self.stages() {
            write!(f, "  ")?;
            write_orb_stage(f, stage, orb)?;
            writeln!(f)?;
        }
        Ok(())
    }
}

fn write_orb_stage(f: &mut fmt::Formatter<'_>, stage: Stage, orb: &OrbSlamPipeline) -> fmt::Result {
    match stage {
        Stage::OrbFrontend => {
            let FrontendConfig::Orb(frontend) = orb.frontend;
            write!(f, "frontend: ORB, {} keypoints", frontend.n_keypoints)
        }
        Stage::StereoDepth => {
            let FrontendConfig::Orb(frontend) = orb.frontend;
            match frontend.stereo_close_depth {
                StereoCloseDepth::Baselines(n) => {
                    write!(
                        f,
                        "stereo depth: rectified pair, close within {n} baselines"
                    )
                }
                StereoCloseDepth::Metres(m) => {
                    write!(f, "stereo depth: rectified pair, close within {m} m")
                }
                StereoCloseDepth::Disabled => write!(f, "stereo depth: rectified pair"),
            }
        }
        Stage::ImuIntegration => write!(f, "IMU integration"),
        Stage::Tracking => write!(f, "tracking"),
        Stage::KeyframeSelection => {
            let k = &orb.keyframes;
            write!(
                f,
                "keyframes: every {}..={} frames, ref ratio {}",
                k.min_frames_between, k.max_frames_between, k.ref_ratio
            )
        }
        Stage::LocalMapping => {
            let execution = match orb.mapping.execution {
                LocalMappingMode::Synchronous => "synchronous",
                LocalMappingMode::Asynchronous => "asynchronous",
            };
            write!(f, "local mapping: {execution}")
        }
        Stage::PlaceRecognition => match orb.loop_closing.vocabulary() {
            Some(vocabulary) => write!(f, "place recognition: {}", vocabulary.display()),
            None => write!(f, "place recognition"),
        },
        Stage::LoopCorrection => write!(f, "loop correction: verification and pose graph"),
    }
}
