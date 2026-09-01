use crate::estimation::map_projection::MapProjectionConfig;
use crate::estimation::two_view::TwoViewInitConfig;
use crate::loop_closure::{LoopEpisodeConfig, LoopFusionConfig, LoopVerificationConfig, PgoConfig};
use crate::map::LocalMappingMode;
use crate::system::{KeyframePolicy, TrackingLossRecoveryPolicy};

/// Runtime preset for [`SlamPipeline`](super::SlamPipeline).
pub struct PipelineConfig {
    pub two_view_init: TwoViewInitConfig,
    pub map_projection: MapProjectionConfig,
    pub keyframe_policy: KeyframePolicy,
    pub tracking_loss_recovery: TrackingLossRecoveryPolicy,
    pub local_mapping: LocalMappingMode,
    /// Near/far depth threshold `mThDepth` (metres). When `Some`, each new
    /// keyframe back-projects its unassociated "close" (`z < threshold`) stereo
    /// keypoints directly into metric map points. `None` disables stereo
    /// densification (monocular, or stereo without per-KF densification).
    pub stereo_close_depth_m: Option<f64>,
    /// Emit per-frame diagnostics: skip reasons in bootstrap, reject reasons
    /// in tracking, keyframe-growth and fuse counters.
    pub debug: bool,
    /// Optional verified loop closure and live pose-graph correction.
    pub pgo: Option<PgoPipelineConfig>,
    /// Optional AprilTag metric anchor: observes a known physical tag in
    /// keyframes and applies one post-hoc Sim3 correction (see
    /// [`SlamPipeline::apply_apriltag_anchor`](super::SlamPipeline)).
    #[cfg(feature = "apriltag")]
    pub apriltag: Option<crate::apriltag_anchor::AprilTagAnchorConfig>,
}

#[derive(Debug, Clone, Default)]
pub struct PgoPipelineConfig {
    /// Mono+IMU maps become metric only after inertial initialization. Stereo
    /// maps are metric from bootstrap and leave this disabled.
    pub require_imu_initialized: bool,
    pub episode: LoopEpisodeConfig,
    pub fusion: LoopFusionConfig,
    pub verification: LoopVerificationConfig,
    pub optimizer: PgoConfig,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        let mut two_view_init = TwoViewInitConfig::default();
        two_view_init.triangulation_config.max_midpoint_gap = 0.25;
        two_view_init.triangulation_config.max_reprojection_error = 3.0;

        Self {
            two_view_init,
            map_projection: MapProjectionConfig::default(),
            keyframe_policy: KeyframePolicy::default(),
            tracking_loss_recovery: TrackingLossRecoveryPolicy::default(),
            local_mapping: LocalMappingMode::Asynchronous,
            stereo_close_depth_m: None,
            debug: false,
            pgo: None,
            #[cfg(feature = "apriltag")]
            apriltag: None,
        }
    }
}
