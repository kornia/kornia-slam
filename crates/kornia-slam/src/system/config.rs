use crate::initialization::TwoViewInitConfig;
pub use crate::loop_closure::LoopClosingConfig;
use crate::map::LocalMappingMode;
use crate::tracking::pose_estimation::map_projection::MapProjectionConfig;
use crate::tracking::{KeyframePolicy, TrackingLossRecoveryPolicy};

/// Runtime configuration for [`SlamSystem`](super::SlamSystem).
pub struct SlamConfig {
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
    pub pgo: Option<LoopClosingConfig>,
}

impl Default for SlamConfig {
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
        }
    }
}
