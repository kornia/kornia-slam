use kornia_slam::estimation::map_projection::MapProjectionConfig;
use kornia_slam::estimation::optical_flow::OpticalFlowConfig;
use kornia_slam::estimation::two_view::TwoViewInitConfig;
use kornia_slam::system::{KeyframePolicy, TrackingLossRecoveryPolicy};

/// Example-local pipeline preset used by the standalone ORB-SLAM binary.
pub struct PipelineConfig {
    pub two_view_init: TwoViewInitConfig,
    pub map_projection: MapProjectionConfig,
    pub keyframe_policy: KeyframePolicy,
    pub tracking_loss_recovery: TrackingLossRecoveryPolicy,
    pub enable_local_ba: bool,
    /// Near/far depth threshold `mThDepth` (metres). When `Some`, each new
    /// keyframe back-projects its unassociated "close" (`z < threshold`) stereo
    /// keypoints directly into metric map points. `None` disables stereo
    /// densification (monocular, or stereo without per-KF densification).
    pub stereo_close_depth_m: Option<f64>,
    /// Emit per-frame diagnostics: skip reasons in bootstrap, reject reasons
    /// in tracking, keyframe-growth and fuse counters.
    pub debug: bool,
    /// Seed `estimate_pose`'s correspondence search with KLT-tracked points
    /// (see `estimation::optical_flow`), tried before the pose-dependent
    /// projection search. A single toggle so a run can be A/B'd against the
    /// pre-existing projection-only path without a rebuild.
    pub enable_optical_flow_tracking: bool,
    pub optical_flow: OpticalFlowConfig,
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
            enable_local_ba: true,
            stereo_close_depth_m: None,
            debug: false,
            enable_optical_flow_tracking: true,
            optical_flow: OpticalFlowConfig::default(),
        }
    }
}
