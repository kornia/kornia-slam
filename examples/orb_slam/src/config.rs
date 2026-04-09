use kornia_slam::estimation::map_projection::MapProjectionConfig;
use kornia_slam::estimation::two_view::TwoViewInitConfig;
use kornia_slam::system::KeyframePolicy;

/// Example-local pipeline preset used by the standalone ORB-SLAM binary.
pub struct PipelineConfig {
    pub two_view_init: TwoViewInitConfig,
    pub map_projection: MapProjectionConfig,
    pub keyframe_policy: KeyframePolicy,
    pub enable_local_ba: bool,
    /// Near/far depth threshold `mThDepth` (metres). When `Some`, each new
    /// keyframe back-projects its unassociated "close" (`z < threshold`) stereo
    /// keypoints directly into metric map points. `None` disables stereo
    /// densification (monocular, or stereo without per-KF densification).
    pub stereo_close_depth_m: Option<f64>,
    /// Emit per-frame diagnostics: skip reasons in bootstrap, reject reasons
    /// in tracking, keyframe-growth and fuse counters.
    pub debug: bool,
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
            enable_local_ba: true,
            stereo_close_depth_m: None,
            debug: false,
        }
    }
}


impl PipelineConfig {
    pub fn for_dataset(dataset: &str) -> Self {
        if dataset.eq_ignore_ascii_case("kitti") {
            return Self::kitti();
        }
        Self::default()
    }

    /// KITTI-tuned defaults: relax acceptance to reduce dropped frames.
    fn kitti() -> Self {
        let mut cfg = Self::default();
        // Two-view bootstrap: allow lower parallax / inlier counts.
        cfg.two_view_init.acceptance_config.min_matches = 60;
        cfg.two_view_init.acceptance_config.min_inliers = 18;
        cfg.two_view_init.acceptance_config.min_triangulated = 24;
        cfg.two_view_init.triangulation_config.min_parallax_deg = 0.5;
        cfg.two_view_init.triangulation_config.max_reprojection_error = 5.0;

        // Tracking/PnP: be more tolerant when motion or illumination changes.
        cfg.map_projection.match_config.nn_ratio = 0.8;
        cfg.map_projection.match_config.th_low = 60;
        cfg.map_projection.projection.search_radius = 30.0;
        cfg.map_projection.projection.max_hamming = 64;
        cfg.map_projection.local_projection.search_radius = 42.0;
        cfg.map_projection.local_projection.max_hamming = 80;
        cfg.map_projection.pnp.final_reproj_threshold_px = 5.0;
        cfg.map_projection.pnp.min_inliers = 15;

        // Insert keyframes earlier so tracking remains anchored.
        cfg.keyframe_policy.min_frames_between = 1;
        cfg.keyframe_policy.max_frames_between = 5;
        cfg.keyframe_policy.ref_ratio = 0.9;

        cfg
    }
}