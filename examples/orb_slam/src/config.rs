use kornia_slam::estimation::map_projection::MapProjectionConfig;
use kornia_slam::estimation::two_view::TwoViewInitConfig;
use kornia_slam::system::KeyframePolicy;

/// Example-local pipeline preset used by the standalone ORB-SLAM binary.
pub struct PipelineConfig {
    pub two_view_init: TwoViewInitConfig,
    pub map_projection: MapProjectionConfig,
    pub keyframe_policy: KeyframePolicy,
    pub enable_local_ba: bool,
    pub enable_neighbor_fusion: bool,
    pub mapping_cooldown_frames: usize,
    pub tracking_orb_keypoints: usize,
    pub initial_orb_keypoint_multiplier: usize,
    pub initial_orb_window_frames: usize,
}

impl PipelineConfig {
    pub fn initial_orb_keypoints(&self) -> usize {
        self.tracking_orb_keypoints * self.initial_orb_keypoint_multiplier
    }
}

impl Default for PipelineConfig {
    fn default() -> Self {
        let mut two_view_init = TwoViewInitConfig::default();
        two_view_init
            .estimation_config
            .triangulation
            .min_parallax_deg = (0.99998_f64).acos().to_degrees();
        two_view_init
            .estimation_config
            .triangulation
            .max_midpoint_gap = 0.25;
        two_view_init
            .estimation_config
            .triangulation
            .max_reprojection_error = 3.0;

        Self {
            two_view_init,
            map_projection: MapProjectionConfig::default(),
            keyframe_policy: KeyframePolicy::default(),
            enable_local_ba: true,
            enable_neighbor_fusion: true,
            mapping_cooldown_frames: 0,
            tracking_orb_keypoints: 1000,
            initial_orb_keypoint_multiplier: 5,
            initial_orb_window_frames: 20,
        }
    }
}
