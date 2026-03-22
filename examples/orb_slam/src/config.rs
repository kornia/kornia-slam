use kornia_slam::estimation::map_projection::MapProjectionConfig;
use kornia_slam::estimation::two_view::TwoViewInitConfig;
use kornia_slam::system::KeyframePolicy;

/// Example-local pipeline preset used by the standalone ORB-SLAM binary.
pub struct PipelineConfig {
    pub two_view_init: TwoViewInitConfig,
    pub map_projection: MapProjectionConfig,
    pub keyframe_policy: KeyframePolicy,
    pub enable_local_ba: bool,
    pub mapping_queue_capacity: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        let mut two_view_init = TwoViewInitConfig::default();
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
            mapping_queue_capacity: 4,
        }
    }
}
