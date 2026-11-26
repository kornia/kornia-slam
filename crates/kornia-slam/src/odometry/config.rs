//! The `pipeline.odometry` knob and the estimator's constants.

use kornia_sensors::imu::ImuCalib;

/// Where each frame's pose comes from.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Odometry {
    /// Visual tracking against the map (today's pipeline).
    #[default]
    MapTracking,
    /// A fixed-lag visual-inertial smoother over the rectified stereo pair (Basalt-style).
    SlidingWindow(SlidingWindowConfig),
}

impl Odometry {
    /// The sliding-window settings, when that estimator is selected.
    pub fn sliding_window(&self) -> Option<&SlidingWindowConfig> {
        match self {
            Self::MapTracking => None,
            Self::SlidingWindow(config) => Some(config),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default, deny_unknown_fields))]
pub struct SlidingWindowConfig {
    pub mode: OdometryMode,
    pub imu_noise: OdometryImuNoise,
    /// Where the visual rows' Jacobians are evaluated (2x-5 K1, A/B 1).
    pub visual_jacobians: VisualJacobians,
}

/// The linearization point of the visual rows' Jacobians.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VisualJacobians {
    /// Every pose at its first estimate once it is in the prior (design card
    /// section 3, open question 1).
    #[default]
    FirstEstimate,
    /// Every pose at its current value. A visual row depends only on the
    /// host-to-target relative pose, so its 6-DOF nullspace holds at any
    /// point; Basalt evaluates that relative pose at the current state
    /// (`linearization_abs_qr.cpp:206-221`).
    Current,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum OdometryMode {
    /// Runs beside the pipeline on the tracking thread and changes nothing it
    /// outputs; its own pose is read with `SlamSystem::odometry_output`.
    #[default]
    Observe,
    /// The estimator's pose is the system's (rung 2x-6, not built).
    Drive,
}

/// The IMU noise densities the estimator preintegrates with.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum OdometryImuNoise {
    /// The rig's (datasheet) values, `SensorRig::imu_noise`.
    #[default]
    Rig,
    Explicit(ImuCalib),
}

/// The estimator's constants, Basalt's EuRoC values (`basalt/src/utils/
/// vio_config.cpp`, `sqrt_keypoint_vio.cpp`, `frame_to_frame_optical_flow.h`)
/// until the cleanup card derives them from the sensor model. Set from Rust
/// only.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OdometryTuning {
    /// Frame states of 15 DOF in the window (`vio_max_states`).
    pub max_states: usize,
    /// Keyframe poses in the window, those still frame states included
    /// (`vio_max_kfs`).
    pub max_keyframes: usize,
    /// The window is optimized once it holds more frame states than this
    /// (`sqrt_keypoint_vio.cpp:1017`).
    pub min_states_to_optimize: usize,
    /// Pixel noise, px (`vio_obs_std_dev`).
    pub pixel_sigma: f64,
    /// Huber knee on the raw residual norm, px (`vio_obs_huber_thresh`).
    pub huber_px: f64,
    /// Least camera-centre distance a landmark is triangulated across, m
    /// (`vio_min_triangulation_dist`).
    pub min_triangulation_m: f64,
    /// Inverse distances at or above this are refused, 1/m (`:444-445`).
    pub max_inverse_distance: f64,
    /// A frame is a keyframe when the share of its camera-0 tracks that have
    /// a landmark is below this (`vio_new_kf_keypoints_thresh`)...
    pub keyframe_tracked_share: f64,
    /// ...and more than this many frames passed since the last
    /// (`vio_min_frames_after_kf`).
    pub keyframe_min_frames: usize,
    /// A keyframe whose landmarks are tracked below this share is dropped
    /// first (`vio_kf_marg_feature_ratio`).
    pub keyframe_marg_feature_ratio: f64,
    /// LM: iterations, start, least and largest damping (`vio_max_iterations`,
    /// `vio_lm_lambda_initial`, `_min`, `_max`).
    pub lm_iterations: usize,
    pub lm_lambda_initial: f64,
    pub lm_lambda_min: f64,
    pub lm_lambda_max: f64,
    /// Information on the first state's position and yaw
    /// (`vio_init_pose_weight`).
    pub gauge_weight: f64,
    /// Standard deviation of the first state's biases, rad/s and m/s²
    /// (`vio_init_bg_weight`, `vio_init_ba_weight`, as Basalt applies them).
    pub initial_gyro_bias_sigma: f64,
    pub initial_accel_bias_sigma: f64,
    /// KLT round trip, px (`optical_flow_max_recovered_dist2` 0.04).
    pub max_round_trip_px: f32,
    /// Stereo row gate as a fraction of `fy` (`optical_flow_epipolar_error`).
    pub row_gate_fy_fraction: f64,
    /// Corners per camera-0 refill: 752×480 over 50 px cells
    /// (`optical_flow_detection_grid_size`).
    pub corners_per_frame: usize,
    pub corner_spacing_px: f32,
    /// KLT pyramid levels below level 0 and iterations per level
    /// (`optical_flow_levels`, `optical_flow_max_iterations`).
    pub klt_levels: usize,
    pub klt_iterations: usize,
    /// LK window side, odd: the window the OpenCV fixture validates.
    pub klt_window: usize,
    /// Set from [`SlidingWindowConfig::visual_jacobians`].
    pub visual_jacobians: VisualJacobians,
}

impl Default for OdometryTuning {
    fn default() -> Self {
        Self {
            max_states: 3,
            max_keyframes: 7,
            min_states_to_optimize: 4,
            pixel_sigma: 0.5,
            huber_px: 1.0,
            min_triangulation_m: 0.05,
            max_inverse_distance: 3.0,
            keyframe_tracked_share: 0.7,
            keyframe_min_frames: 5,
            keyframe_marg_feature_ratio: 0.1,
            lm_iterations: 7,
            lm_lambda_initial: 1e-4,
            lm_lambda_min: 1e-6,
            lm_lambda_max: 1e2,
            gauge_weight: 1e8,
            initial_gyro_bias_sigma: 0.316,
            initial_accel_bias_sigma: 0.1,
            max_round_trip_px: 0.2,
            row_gate_fy_fraction: 0.005,
            corners_per_frame: 144,
            corner_spacing_px: 40.0,
            klt_levels: 3,
            klt_iterations: 5,
            klt_window: 11,
            visual_jacobians: VisualJacobians::FirstEstimate,
        }
    }
}
