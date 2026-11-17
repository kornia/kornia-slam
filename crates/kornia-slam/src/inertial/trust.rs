//! How far the inertial prediction of the current frame can be trusted: by
//! time since initialization, or by its propagated covariance (OpenVINS gates
//! on the marginal, `Propagator.cpp:33-130`).

use kornia_algebra::Mat15F64;

/// The rule that decides [`ImuTrust::trusted`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ImuTrustMode {
    /// The IMU has been initialized for `min_imu_confidence_sec`.
    #[default]
    TimeSinceInit,
    /// Trusted while `search_sigmas` predicted pixel sigmas fit inside the
    /// widest projection search (`lost_search_scale · search_radius`).
    Covariance { search_sigmas: f64 },
}

/// How far the current frame's inertial prediction can be trusted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImuTrust {
    /// The prediction may seed tracking, the pose gate and the KLT guess;
    /// relocalize when false.
    pub trusted: bool,
    /// Projection search scale; `None` keeps `search_scale_for`.
    pub search_scale: Option<f32>,
    /// One-sigma reprojection spread at the reference depth, px; `None` under
    /// [`ImuTrustMode::TimeSinceInit`].
    pub pixel_sigma: Option<f64>,
    /// Predicted state covariance, layout `[ρ ω v b_g b_a]`; `None` under
    /// [`ImuTrustMode::TimeSinceInit`].
    pub covariance: Option<Mat15F64>,
}

impl ImuTrust {
    /// The time rule's verdict, with no covariance behind it.
    pub fn by_time(trusted: bool) -> Self {
        Self {
            trusted,
            search_scale: None,
            pixel_sigma: None,
            covariance: None,
        }
    }
}
