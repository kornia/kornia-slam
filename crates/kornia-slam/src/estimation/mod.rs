//! Initialization algorithms.
//!
//! Pose estimation and optical flow are re-exported from [`crate::tracking`],
//! which owns them.

pub mod imu_init;
pub mod inertial_init_factor;
pub mod two_view;

pub use imu_init::{ImuInitConfig, ImuInitResult, ImuInitializer};

pub use crate::tracking::optical_flow;
pub use crate::tracking::optical_flow::{
    FlowSurvivor, KeypointCorrespondence, KltTracker, MapKeypointMatch, SurvivorFilterConfig,
    Track, TrackId, TrackSet, TrackSetError,
};
pub use crate::tracking::pose_estimation::{Estimate, MapProjectionEstimator, map_projection, pnp};
