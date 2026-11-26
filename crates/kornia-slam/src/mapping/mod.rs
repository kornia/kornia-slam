//! Mapping: the map itself, plus the algorithms and worker that operate on it.
//!
//! [`map`] owns the stored entities, structural queries, and the canonical
//! mutation and correction operations. Growth, culling, bundle adjustment and
//! scheduling are mapping algorithms layered on top; they read the map
//! immutably and write only through its operations.
//!
//! `keyframe_mapping` coordinates the optional work an accepted keyframe
//! earns — neighbour selection, pair growth, forward fusion — so the system
//! keeps admission, core publication and the tracker/IMU lifecycle.
//!
//! [`vi_ba_schur`] is the visual-inertial solver behind [`bundle_adjustment`];
//! its IMU, wheel-odometry and parked-robot factors come from `kornia_factors`,
//! and [`marginalization`] is its boundary prior.

pub mod bundle_adjustment;
mod calibration_states;
pub mod culling;
pub mod dense_llt;
pub mod global_vi_ba;
pub mod growth;
pub(crate) mod keyframe_mapping;
pub mod local_mapping;
pub mod map;
pub mod marginalization;
pub(crate) mod online_calibration;
pub mod outlier_erase;
pub mod session;
#[cfg(any(test, feature = "bench_internals"))]
pub(crate) mod synthetic_walk;
pub mod vi_ba_schur;
pub mod visual_information;

pub use keyframe_mapping::{GrowthNeighbours, KeyframeGrowth};
pub use local_mapping::{KeyframeJob, LocalMapping, LocalMappingMode};
pub use map::{Keyframe, Map, MapPoint, SegmentId};
