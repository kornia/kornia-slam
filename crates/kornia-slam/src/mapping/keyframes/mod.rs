//! Keyframe admission, motion evidence, and redundant-keyframe culling.
//!
//! This module owns decisions about when a tracked frame earns a keyframe and
//! when an already-published keyframe is redundant. Map mutation remains in
//! [`crate::mapping::map`], and accepted-keyframe growth remains coordinated
//! by the mapping publication path.

pub mod culling;
pub mod insertion;
pub mod parallax;

pub use culling::{KeyframeCullReport, KeyframeCullingConfig, cull_redundant_keyframes};
pub use insertion::{KeyframeEvidence, KeyframeInsertionStrategy, KeyframePolicy};
pub use parallax::{median_compensated_parallax_px, median_parallax_px};

/// Selects redundant-keyframe culling independently from keyframe insertion.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum KeyframeCullingStrategy {
    /// Keep the historical runtime behaviour: no redundant-keyframe culling.
    #[default]
    Disabled,
    /// Apply the existing ORB-SLAM-style redundancy policy after accepted
    /// local bundle-adjustment writeback.
    Redundancy(KeyframeCullingConfig),
}
