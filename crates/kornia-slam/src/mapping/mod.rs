//! Mapping: the map itself, plus the algorithms and worker that operate on it.
//!
//! [`map`] owns the stored entities, structural queries, and the canonical
//! mutation and correction operations. Growth, culling and bundle-adjustment
//! scheduling are mapping algorithms layered on top; they read the map
//! immutably and write only through its operations.

pub mod culling;
pub mod map;

pub use map::{Keyframe, KeyframeJob, LocalMapping, LocalMappingMode, Map, MapPoint};
