//! Mapping: the map itself, plus the algorithms and worker that operate on it.
//!
//! [`map`] owns the stored entities, structural queries, and the correction and
//! mutation operations. Growth, culling and bundle-adjustment scheduling are
//! mapping algorithms layered on top.

pub mod map;

pub use map::{Keyframe, KeyframeJob, LocalMapping, LocalMappingMode, Map, MapPoint};
