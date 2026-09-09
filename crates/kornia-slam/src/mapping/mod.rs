//! Mapping: the map itself, plus the algorithms and worker that operate on it.
//!
//! [`map`] owns the stored entities, structural queries, and the canonical
//! mutation and correction operations. Growth, culling, bundle adjustment and
//! scheduling are mapping algorithms layered on top; they read the map
//! immutably and write only through its operations.

pub mod bundle_adjustment;
pub mod culling;
pub mod growth;
pub mod local_mapping;
pub mod map;

pub use local_mapping::{KeyframeJob, LocalMapping, LocalMappingMode};
pub use map::{Keyframe, Map, MapPoint};
