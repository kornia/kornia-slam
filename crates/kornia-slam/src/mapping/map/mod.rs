//! Map storage and the operations that read and change it.
//!
//! [`Map`] owns the keyframes, map points, IMU factors and the world epoch;
//! nothing else stores them. Entities live beside it in [`keyframe`] and
//! [`map_point`], and every operation is an inherent `impl Map` block under the
//! private `ops` module, grouped by responsibility:
//!
//! | Group | Examples |
//! |---|---|
//! | `mutation` | `upsert_keyframe`, `push_map_point`, `register_observation`, `merge_map_points` |
//! | `correction` | `apply_pose_graph_correction`, `apply_inertial_alignment`, `scale_world` |
//! | `queries` | `covisible_keyframes` — the remaining structural query |
//! | `growth` | `add_close_stereo_points`, `grow_map_points_from_keyframe_pair`, `fuse_into_neighbors` |
//! | `culling` | `cull` |
//! | `bundle_adjustment` | `local_ba_snapshot`, `merge_local_ba_snapshot`, `run_local_ba` |
//!
//! Callers are unaffected by the grouping: the methods stay inherent on `Map`,
//! so `map.cull()` and `map::MapPoint` resolve exactly as before. Scheduling the
//! BA worker is separate again, and stays in [`local_mapping`].

mod keyframe;
mod local_mapping;
mod map_point;
mod ops;

pub use keyframe::Keyframe;
pub use local_mapping::{KeyframeJob, LocalMapping, LocalMappingMode};
pub use map_point::MapPoint;
pub use ops::covisible_above_weight;
pub use ops::{
    InertialAlignment, InertialAlignmentError, KeyframeBaCorrection, KeyframeVelocity,
    LocalBaMergeResult, LocalBaSnapshot, MapPointMergeResult, PoseGraphCorrectionError,
    PoseGraphCorrectionResult, STEREO_DEPTH_MIN_SIGMA, STEREO_DEPTH_REL_SIGMA, TriangulatedPoint,
};

use kornia_sensors::imu::{ImuMeasurement, PreintegratedImu};

/// Preintegrated IMU measurements connecting two consecutive keyframes.
#[derive(Debug, Clone)]
pub struct ImuFactor {
    /// Index of the earlier keyframe (`Keyframe::frame.idx`).
    pub prev_kf_idx: usize,
    /// Index of the later keyframe.
    pub curr_kf_idx: usize,
    /// IMU deltas integrated over the interval between the two keyframes.
    pub preintegrated: PreintegratedImu,
    /// Raw measurements covering `[t0, t1]`, retained so this factor can be
    /// repropagated (see `PreintegratedImu::from_measurements`) once the
    /// bias it was linearized at has drifted too far from the current
    /// estimate for the first-order correction to stay valid.
    pub raw_samples: Vec<ImuMeasurement>,
    pub t0: f64,
    pub t1: f64,
}

/// ORB pyramid scale factor between adjacent levels (matches the kornia-imgproc
/// ORB extractor default `downscale = 1.2`, which equals ORB-SLAM3's default).
pub const ORB_SCALE_FACTOR: f64 = 1.2;
/// Number of ORB pyramid levels (matches `OrbDetector` default `n_scales = 8`).
pub const ORB_N_LEVELS: usize = 8;

/// In-memory map storage for keyframes and persistent map points.
#[derive(Debug, Clone, Default)]
pub struct Map {
    keyframes: Vec<Keyframe>,
    map_points: Vec<MapPoint>,
    imu_factors: Vec<ImuFactor>,
    // Incremented whenever the map's world coordinate frame changes. Local BA
    // snapshots from an older epoch must never be merged into the new frame.
    world_epoch: u64,
}

impl Map {
    /// Creates an empty map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns all keyframes.
    pub fn keyframes(&self) -> &[Keyframe] {
        &self.keyframes
    }

    /// Returns mutable access to all keyframes.
    pub fn keyframes_mut(&mut self) -> &mut [Keyframe] {
        &mut self.keyframes
    }

    /// Returns the keyframe with frame index `idx`, if present.
    pub fn get_keyframe(&self, idx: usize) -> Option<&Keyframe> {
        self.keyframes.iter().find(|kf| kf.frame.idx == idx)
    }

    /// Mutable version of [`Map::get_keyframe`]. Needed when a triangulation
    /// or fusion pass produces a new observation that must be recorded on the
    /// live keyframe in the map (not a clone).
    pub fn get_keyframe_mut(&mut self, idx: usize) -> Option<&mut Keyframe> {
        self.keyframes.iter_mut().find(|kf| kf.frame.idx == idx)
    }

    /// Returns all map points.
    pub fn map_points(&self) -> &[MapPoint] {
        &self.map_points
    }

    /// Returns a mutable reference to all map points.
    pub fn map_points_mut(&mut self) -> &mut Vec<MapPoint> {
        &mut self.map_points
    }

    /// Returns the number of persistent map points.
    pub fn num_map_points(&self) -> usize {
        self.map_points.len()
    }

    /// Returns the number of non-culled (active) map points.
    pub fn num_active_map_points(&self) -> usize {
        self.map_points.iter().filter(|mp| !mp.culled).count()
    }

    /// Returns all keyframe-to-keyframe IMU factors in insertion order.
    pub fn imu_factors(&self) -> &[ImuFactor] {
        &self.imu_factors
    }
}

#[cfg(test)]
mod tests {
    use crate::frame::Frame;
    use kornia_3d::pose::Pose3d;
    use kornia_image::ImageSize;
    use kornia_imgproc::features::OrbFeatures;

    pub(super) fn test_frame(idx: usize, descriptors: Vec<[u8; 32]>) -> Frame {
        test_frame_with_pose(idx, descriptors, Pose3d::IDENTITY)
    }

    pub(super) fn test_frame_with_pose(
        idx: usize,
        descriptors: Vec<[u8; 32]>,
        pose: Pose3d,
    ) -> Frame {
        let n = descriptors.len();
        Frame {
            idx,
            features: OrbFeatures {
                keypoints_xy: (0..n).map(|i| [i as f32, i as f32]).collect(),
                orientations: vec![0.0; n],
                descriptors,
                octaves: vec![0; n],
            },
            pose_world_to_cam: pose,
            image_size: ImageSize {
                width: 640,
                height: 480,
            },
            keypoint_colors: vec![[0; 3]; n],
            u_right: Vec::new(),
            depth: Vec::new(),
            keypoints_undist: Vec::new(),
        }
    }
}
