use super::{Keyframe, Map};

/// Narrowly scoped fixture helpers.
///
/// These reach the module-private numeric accessors so tests outside `map` do
/// not need them made public. Each writes estimated quantities only — counters,
/// poses, inertial state — never structure. Anything that changes which feature
/// holds which landmark goes through the canonical operations, including in
/// fixtures.
impl Map {
    /// Sets a landmark's per-frame tracking statistics directly, instead of
    /// replaying the frames that would produce them.
    pub(crate) fn set_tracking_stats_for_test(&mut self, mp_idx: usize, visible: u32, found: u32) {
        if let Some(mp) = self.map_points_mut().get_mut(mp_idx) {
            mp.n_visible = visible;
            mp.n_found = found;
        }
    }

    /// Overrides a stored keyframe's pose, for fixtures that need a specific
    /// geometry rather than one produced by an estimator.
    pub(crate) fn set_keyframe_pose_for_test(
        &mut self,
        kf_idx: usize,
        pose: kornia_3d::pose::Pose3d,
    ) {
        if let Some(kf) = self.get_keyframe_mut(kf_idx) {
            kf.frame.pose_world_to_cam = pose;
        }
    }

    /// A comparable rendering of the complete stored state.
    ///
    /// For unchanged-on-error assertions: comparing two of these catches a
    /// partial write that entity counts alone would miss — a redirected
    /// association, a lost descriptor contribution, a bumped counter, a moved
    /// reference, a stale epoch, a mutated measurement.
    ///
    /// Everything `Map` owns is covered: the epoch, each keyframe's pose,
    /// velocity, bias, association slots and frame data, each landmark's
    /// position, descriptor, colour, reference, geometry, counters and
    /// records, and each IMU factor whole, preintegration and raw samples
    /// included. `Frame` and `PreintegratedImu` are rendered through `Debug`,
    /// so a new field on either joins this automatically.
    pub(crate) fn state_fingerprint_for_test(&self) -> String {
        use std::fmt::Write as _;
        let mut out = String::new();
        let _ = writeln!(out, "epoch {}", self.world_epoch);
        for kf in &self.keyframes {
            let _ = writeln!(
                out,
                "kf {} vel {:?} bias {:?} slots {:?} frame {:?}",
                kf.frame.idx, kf.velocity_world, kf.imu_bias, kf.map_point_by_desc_idx, kf.frame,
            );
        }
        for (idx, mp) in self.map_points.iter().enumerate() {
            let _ = writeln!(
                out,
                "mp {idx} pos {:?} desc {:?} color {:?} ref {} oct {} normal {:?} \
                 dist {:?}..{:?} seen {}/{} culled {} obs {:?}",
                mp.position,
                mp.descriptor,
                mp.color,
                mp.keyframe_idx,
                mp.reference_octave,
                mp.mean_viewing_direction,
                mp.min_distance,
                mp.max_distance,
                mp.n_found,
                mp.n_visible,
                mp.culled,
                mp.observations(),
            );
        }
        for factor in &self.imu_factors {
            let _ = writeln!(
                out,
                "imu {} -> {} [{}, {}] pre {:?} raw {:?}",
                factor.prev_kf_idx,
                factor.curr_kf_idx,
                factor.t0,
                factor.t1,
                factor.preintegrated,
                factor.raw_samples,
            );
        }
        out
    }

    /// Applies `edit` to each stored keyframe in insertion order, for fixtures
    /// that inject deterministic noise into poses or inertial state.
    pub(crate) fn edit_keyframes_for_test(&mut self, mut edit: impl FnMut(usize, &mut Keyframe)) {
        for (index, keyframe) in self.keyframes_mut().iter_mut().enumerate() {
            edit(index, keyframe);
        }
    }
}

use crate::frame::Frame;
use kornia_3d::pose::Pose3d;
use kornia_image::ImageSize;
use kornia_imgproc::features::OrbFeatures;

pub(crate) fn test_frame(idx: usize, descriptors: Vec<[u8; 32]>) -> Frame {
    test_frame_with_pose(idx, descriptors, Pose3d::IDENTITY)
}

pub(crate) fn test_frame_with_pose(idx: usize, descriptors: Vec<[u8; 32]>, pose: Pose3d) -> Frame {
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
