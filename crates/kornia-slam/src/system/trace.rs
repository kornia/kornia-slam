//! Read-only record of what tracking saw on one frame, so a frontend can be
//! replayed and benchmarked without the backend.

use kornia_3d::pose::Pose3d;

use crate::mapping::map::{Keyframe, Map};
use crate::tracking::optical_flow::Track;
use crate::tracking::{LocalMapSelectionConfig, select_local_map_points};

/// One local-map point as the projection search sees it.
#[derive(Debug, Clone, PartialEq)]
pub struct TraceMapPoint {
    /// Index into the map's point list on this frame.
    pub map_point_idx: usize,
    /// World-frame position, metres.
    pub position: [f64; 3],
    pub descriptor: [u8; 32],
    /// Raw matchable distance range, metres (see `MapPoint::min_distance`).
    pub min_distance: f64,
    pub max_distance: f64,
}

/// One optical-flow track at the start of the frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TraceTrack {
    pub track_id: u64,
    /// Raw pixel in the previous primary image.
    pub pixel: [f32; 2],
    pub map_point_idx: Option<usize>,
}

/// What tracking used and decided on one frame.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameTrace {
    pub frame_idx: usize,
    pub timestamp_sec: f64,
    /// World-to-camera pose the motion model predicted.
    pub candidate_pose: Pose3d,
    pub trusted_prediction: bool,
    /// The local map the initial projection search ran against.
    pub local_map: Vec<TraceMapPoint>,
    pub tracks: Vec<TraceTrack>,
    /// World-to-camera pose after tracking; the prediction when skipped.
    pub pose: Pose3d,
    pub tracked: bool,
    pub keyframe: bool,
}

impl FrameTrace {
    pub(crate) fn begin(
        frame_idx: usize,
        timestamp_sec: f64,
        candidate_pose: Pose3d,
        trusted_prediction: bool,
        map: &Map,
        current_kf: Option<&Keyframe>,
        tracks: &[Track],
    ) -> Self {
        let local_indices =
            select_local_map_points(map, &[], current_kf, LocalMapSelectionConfig::default());
        let local_map = local_indices
            .iter()
            .filter_map(|&idx| {
                let mp = map.map_points().get(idx)?;
                Some(TraceMapPoint {
                    map_point_idx: idx,
                    position: [mp.position.x, mp.position.y, mp.position.z],
                    descriptor: mp.descriptor,
                    min_distance: mp.min_distance,
                    max_distance: mp.max_distance,
                })
            })
            .collect();
        let tracks = tracks
            .iter()
            .map(|track| TraceTrack {
                track_id: track.id().as_u64(),
                pixel: track.pixel(),
                map_point_idx: track.map_point_idx(),
            })
            .collect();
        Self {
            frame_idx,
            timestamp_sec,
            candidate_pose,
            trusted_prediction,
            local_map,
            tracks,
            pose: candidate_pose,
            tracked: false,
            keyframe: false,
        }
    }
}
