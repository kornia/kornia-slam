//! Window-owned landmarks, host-anchored in a keyframe's camera 0, after
//! Basalt's landmark database (`landmark_database.h`, `.cpp:70-93`); they are
//! never shared with the map.

use std::collections::BTreeMap;

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::{Pose3d, triangulate_n_view_linear};
use kornia_algebra::Vec2F64;
use kornia_factors::reprojection::AnchoredLandmark;

use super::state::FrameId;

/// A landmark's identity: the camera-0 flow track it was created from.
pub(crate) type LandmarkId = u64;

/// Where an observation was made: frame and camera (0 left, 1 right).
pub(crate) type ObservationKey = (FrameId, u8);

#[derive(Debug, Clone)]
pub(crate) struct Landmark {
    pub host: FrameId,
    pub value: AnchoredLandmark,
    /// Rectified pixels by (frame, camera), the host's own included.
    pub observations: BTreeMap<ObservationKey, [f64; 2]>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct LandmarkStore {
    landmarks: BTreeMap<LandmarkId, Landmark>,
}

impl LandmarkStore {
    pub(crate) fn len(&self) -> usize {
        self.landmarks.len()
    }

    pub(crate) fn contains(&self, id: LandmarkId) -> bool {
        self.landmarks.contains_key(&id)
    }

    pub(crate) fn get(&self, id: LandmarkId) -> Option<&Landmark> {
        self.landmarks.get(&id)
    }

    pub(crate) fn get_mut(&mut self, id: LandmarkId) -> Option<&mut Landmark> {
        self.landmarks.get_mut(&id)
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (&LandmarkId, &Landmark)> {
        self.landmarks.iter()
    }

    pub(crate) fn insert(&mut self, id: LandmarkId, landmark: Landmark) {
        self.landmarks.insert(id, landmark);
    }

    pub(crate) fn remove(&mut self, id: LandmarkId) {
        self.landmarks.remove(&id);
    }

    /// Drops every observation made in a frame of `removed` and every
    /// landmark left with fewer than two (`landmark_database.cpp:70-93`).
    pub(crate) fn remove_frames(&mut self, removed: &[FrameId]) {
        self.landmarks.retain(|_, landmark| {
            landmark
                .observations
                .retain(|(frame, _), _| !removed.contains(frame));
            landmark.observations.len() >= 2
        });
    }

    /// Ids of the landmarks no observation of `frame` refers to.
    pub(crate) fn lost_in(&self, frame: FrameId) -> Vec<LandmarkId> {
        self.landmarks
            .iter()
            .filter(|(_, lm)| {
                !lm.observations.contains_key(&(frame, 0))
                    && !lm.observations.contains_key(&(frame, 1))
            })
            .map(|(id, _)| *id)
            .collect()
    }
}

/// The landmark seen at `host_pixel` by the host camera `host_cw`, from
/// the first `other` observation whose camera centre is at least
/// `min_baseline_m` from the host's (`sqrt_keypoint_vio.cpp:412-445`):
/// two-view DLT, accepted when finite with `0 < ρ < max_inverse_distance`.
/// `other` gives each candidate's world-to-camera pose and pixel, in order.
pub(crate) fn triangulate_from_host(
    host_cw: &Pose3d,
    host_pixel: [f64; 2],
    other: impl IntoIterator<Item = (Pose3d, [f64; 2])>,
    camera: &PinholeCamera,
    min_baseline_m: f64,
    max_inverse_distance: f64,
) -> Option<AnchoredLandmark> {
    let k = [
        camera.fx, 0.0, camera.cx, 0.0, camera.fy, camera.cy, 0.0, 0.0, 1.0,
    ];
    let projection = |pose: &Pose3d| -> [f64; 12] {
        let (r, t) = (pose.rotation, pose.translation);
        let rt = |row: usize, col: usize| match col {
            3 => t.to_array()[row],
            _ => r.row(row).to_array()[col],
        };
        std::array::from_fn(|i| {
            let (row, col) = (i / 4, i % 4);
            (0..3).map(|m| k[row * 3 + m] * rt(m, col)).sum()
        })
    };
    let host_projection = projection(&Pose3d::IDENTITY);
    for (pose_cw, pixel) in other {
        // The other camera in the host camera's frame: X_other = T · X_host.
        let other_from_host = pose_cw.compose(&host_cw.inverse());
        if other_from_host.inverse().translation.length() < min_baseline_m {
            continue;
        }
        let pixels = [
            Vec2F64::new(host_pixel[0], host_pixel[1]),
            Vec2F64::new(pixel[0], pixel[1]),
        ];
        let Some(point_h) =
            triangulate_n_view_linear(&pixels, &[host_projection, projection(&other_from_host)])
        else {
            continue;
        };
        if let Some(lm) = AnchoredLandmark::from_host_point(&point_h)
            && lm.inverse_distance.is_finite()
            && lm.inverse_distance > 0.0
            && lm.inverse_distance < max_inverse_distance
            && point_h.z > 0.0
        {
            return Some(lm);
        }
    }
    None
}
