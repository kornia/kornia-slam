//! Selection of the keyframes and IMU factors that make up one inertial
//! initialization window.
//!
//! Readiness and the solver used to disagree about the window. Readiness summed
//! every factor whose *target* keyframe was in range, so an edge arriving from a
//! keyframe before the window counted toward the duration gate — while the
//! solver skipped exactly those edges, because it needs both endpoints to index
//! a variable. A window that passed the gate on an ineligible edge therefore
//! reached a solve with less integrated time than the gate promised. Both now
//! read the same eligible set from here.

use crate::map::{ImuFactor, Keyframe, Map};

/// The keyframes at or after `start_idx`, plus the IMU factors that connect
/// them.
///
/// Membership is `frame.idx >= start_idx`, the same test the map iterator used.
/// Both orderings are served from this one set: readiness reads insertion order
/// for its first/last motion span, the solver reads frame-ID order to lay out
/// its variables.
pub(crate) struct InitWindow<'a> {
    in_insertion_order: Vec<&'a Keyframe>,
}

impl<'a> InitWindow<'a> {
    pub(crate) fn select(map: &'a Map, start_idx: usize) -> Self {
        Self {
            in_insertion_order: map
                .keyframes()
                .iter()
                .filter(|kf| kf.frame.idx >= start_idx)
                .collect(),
        }
    }

    /// Map insertion order — what readiness measures first-to-last motion over.
    pub(crate) fn in_insertion_order(&self) -> &[&'a Keyframe] {
        &self.in_insertion_order
    }

    /// Ascending `frame.idx` — the solver's variable ordering.
    pub(crate) fn sorted_by_frame_idx(&self) -> Vec<&'a Keyframe> {
        let mut sorted = self.in_insertion_order.clone();
        sorted.sort_by_key(|kf| kf.frame.idx);
        sorted
    }

    pub(crate) fn contains(&self, kf_idx: usize) -> bool {
        self.in_insertion_order
            .iter()
            .any(|kf| kf.frame.idx == kf_idx)
    }

    /// Factors the solver can actually use: both endpoints inside the window,
    /// and a duration that can contribute to a residual.
    pub(crate) fn eligible_imu_factors<'m>(
        &'m self,
        map: &'m Map,
    ) -> impl Iterator<Item = &'m ImuFactor> + 'm {
        map.imu_factors().iter().filter(move |factor| {
            self.contains(factor.prev_kf_idx)
                && self.contains(factor.curr_kf_idx)
                && factor.preintegrated.dt.is_finite()
                && factor.preintegrated.dt > 0.0
        })
    }

    /// Total preintegrated duration over the eligible factors.
    ///
    /// This is integrated IMU time, not elapsed wall time and not a guarantee of
    /// continuous coverage: a gap between two keyframes contributes nothing.
    pub(crate) fn integrated_imu_time_sec(&self, map: &Map) -> f64 {
        self.eligible_imu_factors(map)
            .map(|factor| factor.preintegrated.dt)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::Frame;
    use crate::map::Keyframe;
    use kornia_3d::pose::Pose3d;
    use kornia_image::ImageSize;
    use kornia_imgproc::features::OrbFeatures;
    use kornia_sensors::imu::{ImuCalib, PreintegratedImu};

    fn frame(idx: usize) -> Frame {
        Frame {
            idx,
            features: OrbFeatures {
                keypoints_xy: Vec::new(),
                orientations: Vec::new(),
                descriptors: Vec::new(),
                octaves: Vec::new(),
            },
            pose_world_to_cam: Pose3d::IDENTITY,
            image_size: ImageSize {
                width: 640,
                height: 480,
            },
            keypoint_colors: Vec::new(),
            u_right: Vec::new(),
            depth: Vec::new(),
            keypoints_undist: Vec::new(),
        }
    }

    fn calib() -> ImuCalib {
        ImuCalib {
            gyro_noise: 1.0e-4,
            accel_noise: 1.0e-3,
            gyro_bias_noise: 1.0e-5,
            accel_bias_noise: 1.0e-3,
        }
    }

    fn edge(map: &mut Map, prev: usize, curr: usize, dt: f64) {
        let mut pre = PreintegratedImu::new(Default::default(), calib());
        pre.dt = dt;
        map.add_imu_factor(prev, curr, pre, Vec::new(), 0.0, dt);
    }

    fn map_with_edges() -> Map {
        let mut map = Map::new();
        for idx in [9usize, 10, 11, 12] {
            map.insert_keyframe(Keyframe::from_frame(frame(idx)))
                .unwrap();
        }
        edge(&mut map, 9, 10, 0.5);
        edge(&mut map, 10, 11, 0.25);
        edge(&mut map, 11, 12, 0.25);
        map
    }

    #[test]
    fn incoming_edge_does_not_count_toward_window_duration() {
        let map = map_with_edges();
        let window = InitWindow::select(&map, 10);

        assert_eq!(window.in_insertion_order().len(), 3);
        let eligible: Vec<_> = window
            .eligible_imu_factors(&map)
            .map(|f| (f.prev_kf_idx, f.curr_kf_idx))
            .collect();
        // 9 -> 10 enters the window but starts outside it, so the solver cannot
        // use it and readiness must not credit its 0.5 s.
        assert_eq!(eligible, vec![(10, 11), (11, 12)]);
        assert!((window.integrated_imu_time_sec(&map) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn nonpositive_and_non_finite_durations_are_ineligible() {
        let mut map = map_with_edges();
        edge(&mut map, 12, 13, 0.0);
        edge(&mut map, 13, 14, f64::NAN);
        for idx in [13usize, 14] {
            map.insert_keyframe(Keyframe::from_frame(frame(idx)))
                .unwrap();
        }
        let window = InitWindow::select(&map, 10);

        let eligible: Vec<_> = window
            .eligible_imu_factors(&map)
            .map(|f| (f.prev_kf_idx, f.curr_kf_idx))
            .collect();
        assert_eq!(eligible, vec![(10, 11), (11, 12)]);
        assert!(window.integrated_imu_time_sec(&map).is_finite());
    }

    #[test]
    fn both_orderings_come_from_one_membership_set() {
        let mut map = Map::new();
        // Inserted out of frame-index order on purpose.
        for idx in [12usize, 10, 11] {
            map.insert_keyframe(Keyframe::from_frame(frame(idx)))
                .unwrap();
        }
        let window = InitWindow::select(&map, 10);

        let insertion: Vec<_> = window
            .in_insertion_order()
            .iter()
            .map(|kf| kf.frame.idx)
            .collect();
        let sorted: Vec<_> = window
            .sorted_by_frame_idx()
            .iter()
            .map(|kf| kf.frame.idx)
            .collect();

        assert_eq!(insertion, vec![12, 10, 11]);
        assert_eq!(sorted, vec![10, 11, 12]);
    }

    #[test]
    fn membership_is_inclusive_of_the_start_index() {
        let map = map_with_edges();
        assert!(InitWindow::select(&map, 10).contains(10));
        assert!(!InitWindow::select(&map, 10).contains(9));
    }
}
