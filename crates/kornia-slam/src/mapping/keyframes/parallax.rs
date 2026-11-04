//! Rotation-compensated parallax between a keyframe and the current frame,
//! the motion measure behind [`super::KeyframeInsertionStrategy::Parallax`].

use crate::frame::Frame;
use crate::mapping::map::Map;
use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, Vec3F64};

/// Median parallax, pixels, of the tracked features between keyframe
/// `keyframe_idx` and `current`, with the relative camera rotation removed.
///
/// `matches` are `(map point id, current feature index)` pairs, as tracking
/// produces them; a match whose landmark is retired or not observed by the
/// keyframe is skipped. The relative rotation comes from the keyframe's
/// world-to-camera pose and `current_pose_world_to_cam`, the tracked pose of
/// `current`. `None` when the keyframe is unknown or no match survives.
pub fn median_parallax_px(
    map: &Map,
    keyframe_idx: usize,
    current: &Frame,
    current_pose_world_to_cam: &Pose3d,
    matches: &[(usize, usize)],
    camera: &PinholeCamera,
) -> Option<f64> {
    let keyframe = map.get_keyframe(keyframe_idx)?;
    let rotation_cur_from_kf = current_pose_world_to_cam
        .compose(&keyframe.frame.pose_world_to_cam.inverse())
        .rotation;
    let pairs = matches.iter().filter_map(|&(mp_idx, feature)| {
        let mp = map.map_points().get(mp_idx).filter(|mp| !mp.culled)?;
        let kf_feature = mp
            .observations()
            .iter()
            .find(|o| o.key.keyframe_idx == keyframe_idx)?
            .key
            .feature_idx;
        Some((
            keyframe.frame.undistorted_xy(kf_feature, camera)?,
            current.undistorted_xy(feature, camera)?,
        ))
    });
    median_compensated_parallax_px(camera, &rotation_cur_from_kf, pairs)
}

/// Median pixel distance between each current-frame pixel and its keyframe
/// pixel rotated into the current camera (VINS-Mono's compensated parallax).
///
/// Each keyframe bearing is rotated by `rotation_cur_from_kf` and reprojected
/// before the distance is taken, so a pure rotation reads zero and only the
/// translation-induced parallax that triangulation needs remains. Pixels are
/// undistorted `(keyframe, current)` pairs. A bearing rotated behind the
/// camera is skipped; `None` when nothing is left.
pub fn median_compensated_parallax_px(
    camera: &PinholeCamera,
    rotation_cur_from_kf: &Mat3F64,
    pairs: impl IntoIterator<Item = ([f32; 2], [f32; 2])>,
) -> Option<f64> {
    let mut distances: Vec<f64> = pairs
        .into_iter()
        .filter_map(|(kf_px, cur_px)| {
            let bearing = Vec3F64::new(
                (kf_px[0] as f64 - camera.cx) / camera.fx,
                (kf_px[1] as f64 - camera.cy) / camera.fy,
                1.0,
            );
            let rotated = camera.project_to_pixel(&(*rotation_cur_from_kf * bearing), 1e-6)?;
            let d = (cur_px[0] as f64 - rotated.x).hypot(cur_px[1] as f64 - rotated.y);
            d.is_finite().then_some(d)
        })
        .collect();
    if distances.is_empty() {
        return None;
    }
    let mid = distances.len() / 2;
    distances.select_nth_unstable_by(mid, f64::total_cmp);
    Some(distances[mid])
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::mapping::map::tests::test_frame_with_pose;
    use crate::mapping::map::{Keyframe, LandmarkSeed, ObservationKey};
    use kornia_algebra::SO3F64;

    pub(crate) fn camera() -> PinholeCamera {
        PinholeCamera {
            fx: 400.0,
            fy: 400.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

    /// Landmarks 4 m in front of the camera at `kf_pose` (world to camera).
    fn points_before(kf_pose: &Pose3d) -> Vec<Vec3F64> {
        (0..9)
            .map(|i| {
                let in_cam =
                    Vec3F64::new((i % 3) as f64 * 0.5 - 0.5, (i / 3) as f64 * 0.4 - 0.4, 4.0);
                kf_pose.inverse().transform_point(&in_cam)
            })
            .collect()
    }

    fn points() -> Vec<Vec3F64> {
        points_before(&Pose3d::IDENTITY)
    }

    /// A frame at `pose` whose keypoints are the exact projections of `pts`.
    pub(crate) fn frame_seeing_points(idx: usize, pose: Pose3d, pts: &[Vec3F64]) -> Frame {
        let mut frame = test_frame_with_pose(idx, vec![[0; 32]; pts.len()], pose);
        for (i, p) in pts.iter().enumerate() {
            let px = camera()
                .project_to_pixel(&pose.transform_point(p), 1e-6)
                .unwrap();
            frame.features.keypoints_xy[i] = [px.x as f32, px.y as f32];
        }
        frame
    }

    /// A frame at `pose` whose keypoints are the exact projections of `points()`.
    fn frame_seeing(idx: usize, pose: Pose3d) -> Frame {
        frame_seeing_points(idx, pose, &points())
    }

    fn map_with_keyframe() -> Map {
        map_with_keyframe_at(Pose3d::IDENTITY)
    }

    /// A map whose keyframe 0 sits at `kf_pose` and observes
    /// `points_before(kf_pose)` as landmarks `0..9`, feature `i` = landmark `i`.
    pub(crate) fn map_with_keyframe_at(kf_pose: Pose3d) -> Map {
        let pts = points_before(&kf_pose);
        let mut map = Map::new();
        map.insert_keyframe(Keyframe::from_frame(frame_seeing_points(0, kf_pose, &pts)))
            .unwrap();
        for (i, p) in pts.into_iter().enumerate() {
            map.insert_landmark(LandmarkSeed {
                position: p,
                color: [0; 3],
                reference: ObservationKey {
                    keyframe_idx: 0,
                    feature_idx: i,
                },
            })
            .unwrap();
        }
        map
    }

    pub(crate) fn all_matches() -> Vec<(usize, usize)> {
        (0..points().len()).map(|i| (i, i)).collect()
    }

    #[test]
    fn a_pure_rotation_reads_zero_parallax() {
        let map = map_with_keyframe();
        let yaw = SO3F64::exp(Vec3F64::new(0.0, 0.08, 0.0)).matrix();
        let pose = Pose3d::new(yaw, Vec3F64::ZERO);
        let current = frame_seeing(5, pose);
        let parallax =
            median_parallax_px(&map, 0, &current, &pose, &all_matches(), &camera()).unwrap();
        // Uncompensated, a 0.08 rad yaw moves every pixel by about 32 px.
        assert!(parallax < 1e-3, "{parallax}");
    }

    #[test]
    fn a_pure_rotation_from_a_rotated_keyframe_reads_zero_parallax() {
        // The relative rotation is R_cur * R_kf^T. With the keyframe at
        // identity the reversed order R_kf^T * R_cur is the same matrix, so
        // the keyframe here is pitched, and the head then yaws on top of it:
        // the two orders differ by tens of pixels.
        let kf_pose = Pose3d::new(
            SO3F64::exp(Vec3F64::new(0.4, 0.0, 0.0)).matrix(),
            Vec3F64::new(0.3, -0.2, 0.1),
        );
        let map = map_with_keyframe_at(kf_pose);
        let yaw = Pose3d::new(
            SO3F64::exp(Vec3F64::new(0.0, 0.15, 0.0)).matrix(),
            Vec3F64::ZERO,
        );
        let pose = yaw.compose(&kf_pose);
        let current = frame_seeing_points(5, pose, &points_before(&kf_pose));
        let parallax =
            median_parallax_px(&map, 0, &current, &pose, &all_matches(), &camera()).unwrap();
        assert!(parallax < 1e-3, "{parallax}");
    }

    #[test]
    fn a_translation_reads_its_pixel_shift() {
        let map = map_with_keyframe();
        // 0.2 m sideways at 4 m depth with f = 400 px is 20 px for every point.
        let pose = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.2, 0.0, 0.0));
        let current = frame_seeing(5, pose);
        let parallax =
            median_parallax_px(&map, 0, &current, &pose, &all_matches(), &camera()).unwrap();
        assert!((parallax - 20.0).abs() < 1e-3, "{parallax}");
    }

    #[test]
    fn rotation_and_translation_leave_only_the_translation() {
        let map = map_with_keyframe();
        let r = SO3F64::exp(Vec3F64::new(0.02, -0.06, 0.03)).matrix();
        let t = Vec3F64::new(0.2, 0.0, 0.0);
        let pose = Pose3d::new(r, t);
        let current = frame_seeing(5, pose);
        let parallax =
            median_parallax_px(&map, 0, &current, &pose, &all_matches(), &camera()).unwrap();
        // Same translation as above, so about the same 20 px, give or take the
        // depth change the rotation causes.
        assert!((parallax - 20.0).abs() < 2.0, "{parallax}");
    }

    #[test]
    fn unusable_matches_and_unknown_keyframes_yield_none() {
        let mut map = map_with_keyframe();
        let current = frame_seeing(5, Pose3d::IDENTITY);
        assert_eq!(
            median_parallax_px(
                &map,
                9,
                &current,
                &Pose3d::IDENTITY,
                &all_matches(),
                &camera()
            ),
            None
        );
        map.remove_landmark(0).unwrap();
        assert_eq!(
            median_parallax_px(
                &map,
                0,
                &current,
                &Pose3d::IDENTITY,
                &[(0, 0), (99, 1)],
                &camera()
            ),
            None
        );
        assert_eq!(
            median_compensated_parallax_px(&camera(), &Mat3F64::IDENTITY, []),
            None
        );
    }
}
