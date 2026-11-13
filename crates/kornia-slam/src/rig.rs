//! Geometry of a non-rectified multi-camera rig: each camera's model, its pose
//! relative to the primary camera and the image region it shares with the
//! others (ORB-SLAM3's non-rectified stereo, `Frame` with `Nleft`/`Nright`).
//!
//! Poses tracked by the system stay those of the primary camera; a feature of
//! camera `c` is projected with `camera_from_primary[c] ∘ pose_world_to_primary`.

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;
use kornia_image::ImageSize;
use kornia_sensors::SensorRig;

/// Depths (metres) sampled when mapping where two cameras' views overlap.
const OVERLAP_DEPTHS_M: [f64; 5] = [1.0, 2.0, 5.0, 15.0, 50.0];
/// Pixel step of the overlap sampling grid.
const OVERLAP_GRID_STEP_PX: usize = 8;

/// An axis-aligned pixel box `[x0, y0, x1, y1)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PixelBox {
    pub min: [f32; 2],
    pub max: [f32; 2],
}

impl PixelBox {
    pub fn contains(&self, xy: [f32; 2]) -> bool {
        xy[0] >= self.min[0] && xy[0] < self.max[0] && xy[1] >= self.min[1] && xy[1] < self.max[1]
    }
}

/// One camera of the rig as tracking sees it.
#[derive(Debug, Clone)]
pub struct RigView {
    pub camera: PinholeCamera,
    pub size: ImageSize,
    /// `T_CP`: `X_camera = T_CP * X_primary`.
    pub camera_from_primary: Pose3d,
    /// Pixels of this camera that some other camera also sees, between 1 and
    /// 50 m; `None` when no other camera shares its view.
    pub overlap: Option<PixelBox>,
}

impl RigView {
    /// Pose of this camera for a primary-camera pose (both world-to-camera).
    pub fn pose(&self, pose_world_to_primary: &Pose3d) -> Pose3d {
        self.camera_from_primary.compose(pose_world_to_primary)
    }
}

/// Every camera of a multi-camera rig, the primary first.
#[derive(Debug, Clone)]
pub struct RigGeometry {
    views: Vec<RigView>,
}

impl RigGeometry {
    /// `None` unless the rig has more than one camera.
    pub fn from_rig(rig: &SensorRig) -> Option<Self> {
        if !rig.is_multi_camera() {
            return None;
        }
        let mut views: Vec<RigView> = rig
            .cameras
            .iter()
            .map(|cam| RigView {
                camera: cam.camera.clone(),
                size: cam.size,
                camera_from_primary: cam.primary_from_camera.inverse(),
                overlap: None,
            })
            .collect();
        let overlaps: Vec<Option<PixelBox>> =
            (0..views.len()).map(|c| overlap_box(&views, c)).collect();
        for (view, overlap) in views.iter_mut().zip(overlaps) {
            view.overlap = overlap;
        }
        Some(Self { views })
    }

    pub fn views(&self) -> &[RigView] {
        &self.views
    }

    pub fn view(&self, camera: usize) -> Option<&RigView> {
        self.views.get(camera)
    }

    /// `T_BA`: maps points in camera `a` into camera `b`.
    pub fn b_from_a(&self, a: usize, b: usize) -> Option<Pose3d> {
        let a = self.views.get(a)?;
        let b = self.views.get(b)?;
        Some(
            b.camera_from_primary
                .compose(&a.camera_from_primary.inverse()),
        )
    }
}

/// Bounding box of the pixels of camera `c` whose rays, at some sampled
/// depth, land inside another camera's image.
fn overlap_box(views: &[RigView], c: usize) -> Option<PixelBox> {
    let view = &views[c];
    let primary_from_c = view.camera_from_primary.inverse();
    let mut found: Option<PixelBox> = None;
    for y in (0..view.size.height).step_by(OVERLAP_GRID_STEP_PX) {
        for x in (0..view.size.width).step_by(OVERLAP_GRID_STEP_PX) {
            let ray = Vec3F64::new(
                (x as f64 - view.camera.cx) / view.camera.fx,
                (y as f64 - view.camera.cy) / view.camera.fy,
                1.0,
            );
            let seen = views.iter().enumerate().any(|(o, other)| {
                o != c
                    && OVERLAP_DEPTHS_M.iter().any(|&depth| {
                        let p_primary = primary_from_c.transform_point(&(ray * depth));
                        let p_other = other.camera_from_primary.transform_point(&p_primary);
                        other
                            .camera
                            .project_to_image(&p_other, 0.05, other.size)
                            .is_ok()
                    })
            });
            if seen {
                let (x, y) = (x as f32, y as f32);
                let step = OVERLAP_GRID_STEP_PX as f32;
                found = Some(match found {
                    None => PixelBox {
                        min: [x, y],
                        max: [x + step, y + step],
                    },
                    Some(b) => PixelBox {
                        min: [b.min[0].min(x), b.min[1].min(y)],
                        max: [b.max[0].max(x + step), b.max[1].max(y + step)],
                    },
                });
            }
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_algebra::Mat3F64;
    use kornia_sensors::RigCamera;

    fn camera() -> PinholeCamera {
        PinholeCamera {
            fx: 240.0,
            fy: 240.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

    fn rig_camera(primary_from_camera: Pose3d) -> RigCamera {
        RigCamera {
            camera: camera(),
            size: ImageSize {
                width: 640,
                height: 480,
            },
            primary_from_camera,
        }
    }

    fn yaw(deg: f64) -> Mat3F64 {
        let (s, c) = deg.to_radians().sin_cos();
        Mat3F64::from_cols(
            Vec3F64::new(c, 0.0, -s),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(s, 0.0, c),
        )
    }

    fn two_camera_rig(yaw_deg: f64) -> RigGeometry {
        let rig = SensorRig::new(camera()).with_rig_cameras(vec![
            rig_camera(Pose3d::IDENTITY),
            rig_camera(Pose3d::new(yaw(yaw_deg), Vec3F64::new(0.1, 0.0, 0.0))),
        ]);
        RigGeometry::from_rig(&rig).unwrap()
    }

    #[test]
    fn a_single_camera_is_not_a_rig() {
        let rig = SensorRig::new(camera()).with_rig_cameras(vec![rig_camera(Pose3d::IDENTITY)]);
        assert!(RigGeometry::from_rig(&rig).is_none());
    }

    #[test]
    fn diverging_cameras_overlap_on_their_inner_sides() {
        // Camera 1 is yawed 75 deg towards +x (its optical axis turns right),
        // so only the right part of camera 0 and the left of camera 1 overlap.
        let rig = two_camera_rig(75.0);
        let left = rig.view(0).unwrap().overlap.unwrap();
        let right = rig.view(1).unwrap().overlap.unwrap();
        assert!(left.min[0] > 200.0, "{left:?}");
        assert!(right.max[0] < 440.0, "{right:?}");
    }

    #[test]
    fn cameras_facing_apart_share_no_view() {
        let rig = two_camera_rig(180.0);
        assert!(rig.view(0).unwrap().overlap.is_none());
    }

    #[test]
    fn b_from_a_chains_through_the_primary() {
        let rig = two_camera_rig(30.0);
        let p0 = Vec3F64::new(0.3, -0.2, 4.0);
        let p1 = rig.b_from_a(0, 1).unwrap().transform_point(&p0);
        let back = rig.b_from_a(1, 0).unwrap().transform_point(&p1);
        assert!((back - p0).length() < 1e-12);
        let view1 = rig.view(1).unwrap();
        assert!((view1.camera_from_primary.transform_point(&p0) - p1).length() < 1e-12);
    }
}
