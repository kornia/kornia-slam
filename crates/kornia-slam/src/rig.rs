//! Geometry of a non-rectified multi-camera rig: each camera's model, its pose
//! relative to the primary camera and the image region it shares with the
//! others (ORB-SLAM3's non-rectified stereo, `Frame` with `Nleft`/`Nright`).
//!
//! Poses tracked by the system stay those of the primary camera; a feature of
//! camera `c` is projected with `camera_from_primary[c] ∘ pose_world_to_primary`.

use kornia_3d::camera::{CameraModel, PinholeCamera};
use kornia_3d::pose::Pose3d;
use kornia_algebra::{Vec2F64, Vec3F64};
use kornia_image::ImageSize;
use kornia_sensors::{Fisheye624Camera, SensorRig};

/// Depths (metres) sampled when mapping where two cameras' views overlap.
const OVERLAP_DEPTHS_M: [f64; 5] = [1.0, 2.0, 5.0, 15.0, 50.0];
/// Pixel step of the overlap sampling grid.
const OVERLAP_GRID_STEP_PX: usize = 8;
/// Incidence cap (degrees) of raw fisheye views. The virtual pinhole radius
/// `f·tanθ` diverges at 90°, and past 80° one raw pixel spans over 40 virtual
/// pixels on the Aria lens, so wider rays would carry no usable geometry.
pub(crate) const RAW_MAX_INCIDENCE_DEG: f64 = 80.0;

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
    /// `T_PC` as the rig gave it, not the inverse of `camera_from_primary`, so
    /// rays of one camera share a bit-equal origin.
    pub primary_from_camera: Pose3d,
    /// Pixels of this camera that some other camera also sees, between 1 and
    /// 50 m; `None` when no other camera shares its view.
    pub overlap: Option<PixelBox>,
    /// Model of raw fisheye images ([`kornia_sensors::RigCamera::raw_model`]);
    /// `camera` is then a virtual pinhole and `size` the raw image size.
    pub raw_model: Option<Fisheye624Camera>,
}

impl RigView {
    /// Pose of this camera for a primary-camera pose (both world-to-camera).
    pub fn pose(&self, pose_world_to_primary: &Pose3d) -> Pose3d {
        self.camera_from_primary.compose(pose_world_to_primary)
    }

    /// Ray through an undistorted pixel of this camera in the primary frame:
    /// `(origin, unit direction)` ([`kornia_sensors::camera_ray`]).
    pub fn ray(&self, pixel_undist: [f64; 2]) -> (Vec3F64, Vec3F64) {
        kornia_sensors::camera_ray(&self.camera, &self.primary_from_camera, pixel_undist)
    }

    /// Whether this camera's images are raw fisheye.
    pub fn is_raw(&self) -> bool {
        self.raw_model.is_some()
    }

    /// Raw image pixel to virtual-pinhole pixel; `None` past the 80° cap or outside the lens.
    /// The identity without a raw model.
    pub fn raw_to_virtual(&self, raw: [f32; 2]) -> Option<[f32; 2]> {
        let Some(model) = &self.raw_model else {
            return Some(raw);
        };
        let bearing = unproject_capped(model, raw)?;
        let pixel = self.camera.project_to_pixel(&bearing, 0.0)?;
        Some([pixel.x as f32, pixel.y as f32])
    }

    /// Unit bearing of a raw pixel in this camera's frame; `None` as above.
    pub fn unproject_raw(&self, raw: [f32; 2]) -> Option<Vec3F64> {
        match &self.raw_model {
            Some(model) => unproject_capped(model, raw),
            None => Some(
                Vec3F64::new(
                    (raw[0] as f64 - self.camera.cx) / self.camera.fx,
                    (raw[1] as f64 - self.camera.cy) / self.camera.fy,
                    1.0,
                )
                .normalize(),
            ),
        }
    }

    /// Raw pixel of a camera-frame point; `None` behind the camera or past the cap.
    /// The pinhole projection without a raw model.
    pub fn project_raw(&self, p_cam: Vec3F64) -> Option<[f32; 2]> {
        let pixel = match &self.raw_model {
            Some(model) if within_raw_cap(&p_cam) => model.project(&p_cam)?,
            Some(_) => return None,
            None => self.camera.project_to_pixel(&p_cam, 0.0)?,
        };
        Some([pixel.x as f32, pixel.y as f32])
    }

    /// Incidence angle (rad) of an undistorted (virtual-pinhole) pixel.
    pub fn incidence_rad(&self, pixel_undist: [f32; 2]) -> f64 {
        let x = (pixel_undist[0] as f64 - self.camera.cx) / self.camera.fx;
        let y = (pixel_undist[1] as f64 - self.camera.cy) / self.camera.fy;
        x.hypot(y).atan()
    }

    /// (radial, tangential) raw-to-virtual magnification at incidence `theta`
    /// (rad), for this view's virtual focal length; `None` without a raw model.
    pub fn pinhole_magnification(&self, theta: f64) -> Option<(f64, f64)> {
        let model = self.raw_model.as_ref()?;
        let (radial, tangential) = model.pinhole_magnification(theta);
        let scale = self.camera.fx / model.f;
        Some((radial * scale, tangential * scale))
    }
}

fn within_raw_cap(p_cam: &Vec3F64) -> bool {
    p_cam.z > p_cam.length() * RAW_MAX_INCIDENCE_DEG.to_radians().cos()
}

fn unproject_capped(model: &Fisheye624Camera, raw: [f32; 2]) -> Option<Vec3F64> {
    let bearing = model.unproject(&Vec2F64::new(raw[0] as f64, raw[1] as f64))?;
    within_raw_cap(&bearing).then_some(bearing)
}

/// Virtual-pinhole pixel of a camera-frame point that the camera images.
///
/// Without a raw model, `camera.project_to_image`. With one, the point must lie
/// within the 80° cap and its raw projection inside `size`.
pub(crate) fn project_in_view(
    camera: &PinholeCamera,
    raw: Option<&Fisheye624Camera>,
    size: ImageSize,
    p_cam: &Vec3F64,
    min_depth: f64,
) -> Option<Vec2F64> {
    let Some(model) = raw else {
        return camera.project_to_image(p_cam, min_depth, size).ok();
    };
    if p_cam.z <= min_depth || !within_raw_cap(p_cam) {
        return None;
    }
    let pixel = model.project(p_cam)?;
    if pixel.x < 0.0
        || pixel.y < 0.0
        || pixel.x >= size.width as f64
        || pixel.y >= size.height as f64
    {
        return None;
    }
    camera.project_to_pixel(p_cam, min_depth)
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
                primary_from_camera: cam.primary_from_camera,
                overlap: None,
                raw_model: cam.raw_model,
            })
            .collect();
        let overlaps: Vec<Option<PixelBox>> =
            (0..views.len()).map(|c| overlap_box(&views, c)).collect();
        for (view, overlap) in views.iter_mut().zip(overlaps) {
            view.overlap = overlap;
        }
        Some(Self { views })
    }

    /// Whether any camera supplies raw fisheye images.
    pub fn has_raw_views(&self) -> bool {
        self.views.iter().any(RigView::is_raw)
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
            let ray = if view.is_raw() {
                // Scaled to z = 1, so `ray * depth` sits at z = depth as below.
                match view.unproject_raw([x as f32, y as f32]) {
                    Some(bearing) => bearing * (1.0 / bearing.z),
                    None => continue,
                }
            } else {
                Vec3F64::new(
                    (x as f64 - view.camera.cx) / view.camera.fx,
                    (y as f64 - view.camera.cy) / view.camera.fy,
                    1.0,
                )
            };
            let seen = views.iter().enumerate().any(|(o, other)| {
                o != c
                    && OVERLAP_DEPTHS_M.iter().any(|&depth| {
                        let p_primary = primary_from_c.transform_point(&(ray * depth));
                        let p_other = other.camera_from_primary.transform_point(&p_primary);
                        project_in_view(
                            &other.camera,
                            other.raw_model.as_ref(),
                            other.size,
                            &p_other,
                            0.05,
                        )
                        .is_some()
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
pub(crate) mod tests {
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
            raw_model: None,
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

    /// Aria-like lens (LaMAria R_11 `cam0` distortion) on [`camera`]'s intrinsics.
    pub(crate) fn raw_model() -> Fisheye624Camera {
        Fisheye624Camera {
            f: 240.0,
            cx: 320.0,
            cy: 240.0,
            k: [
                -0.02484243376501498,
                0.09628938752134535,
                -0.063325466945624,
                0.006200726341569274,
                0.003488761399963396,
                -0.0007297520056856479,
            ],
            p: [0.00089532032215904, 0.000361066580790014],
            s: [
                -0.00136879244387001,
                -0.0001054083415853531,
                -0.0004635611298488213,
                -0.0001049856461262643,
            ],
        }
    }

    pub(crate) fn raw_two_camera_rig(yaw_deg: f64) -> RigGeometry {
        let raw = |primary_from_camera| RigCamera {
            raw_model: Some(raw_model()),
            ..rig_camera(primary_from_camera)
        };
        let rig = SensorRig::new(camera()).with_rig_cameras(vec![
            raw(Pose3d::IDENTITY),
            raw(Pose3d::new(yaw(yaw_deg), Vec3F64::new(0.1, 0.0, 0.0))),
        ]);
        RigGeometry::from_rig(&rig).unwrap()
    }

    #[test]
    fn raw_fisheye_views_overlap_wider() {
        let rig = raw_two_camera_rig(75.0);
        assert!(rig.has_raw_views());
        assert!(!two_camera_rig(75.0).has_raw_views());
        let left = rig.view(0).unwrap().overlap.unwrap();
        assert!(left.min[0] < 345.0, "{left:?}");
    }

    #[test]
    fn raw_fisheye_project_in_view_sees_past_the_pinhole_frustum() {
        let size = ImageSize {
            width: 640,
            height: 480,
        };
        let (s, c) = 70f64.to_radians().sin_cos();
        let p = Vec3F64::new(s, 0.0, c) * 3.0;
        let raw = raw_model();
        let pixel = project_in_view(&camera(), Some(&raw), size, &p, 0.05).unwrap();
        assert!((pixel - camera().project_to_pixel(&p, 0.05).unwrap()).length() < 1e-12);
        assert!(project_in_view(&camera(), None, size, &p, 0.05).is_none());
        let (s, c) = 85f64.to_radians().sin_cos();
        assert!(
            project_in_view(&camera(), Some(&raw), size, &Vec3F64::new(s, 0.0, c), 0.0).is_none()
        );
    }

    #[test]
    fn raw_fisheye_view_maps_raw_pixels_to_the_virtual_pinhole() {
        let rig = raw_two_camera_rig(75.0);
        let view = rig.view(0).unwrap();
        let p = Vec3F64::new(0.9, -0.4, 1.0);
        let raw = view.project_raw(p).unwrap();
        let virtual_px = view.raw_to_virtual(raw).unwrap();
        let expected = view.camera.project_to_pixel(&p, 0.0).unwrap();
        assert!(
            (virtual_px[0] as f64 - expected.x).abs() < 1e-2,
            "{virtual_px:?} vs {expected:?}"
        );
        assert!(
            (virtual_px[1] as f64 - expected.y).abs() < 1e-2,
            "{virtual_px:?} vs {expected:?}"
        );
        let bearing = view.unproject_raw(raw).unwrap();
        assert!((bearing - p.normalize()).length() < 1e-5);
        assert!(view.project_raw(Vec3F64::new(1.0, 0.0, 0.05)).is_none());
        let (radial, tangential) = view.pinhole_magnification(70f64.to_radians()).unwrap();
        assert!(radial > tangential && tangential > 1.0);
        assert!(
            two_camera_rig(75.0)
                .view(0)
                .unwrap()
                .pinhole_magnification(1.0)
                .is_none()
        );
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
