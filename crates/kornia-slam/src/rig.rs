//! Geometry of a non-rectified multi-camera rig: each camera's model, its pose
//! relative to the primary camera and the image region it shares with the
//! others (ORB-SLAM3's non-rectified stereo, `Frame` with `Nleft`/`Nright`).
//!
//! Poses tracked by the system stay those of the primary camera; a feature of
//! camera `c` is projected with `camera_from_primary[c] ∘ pose_world_to_primary`.

use kornia_3d::camera::PinholeCamera;
use kornia_3d::camera::{Camera, CameraModel};
use kornia_3d::pose::Pose3d;
use kornia_algebra::{Vec2F64, Vec3F64};
use kornia_image::ImageSize;
use kornia_sensors::{RigCamera, SensorRig};

/// Depths (metres) sampled when mapping where two cameras' views overlap.
const OVERLAP_DEPTHS_M: [f64; 5] = [1.0, 2.0, 5.0, 15.0, 50.0];
/// Pixel step of the overlap sampling grid.
const OVERLAP_GRID_STEP_PX: usize = 8;
/// Nearest point (metres) a camera counts as imaging.
const OVERLAP_MIN_DEPTH_M: f64 = 0.05;
/// Incidence cap (degrees) of raw fisheye views. The virtual pinhole radius
/// `f·tanθ` diverges at 90°, and past 80° one raw pixel spans over 40 virtual
/// pixels on the Aria lens, so wider rays would carry no usable geometry.
pub(crate) const RAW_MAX_INCIDENCE_DEG: f64 = 80.0;

/// An axis-aligned box `[x0, y0, x1, y1)` in raw image pixels.
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
#[derive(Debug, Clone, Copy)]
pub struct RigView<'a> {
    pub camera: &'a RigCamera,
    /// [`RigCamera::pinhole`] of `camera`, the pinhole keypoints are measured in.
    pub pinhole: &'a PinholeCamera,
    /// `T_CP`: `X_camera = T_CP * X_primary`.
    pub camera_from_primary: &'a Pose3d,
    /// Raw image pixels of this camera that some other camera also sees,
    /// between 1 and 50 m, to test raw keypoints against; `None` when no other
    /// camera shares its view.
    pub overlap: Option<PixelBox>,
}

impl RigView<'_> {
    /// Pose of this camera for a primary-camera pose (both world-to-camera).
    pub fn pose(&self, pose_world_to_primary: &Pose3d) -> Pose3d {
        self.camera_from_primary.compose(pose_world_to_primary)
    }

    /// Whether this camera's images are raw fisheye; `pinhole` is then its
    /// virtual pinhole and `camera.size` the raw image size.
    pub fn is_raw(&self) -> bool {
        self.camera.is_fisheye()
    }

    /// The raw fisheye model, `None` for a pinhole camera.
    pub fn fisheye_model(&self) -> Option<&Camera> {
        self.is_raw().then_some(&self.camera.model)
    }

    /// Raw image pixel to virtual-pinhole pixel; `None` past the 80° cap or
    /// outside the lens. The identity for a pinhole camera.
    pub fn raw_to_virtual(&self, raw: [f32; 2]) -> Option<[f32; 2]> {
        let Some(model) = self.fisheye_model() else {
            return Some(raw);
        };
        let bearing = model.unproject(&Vec2F64::new(raw[0] as f64, raw[1] as f64))?;
        if !within_raw_cap(&bearing) {
            return None;
        }
        let pixel = self.pinhole.project_to_pixel(&bearing, 0.0)?;
        Some([pixel.x as f32, pixel.y as f32])
    }
}

fn within_raw_cap(p_cam: &Vec3F64) -> bool {
    p_cam.z > p_cam.length() * RAW_MAX_INCIDENCE_DEG.to_radians().cos()
}

/// Virtual-pinhole pixel of a camera-frame point that the camera images.
///
/// Without a raw model, `camera.project_to_image`. With one, the point must lie
/// within the 80° cap and its raw projection inside `size`.
pub(crate) fn project_in_view(
    camera: &PinholeCamera,
    raw: Option<&Camera>,
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

/// Every camera of a multi-camera rig, the primary first, with the per-camera
/// values tracking reads on every frame computed once.
#[derive(Debug, Clone)]
pub struct RigGeometry {
    cameras: Vec<RigCamera>,
    pinholes: Vec<PinholeCamera>,
    camera_from_primary: Vec<Pose3d>,
    overlap: Vec<Option<PixelBox>>,
}

impl RigGeometry {
    /// `None` unless the rig has more than one camera.
    pub fn from_rig(rig: &SensorRig) -> Option<Self> {
        if rig.cameras().len() < 2 {
            return None;
        }
        let cameras = rig.cameras().to_vec();
        let mut geometry = Self {
            pinholes: cameras.iter().map(RigCamera::pinhole).collect(),
            camera_from_primary: cameras.iter().map(RigCamera::camera_from_primary).collect(),
            overlap: vec![None; cameras.len()],
            cameras,
        };
        geometry.overlap = (0..geometry.cameras.len())
            .map(|c| overlap_box(&geometry, c))
            .collect();
        Some(geometry)
    }

    /// Whether any camera supplies raw fisheye images.
    pub fn has_raw_views(&self) -> bool {
        self.cameras.iter().any(RigCamera::is_fisheye)
    }

    /// Whether some camera's pixels are not ideal pinhole pixels: a raw
    /// fisheye view, or a pinhole with radtan distortion (EuRoC).
    pub fn needs_undistortion(&self) -> bool {
        self.has_raw_views()
            || self
                .pinholes
                .iter()
                .any(|c| c.k1 != 0.0 || c.k2 != 0.0 || c.p1 != 0.0 || c.p2 != 0.0)
    }

    pub fn len(&self) -> usize {
        self.cameras.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cameras.is_empty()
    }

    pub fn views(&self) -> impl ExactSizeIterator<Item = RigView<'_>> {
        (0..self.cameras.len()).map(|c| self.view_unchecked(c))
    }

    pub fn view(&self, camera: usize) -> Option<RigView<'_>> {
        (camera < self.cameras.len()).then(|| self.view_unchecked(camera))
    }

    fn view_unchecked(&self, camera: usize) -> RigView<'_> {
        RigView {
            camera: &self.cameras[camera],
            pinhole: &self.pinholes[camera],
            camera_from_primary: &self.camera_from_primary[camera],
            overlap: self.overlap[camera],
        }
    }
}

/// Whether `view` images the camera-frame point `p_cam` inside its raw image
/// (and, for a raw fisheye view, within its incidence cap).
fn images(view: &RigView<'_>, p_cam: &Vec3F64) -> bool {
    let size = view.camera.size;
    p_cam.z > OVERLAP_MIN_DEPTH_M
        && (!view.is_raw() || within_raw_cap(p_cam))
        && view.camera.model.project(p_cam).is_some_and(|pixel| {
            pixel.x >= 0.0
                && pixel.y >= 0.0
                && pixel.x < size.width as f64
                && pixel.y < size.height as f64
        })
}

/// Bounding box of the raw pixels of camera `c` whose rays, through the
/// camera's model and at some sampled depth, land inside another camera's
/// raw image.
fn overlap_box(rig: &RigGeometry, c: usize) -> Option<PixelBox> {
    let view = rig.view(c)?;
    let primary_from_c = &view.camera.primary_from_camera;
    let size = view.camera.size;
    let mut found: Option<PixelBox> = None;
    for y in (0..size.height).step_by(OVERLAP_GRID_STEP_PX) {
        for x in (0..size.width).step_by(OVERLAP_GRID_STEP_PX) {
            // Scaled to z = 1, so `ray * depth` sits at z = depth.
            let ray = match view
                .camera
                .model
                .unproject(&Vec2F64::new(x as f64, y as f64))
            {
                Some(bearing)
                    if bearing.z > 0.0 && (!view.is_raw() || within_raw_cap(&bearing)) =>
                {
                    bearing * (1.0 / bearing.z)
                }
                _ => continue,
            };
            let seen = rig.views().enumerate().any(|(o, other)| {
                o != c
                    && OVERLAP_DEPTHS_M.iter().any(|&depth| {
                        let p_primary = primary_from_c.transform_point(&(ray * depth));
                        let p_other = other.camera_from_primary.transform_point(&p_primary);
                        images(&other, &p_other)
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
    use kornia_3d::camera::Fisheye624Camera;
    use kornia_algebra::Mat3F64;

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

    const SIZE: ImageSize = ImageSize {
        width: 640,
        height: 480,
    };

    pub(crate) fn yaw(deg: f64) -> Mat3F64 {
        let (s, c) = deg.to_radians().sin_cos();
        Mat3F64::from_cols(
            Vec3F64::new(c, 0.0, -s),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(s, 0.0, c),
        )
    }

    fn rig_of(cameras: Vec<RigCamera>) -> RigGeometry {
        RigGeometry::from_rig(&SensorRig::from_cameras(cameras).unwrap()).unwrap()
    }

    pub(crate) fn two_camera_rig(yaw_deg: f64) -> RigGeometry {
        rig_of(vec![
            RigCamera::primary(camera(), SIZE),
            RigCamera::new(
                camera(),
                SIZE,
                Pose3d::new(yaw(yaw_deg), Vec3F64::new(0.1, 0.0, 0.0)),
            ),
        ])
    }

    /// Aria-like lens (LaMAria R_11 `cam0` distortion) on [`camera`]'s intrinsics.
    pub(crate) fn aria_lens() -> Fisheye624Camera {
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
        rig_of(vec![
            RigCamera::primary(aria_lens(), SIZE),
            RigCamera::new(
                aria_lens(),
                SIZE,
                Pose3d::new(yaw(yaw_deg), Vec3F64::new(0.1, 0.0, 0.0)),
            ),
        ])
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
        let (s, c) = 70f64.to_radians().sin_cos();
        let p = Vec3F64::new(s, 0.0, c) * 3.0;
        let raw = Camera::from(aria_lens());
        let pixel = project_in_view(&camera(), Some(&raw), SIZE, &p, 0.05).unwrap();
        assert!((pixel - camera().project_to_pixel(&p, 0.05).unwrap()).length() < 1e-12);
        assert!(project_in_view(&camera(), None, SIZE, &p, 0.05).is_none());
        let (s, c) = 85f64.to_radians().sin_cos();
        assert!(
            project_in_view(&camera(), Some(&raw), SIZE, &Vec3F64::new(s, 0.0, c), 0.0).is_none()
        );
    }

    #[test]
    fn raw_fisheye_view_maps_raw_pixels_to_the_virtual_pinhole() {
        let rig = raw_two_camera_rig(75.0);
        let view = rig.view(0).unwrap();
        let p = Vec3F64::new(0.9, -0.4, 1.0);
        let raw = view.camera.model.project(&p).unwrap();
        let virtual_px = view.raw_to_virtual([raw.x as f32, raw.y as f32]).unwrap();
        let expected = view.pinhole.project_to_pixel(&p, 0.0).unwrap();
        assert!(
            (virtual_px[0] as f64 - expected.x).abs() < 1e-2,
            "{virtual_px:?} vs {expected:?}"
        );
        assert!(
            (virtual_px[1] as f64 - expected.y).abs() < 1e-2,
            "{virtual_px:?} vs {expected:?}"
        );
        let past_cap = view.camera.model.project(&Vec3F64::new(1.0, 0.0, 0.05));
        assert!(
            past_cap.is_none_or(|px| view.raw_to_virtual([px.x as f32, px.y as f32]).is_none())
        );
    }

    #[test]
    fn a_single_camera_is_not_a_rig() {
        assert!(RigGeometry::from_rig(&SensorRig::mono(camera(), SIZE)).is_none());
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
    fn only_raw_or_distorted_views_need_undistortion() {
        assert!(!two_camera_rig(75.0).needs_undistortion());
        assert!(raw_two_camera_rig(75.0).needs_undistortion());
        let radtan = PinholeCamera {
            k1: -0.28,
            ..camera()
        };
        let rig = rig_of(vec![
            RigCamera::primary(camera(), SIZE),
            RigCamera::new(
                radtan,
                SIZE,
                Pose3d::new(yaw(0.0), Vec3F64::new(0.1, 0.0, 0.0)),
            ),
        ]);
        assert!(rig.needs_undistortion());
    }

    /// Two EuRoC radtan cameras, camera 1 yawed 40 deg and 0.11 m along x.
    fn euroc_radtan_rig() -> RigGeometry {
        let euroc = PinholeCamera {
            fx: 458.654,
            fy: 457.296,
            cx: 367.215,
            cy: 248.375,
            k1: -0.28340811,
            k2: 0.07395907,
            p1: 0.00019359,
            p2: 1.76187114e-05,
        };
        let size = ImageSize {
            width: 752,
            height: 480,
        };
        rig_of(vec![
            RigCamera::primary(euroc.clone(), size),
            RigCamera::new(
                euroc,
                size,
                Pose3d::new(yaw(40.0), Vec3F64::new(0.11, 0.0, 0.0)),
            ),
        ])
    }

    #[test]
    fn a_point_both_distorted_cameras_see_lies_in_both_boxes() {
        let rig = euroc_radtan_rig();
        let views: Vec<RigView<'_>> = rig.views().collect();
        let inside = |view: &RigView<'_>, p_cam: &Vec3F64| {
            let pixel = view.camera.model.project(p_cam)?;
            let size = view.camera.size;
            let margin = 8.0;
            (pixel.x >= margin
                && pixel.y >= margin
                && pixel.x < size.width as f64 - margin
                && pixel.y < size.height as f64 - margin)
                .then_some([pixel.x as f32, pixel.y as f32])
        };
        let mut checked = 0;
        for depth in [1.0, 2.0, 5.0, 15.0] {
            for i in -40..=40 {
                for j in -25..=25 {
                    let p_primary =
                        Vec3F64::new(i as f64 * 0.025 * depth, j as f64 * 0.025 * depth, depth);
                    let seen: Vec<Option<[f32; 2]>> = views
                        .iter()
                        .map(|view| {
                            let p_cam = view.camera_from_primary.transform_point(&p_primary);
                            if p_cam.z <= 0.0 {
                                return None;
                            }
                            inside(view, &p_cam)
                        })
                        .collect();
                    let (Some(pixel0), Some(pixel1)) = (seen[0], seen[1]) else {
                        continue;
                    };
                    checked += 1;
                    for (view, pixel) in views.iter().zip([pixel0, pixel1]) {
                        let overlap = view.overlap.unwrap();
                        assert!(
                            overlap.contains(pixel),
                            "{p_primary:?} at {pixel:?} outside {overlap:?}"
                        );
                    }
                }
            }
        }
        assert!(checked > 100, "only {checked} points in both images");
    }

    #[test]
    fn cameras_facing_apart_share_no_view() {
        let rig = two_camera_rig(180.0);
        assert!(rig.view(0).unwrap().overlap.is_none());
    }

    #[test]
    fn views_cache_each_cameras_pinhole_and_inverse_extrinsic() {
        let rig = two_camera_rig(30.0);
        for view in rig.views() {
            let pinhole = view.camera.pinhole();
            assert_eq!(
                [
                    view.pinhole.fx,
                    view.pinhole.fy,
                    view.pinhole.cx,
                    view.pinhole.cy
                ],
                [pinhole.fx, pinhole.fy, pinhole.cx, pinhole.cy]
            );
            assert_eq!(view.camera_from_primary, &view.camera.camera_from_primary());
        }
        assert!(rig.view(2).is_none());
    }
}
