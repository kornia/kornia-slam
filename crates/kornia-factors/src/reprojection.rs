//! Pinhole reprojection into a camera pose `T_cw` perturbed on the right,
//! `T_cw ← T_cw · Exp([ρ; ω])`: of a fixed world point, or of a landmark
//! anchored in a host camera by bearing and inverse distance.

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::{SMatF64, SO3F64, SVecF64, Vec3F64};

/// The pixel residual `π(T_cw · p_w) − u` and its Jacobian with respect to the
/// pose, or `None` when the point is not more than `min_depth` in front of the
/// camera. `pixel` is undistorted.
pub fn residual_and_pose_jacobian(
    pose_cw: &Pose3d,
    point_w: &Vec3F64,
    pixel: [f64; 2],
    camera: &PinholeCamera,
    min_depth: f64,
) -> Option<(SVecF64<2>, SMatF64<2, 6>)> {
    let p_c = pose_cw.transform_point(point_w);
    if p_c.z <= min_depth {
        return None;
    }
    let inv_z = 1.0 / p_c.z;
    let residual = SVecF64::<2>::new(
        camera.fx * p_c.x * inv_z + camera.cx - pixel[0],
        camera.fy * p_c.y * inv_z + camera.cy - pixel[1],
    );
    let d_pixel_d_cam = SMatF64::<2, 3>::new(
        camera.fx * inv_z,
        0.0,
        -camera.fx * p_c.x * inv_z * inv_z,
        0.0,
        camera.fy * inv_z,
        -camera.fy * p_c.y * inv_z * inv_z,
    );
    // Exp([ρ; ω])·p ≈ p + ρ + ω×p, so ∂p_c/∂ρ = R and ∂p_c/∂ω = −R·[p_w]×.
    let rotation: SMatF64<3, 3> = pose_cw.rotation.into();
    let rotation_hat: SMatF64<3, 3> = (pose_cw.rotation * SO3F64::hat(*point_w) * -1.0).into();
    let mut jacobian = SMatF64::<2, 6>::zeros();
    jacobian
        .fixed_view_mut::<2, 3>(0, 0)
        .copy_from(&(d_pixel_d_cam * rotation));
    jacobian
        .fixed_view_mut::<2, 3>(0, 3)
        .copy_from(&(d_pixel_d_cam * rotation_hat));
    Some((residual, jacobian))
}

/// A landmark anchored in its host camera: the stereographic coordinates of
/// its unit bearing and its inverse distance (1/m) along it, as Basalt's
/// keypoint landmarks (`landmark_database.h:55-66`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnchoredLandmark {
    pub bearing: [f64; 2],
    pub inverse_distance: f64,
}

impl AnchoredLandmark {
    /// The landmark at `point_h`, a point in the host camera frame; `None`
    /// at the camera centre or straight behind it, where the stereographic
    /// chart is singular.
    pub fn from_host_point(point_h: &Vec3F64) -> Option<Self> {
        let distance = point_h.length();
        if !distance.is_finite() || distance <= 0.0 {
            return None;
        }
        let d = *point_h * (1.0 / distance);
        let denominator = 1.0 + d.z;
        (denominator > 1e-9).then(|| Self {
            bearing: [d.x / denominator, d.y / denominator],
            inverse_distance: 1.0 / distance,
        })
    }

    /// The unit bearing `[2u, 2v, 1 − u² − v²] / (1 + u² + v²)`.
    pub fn bearing_vector(&self) -> Vec3F64 {
        stereographic(self.bearing).0
    }
}

/// The stereographic bearing of `[u, v]` and its 3×2 Jacobian.
fn stereographic(uv: [f64; 2]) -> (Vec3F64, SMatF64<3, 2>) {
    let [u, v] = uv;
    let n = 1.0 + u * u + v * v;
    let inv_n = 1.0 / n;
    let inv_n2 = inv_n * inv_n;
    let bearing = Vec3F64::new(
        2.0 * u * inv_n,
        2.0 * v * inv_n,
        (1.0 - u * u - v * v) * inv_n,
    );
    let jacobian = SMatF64::<3, 2>::new(
        2.0 * (n - 2.0 * u * u) * inv_n2,
        -4.0 * u * v * inv_n2,
        -4.0 * u * v * inv_n2,
        2.0 * (n - 2.0 * v * v) * inv_n2,
        -4.0 * u * inv_n2,
        -4.0 * v * inv_n2,
    );
    (bearing, jacobian)
}

/// One anchored observation: the pixel residual and its Jacobians with
/// respect to the target pose (right perturbation) and the landmark
/// `[u, v, ρ]`. The host pose's Jacobian is `−d_target`.
#[derive(Debug, Clone, Copy)]
pub struct AnchoredRow {
    pub residual: SVecF64<2>,
    pub d_target: SMatF64<2, 6>,
    pub d_landmark: SMatF64<2, 3>,
}

/// `(q_w, R_cw, ρ t_cw)`: the landmark's world point times its inverse
/// distance, `R_wh · bearing + ρ · t_wh`, finite at `ρ = 0`, and the target
/// pose with its translation scaled the same way, so `π` sees the scaled
/// camera point.
fn scaled_point_and_pose(
    host_cw: &Pose3d,
    target_cw: &Pose3d,
    lm: &AnchoredLandmark,
) -> (Vec3F64, Pose3d) {
    let host_wc = host_cw.inverse();
    let rho = lm.inverse_distance;
    let q_w = host_wc.rotation * lm.bearing_vector() + host_wc.translation * rho;
    (
        q_w,
        Pose3d::new(target_cw.rotation, target_cw.translation * rho),
    )
}

/// The residual `π(T_cw(target) · p_w) − pixel` of a landmark anchored in
/// `host_cw` (the host camera's world-to-camera pose); `target_cw` is the
/// observing camera's. `None` for a negative inverse distance or a point
/// not more than `min_depth` in front of the target camera.
pub fn anchored_residual(
    host_cw: &Pose3d,
    target_cw: &Pose3d,
    lm: &AnchoredLandmark,
    pixel: [f64; 2],
    camera: &PinholeCamera,
    min_depth: f64,
) -> Option<SVecF64<2>> {
    anchored_residual_and_jacobians(host_cw, target_cw, lm, pixel, camera, min_depth)
        .map(|row| row.residual)
}

/// [`anchored_residual`] and its Jacobians. The scaled camera point is
/// `s = R_cw q_w + ρ t_cw`, so the target's translation columns of
/// [`residual_and_pose_jacobian`] at `(R_cw, ρ t_cw)` and `q_w` are scaled
/// by `ρ`; `∂s/∂(u, v) = R_cw R_wh ∂bearing` and `∂s/∂ρ = t_ch`.
pub fn anchored_residual_and_jacobians(
    host_cw: &Pose3d,
    target_cw: &Pose3d,
    lm: &AnchoredLandmark,
    pixel: [f64; 2],
    camera: &PinholeCamera,
    min_depth: f64,
) -> Option<AnchoredRow> {
    let rho = lm.inverse_distance;
    if !rho.is_finite() || rho < 0.0 {
        return None;
    }
    let (q_w, scaled_pose) = scaled_point_and_pose(host_cw, target_cw, lm);
    let (residual, pose_jacobian) =
        residual_and_pose_jacobian(&scaled_pose, &q_w, pixel, camera, min_depth * rho)?;
    let rotation: SMatF64<3, 3> = target_cw.rotation.into();
    // `pose_jacobian[:, 0..3] = dπ · R_cw`.
    let d_pi = pose_jacobian.fixed_view::<2, 3>(0, 0) * rotation.transpose();
    let mut d_target = pose_jacobian;
    d_target
        .fixed_view_mut::<2, 3>(0, 0)
        .copy_from(&(pose_jacobian.fixed_view::<2, 3>(0, 0) * rho));

    let host_wc = host_cw.inverse();
    let (_, d_bearing) = stereographic(lm.bearing);
    let r_ch: SMatF64<3, 3> = (target_cw.rotation * host_wc.rotation).into();
    let t_ch: SVecF64<3> =
        (target_cw.rotation * host_wc.translation + target_cw.translation).into();
    let mut d_landmark = SMatF64::<2, 3>::zeros();
    d_landmark
        .fixed_view_mut::<2, 2>(0, 0)
        .copy_from(&(d_pi * r_ch * d_bearing));
    d_landmark
        .fixed_view_mut::<2, 1>(0, 2)
        .copy_from(&(d_pi * t_ch));
    Some(AnchoredRow {
        residual,
        d_target,
        d_landmark,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{Lcg, perturbed};

    fn camera() -> PinholeCamera {
        PinholeCamera {
            fx: 450.0,
            fy: 455.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

    #[test]
    fn jacobian_matches_finite_differences() {
        let mut rng = Lcg(7);
        let camera = camera();
        let pose = rng.pose(0.3, 0.2);
        let point = pose
            .inverse()
            .transform_point(&Vec3F64::new(0.4, -0.2, 3.0));
        let pixel = [300.0, 250.0];

        let (r0, jacobian) =
            residual_and_pose_jacobian(&pose, &point, pixel, &camera, 1e-3).unwrap();
        let eps = 1e-6;
        for k in 0..6 {
            let (r1, _) =
                residual_and_pose_jacobian(&perturbed(&pose, k, eps), &point, pixel, &camera, 1e-3)
                    .unwrap();
            let numeric = (r1 - r0) / eps;
            for row in 0..2 {
                assert!(
                    (numeric[row] - jacobian[(row, k)]).abs() < 1e-3,
                    "column {k} row {row}: {} vs {}",
                    numeric[row],
                    jacobian[(row, k)]
                );
            }
        }
    }

    /// A host at `host`, a target 0.4 m to its right and turned, and a
    /// landmark 3-6 m ahead of the host, seen by the target.
    fn anchored_case(rng: &mut Lcg) -> (Pose3d, Pose3d, AnchoredLandmark, [f64; 2]) {
        let host = rng.pose(1.0, 0.3);
        let target = host.compose(&rng.pose(0.4, 0.1).inverse());
        let point_h = Vec3F64::new(0.3 + 0.2 * rng.next(), -0.2 + 0.1 * rng.next(), 4.5)
            * (1.0 + 0.3 * rng.next());
        let lm = AnchoredLandmark::from_host_point(&point_h).unwrap();
        (host, target, lm, [310.0, 245.0])
    }

    /// Largest entry of `numeric − analytic` over the largest of `analytic`.
    fn relative_error<const C: usize>(numeric: &SMatF64<2, C>, analytic: &SMatF64<2, C>) -> f64 {
        (numeric - analytic).abs().max() / analytic.abs().max()
    }

    /// T0: every anchored Jacobian against central differences, on several
    /// random geometries.
    #[test]
    fn anchored_jacobians_match_central_differences() {
        let mut rng = Lcg(11);
        let camera = camera();
        let eps = 1e-6;
        for _ in 0..8 {
            let (host, target, lm, pixel) = anchored_case(&mut rng);
            let row =
                anchored_residual_and_jacobians(&host, &target, &lm, pixel, &camera, 1e-3).unwrap();
            let at = |h: &Pose3d, t: &Pose3d, l: &AnchoredLandmark| {
                anchored_residual(h, t, l, pixel, &camera, 1e-3).unwrap()
            };
            let mut d_target = SMatF64::<2, 6>::zeros();
            let mut d_host = SMatF64::<2, 6>::zeros();
            for k in 0..6 {
                let central = |f: &dyn Fn(f64) -> SVecF64<2>| (f(eps) - f(-eps)) / (2.0 * eps);
                d_target.set_column(k, &central(&|e| at(&host, &perturbed(&target, k, e), &lm)));
                d_host.set_column(k, &central(&|e| at(&perturbed(&host, k, e), &target, &lm)));
            }
            let mut d_landmark = SMatF64::<2, 3>::zeros();
            for k in 0..3 {
                let shifted = |e: f64| {
                    let mut l = lm;
                    match k {
                        0 | 1 => l.bearing[k] += e,
                        _ => l.inverse_distance += e,
                    }
                    at(&host, &target, &l)
                };
                d_landmark.set_column(k, &((shifted(eps) - shifted(-eps)) / (2.0 * eps)));
            }
            assert!(relative_error(&d_target, &row.d_target) < 1e-6, "target");
            assert!(relative_error(&d_host, &(-row.d_target)) < 1e-6, "host");
            assert!(
                relative_error(&d_landmark, &row.d_landmark) < 1e-6,
                "landmark"
            );
        }
    }

    /// The host's own camera sees the landmark at its bearing whatever the
    /// host pose and the inverse distance: the pose columns cancel exactly.
    #[test]
    fn the_host_row_is_pose_free() {
        let mut rng = Lcg(5);
        let camera = camera();
        let (host, _, lm, _) = anchored_case(&mut rng);
        let b = lm.bearing_vector();
        let pixel = [
            camera.fx * b.x / b.z + camera.cx,
            camera.fy * b.y / b.z + camera.cy,
        ];
        for rho in [0.0, 0.2, 2.0] {
            let lm = AnchoredLandmark {
                inverse_distance: rho,
                ..lm
            };
            let r = anchored_residual(&host, &host, &lm, pixel, &camera, 1e-3).unwrap();
            assert!(r.abs().max() < 1e-9, "rho {rho}: {r:?}");
        }
    }

    /// T0: a landmark at infinity (`ρ = 0`) still projects, along its
    /// bearing rotated into the target; a negative one does not.
    #[test]
    fn a_landmark_at_infinity_projects() {
        let mut rng = Lcg(9);
        let camera = camera();
        let (host, target, lm, pixel) = anchored_case(&mut rng);
        let lm = AnchoredLandmark {
            inverse_distance: 0.0,
            ..lm
        };
        let row =
            anchored_residual_and_jacobians(&host, &target, &lm, pixel, &camera, 1e-3).unwrap();
        let ray = target.rotation * (host.inverse().rotation * lm.bearing_vector());
        let want = [
            camera.fx * ray.x / ray.z + camera.cx - pixel[0],
            camera.fy * ray.y / ray.z + camera.cy - pixel[1],
        ];
        assert!((row.residual[0] - want[0]).abs() < 1e-9);
        assert!((row.residual[1] - want[1]).abs() < 1e-9);
        assert!(row.d_target.fixed_view::<2, 3>(0, 0).abs().max() == 0.0);
        let negative = AnchoredLandmark {
            inverse_distance: -0.1,
            ..lm
        };
        assert!(anchored_residual(&host, &target, &negative, pixel, &camera, 1e-3).is_none());
    }

    /// T0: the stereographic chart and its inverse round trip, and the
    /// bearing is a unit vector.
    #[test]
    fn the_stereographic_bearing_round_trips() {
        let mut rng = Lcg(3);
        for _ in 0..50 {
            let point = Vec3F64::new(
                rng.next() * 3.0,
                rng.next() * 3.0,
                0.5 + rng.next().abs() * 5.0,
            );
            let lm = AnchoredLandmark::from_host_point(&point).unwrap();
            let back = lm.bearing_vector() * (1.0 / lm.inverse_distance);
            assert!((back - point).length() < 1e-12 * point.length().max(1.0));
            assert!((lm.bearing_vector().length() - 1.0).abs() < 1e-15);
        }
        assert!(AnchoredLandmark::from_host_point(&Vec3F64::new(0.0, 0.0, -1.0)).is_none());
        assert!(AnchoredLandmark::from_host_point(&Vec3F64::ZERO).is_none());
    }

    #[test]
    fn a_point_behind_the_camera_has_no_residual() {
        let point = Vec3F64::new(0.0, 0.0, -1.0);
        assert!(
            residual_and_pose_jacobian(&Pose3d::IDENTITY, &point, [0.0, 0.0], &camera(), 1e-3)
                .is_none()
        );
    }
}
