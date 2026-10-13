//! A parked window cannot observe the depth of a landmark without a stereo
//! measurement: every keyframe sees it from the same place. These tests pin
//! the guards against what that did to VI-BA: an undamped point block flung
//! such a landmark ~1e7 m along its ray, and a later stereo depth row on it
//! (a ~1e8-sigma residual) was lowered by turning a keyframe by ~90°.

use super::*;

fn camera() -> PinholeCamera {
    PinholeCamera {
        fx: 400.7,
        fy: 400.7,
        cx: 320.0,
        cy: 200.0,
        k1: 0.0,
        k2: 0.0,
        p1: 0.0,
        p2: 0.0,
    }
}

/// The parked camera: pitched 22° about x (the VIO world is gravity-on-+Y).
fn parked_pose() -> Pose3d {
    let r_cw = SO3F64::exp(Vec3F64::new(22f64.to_radians(), 0.0, 0.0)).matrix();
    let centre = Vec3F64::new(0.05, -0.02, 0.13);
    Pose3d::new(r_cw, -(r_cw * centre))
}

fn rotation_deg(a: &Pose3d, b: &Pose3d) -> f64 {
    SO3F64::from_matrix(&a.inverse().compose(b).rotation)
        .log()
        .length()
        .to_degrees()
}

/// World point at camera depth `z` behind pixel `(u, v)` of `pose`.
fn point_at(pose: &Pose3d, u: f64, v: f64, z: f64) -> Vec3F64 {
    let cam = camera();
    let pc = Vec3F64::new((u - cam.cx) / cam.fx * z, (v - cam.cy) / cam.fy * z, z);
    pose.inverse().transform_point(&pc)
}

fn pixel(pose: &Pose3d, p: &Vec3F64) -> [f32; 2] {
    let cam = camera();
    let pc = pose.transform_point(p);
    [
        (cam.fx * pc.x / pc.z + cam.cx) as f32,
        (cam.fy * pc.y / pc.z + cam.cy) as f32,
    ]
}

fn stereo(z: f64) -> (Option<f32>, f32) {
    (Some(z as f32), (0.05 * z).max(0.02) as f32)
}

const N_KF: usize = 6;
const N_FIXED: usize = 3;
/// Where the flung landmark sits in the scene, metres.
const TRUE_DEPTH: f64 = 0.94;
/// How far out along its ray the earlier window left it.
const FLUNG_DEPTH: f64 = 1e7;

/// A parked window (six keyframes at one pose, the first three fixed) over
/// a grid of well-measured landmarks, plus landmark 0: a depth-free landmark
/// an earlier window flung to [`FLUNG_DEPTH`] along its ray, now observed by
/// the newest keyframe with a stereo depth of [`TRUE_DEPTH`].
fn flung_window() -> (Vec<ViBaKeyframe>, Vec<Vec3F64>, Vec<BaObservation>) {
    let pose = parked_pose();
    let keyframes: Vec<ViBaKeyframe> = (0..N_KF)
        .map(|k| ViBaKeyframe {
            pose,
            velocity: Vec3F64::ZERO,
            bias: ImuBias::default(),
            fixed: k < N_FIXED,
        })
        .collect();

    let (u0, v0) = (410.0, 150.0);
    let mut truth = vec![point_at(&pose, u0, v0, TRUE_DEPTH)];
    let mut points = vec![point_at(&pose, u0, v0, FLUNG_DEPTH)];
    for i in 0..6 {
        for j in 0..5 {
            let (u, v) = (60.0 + 100.0 * i as f64, 40.0 + 80.0 * j as f64);
            let z = 1.0 + 0.4 * ((i + 2 * j) % 7) as f64;
            let p = point_at(&pose, u, v, z);
            truth.push(p);
            points.push(p);
        }
    }

    let mut observations = Vec::new();
    for (k, kf) in keyframes.iter().enumerate() {
        for (j, p) in truth.iter().enumerate() {
            let z = kf.pose.transform_point(p).z;
            let (depth_meas, depth_sigma) = if j == 0 {
                // Depth-free in every window but the newest keyframe.
                if k == N_KF - 1 {
                    stereo(TRUE_DEPTH)
                } else {
                    (None, 1.0)
                }
            } else if j % 2 == 0 {
                stereo(z)
            } else {
                (None, 1.0)
            };
            observations.push(BaObservation {
                pose_idx: k,
                point_idx: j,
                pixel: pixel(&kf.pose, p),
                fixed_pose: kf.fixed,
                fixed_point: false,
                depth_meas,
                depth_sigma,
            });
        }
    }
    (keyframes, points, observations)
}

fn params() -> ViBaParams {
    ViBaParams {
        max_iterations: 50,
        ..ViBaParams::default()
    }
}

fn max_active_rotation_deg(result: &ViBaResult) -> f64 {
    result.keyframes[N_FIXED..]
        .iter()
        .map(|kf| rotation_deg(&kf.pose, &parked_pose()))
        .fold(0.0, f64::max)
}

/// The mechanism end to end, at the solver: a landmark flung 1e7 m along
/// its ray and then measured at 0.94 m by stereo must not turn a keyframe.
///
/// Red without the fix (2026-10-13, guards reverted in place): with the point
/// block undamped, no depth gate and no cheirality check an active keyframe
/// turns by 136.7°; with only the gate reverted, by 90.1°. The gate is the
/// guard that rescues a landmark already flung; damping is what keeps a solve
/// from flinging it ([`damping_makes_a_rank_two_point_block_invertible`]).
#[test]
fn a_flung_landmark_with_a_stereo_depth_does_not_rotate_the_window() {
    let (keyframes, points, observations) = flung_window();
    let result = visual_inertial_bundle_adjust(
        &keyframes,
        &points,
        &observations,
        &[],
        &[],
        &[],
        &camera(),
        &params(),
    )
    .unwrap();
    let rot = max_active_rotation_deg(&result);
    assert!(rot < 1.0, "an active keyframe turned by {rot:.2} deg");
    assert_eq!(result.depth_rows_gated, 1);
    assert!(result.initial_cost.is_finite() && result.initial_cost > 0.0);
}

/// The gate drops exactly the rows whose landmark depth disagrees with the
/// measurement at the start, and `f64::INFINITY` keeps every row in front of
/// the camera.
#[test]
fn the_depth_gate_drops_only_rows_far_from_their_measurement() {
    let (keyframes, mut points, observations) = flung_window();
    let pose = parked_pose();
    let rows = |pts: &[Vec3F64], factor: f64| depth_gate(&keyframes, pts, &observations, factor);

    let (keep, gated) = rows(&points, 10.0);
    assert_eq!(gated, 1);
    let flung_row = observations
        .iter()
        .position(|o| o.point_idx == 0 && o.depth_meas.is_some())
        .unwrap();
    assert!(!keep[flung_row]);
    assert_eq!(
        keep.iter().filter(|&&k| k).count(),
        observations
            .iter()
            .filter(|o| o.depth_meas.is_some())
            .count()
            - 1
    );
    // A row without a depth measurement is never "kept".
    assert!(
        observations
            .iter()
            .zip(&keep)
            .all(|(o, &k)| o.depth_meas.is_some() || !k)
    );

    // Inside the bound (10·0.94 + 1 m = 10.4 m) the row stays.
    points[0] = point_at(&pose, 410.0, 150.0, 10.3);
    assert_eq!(rows(&points, 10.0).1, 0);
    points[0] = point_at(&pose, 410.0, 150.0, 10.5);
    assert_eq!(rows(&points, 10.0).1, 1);
    assert_eq!(rows(&points, f64::INFINITY).1, 0);
    // Behind the camera is gated whatever the factor.
    points[0] = point_at(&pose, 410.0, 150.0, -2.0);
    assert_eq!(rows(&points, f64::INFINITY).1, 1);
}

/// Flips are listed per observation: a landmark moved behind every camera
/// flips once per observer, one already behind does not flip.
#[test]
fn cheirality_flips_are_counted_per_observation() {
    let (keyframes, points, observations) = flung_window();
    assert_eq!(
        cheirality_flips(&observations, (&keyframes, &points), (&keyframes, &points)).len(),
        0
    );
    let mut behind = points.clone();
    behind[3] = point_at(&parked_pose(), 300.0, 200.0, -1.0);
    // Landmark 3 is seen by all six keyframes.
    assert_eq!(
        cheirality_flips(&observations, (&keyframes, &points), (&keyframes, &behind)),
        observations
            .iter()
            .enumerate()
            .filter(|(_, o)| o.point_idx == 3)
            .map(|(i, _)| i)
            .collect::<Vec<_>>()
    );
    // Already behind before the step: not a flip.
    assert_eq!(
        cheirality_flips(&observations, (&keyframes, &behind), (&keyframes, &behind)).len(),
        0
    );
}

/// A landmark that starts 2 cm in front of the camera, off its own ray, with
/// a stereo depth of 0.94 m: the undamped Gauss-Newton step overshoots
/// through the image plane. The damped point block or the cheirality check
/// each keep every landmark in front; with both reverted (2026-10-13) one
/// ends at z = −0.73 m.
#[test]
fn the_solver_never_accepts_a_step_through_the_image_plane() {
    let (keyframes, mut points, mut observations) = flung_window();
    let pose = parked_pose();
    // Landmark 0 starts 2 cm in front of the camera while its pixels and its
    // (ungated) stereo depth put it 0.94 m out, at a pixel 60 px off its own
    // ray: the Gauss-Newton step overshoots through z = 0.
    points[0] = point_at(&pose, 150.0, 330.0, 0.02);
    for o in observations.iter_mut().filter(|o| o.point_idx == 0) {
        let (d, s) = stereo(TRUE_DEPTH);
        o.depth_meas = d;
        o.depth_sigma = s;
    }
    let result = visual_inertial_bundle_adjust(
        &keyframes,
        &points,
        &observations,
        &[],
        &[],
        &[],
        &camera(),
        &params(),
    )
    .unwrap();
    for o in &observations {
        let z = result.keyframes[o.pose_idx]
            .pose
            .transform_point(&result.points[o.point_idx])
            .z;
        assert!(z > 0.0, "landmark {} ends at z = {z}", o.point_idx);
    }
}

/// With the depth gate off, the flung landmark's depth row asks for a turn
/// that would put the window's landmarks behind the cameras. The cheirality
/// check refuses every such step: no landmark that starts in front of an
/// observing camera ends behind it.
///
/// Red without the check (reverted in place, 2026-10-13): the solve turns a
/// keyframe by ~150° and leaves landmarks behind it (landmark 21 at z =
/// -192 m). The flips come from the camera turn, so holding landmarks does
/// not prevent them: the step is refused.
#[test]
fn cheirality_refuses_steps_that_turn_landmarks_behind_the_camera() {
    let (keyframes, points, observations) = flung_window();
    let ungated = ViBaParams {
        depth_gate_factor: f64::INFINITY,
        ..params()
    };
    let result = visual_inertial_bundle_adjust(
        &keyframes,
        &points,
        &observations,
        &[],
        &[],
        &[],
        &camera(),
        &ungated,
    )
    .unwrap();
    assert_eq!(result.depth_rows_gated, 0);
    for o in &observations {
        let z = result.keyframes[o.pose_idx]
            .pose
            .transform_point(&result.points[o.point_idx])
            .z;
        assert!(z > 0.0, "landmark {} ends at z = {z}", o.point_idx);
    }
    // The turn moves the camera, not one landmark: refused, not held.
    assert!(result.cheirality_rejections > 0);
}

/// The point block of a landmark seen from one viewpoint is rank 2; damped,
/// it inverts, and the inverse is bounded by `1 / (λ·floor)` along the ray.
#[test]
fn damping_makes_a_rank_two_point_block_invertible() {
    // J has rows perpendicular to the ray (0, 0, 1): C = JᵀJ = diag(a, a, 0).
    let a = 400.0f64.powi(2);
    let mut c = [[a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, 0.0]];
    assert!(invert_3x3_f64(&c[0]).is_none());
    let lambda = 1e-2;
    damp_point_blocks(&mut c, lambda);
    let inv = invert_3x3_f64(&c[0]).expect("damped C inverts");
    assert!((inv[0] - 1.0 / (a * (1.0 + lambda))).abs() < 1e-15);
    assert!((inv[8] - 1.0 / (lambda * MIN_DAMPING_FLOOR)).abs() < 1e-3 * inv[8]);

    // Rotated off the axes, so every diagonal entry is non-zero.
    let rot = SO3F64::exp(Vec3F64::new(0.3, -0.5, 0.2)).matrix();
    let d = Mat3F64::from_cols_array(&[a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, 0.0]);
    let full = Mat3F64(rot.mul_mat3(&d.mul_mat3(&rot.transpose())));
    let cols = full.to_cols_array();
    let mut c = [[
        cols[0], cols[3], cols[6], cols[1], cols[4], cols[7], cols[2], cols[5], cols[8],
    ]];
    let ray = rot * Vec3F64::new(0.0, 0.0, 1.0);
    let min_diag = (0..3).map(|d| c[0][d * 4]).fold(f64::INFINITY, f64::min);
    damp_point_blocks(&mut c, lambda);
    let inv = invert_3x3_f64(&c[0]).expect("damped C inverts");
    // C is PSD, so the damped block's smallest eigenvalue is at least
    // λ·min diag(C): the step along the ray is bounded by its inverse.
    let v = [ray.x, ray.y, ray.z];
    let along: f64 = (0..3)
        .map(|r| (0..3).map(|k| v[r] * inv[r * 3 + k] * v[k]).sum::<f64>())
        .sum();
    assert!(
        along > 0.0 && along <= 1.0 / (lambda * min_diag) * (1.0 + 1e-9),
        "{along} vs {}",
        1.0 / (lambda * min_diag)
    );
}

/// The check is per landmark: a free landmark whose own step crosses the
/// image plane is held where it was, a camera step that puts an inlier
/// observation's landmark behind refuses the step, and one that does it to a
/// Huber outlier only does not.
///
/// Red with the holds removed (refusal instead) and with a refusal on any
/// flip (2026-10-13, mutated in place).
#[test]
fn cheirality_holds_the_landmark_and_refuses_only_for_inliers() {
    let (keyframes, points, observations) = flung_window();
    let pose = parked_pose();
    let all_free: Vec<i64> = (0..points.len() as i64).collect();
    let inliers = vec![true; observations.len()];

    // Landmark 3's own step goes through the plane: held, step kept.
    let mut after = points.clone();
    after[3] = point_at(&pose, 300.0, 200.0, -1.0);
    after[4] = points[4] + Vec3F64::new(0.0, 0.0, 0.01);
    let out = enforce_cheirality(
        &observations,
        &inliers,
        &all_free,
        (&keyframes, &points),
        &keyframes,
        &mut after,
    );
    assert_eq!(
        out,
        CheiralityOutcome {
            points_held: 1,
            refuse: false
        }
    );
    assert_eq!(after[3], points[3]);
    assert_eq!(after[4], points[4] + Vec3F64::new(0.0, 0.0, 0.01));

    // A fixed landmark cannot be held: its flip comes from the step, and
    // refuses it while the observation is an inlier...
    let mut fixed = all_free.clone();
    fixed[3] = -1;
    let mut after = points.clone();
    after[3] = point_at(&pose, 300.0, 200.0, -1.0);
    let out = enforce_cheirality(
        &observations,
        &inliers,
        &fixed,
        (&keyframes, &points),
        &keyframes,
        &mut after,
    );
    assert_eq!(
        out,
        CheiralityOutcome {
            points_held: 0,
            refuse: true
        }
    );

    // ...and not when every flipped observation is a Huber outlier.
    let outliers: Vec<bool> = observations.iter().map(|o| o.point_idx != 3).collect();
    let out = enforce_cheirality(
        &observations,
        &outliers,
        &fixed,
        (&keyframes, &points),
        &keyframes,
        &mut after,
    );
    assert!(!out.refuse);
}

/// Keyframes in a moving window.
const N_MOVING: usize = 5;

/// A moving window: five keyframes 8 cm apart sideways, the first two fixed,
/// over a landmark grid 2-4.4 m out. The active keyframes start a few cm off
/// their true pose, the newest 6 cm back along its optical axis. With
/// `outlier`, one more landmark (held fixed, as one pinned by other views
/// would be) sits on the newest keyframe's optical axis 3 cm in front of its
/// start, so its true pose has the landmark 3 cm behind it, matched to a
/// feature 40 px off (chi² 1600, far past Huber's 25). Returns the solve,
/// the true poses and the outlier's landmark index.
fn moving_window(outlier: bool) -> (ViBaResult, Vec<Pose3d>, usize) {
    let base = parked_pose();
    let truth: Vec<Pose3d> = (0..N_MOVING)
        .map(|k| {
            let t = base.translation - Vec3F64::new(0.08 * k as f64, 0.0, 0.0);
            Pose3d::new(base.rotation, t)
        })
        .collect();
    let mut points = Vec::new();
    for i in 0..6 {
        for j in 0..5 {
            let (u, v) = (60.0 + 100.0 * i as f64, 40.0 + 80.0 * j as f64);
            let z = 2.0 + 0.4 * ((i + 2 * j) % 7) as f64;
            points.push(point_at(&truth[N_MOVING / 2], u, v, z));
        }
    }
    let start: Vec<Pose3d> = truth
        .iter()
        .enumerate()
        .map(|(k, p)| {
            let off = match k {
                k if k == N_MOVING - 1 => Vec3F64::new(0.0, 0.0, 0.06),
                k if k >= 2 => Vec3F64::new(0.01, -0.01, 0.0),
                _ => Vec3F64::ZERO,
            };
            Pose3d::new(p.rotation, p.translation + off)
        })
        .collect();
    let keyframes: Vec<ViBaKeyframe> = start
        .iter()
        .enumerate()
        .map(|(k, &pose)| ViBaKeyframe {
            pose,
            velocity: Vec3F64::ZERO,
            bias: ImuBias::default(),
            fixed: k < 2,
        })
        .collect();
    let mut observations = Vec::new();
    for (k, pose) in truth.iter().enumerate() {
        for (j, p) in points.iter().enumerate() {
            observations.push(BaObservation {
                pose_idx: k,
                point_idx: j,
                pixel: pixel(pose, p),
                fixed_pose: k < 2,
                fixed_point: false,
                depth_meas: None,
                depth_sigma: 1.0,
            });
        }
    }
    let outlier_idx = points.len();
    if outlier {
        let cam = camera();
        points.push(point_at(&start[N_MOVING - 1], cam.cx, cam.cy, 0.03));
        observations.push(BaObservation {
            pose_idx: N_MOVING - 1,
            point_idx: outlier_idx,
            pixel: [(cam.cx + 40.0) as f32, cam.cy as f32],
            fixed_pose: false,
            fixed_point: true,
            depth_meas: None,
            depth_sigma: 1.0,
        });
    }
    let result = visual_inertial_bundle_adjust(
        &keyframes,
        &points,
        &observations,
        &[],
        &[],
        &[],
        &camera(),
        &ViBaParams {
            // Near Gauss-Newton: the first step takes the newest keyframe
            // most of the 6 cm, through the outlier's plane.
            initial_lambda: 1e-3,
            ..params()
        },
    )
    .unwrap();
    (result, truth, outlier_idx)
}

fn centre_error_m(a: &Pose3d, b: &Pose3d) -> f64 {
    (a.inverse().translation - b.inverse().translation).length()
}

/// A moving window with one near-plane mismatched association
/// ([`moving_window`]): moving the newest keyframe to its true pose takes
/// the outlier's landmark through its image plane. The observation is a
/// Huber outlier, so that step must not be refused: the keyframe converges
/// to its true pose, short only of the outlier's Huber-bounded pull.
///
/// Red with the check refusing a step on any flip, as it first did
/// (2026-10-13, mutated in place): the keyframe stalls 31 mm short of its true
/// pose. Unchanged with the landmark holds removed: the flip here comes from
/// the camera, not the (fixed) landmark.
#[test]
fn a_moving_window_converges_past_a_near_plane_outlier() {
    let newest = N_MOVING - 1;
    let (clean, truth, _) = moving_window(false);
    let err = centre_error_m(&clean.keyframes[newest].pose, &truth[newest]);
    assert!(err < 1e-4, "without the outlier: {:.2} mm", err * 1e3);

    let (result, truth, outlier) = moving_window(true);
    let err = centre_error_m(&result.keyframes[newest].pose, &truth[newest]);
    // The outlier's Huber-bounded pull leaves 3 mm; refusing every flip
    // left 31 mm.
    assert!(
        err < 0.005,
        "the newest keyframe ended {:.1} mm from its true pose",
        err * 1e3
    );
    let z = result.keyframes[newest]
        .pose
        .transform_point(&result.points[outlier])
        .z;
    assert!(z < 0.0, "the keyframe never passed the outlier (z = {z})");
    assert_eq!(result.cheirality_rejections, 0);
}
