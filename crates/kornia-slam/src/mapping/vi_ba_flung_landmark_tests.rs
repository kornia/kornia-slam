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
/// turns by 136.7°; with only the gate reverted, by 90.1° (before the
/// re-seed existed). Now the re-seed moves the landmark back to its measured
/// depth before the gate sees it; a landmark that cannot be re-seeded is the
/// gate's ([`a_fixed_flung_landmark_is_gated_and_does_not_rotate_the_window`]).
/// Damping alone does not do it: it bounds a solve's point step, but a
/// depth-free landmark still walks far along its ray over parked windows
/// ([`damping_bounds_how_far_a_parked_chain_walks_a_depth_free_landmark`]).
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
    assert_eq!(result.points_reseeded, 1);
    assert_eq!(result.depth_rows_gated, 0);
    let z = parked_pose().transform_point(&result.points[0]).z;
    assert!(
        (z / TRUE_DEPTH - 1.0).abs() < 0.05,
        "the flung landmark ends at depth {z:.3} m"
    );
    assert!(result.initial_cost.is_finite() && result.initial_cost > 0.0);
}

/// The same window with the flung landmark held fixed, so it cannot be
/// re-seeded: its depth row is gated and still no keyframe turns.
#[test]
fn a_fixed_flung_landmark_is_gated_and_does_not_rotate_the_window() {
    let (keyframes, points, mut observations) = flung_window();
    for o in observations.iter_mut().filter(|o| o.point_idx == 0) {
        o.fixed_point = true;
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
    let rot = max_active_rotation_deg(&result);
    assert!(rot < 1.0, "an active keyframe turned by {rot:.2} deg");
    assert_eq!(result.points_reseeded, 0);
    assert_eq!(result.depth_rows_gated, 1);
    assert_eq!(result.points[0], points[0]);
}

/// The re-seed moves a free landmark only when every depth row on it is
/// beyond the gate, along the best measured row's ray, to its depth. (The
/// flung landmark here is 1e7 m out, past the uncorroborated bound, so its
/// one row is enough; see
/// [`a_single_bad_depth_row_does_not_reseed_a_plausible_landmark`].)
#[test]
fn reseed_moves_only_landmarks_every_depth_row_disagrees_with() {
    let (keyframes, points, mut observations) = flung_window();
    let pose = parked_pose();
    let free = vec![true; points.len()];
    let reseed = |pts: &mut Vec<Vec3F64>, obs: &[BaObservation], free: &[bool]| {
        reseed_walked_landmarks(&keyframes, pts, obs, free, 10.0)
    };

    let mut pts = points.clone();
    assert_eq!(reseed(&mut pts, &observations, &free), 1);
    let truth = point_at(&pose, 410.0, 150.0, TRUE_DEPTH);
    // Along the ray, to the f32 measured depth.
    assert!((pts[0] - truth).length() < 1e-6, "{:?}", pts[0] - truth);
    assert_eq!(pts[1..], points[1..]);

    // Fixed: left alone.
    let mut fixed = free.clone();
    fixed[0] = false;
    let mut pts = points.clone();
    assert_eq!(reseed(&mut pts, &observations, &fixed), 0);
    assert_eq!(pts, points);

    // A second depth row that agrees with the landmark: left alone.
    let agreeing = observations
        .iter()
        .position(|o| o.point_idx == 0 && o.depth_meas.is_none())
        .unwrap();
    observations[agreeing].depth_meas = Some(FLUNG_DEPTH as f32);
    let mut pts = points.clone();
    assert_eq!(reseed(&mut pts, &observations, &free), 0);
    observations[agreeing].depth_meas = None;

    // Behind the camera: left alone (the cheirality guards own it).
    let mut pts = points.clone();
    pts[0] = point_at(&pose, 410.0, 150.0, -2.0);
    let behind = pts[0];
    assert_eq!(reseed(&mut pts, &observations, &free), 0);
    assert_eq!(pts[0], behind);
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

/// Deterministic standard-normal draws (xorshift64* and Box-Muller), so the
/// parked chain below is reproducible without a `rand` dependency.
struct Gauss(u64);

impl Gauss {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        ((self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (a, b) = (self.uniform(), self.uniform());
        (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
    }

    fn vec3(&mut self, sigma: f64) -> Vec3F64 {
        Vec3F64::new(self.normal(), self.normal(), self.normal()) * sigma
    }
}

/// A camera parked for many windows. Every window has six keyframes (the
/// first three fixed) whose pose estimates are the true parked pose plus
/// tracking noise (1 mm, 0.02°), and pixels that are the true projections
/// plus 0.3 px. The landmarks are [`flung_window`]'s grid: the even ones have
/// stereo depth, the odd ones none, and every one starts at its true
/// position. Each window starts from the landmarks the previous one left.
struct ParkedChain {
    rng: Gauss,
    truth: Vec<Vec3F64>,
    points: Vec<Vec3F64>,
}

impl ParkedChain {
    fn new(seed: u64) -> Self {
        let pose = parked_pose();
        let mut truth = Vec::new();
        for i in 0..6 {
            for j in 0..5 {
                let (u, v) = (60.0 + 100.0 * i as f64, 40.0 + 80.0 * j as f64);
                let z = 1.0 + 0.4 * ((i + 2 * j) % 7) as f64;
                truth.push(point_at(&pose, u, v, z));
            }
        }
        Self {
            rng: Gauss(0x9e37_79b9_7f4a_7c15 ^ seed),
            points: truth.clone(),
            truth,
        }
    }

    /// Solves the next window with the default parameters (so the default λ
    /// schedule) and takes its landmarks: all of them, or with `guard` only
    /// those [`landmarks_to_write_back`] keeps, as `run_local_inertial_ba`
    /// does. The newest `stereo_rows` keyframes also measure every
    /// depth-free landmark by stereo, at its true depth (two or more rows
    /// corroborate a re-seed; one does not).
    fn window(&mut self, guard: bool, stereo_rows: usize) -> ViBaResult {
        self.window_moving(guard, stereo_rows, 0.0)
    }

    /// [`Self::window`] with keyframe `k` moved `k · step_m` along the
    /// camera's x axis from the parked pose (0: parked).
    fn window_moving(&mut self, guard: bool, stereo_rows: usize, step_m: f64) -> ViBaResult {
        let parked = parked_pose();
        let truth_poses: Vec<Pose3d> = (0..N_KF)
            .map(|k| {
                let centre =
                    parked
                        .inverse()
                        .transform_point(&Vec3F64::new(k as f64 * step_m, 0.0, 0.0));
                Pose3d::new(parked.rotation, -(parked.rotation * centre))
            })
            .collect();
        let rng = &mut self.rng;
        let keyframes: Vec<ViBaKeyframe> = truth_poses
            .iter()
            .enumerate()
            .map(|(k, truth_pose)| {
                let r_cw =
                    SO3F64::exp(rng.vec3(0.02f64.to_radians())).matrix() * truth_pose.rotation;
                let centre = truth_pose.inverse().translation + rng.vec3(1e-3);
                ViBaKeyframe {
                    pose: Pose3d::new(r_cw, -(r_cw * centre)),
                    velocity: Vec3F64::ZERO,
                    bias: ImuBias::default(),
                    fixed: k < N_FIXED,
                }
            })
            .collect();
        let mut observations = Vec::new();
        for (k, kf) in keyframes.iter().enumerate() {
            let truth_pose = &truth_poses[k];
            for (j, p) in self.truth.iter().enumerate() {
                let px = pixel(truth_pose, p);
                let measured = j % 2 == 0 || k + stereo_rows >= N_KF;
                let (depth_meas, depth_sigma) = if measured {
                    stereo(truth_pose.transform_point(p).z)
                } else {
                    (None, 1.0)
                };
                observations.push(BaObservation {
                    pose_idx: k,
                    point_idx: j,
                    pixel: [
                        px[0] + (0.3 * rng.normal()) as f32,
                        px[1] + (0.3 * rng.normal()) as f32,
                    ],
                    fixed_pose: kf.fixed,
                    fixed_point: false,
                    depth_meas,
                    depth_sigma,
                });
            }
        }
        let result = visual_inertial_bundle_adjust(
            &keyframes,
            &self.points,
            &observations,
            &[],
            &[],
            &[],
            &camera(),
            &ViBaParams::default(),
        )
        .unwrap();
        let keep = if guard {
            crate::mapping::bundle_adjustment::landmarks_to_write_back(
                &observations,
                (&keyframes, &self.points),
                (&result.keyframes, &result.points),
            )
        } else {
            vec![true; result.points.len()]
        };
        for ((p, new), keep) in self.points.iter_mut().zip(&result.points).zip(keep) {
            if keep {
                *p = *new;
            }
        }
        result
    }

    /// The largest factor by which a depth-free landmark's camera depth is
    /// off its true depth, either way (1 = every one where it should be).
    fn worst_depth_ratio(&self) -> f64 {
        let pose = parked_pose();
        let z = |p: &Vec3F64| pose.transform_point(p).z;
        (1..self.truth.len())
            .step_by(2)
            .map(|j| (z(&self.points[j]) / z(&self.truth[j])).abs())
            .map(|r| r.max(1.0 / r))
            .fold(1.0, f64::max)
    }
}

const CHAIN_WINDOWS: usize = 12;

/// Damping at the solver, from landmarks at their true depth: a parked chain
/// with tracking noise and the default λ schedule. Damping does NOT keep a
/// depth-free landmark at its depth: none is observable from one place, and
/// across windows these walk out along their rays with the noise, to tens or
/// hundreds of times their depth (measured 2026-10-13: up to 542x and 2315x
/// for the two seeds). What it bounds is the step: with the point blocks
/// undamped (mutated in place, same date) the same chain flings them
/// 4.3e6x-1.7e7x, the ~1e7 m the parked robot showed, and this test is red.
/// A landmark that got that far is the depth gate's job
/// ([`a_flung_landmark_with_a_stereo_depth_does_not_rotate_the_window`]).
#[test]
fn damping_bounds_how_far_a_parked_chain_walks_a_depth_free_landmark() {
    for seed in [1, 2] {
        let mut chain = ParkedChain::new(seed);
        let mut worst = 1.0f64;
        for _ in 0..CHAIN_WINDOWS {
            chain.window(false, 0);
            worst = worst.max(chain.worst_depth_ratio());
        }
        assert!(
            worst < 1e5,
            "seed {seed}: a depth-free landmark walked {worst:.3e}x its depth"
        );
    }
}

/// The walk damping leaves ([`damping_bounds_how_far_a_parked_chain_walks_a_depth_free_landmark`]),
/// through the write-back guard as `run_local_inertial_ba` applies it: the
/// guard does not hold the depth either (each window may grow a landmark up
/// to tenfold, or to 50 m). When only the newest keyframe of the next
/// window measures every depth-free landmark by stereo, one row is no
/// corroboration: the walked landmarks are gated, not moved, and no keyframe
/// turns. When the newest two then measure them at their true depth, the
/// landmarks that walked past the depth gate are re-seeded at the measured
/// depth instead of gated, no keyframe turns, and every depth-free landmark
/// ends near its true depth. A moving window after that (the robot drives
/// off, no stereo on those landmarks) keeps them there.
///
/// Red with the re-seed removed (2026-10-13, mutated in place): the walked
/// landmarks' rows are gated (seed 1: 4 rows), the worst landmark stays
/// 166x off its depth, and a moving window of 5 cm steps brings it only to
/// 16x.
#[test]
fn a_parked_chain_s_walked_landmarks_are_restored_when_stereo_measures_them() {
    for seed in [1, 2] {
        let mut chain = ParkedChain::new(seed);
        for _ in 0..CHAIN_WINDOWS {
            chain.window(true, 0);
        }
        let walked = chain.worst_depth_ratio();
        assert!(walked > 10.0, "seed {seed}: walked only {walked:.1}x");
        // One stereo row is no corroboration: the walked landmarks' rows are
        // gated, nothing moves, and no keyframe turns.
        let single = chain.window(true, 1);
        assert_eq!(single.points_reseeded, 0, "seed {seed}");
        assert!(single.depth_rows_gated > 0, "seed {seed}: nothing gated");
        let rot = max_active_rotation_deg(&single);
        assert!(
            rot < 1.0,
            "seed {seed}: an active keyframe turned by {rot:.2} deg"
        );
        assert!(chain.worst_depth_ratio() > 10.0, "seed {seed}");

        let result = chain.window(true, 2);
        assert!(result.points_reseeded > 0, "seed {seed}: nothing re-seeded");
        assert_eq!(result.depth_rows_gated, 0, "seed {seed}");
        let rot = max_active_rotation_deg(&result);
        assert!(
            rot < 1.0,
            "seed {seed}: an active keyframe turned by {rot:.2} deg"
        );
        let restored = chain.worst_depth_ratio();
        assert!(
            restored < 1.2,
            "seed {seed}: a landmark is still {restored:.2}x off its depth"
        );

        let moving = chain.window_moving(true, 0, 0.05);
        assert_eq!(moving.points_reseeded, 0, "seed {seed}");
        let after = chain.worst_depth_ratio();
        assert!(
            after < 1.2,
            "seed {seed}: after driving off a landmark is {after:.2}x off its depth"
        );
    }
}

/// One erroneous stereo depth must not pull a correct far landmark onto
/// itself. A landmark at 30 m, measured once at 2 m: its row is gated (bound
/// 21 m) but the landmark stays where it is. A second gated row that agrees
/// with the 2 m re-seed corroborates it and the landmark moves; a second row
/// that disagrees (2.6 m, beyond three combined sigmas) does not.
///
/// Red before corroboration was required (2026-10-13): the single row
/// re-seeded the landmark to 2 m.
#[test]
fn a_single_bad_depth_row_does_not_reseed_a_plausible_landmark() {
    let (keyframes, mut points, mut observations) = flung_window();
    let pose = parked_pose();
    points[0] = point_at(&pose, 410.0, 150.0, 30.0);
    let start = points.clone();
    let free = vec![true; points.len()];
    let row_of = |k: usize| {
        observations
            .iter()
            .position(|o| o.point_idx == 0 && o.pose_idx == k)
            .unwrap()
    };
    let (newest, second) = (row_of(N_KF - 1), row_of(N_KF - 2));
    (
        observations[newest].depth_meas,
        observations[newest].depth_sigma,
    ) = stereo(2.0);
    let reseed = |pts: &mut Vec<Vec3F64>, obs: &[BaObservation]| {
        reseed_walked_landmarks(&keyframes, pts, obs, &free, 10.0)
    };

    let mut pts = start.clone();
    assert_eq!(reseed(&mut pts, &observations), 0);
    assert_eq!(pts, start);
    let (_, gated) = depth_gate(&keyframes, &pts, &observations, 10.0);
    assert_eq!(gated, 1, "the bad row is still kept out of the solve");

    // Disagreeing second row: still left alone.
    (
        observations[second].depth_meas,
        observations[second].depth_sigma,
    ) = stereo(2.6);
    let mut pts = start.clone();
    assert_eq!(reseed(&mut pts, &observations), 0);
    assert_eq!(pts, start);

    // Agreeing second row: corroborated, moved to 2 m along the ray.
    (
        observations[second].depth_meas,
        observations[second].depth_sigma,
    ) = stereo(2.05);
    let mut pts = start.clone();
    assert_eq!(reseed(&mut pts, &observations), 1);
    let z = pose.transform_point(&pts[0]).z;
    assert!((z - 2.0).abs() < 1e-5, "re-seeded to {z} m");
}

/// `depth_gate_factor` must be positive: NaN would gate (and first re-seed)
/// every depth row, zero or a negative value would gate every correct one.
/// `f64::INFINITY` is the documented "no upper bound" and is accepted.
#[test]
fn depth_gate_factor_must_be_positive() {
    let (keyframes, points, observations) = flung_window();
    let solve = |factor: f64| {
        visual_inertial_bundle_adjust(
            &keyframes,
            &points,
            &observations,
            &[],
            &[],
            &[],
            &camera(),
            &ViBaParams {
                max_iterations: 2,
                depth_gate_factor: factor,
                ..ViBaParams::default()
            },
        )
    };
    for bad in [f64::NAN, 0.0, -1.0, f64::NEG_INFINITY] {
        let refused = matches!(
            solve(bad),
            Err(ViBaError::InvalidDepthGateFactor(f)) if f.to_bits() == bad.to_bits()
        );
        assert!(refused, "{bad} was accepted");
    }
    for good in [1e-3, 10.0, f64::INFINITY] {
        assert!(solve(good).is_ok(), "{good} was refused");
    }
}
