//! T2-T6: the window on the synthetic stereo-inertial walk, its bookkeeping
//! and its solver.

use std::collections::BTreeSet;
use std::sync::OnceLock;

use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, SO3F64, Vec3F64};
use kornia_sensors::imu::{ImuBias, ImuCalib, PreintegratedImu};

use super::Ledger;
use crate::mapping::synthetic_walk::Gauss;
use crate::odometry::config::OdometryTuning;
use crate::odometry::keyframes::{KeyframeCounts, MargPlan, is_keyframe, plan_marginalization};
use crate::odometry::linearize::{VarKind, cost};
use crate::odometry::state::{FrameId, PoseWithLin, StateWithLin};
use crate::odometry::synthetic::{StereoWalk, camera_to_body, gravity, stereo_views};
use crate::odometry::window::{FrameInput, FrameReport, Window};
use crate::tracking::pose_estimation::pose_inertial::FrameNavState;

fn biased() -> ImuBias {
    ImuBias {
        gyro: Vec3F64::new(-0.002, 0.020, 0.077),
        accel: Vec3F64::new(-0.026, 0.137, 0.076),
    }
}

fn datasheet() -> ImuCalib {
    ImuCalib {
        gyro_noise: 1.6968e-4,
        accel_noise: 2.0e-3,
        gyro_bias_noise: 1.9393e-5,
        accel_bias_noise: 3.0e-3,
    }
}

/// Runs `frames` frames of `walk` from frame `start` through a new window,
/// calling `each` after every frame.
fn drive(
    walk: &StereoWalk,
    tuning: &OdometryTuning,
    start: usize,
    frames: usize,
    pixel_noise: Option<f64>,
    ledger: bool,
    mut each: impl FnMut(usize, &Window, &FrameReport),
) -> Window {
    let mut window = Window::new(tuning, stereo_views(), camera_to_body(), gravity());
    if ledger {
        window.ledger = Some(Ledger::default());
    }
    let mut gauss = Gauss(0x5eed);
    for k in start..start + frames {
        let t = StereoWalk::time(k);
        let observations = walk.observations(k, pixel_noise.map(|s| (&mut gauss, s)));
        let id = StereoWalk::id(k);
        let imu = if k == start {
            window.initialize(id, walk.accel_at(t)).unwrap();
            None
        } else {
            let t0 = StereoWalk::time(k - 1);
            let bias = window.newest().unwrap().1.value().bias;
            Some(PreintegratedImu::from_measurements(
                bias,
                datasheet(),
                walk.samples(t0, t),
                t0,
                t,
            ))
        };
        let report = window
            .add_frame(FrameInput {
                id,
                imu,
                observations: &observations,
            })
            .unwrap();
        each(k, &window, &report);
    }
    window
}

fn r_wb(state: &FrameNavState) -> Mat3F64 {
    camera_to_body()
        .compose(&state.pose_cw)
        .rotation
        .transpose()
}

fn body_position(state: &FrameNavState) -> Vec3F64 {
    camera_to_body()
        .compose(&state.pose_cw)
        .inverse()
        .translation
}

/// Angle between the estimated and the true up direction in the body, deg.
fn tilt_deg(estimate: &FrameNavState, truth: &FrameNavState) -> f64 {
    let z = Vec3F64::new(0.0, 0.0, 1.0);
    let (a, b) = (r_wb(estimate).transpose() * z, r_wb(truth).transpose() * z);
    a.cross(b).length().atan2(a.dot(b)).to_degrees()
}

/// Yaw-and-translation fit of `truth` onto `estimate` positions; returns the
/// aligned error of the last one.
fn aligned_last_error(estimate: &[Vec3F64], truth: &[Vec3F64]) -> f64 {
    let n = estimate.len() as f64;
    let mean = |v: &[Vec3F64]| v.iter().fold(Vec3F64::ZERO, |a, p| a + *p) * (1.0 / n);
    let (me, mt) = (mean(estimate), mean(truth));
    let (mut s, mut c) = (0.0, 0.0);
    for (e, t) in estimate.iter().zip(truth) {
        let (e, t) = (*e - me, *t - mt);
        s += t.x * e.y - t.y * e.x;
        c += t.x * e.x + t.y * e.y;
    }
    let yaw = SO3F64::exp(Vec3F64::new(0.0, 0.0, s.atan2(c))).matrix();
    let shift = me - yaw * mt;
    let (e, t) = (estimate[estimate.len() - 1], truth[truth.len() - 1]);
    (e - (yaw * t + shift)).length()
}

/// What T3b, T3c and T4a read from one 20 s run from rest.
struct WalkRun {
    accel_error_10s: f64,
    gyro_error_10s: f64,
    tilt_10s_deg: f64,
    position_error_20s: f64,
    nullspace_final: [f64; 6],
    nullspace_worst_gauge: f64,
    attempts_max: usize,
    resets: usize,
}

fn walk_run() -> &'static WalkRun {
    static RUN: OnceLock<WalkRun> = OnceLock::new();
    RUN.get_or_init(|| {
        let walk = StereoWalk::new(20.0, biased(), None);
        let frames = walk.frames();
        let mut run = WalkRun {
            accel_error_10s: f64::NAN,
            gyro_error_10s: f64::NAN,
            tilt_10s_deg: f64::NAN,
            position_error_20s: f64::NAN,
            nullspace_final: [f64::NAN; 6],
            nullspace_worst_gauge: 0.0,
            attempts_max: 0,
            resets: 0,
        };
        let (mut est, mut truth) = (Vec::new(), Vec::new());
        drive(
            &walk,
            &OdometryTuning::default(),
            0,
            frames,
            None,
            false,
            |k, window, report| {
                let state = window.newest().unwrap().1.value();
                est.push(body_position(&state));
                truth.push(body_position(&walk.truth(k)));
                if let Some(solve) = report.solve {
                    run.attempts_max = run.attempts_max.max(solve.attempts);
                    run.resets += usize::from(solve.failed);
                }
                if let Some(n) = report.nullspace {
                    run.nullspace_final = n;
                    for d in [0, 1, 2, 5] {
                        run.nullspace_worst_gauge = run.nullspace_worst_gauge.max(n[d]);
                    }
                }
                if k == 200 {
                    run.accel_error_10s = (state.bias.accel - walk.bias.accel).length();
                    run.gyro_error_10s = (state.bias.gyro - walk.bias.gyro).length();
                    run.tilt_10s_deg = tilt_deg(&state, &walk.truth(k));
                }
            },
        );
        run.position_error_20s = aligned_last_error(&est, &truth);
        run
    })
}

/// T4a: from rest and zero biases, exact pixels and IMU, the window finds
/// the biases and gravity within 10 s and holds the path.
#[test]
fn t4a_the_synthetic_walk_recovers_bias_gravity_and_path() {
    let run = walk_run();
    eprintln!(
        "T4a: accel bias error {:.5} m/s², gyro {:.2e} rad/s, tilt {:.4} deg at 10 s; \
         position {:.4} m at 20 s; resets {}",
        run.accel_error_10s,
        run.gyro_error_10s,
        run.tilt_10s_deg,
        run.position_error_20s,
        run.resets
    );
    assert!(
        run.accel_error_10s <= 0.01,
        "accel bias {}",
        run.accel_error_10s
    );
    assert!(
        run.gyro_error_10s <= 1e-4,
        "gyro bias {}",
        run.gyro_error_10s
    );
    assert!(run.tilt_10s_deg <= 0.1, "tilt {}", run.tilt_10s_deg);
    assert!(
        run.position_error_20s <= 0.02,
        "position {}",
        run.position_error_20s
    );
    assert_eq!(run.resets, 0);
}

/// T3b, T3c: after 400 frames the twin prior keeps x, y, z and yaw
/// unobservable, while gravity makes roll and pitch observable.
#[test]
fn t3bc_the_prior_keeps_the_four_dof_nullspace() {
    let run = walk_run();
    let n = run.nullspace_final;
    eprintln!(
        "T3b/c: final x {:.2e} y {:.2e} z {:.2e} roll {:.2e} pitch {:.2e} yaw {:.2e}; worst gauge {:.2e}",
        n[0], n[1], n[2], n[3], n[4], n[5], run.nullspace_worst_gauge
    );
    for d in [0, 1, 2, 5] {
        assert!(n[d] <= 1e-9, "direction {d}: {:.3e}", n[d]);
    }
    assert!(
        run.nullspace_worst_gauge <= 1e-9,
        "worst {:.3e}",
        run.nullspace_worst_gauge
    );
    for d in [3, 4] {
        assert!(
            n[d] >= 1e3 * n[5],
            "direction {d}: {:.3e} against yaw {:.3e}",
            n[d],
            n[5]
        );
    }
}

/// T6: the 8-attempt cap holds on every solve of the walk.
#[test]
fn t6_no_solve_takes_more_than_eight_attempts() {
    let run = walk_run();
    assert!(run.attempts_max <= 8, "{}", run.attempts_max);
    assert!(run.attempts_max >= 1);
}

/// T4b: a moving start at 5 s, 0.5 px pixel noise and datasheet IMU noise.
#[test]
fn t4b_a_moving_noisy_start_converges() {
    let walk = StereoWalk::new(15.5, biased(), Some((datasheet(), 99)));
    let start = 100;
    let mut chi2 = Vec::new();
    let mut at_10s = None;
    drive(
        &walk,
        &OdometryTuning::default(),
        start,
        201,
        Some(0.5),
        false,
        |k, window, report| {
            if let Some(c) = report.imu_chi2_nav {
                chi2.push(c);
            }
            if k == start + 200 {
                let state = window.newest().unwrap().1.value();
                at_10s = Some((
                    (state.bias.accel - walk.bias.accel).length(),
                    (state.bias.gyro - walk.bias.gyro).length(),
                    tilt_deg(&state, &walk.truth(k)),
                ));
            }
        },
    );
    let (accel, gyro, tilt) = at_10s.unwrap();
    let newest_edge = chi2.iter().sum::<f64>() / chi2.len() as f64;
    let truth = chi2_at_truth(&walk, start + 1..start + 201);
    eprintln!(
        "T4b: accel {accel:.4} m/s², gyro {gyro:.2e} rad/s, tilt {tilt:.3} deg; \
         imu_chi2_nav {newest_edge:.3} (newest edge after the solve), {truth:.3} at the true states"
    );
    assert!(accel <= 0.03, "accel {accel}");
    assert!(gyro <= 3e-4, "gyro {gyro}");
    assert!(tilt <= 0.3, "tilt {tilt}");
    // The prediction was 0.5-2.0 for `imu_chi2_nav`; it reads 0.05 (miss,
    // results doc 2x-4): the newest state's free velocity and its weakly
    // seen pose absorb the newest edge's residual, so that statistic is not
    // a noise-model test. The model itself is consistent at the truth.
    assert!(
        (0.5..=2.0).contains(&truth),
        "chi2 at the true states {truth}"
    );
}

/// Mean `r_navᵀ Ω r_nav / 9` of the edges between the true states of
/// `frames`, with the walk's noisy readings.
fn chi2_at_truth(walk: &StereoWalk, frames: std::ops::Range<usize>) -> f64 {
    let n = frames.len() as f64;
    frames
        .map(|k| {
            let (t0, t) = (StereoWalk::time(k - 1), StereoWalk::time(k));
            let pim = PreintegratedImu::from_measurements(
                walk.bias,
                datasheet(),
                walk.samples(t0, t),
                t0,
                t,
            );
            let omega = kornia_factors::inertial::preintegration::information(&pim, 1.0).omega;
            let (a, b) = (walk.truth(k - 1), walk.truth(k));
            let (r, _, _) = kornia_factors::inertial::preintegration::residual_and_jacobians(
                a.nav_state(),
                b.nav_state(),
                &pim,
                &gravity(),
                Some(&camera_to_body()),
            );
            let nav = r.fixed_rows::<9>(0);
            nav.dot(&(omega.fixed_view::<9, 9>(0, 0) * nav)) / 9.0
        })
        .sum::<f64>()
        / n
}

/// T2: after 60 noisy frames the sequential prior equals the one-shot
/// Schur complement of every factor the window absorbed.
#[test]
fn t2_the_sequential_prior_is_the_one_shot_schur_complement_of_its_ledger() {
    let walk = StereoWalk::new(3.2, biased(), Some((datasheet(), 5)));
    let window = drive(
        &walk,
        &OdometryTuning::default(),
        0,
        60,
        Some(0.5),
        true,
        |_, _, _| {},
    );
    let ledger = window.ledger.as_ref().unwrap();
    assert!(ledger.priors > 40, "{} marginalizations", ledger.priors);
    let (dh, db) = ledger.compare(&window.prior);
    eprintln!("T2: {} priors, H {dh:.2e}, b {db:.2e}", ledger.priors);
    assert!(dh <= 1e-9, "H {dh:.3e}");
    assert!(db <= 1e-9, "b {db:.3e}");
}

/// `G · x` for the world transform `G = (R, t)`: poses `T_cw G⁻¹`,
/// velocities rotated.
fn transformed(state: &FrameNavState, r: &Mat3F64, t: &Vec3F64) -> FrameNavState {
    let g_inv = Pose3d::new(*r, *t).inverse();
    FrameNavState {
        pose_cw: state.pose_cw.compose(&g_inv),
        velocity: *r * state.velocity,
        bias: state.bias,
    }
}

fn transform_window(window: &mut Window, r: &Mat3F64, t: &Vec3F64) {
    for state in window.states.values_mut() {
        *state = StateWithLin::new(transformed(&state.value(), r, t));
    }
    for pose in window.poses.values_mut() {
        *pose = PoseWithLin::new(pose.value().compose(&Pose3d::new(*r, *t).inverse()));
    }
}

/// T3a: the vision and IMU cost does not change under a finite world
/// translation and yaw, and an infinitesimal one is the nullspace column.
#[test]
fn t3a_the_cost_is_invariant_under_world_translation_and_yaw() {
    let walk = StereoWalk::new(3.2, biased(), Some((datasheet(), 11)));
    let window = drive(
        &walk,
        &OdometryTuning::default(),
        0,
        60,
        Some(0.5),
        false,
        |_, _, _| {},
    );
    let base = cost(&window);
    let factors = base.vision + base.imu;
    let mut moved = drive(
        &walk,
        &OdometryTuning::default(),
        0,
        60,
        Some(0.5),
        false,
        |_, _, _| {},
    );
    let yaw = SO3F64::exp(Vec3F64::new(0.0, 0.0, 0.7)).matrix();
    transform_window(&mut moved, &yaw, &Vec3F64::new(0.3, -0.2, 0.5));
    let after = cost(&moved);
    let relative = ((after.vision + after.imu) - factors).abs() / factors;
    eprintln!(
        "T3a: cost {factors:.6e} → {:.6e}, relative {relative:.2e}",
        after.vision + after.imu
    );
    assert!(relative <= 1e-9, "{relative:.3e}");

    // The infinitesimal transform against the columns: ρ = −e_k for a
    // translation; ρ = a × m, ω = −a, δv = a × v for a rotation about a
    // through m.
    let eps = 1e-6;
    let m = Vec3F64::new(0.4, 1.0, -0.3);
    for state in window.states.values() {
        let s = state.value();
        for axis in [Vec3F64::new(0.0, 0.0, 1.0), Vec3F64::new(1.0, 0.0, 0.0)] {
            let r = SO3F64::exp(axis * eps).matrix();
            let t = m - r * m;
            let moved = transformed(&s, &r, &t);
            let omega =
                SO3F64::from_matrix(&(s.pose_cw.rotation.transpose() * moved.pose_cw.rotation))
                    .log();
            let rho = s.pose_cw.rotation.transpose()
                * (moved.pose_cw.translation - s.pose_cw.translation);
            let dv = moved.velocity - s.velocity;
            let want_rho = axis.cross(m);
            let want_dv = axis.cross(s.velocity);
            assert!(((rho * (1.0 / eps)) - want_rho).length() < 1e-5 * want_rho.length().max(1.0));
            assert!(((omega * (1.0 / eps)) + axis).length() < 1e-5);
            assert!(((dv * (1.0 / eps)) - want_dv).length() < 1e-5 * want_dv.length().max(1.0));
        }
        let t = Vec3F64::new(eps, 0.0, 0.0);
        let moved = transformed(&s, &Mat3F64::IDENTITY, &t);
        let rho =
            s.pose_cw.rotation.transpose() * (moved.pose_cw.translation - s.pose_cw.translation);
        assert!(((rho * (1.0 / eps)) + Vec3F64::new(1.0, 0.0, 0.0)).length() < 1e-6);
    }
}

fn tuning() -> OdometryTuning {
    OdometryTuning::default()
}

fn at(x: f64) -> impl Fn(FrameId) -> Option<Vec3F64> {
    move |id| Some(Vec3F64::new(id as f64 * x, 0.0, 0.0))
}

/// T5, M0-M4: when a marginalization runs and which states leave how.
#[test]
fn t5_the_marginalization_plan_follows_m0_to_m4() {
    let counts = KeyframeCounts::default();
    let kfs: BTreeSet<FrameId> = [1].into();
    assert_eq!(
        plan_marginalization(&[1, 2, 3], &kfs, &counts, at(1.0), false, &tuning()),
        None
    );
    assert_eq!(
        plan_marginalization(&[1, 2], &kfs, &counts, at(1.0), true, &tuning()),
        None
    );
    assert_eq!(
        plan_marginalization(&[1, 2, 3], &kfs, &counts, at(1.0), true, &tuning()),
        Some(MargPlan {
            last_state_to_marg: 2,
            states_vel_bias: vec![1],
            states_all: vec![],
            keyframes_to_marg: vec![],
        })
    );
    assert_eq!(
        plan_marginalization(&[1, 2, 3, 4, 5], &kfs, &counts, at(1.0), true, &tuning()),
        Some(MargPlan {
            last_state_to_marg: 4,
            states_vel_bias: vec![1],
            states_all: vec![2, 3],
            keyframes_to_marg: vec![],
        })
    );
}

/// T5, M5-M6: past seven keyframes, the oldest poorly tracked one goes
/// first, else the DSO score picks; never one of the newest two.
#[test]
fn t5_keyframes_leave_by_tracked_share_then_by_score() {
    // Keyframe poses 10..=70, keyframe states 80 (converting) and 90.
    let kfs: BTreeSet<FrameId> = [10, 20, 30, 40, 50, 60, 70, 80].into();
    let states = [80, 85, 90];
    let mut counts = KeyframeCounts::default();
    for &k in &kfs {
        counts.created.insert(k, 100);
        counts.connected.insert(k, 50);
    }
    counts.connected.insert(30, 5);
    let plan = plan_marginalization(&states, &kfs, &counts, at(1.0), true, &tuning()).unwrap();
    assert_eq!(plan.keyframes_to_marg, vec![30]);
    assert_eq!(plan.states_vel_bias, vec![80]);

    counts.connected.insert(30, 50);
    // Every share is fine: the DSO score decides, checked against a direct
    // evaluation; keyframe 20 sits apart.
    let position = |id: FrameId| {
        Some(Vec3F64::new(
            if id == 20 { 0.0 } else { id as f64 * 0.1 },
            0.0,
            0.0,
        ))
    };
    let plan = plan_marginalization(&states, &kfs, &counts, position, true, &tuning()).unwrap();
    let candidates: Vec<FrameId> = kfs.iter().copied().take(kfs.len() - 2).collect();
    let newest = position(80).unwrap();
    let score = |i: FrameId| {
        let pi = position(i).unwrap();
        let d: f64 = candidates
            .iter()
            .map(|&j| 1.0 / ((pi - position(j).unwrap()).length() + 1e-5))
            .sum();
        (pi - newest).length().sqrt() * d
    };
    let best = candidates
        .iter()
        .copied()
        .min_by(|a, b| score(*a).total_cmp(&score(*b)))
        .unwrap();
    assert_eq!(plan.keyframes_to_marg, vec![best]);
    assert!(!plan.keyframes_to_marg.contains(&70) && !plan.keyframes_to_marg.contains(&80));
}

/// T5: the keyframe rule.
#[test]
fn t5_the_keyframe_rule() {
    let t = tuning();
    assert!(is_keyframe(6, 4, 6, &t));
    assert!(!is_keyframe(6, 4, 5, &t));
    assert!(!is_keyframe(7, 3, 9, &t));
    assert!(!is_keyframe(0, 0, 9, &t));
}

/// T5, M7-M10: after every frame of a run, no observation, edge or prior
/// variable refers to a frame outside the window, the prior is the
/// keyframe poses then the oldest state, all linearized, and the window
/// holds at most 2 states and one keyframe over the bound after it.
#[test]
fn t5_the_window_bookkeeping_holds_on_every_frame() {
    let walk = StereoWalk::new(6.0, biased(), Some((datasheet(), 3)));
    let mut marginalized = 0;
    let mut dropped_keyframes = 0;
    let mut previous_kfs: BTreeSet<FrameId> = BTreeSet::new();
    // The synthetic tracks outlive real ones (5 keyframes in 10 s), so a
    // smaller window and a stricter keyframe share exercise M5-M6 here.
    let tuning = OdometryTuning {
        max_keyframes: 4,
        keyframe_tracked_share: 0.95,
        ..OdometryTuning::default()
    };
    drive(&walk, &tuning, 0, 120, Some(0.5), false, |_, w, report| {
        let frames: BTreeSet<FrameId> = w.states.keys().chain(w.poses.keys()).copied().collect();
        for (_, lm) in w.landmarks.iter() {
            assert!(frames.contains(&lm.host), "host {} outside", lm.host);
            assert!(lm.observations.len() >= 2 || report.keyframe);
            for (frame, _) in lm.observations.keys() {
                assert!(frames.contains(frame), "observation in {frame} outside");
            }
        }
        for (from, edge) in &w.imu {
            assert!(w.states.contains_key(from) && w.states.contains_key(&edge.to));
        }
        assert!(w.keyframes.iter().all(|k| frames.contains(k)));
        if report.marginalized {
            marginalized += 1;
            assert!(w.states.len() <= 2, "{} states", w.states.len());
            // A keyframe leaves only while a keyframe state converts (M5), so
            // a new keyframe may hold one more until then.
            assert!(
                w.keyframes.len() <= tuning.max_keyframes + 1,
                "{} keyframes",
                w.keyframes.len()
            );
            let n = w.prior.order.len();
            let oldest = *w.states.keys().next().unwrap();
            assert_eq!(w.prior.order[n - 1].id, oldest);
            assert_eq!(w.prior.order[n - 1].kind, VarKind::State);
            let poses: Vec<FrameId> = w.prior.order[..n - 1].iter().map(|e| e.id).collect();
            assert_eq!(poses, w.poses.keys().copied().collect::<Vec<_>>());
            assert!(w.poses.values().all(PoseWithLin::is_linearized));
            assert!(w.states[&oldest].is_linearized());
            dropped_keyframes += previous_kfs
                .iter()
                .filter(|k| !w.keyframes.contains(k) && **k < oldest)
                .count();
        }
        previous_kfs = w.keyframes.clone();
    });
    assert!(marginalized > 100, "{marginalized}");
    assert!(dropped_keyframes > 0, "no keyframe was dropped");
}
