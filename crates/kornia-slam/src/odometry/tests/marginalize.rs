//! T1: sequential marginalization equals the batch solve on a linear
//! problem.

use faer::Mat;
use faer::prelude::Solve;
use kornia_algebra::{SMatF64, SVecF64};

use crate::odometry::linearize::{DenseSystem, LandmarkBlock, Order, VarKind};
use crate::odometry::marginalize::{Fate, MargPrior, marginalize_system, nullspace_residual};
use crate::odometry::state::{PoseWithLin, STATE_DOF, StateWithLin, pose_plus};
use crate::tracking::pose_estimation::pose_inertial::FrameNavState;
use kornia_3d::pose::Pose3d;
use kornia_algebra::{SO3F64, Vec3F64};
use kornia_sensors::imu::ImuBias;

const STATES: usize = 40;
const WINDOW: usize = 7;

/// Deterministic uniform samples in [-1, 1).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

/// `r = J_a x_a + J_b x_b − z` between consecutive states, unit information.
struct Chain {
    from: usize,
    j_from: SMatF64<15, 15>,
    j_to: SMatF64<15, 15>,
    z: SVecF64<15>,
}

/// `r = A x_state[0..6] + B l − z` per observation, unit weight.
struct Landmark {
    rows: Vec<(usize, SMatF64<2, 6>, SMatF64<2, 3>, SVecF64<2>)>,
}

impl Landmark {
    fn last(&self) -> usize {
        self.rows.iter().map(|r| r.0).max().unwrap_or(0)
    }
}

struct Problem {
    prior_z: SVecF64<15>,
    chains: Vec<Chain>,
    landmarks: Vec<Landmark>,
}

fn problem() -> Problem {
    let mut rng = Lcg(17);
    let mut mat =
        |rows: usize, cols: usize| -> Vec<f64> { (0..rows * cols).map(|_| rng.next()).collect() };
    let chains = (0..STATES - 1)
        .map(|from| {
            let a = mat(15, 15);
            let b = mat(15, 15);
            let z = mat(15, 1);
            Chain {
                from,
                j_from: SMatF64::from_row_slice(&a),
                j_to: SMatF64::<15, 15>::from_row_slice(&b) + SMatF64::identity() * 3.0,
                z: SVecF64::from_row_slice(&z),
            }
        })
        .collect();
    let prior_z = SVecF64::from_row_slice(&mat(15, 1));
    let mut landmarks = Vec::new();
    for first in 0..STATES - 1 {
        for k in 0..3 {
            let span = 2 + (first + k) % 5;
            let rows = (first..(first + span).min(STATES))
                .map(|state| {
                    (
                        state,
                        SMatF64::from_row_slice(&mat(2, 6)),
                        SMatF64::from_row_slice(&mat(2, 3)),
                        SVecF64::from_row_slice(&mat(2, 1)),
                    )
                })
                .collect::<Vec<_>>();
            if rows.len() >= 2 {
                landmarks.push(Landmark { rows });
            }
        }
    }
    Problem {
        prior_z,
        chains,
        landmarks,
    }
}

/// Adds a chain factor evaluated at `x` to `system` over `order`.
fn add_chain(system: &mut DenseSystem, order: &Order, chain: &Chain, x: &[SVecF64<15>]) {
    let r = chain.j_from * x[chain.from] + chain.j_to * x[chain.from + 1] - chain.z;
    let a = order
        .pose_offset(chain.from as i64)
        .map(|o| (o, &chain.j_from));
    let b = order
        .pose_offset(chain.from as i64 + 1)
        .map(|o| (o, &chain.j_to));
    system.add_pair(a, b, &SMatF64::<15, 15>::identity(), &r);
}

/// The landmark's rows at `x`, `l` into `block`.
fn landmark_block(
    block: &mut LandmarkBlock,
    order: &Order,
    lm: &Landmark,
    x: &[SVecF64<15>],
    l: &SVecF64<3>,
) {
    block.begin(lm.rows.iter().filter_map(|r| order.pose_offset(r.0 as i64)));
    for (state, a, b, z) in &lm.rows {
        let pose = x[*state].fixed_rows::<6>(0).into_owned();
        let r = a * pose + b * l - z;
        block.add_row(order.pose_offset(*state as i64), None, a, b, &r, 1.0);
    }
}

fn add_initial_prior(system: &mut DenseSystem, z: &SVecF64<15>, x0: &SVecF64<15>, offset: usize) {
    let r = x0 - z;
    system.add_pair::<15, 15, 15>(
        Some((offset, &SMatF64::identity())),
        None,
        &SMatF64::identity(),
        &r,
    );
}

fn solve(system: &DenseSystem) -> Vec<f64> {
    let n = system.dim();
    let rhs = Mat::<f64>::from_fn(n, 1, |r, _| -system.b[r]);
    let x = system.h.llt(faer::Side::Lower).unwrap().solve(&rhs);
    (0..n).map(|r| x[(r, 0)]).collect()
}

/// T1: forty states, eliminated oldest first with every landmark whose
/// observations are complete, against one dense solve of everything. The
/// window takes one Gauss-Newton step per state (exact on a linear problem),
/// so the prior is re-centred at a moved point every time.
#[test]
fn sequential_marginalization_is_the_batch_solve() {
    let p = problem();
    let mut x = vec![SVecF64::<15>::zeros(); STATES];
    let mut l = vec![SVecF64::<3>::zeros(); p.landmarks.len()];
    let mut lin: Vec<Option<SVecF64<15>>> = vec![None; STATES];
    let mut landmark_done = vec![false; p.landmarks.len()];
    let mut chain_done = vec![false; p.chains.len()];
    let mut prior: Option<MargPrior> = None;
    let mut window: Vec<usize> = Vec::new();
    let mut absorbed_landmarks: Vec<usize> = Vec::new();
    let mut absorbed_chains: Vec<usize> = Vec::new();
    let mut block = LandmarkBlock::default();
    let mut system = DenseSystem::default();
    let mut order = Order::default();

    let delta_of = |prior: &MargPrior, x: &[SVecF64<15>], lin: &[Option<SVecF64<15>>]| {
        prior
            .order
            .iter()
            .flat_map(|e| {
                let i = e.id as usize;
                let d = x[i] - lin[i].unwrap_or(x[i]);
                (0..e.kind.dim()).map(move |k| d[k]).collect::<Vec<_>>()
            })
            .collect::<Vec<f64>>()
    };

    for k in 0..STATES {
        window.push(k);
        if k > 0 {
            x[k] = x[k - 1];
        }
        order.clear();
        for &s in &window {
            order.push(s as i64, VarKind::State);
        }
        // One GN step on the window.
        system.reset(order.dim());
        match &prior {
            Some(prior) => {
                prior
                    .accumulate(&delta_of(prior, &x, &lin), &order, &mut system)
                    .unwrap();
            }
            None => add_initial_prior(&mut system, &p.prior_z, &x[0], 0),
        }
        for (c, chain) in p.chains.iter().enumerate() {
            if !chain_done[c] && chain.from < k {
                add_chain(&mut system, &order, chain, &x);
            }
        }
        let active: Vec<usize> = (0..p.landmarks.len())
            .filter(|&i| !landmark_done[i] && p.landmarks[i].rows[0].0 <= k)
            .collect();
        let mut blocks = Vec::new();
        for &i in &active {
            landmark_block(&mut block, &order, &p.landmarks[i], &x, &l[i]);
            block.eliminate_into(&mut system);
            blocks.push((i, block.clone()));
        }
        let inc = solve(&system);
        for (slot, &s) in window.iter().enumerate() {
            for d in 0..STATE_DOF {
                x[s][d] += inc[slot * STATE_DOF + d];
            }
        }
        for (i, b) in &blocks {
            l[*i] += b.back_substitute(&inc);
        }

        if window.len() > WINDOW || k == STATES - 1 {
            let oldest = window[0];
            order.clear();
            for &s in &window {
                order.push(s as i64, VarKind::State);
            }
            system.reset(order.dim());
            match &prior {
                Some(prior) => {
                    prior
                        .accumulate(&delta_of(prior, &x, &lin), &order, &mut system)
                        .unwrap();
                }
                None => add_initial_prior(&mut system, &p.prior_z, &x[0], 0),
            }
            for (c, chain) in p.chains.iter().enumerate() {
                if !chain_done[c] && chain.from < k {
                    add_chain(&mut system, &order, chain, &x);
                    chain_done[c] = true;
                    absorbed_chains.push(c);
                }
            }
            for i in 0..p.landmarks.len() {
                if !landmark_done[i] && p.landmarks[i].last() <= k {
                    landmark_block(&mut block, &order, &p.landmarks[i], &x, &l[i]);
                    assert_eq!(block.rows(), p.landmarks[i].rows.len());
                    assert!(block.eliminate_reduced_into(&mut system));
                    landmark_done[i] = true;
                    absorbed_landmarks.push(i);
                }
            }
            let last = k == STATES - 1;
            let (mut next, pinv) = marginalize_system(&system, &order, |id| {
                if id as usize == oldest && !last {
                    Fate::Drop
                } else {
                    Fate::Keep
                }
            })
            .unwrap();
            assert!(!pinv);
            for e in &next.order {
                lin[e.id as usize].get_or_insert(x[e.id as usize]);
            }
            next.recentre(&delta_of(&next, &x, &lin));
            prior = Some(next);
            if !last {
                window.remove(0);
            }
        }
    }

    // The batch: every factor over every variable, at x = 0.
    let n_states = STATES * STATE_DOF;
    let n = n_states + 3 * p.landmarks.len();
    let mut batch = DenseSystem::default();
    batch.reset(n);
    let mut all = Order::default();
    for s in 0..STATES {
        all.push(s as i64, VarKind::State);
    }
    let zeros = vec![SVecF64::<15>::zeros(); STATES];
    add_initial_prior(&mut batch, &p.prior_z, &zeros[0], 0);
    for chain in &p.chains {
        add_chain(&mut batch, &all, chain, &zeros);
    }
    for (i, lm) in p.landmarks.iter().enumerate() {
        let lo = n_states + 3 * i;
        for (state, a, b, z) in &lm.rows {
            let mut j = SMatF64::<2, 21>::zeros();
            j.fixed_view_mut::<2, 6>(0, 0).copy_from(a);
            j.fixed_view_mut::<2, 3>(0, 6).copy_from(b);
            let jtj = j.transpose() * j;
            let jtr = j.transpose() * (-z);
            let idx: Vec<usize> = (0..6)
                .map(|d| state * STATE_DOF + d)
                .chain((0..3).map(|d| lo + d))
                .collect();
            for (r, &ir) in idx.iter().enumerate() {
                batch.b[ir] += jtr[r];
                for (c, &ic) in idx.iter().enumerate() {
                    batch.h[(ir, ic)] += jtj[(r, c)];
                }
            }
        }
    }
    let x_batch = solve(&batch);
    let newest = STATES - 1;
    let scale = (0..STATE_DOF)
        .map(|d| x_batch[newest * STATE_DOF + d].abs())
        .fold(0.0, f64::max);
    for d in 0..STATE_DOF {
        let (b, s) = (x_batch[newest * STATE_DOF + d], x[newest][d]);
        assert!(
            (b - s).abs() <= 1e-9 * scale,
            "newest state dof {d}: batch {b} sequential {s}"
        );
    }

    // The prior against the batch Schur complement of what it absorbed.
    let prior = prior.unwrap();
    let kept: Vec<usize> = prior.order.iter().map(|e| e.id as usize).collect();
    let mut absorbed = DenseSystem::default();
    absorbed.reset(n);
    add_initial_prior(&mut absorbed, &p.prior_z, &zeros[0], 0);
    for &c in &absorbed_chains {
        add_chain(&mut absorbed, &all, &p.chains[c], &zeros);
    }
    for &i in &absorbed_landmarks {
        let lo = n_states + 3 * i;
        for (state, a, b, z) in &p.landmarks[i].rows {
            let mut j = SMatF64::<2, 21>::zeros();
            j.fixed_view_mut::<2, 6>(0, 0).copy_from(a);
            j.fixed_view_mut::<2, 3>(0, 6).copy_from(b);
            let jtj = j.transpose() * j;
            let jtr = j.transpose() * (-z);
            let idx: Vec<usize> = (0..6)
                .map(|d| state * STATE_DOF + d)
                .chain((0..3).map(|d| lo + d))
                .collect();
            for (r, &ir) in idx.iter().enumerate() {
                absorbed.b[ir] += jtr[r];
                for (c, &ic) in idx.iter().enumerate() {
                    absorbed.h[(ir, ic)] += jtj[(r, c)];
                }
            }
        }
    }
    let keep: Vec<usize> = kept
        .iter()
        .flat_map(|s| (0..STATE_DOF).map(move |d| s * STATE_DOF + d))
        .collect();
    let marg: Vec<usize> = (0..n).filter(|i| !keep.contains(i)).collect();
    let schur =
        crate::mapping::marginalization::schur_complement(&absorbed.h, &absorbed.b, &keep, &marg)
            .unwrap();
    let dim = keep.len();
    let lin_stack: Vec<f64> = kept
        .iter()
        .flat_map(|&s| (0..STATE_DOF).map(move |d| (s, d)))
        .map(|(s, d)| lin[s].map_or(0.0, |v| v[d]))
        .collect();
    let h_scale = (0..dim).map(|i| schur.h[(i, i)].abs()).fold(0.0, f64::max);
    let b_scale = schur.b.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    for r in 0..dim {
        for c in 0..dim {
            let (s, q) = (schur.h[(r, c)], prior.h[(r, c)]);
            assert!(
                (s - q).abs() <= 1e-9 * h_scale,
                "H[{r},{c}]: batch {s} sequential {q}"
            );
        }
        // In x: the prior's linear term is b − H·lin.
        let ours = prior.b[r]
            - (0..dim)
                .map(|c| prior.h[(r, c)] * lin_stack[c])
                .sum::<f64>();
        assert!(
            (schur.b[r] - ours).abs() <= 1e-9 * b_scale.max(h_scale),
            "b[{r}]: batch {} sequential {ours}",
            schur.b[r]
        );
    }
}

/// A free state moves its linearization point; a linearized one keeps it
/// and accumulates its delta; its pose keeps the same linearization.
#[test]
fn a_linearized_state_keeps_its_first_estimate() {
    let start = FrameNavState {
        pose_cw: Pose3d::new(
            SO3F64::exp(Vec3F64::new(0.1, -0.2, 0.3)).matrix(),
            Vec3F64::new(1.0, 2.0, 3.0),
        ),
        velocity: Vec3F64::new(0.5, 0.0, -0.1),
        bias: ImuBias::default(),
    };
    let step = SVecF64::<15>::from_fn(|r, _| 1e-2 * (r as f64 + 1.0));
    let mut free = StateWithLin::new(start);
    free.apply_increment(&step);
    assert!(!free.is_linearized());
    assert_eq!(free.jacobian_point().pose_cw, free.value().pose_cw);
    assert_eq!(free.value().pose_cw, start.plus(&step).pose_cw);

    let mut fixed = StateWithLin::new(start);
    fixed.set_linearized();
    fixed.apply_increment(&step);
    fixed.apply_increment(&step);
    assert_eq!(fixed.jacobian_point().pose_cw, start.pose_cw);
    assert_eq!(*fixed.delta(), step * 2.0);
    assert_eq!(fixed.value().pose_cw, start.plus(&(step * 2.0)).pose_cw);
    fixed.set_linearized();
    assert_eq!(
        *fixed.delta(),
        step * 2.0,
        "linearizing again keeps the point"
    );

    let pose = fixed.to_pose();
    assert!(pose.is_linearized());
    assert_eq!(pose.jacobian_point(), start.pose_cw);
    let head = (step * 2.0).fixed_rows::<6>(0).into_owned();
    assert_eq!(pose.value(), pose_plus(&start.pose_cw, &head));
    assert_eq!(*pose.delta(), head);

    let six = step.fixed_rows::<6>(0).into_owned();
    let mut keyframe = PoseWithLin::new(start.pose_cw);
    keyframe.apply_increment(&six);
    assert_eq!(keyframe.value(), pose_plus(&start.pose_cw, &six));
    keyframe.set_linearized();
    keyframe.apply_increment(&six);
    assert_eq!(keyframe.jacobian_point(), pose_plus(&start.pose_cw, &six));
    assert_eq!(*keyframe.delta(), six);
}

/// The first state's prior fixes position and yaw only: under it roll and
/// pitch are free, the gauge directions are not.
#[test]
fn the_initial_prior_fixes_position_and_yaw() {
    let prior = MargPrior::initial(0, [0.0, 0.0, -1.0], 1e8, 0.316, 0.1);
    assert_eq!(prior.dim(), STATE_DOF);
    assert!((prior.h[(9, 9)] - 0.316f64.powi(-2)).abs() < 1e-9);
    assert!((prior.h[(14, 14)] - 100.0).abs() < 1e-9);
    let n = nullspace_residual(&prior, [0.0; 3], &[Some([0.0; 3])]);
    for d in [0, 1, 2, 5] {
        assert!(n[d] > 0.1, "direction {d}: {}", n[d]);
    }
    assert_eq!(n[3], 0.0);
    assert_eq!(n[4], 0.0);
    assert_eq!(MargPrior::zero(3).h[(0, 0)], 0.0);
}

/// `KeepPose` keeps the first six dimensions of a state and marginalizes
/// the rest: on a block-diagonal system the kept block is unchanged.
#[test]
fn keep_pose_keeps_the_pose_of_a_state() {
    let mut order = Order::default();
    order.push(1, VarKind::State);
    order.push(2, VarKind::Pose);
    let mut system = DenseSystem::default();
    system.reset(order.dim());
    for i in 0..order.dim() {
        system.h[(i, i)] = 1.0 + i as f64;
        system.b[i] = i as f64;
    }
    let (prior, _) = marginalize_system(&system, &order, |id| {
        if id == 1 { Fate::KeepPose } else { Fate::Drop }
    })
    .unwrap();
    assert_eq!(prior.order.len(), 1);
    assert_eq!(prior.order[0].kind, VarKind::Pose);
    for i in 0..6 {
        assert_eq!(prior.h[(i, i)], 1.0 + i as f64);
        assert_eq!(prior.b[i], i as f64);
    }
}
