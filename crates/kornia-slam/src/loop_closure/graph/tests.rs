use kornia_3d::pose::Pose3d;
use kornia_algebra::{SO3F64, Vec3F64};

use super::{
    GraphManifold, KeyframeOutcome, LoopCandidate, LoopMeasurement, LoopQuery, LoopVerifier,
    PoseGraph, PoseGraphConfig, RejectAll,
};

const RADIUS_M: f64 = 2.0;
const STEP_M: f64 = 0.1;
const STEP_S: f64 = 0.2;
/// Injected odometry yaw error per radian turned: 0.19 rad (10.8 deg) a lap.
const YAW_DRIFT: f64 = 0.03;

fn config() -> PoseGraphConfig {
    PoseGraphConfig {
        manifold: GraphManifold::Gravity(Vec3F64::new(0.0, 0.0, 1.0)),
        ..PoseGraphConfig::default()
    }
}

fn keyframes_per_lap() -> usize {
    (std::f64::consts::TAU * RADIUS_M / STEP_M).round() as usize
}

/// A body-to-world transform: a yaw about +Z, then `translation`.
fn body_to_world(yaw: f64, translation: Vec3F64) -> Pose3d {
    Pose3d::new(
        SO3F64::exp(Vec3F64::new(0.0, 0.0, yaw)).matrix(),
        translation,
    )
}

fn yaw_of(transform: &Pose3d) -> f64 {
    let x_axis = transform.rotation.col(0);
    x_axis.y.atan2(x_axis.x)
}

/// The true body-to-world at keyframe `index` on a counter-clockwise circle through the origin.
fn truth(index: usize) -> Pose3d {
    let arc = index as f64 * STEP_M / RADIUS_M;
    body_to_world(
        arc,
        Vec3F64::new(RADIUS_M * arc.sin(), RADIUS_M * (1.0 - arc.cos()), 0.0),
    )
}

/// World-to-camera odometry: the truth's increments with each turn scaled by `1 + YAW_DRIFT`,
/// integrated, as with a gyro scale error.
fn odometry_trajectory(keyframes: usize) -> Vec<Pose3d> {
    let mut poses = vec![truth(0)];
    for index in 1..keyframes {
        let step = truth(index - 1).inverse().compose(&truth(index));
        let drifted = body_to_world(yaw_of(&step) * (1.0 + YAW_DRIFT), step.translation);
        poses.push(poses[index - 1].compose(&drifted));
    }
    poses.iter().map(Pose3d::inverse).collect()
}

fn t_s(index: usize) -> f64 {
    index as f64 * STEP_S
}

fn index_of(t_s: f64) -> usize {
    (t_s / STEP_S).round() as usize
}

/// An ORACLE, for graph mechanics only: it measures the loop from the truth and accepts when the
/// candidate truly lies within 0.3 m.
struct TruthVerifier {
    accepts_left: usize,
}

impl LoopVerifier<()> for TruthVerifier {
    fn verify(
        &mut self,
        query: &LoopQuery<'_, ()>,
        candidate: &LoopCandidate<'_, ()>,
    ) -> Option<LoopMeasurement> {
        let query_truth = truth(index_of(query.t_s));
        let candidate_truth = truth(index_of(candidate.t_s));
        let candidate_from_query = candidate_truth.inverse().compose(&query_truth);
        if self.accepts_left == 0 || candidate_from_query.translation.length() > 0.3 {
            return None;
        }
        self.accepts_left -= 1;
        Some(LoopMeasurement {
            candidate_to_query: candidate_from_query.inverse(),
            sigma_m: 0.02,
            sigma_rad: 0.01,
        })
    }
}

/// Consulted, refuses every candidate.
struct RefuseAll {
    consulted: usize,
}

impl LoopVerifier<()> for RefuseAll {
    fn verify(
        &mut self,
        _: &LoopQuery<'_, ()>,
        _: &LoopCandidate<'_, ()>,
    ) -> Option<LoopMeasurement> {
        self.consulted += 1;
        None
    }
}

fn feed(
    graph: &mut PoseGraph<()>,
    odometry: &[Pose3d],
    range: std::ops::Range<usize>,
    verifier: &mut dyn LoopVerifier<()>,
) {
    for index in range {
        graph
            .add_keyframe(index, t_s(index), odometry[index], (), verifier)
            .expect("the solve converges");
    }
}

fn end_error_m(graph: &PoseGraph<()>, odometry: &[Pose3d]) -> f64 {
    let last = odometry.len() - 1;
    let estimate = graph.estimate(&odometry[last]).inverse().translation;
    (estimate - truth(last).translation).length()
}

#[test]
fn one_correct_loop_edge_removes_a_lap_of_yaw_drift() {
    let keyframes = keyframes_per_lap() + 5;
    let odometry = odometry_trajectory(keyframes);
    let raw_error =
        (odometry[keyframes - 1].inverse().translation - truth(keyframes - 1).translation).length();
    assert!(
        raw_error > 0.3,
        "the lap must drift for the test to mean anything: {raw_error}"
    );

    let mut graph = PoseGraph::new(config());
    feed(
        &mut graph,
        &odometry,
        0..keyframes,
        &mut TruthVerifier { accepts_left: 1 },
    );

    assert_eq!(graph.loops_accepted(), 1);
    let error = end_error_m(&graph, &odometry);
    assert!(error < 0.05, "end error {error} m, raw {raw_error} m");
}

#[test]
fn fifty_laps_make_at_most_twice_the_nodes_of_two() {
    let per_lap = keyframes_per_lap();
    let odometry = odometry_trajectory(50 * per_lap + 1);
    let mut graph = PoseGraph::new(config());
    let mut verifier = TruthVerifier {
        accepts_left: usize::MAX,
    };

    feed(&mut graph, &odometry, 0..2 * per_lap + 1, &mut verifier);
    let after_two = graph.nodes().len();
    feed(
        &mut graph,
        &odometry,
        2 * per_lap + 1..odometry.len(),
        &mut verifier,
    );
    let after_fifty = graph.nodes().len();

    assert!(
        after_fifty <= 2 * after_two,
        "{after_fifty} nodes after 50 laps, {after_two} after 2"
    );
    let error = end_error_m(&graph, &odometry);
    assert!(error < 0.05, "end error after 50 laps {error} m");
}

#[test]
fn a_refused_candidate_changes_nothing() {
    let odometry = odometry_trajectory(keyframes_per_lap() * 3 / 2);
    let mut refused = PoseGraph::new(config());
    let mut verifier = RefuseAll { consulted: 0 };
    feed(&mut refused, &odometry, 0..odometry.len(), &mut verifier);
    let mut ungated = PoseGraph::new(PoseGraphConfig {
        loop_gate_sigmas: 0.0,
        ..config()
    });
    feed(
        &mut ungated,
        &odometry,
        0..odometry.len(),
        &mut RefuseAll { consulted: 0 },
    );

    assert!(verifier.consulted > 0, "no candidate was ever offered");
    assert_eq!(refused.loops_accepted(), 0);
    assert_eq!(refused.nodes(), ungated.nodes());
    assert!(refused.edges().eq(ungated.edges()));
    assert_eq!(refused.correction(), ungated.correction());
}

#[test]
fn dedup_skips_a_keyframe_inside_an_existing_node() {
    let at = |x: f64, yaw: f64| body_to_world(yaw, Vec3F64::new(x, 0.0, 0.0)).inverse();
    let mut graph = PoseGraph::new(config());
    let first = graph
        .add_keyframe(0, 0.0, at(0.0, 0.0), (), &mut RejectAll)
        .unwrap();
    let near = graph
        .add_keyframe(1, 0.2, at(0.1, 0.05), (), &mut RejectAll)
        .unwrap();
    let turned = graph
        .add_keyframe(2, 0.4, at(0.1, 0.5), (), &mut RejectAll)
        .unwrap();
    let far = graph
        .add_keyframe(3, 0.6, at(0.6, 0.5), (), &mut RejectAll)
        .unwrap();

    assert!(matches!(first, KeyframeOutcome::NewNode(_)));
    assert_eq!(near, KeyframeOutcome::Skipped);
    assert!(matches!(turned, KeyframeOutcome::NewNode(_)));
    assert!(matches!(far, KeyframeOutcome::NewNode(_)));
    assert_eq!(graph.nodes().len(), 3);
    assert_eq!(graph.node_of_keyframe(1), None);
    assert_eq!(graph.edges().count(), 2);
    let (keyframes, _, _) = graph.poses_before_after();
    assert_eq!(keyframes, vec![0, 2, 3]);
}
