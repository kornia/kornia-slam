use kornia_3d::pose::Pose3d;
use kornia_algebra::{SO3F64, Vec3F64};

use super::{GraphManifold, LoopMeasurement, PoseGraph, PoseGraphConfig};

const RADIUS_M: f64 = 2.0;
const STEP_M: f64 = 0.1;
/// Injected odometry yaw error per radian turned: 0.19 rad (10.8 deg) a lap.
const YAW_DRIFT: f64 = 0.03;

fn config() -> PoseGraphConfig {
    PoseGraphConfig {
        manifold: GraphManifold::Gravity(Vec3F64::new(0.0, 0.0, 1.0)),
        walk_m2_per_m: 0.0,
        walk_rad2_per_m: 0.0,
        min_sigma_m: 0.01,
        min_sigma_rad: 0.002,
        loop_cauchy_scale: Some(3.0),
        max_iterations: 50,
        cost_tolerance: 1e-7,
        gradient_tolerance: 1e-7,
        initial_lambda: 1e-3,
    }
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

/// Summed odometry-edge variance along a straight 8 m run, keyframes every `spacing_m`.
fn chain_variance_m2(config: PoseGraphConfig, spacing_m: f64) -> f64 {
    let mut graph = PoseGraph::new(config);
    let keyframes = (8.0 / spacing_m).round() as usize;
    for index in 0..=keyframes {
        let centre = Vec3F64::new(index as f64 * spacing_m, 0.0, 0.0);
        let pose = body_to_world(0.0, centre).inverse();
        graph.add_keyframe(index, pose).unwrap();
    }
    graph.edges().map(|edge| edge.sigma_m * edge.sigma_m).sum()
}

/// A random walk composes to `q * path` at any spacing; a floored per-edge sigma does not, so
/// denser keyframes make a stiffer chain.
#[test]
fn random_walk_odometry_composes_independently_of_keyframe_spacing() {
    let walk = PoseGraphConfig {
        walk_m2_per_m: 0.01,
        min_sigma_m: 1e-6,
        ..config()
    };
    for spacing in [0.1, 0.2, 0.5] {
        let variance = chain_variance_m2(walk.clone(), spacing);
        assert!(
            (variance - 0.08).abs() < 1e-9,
            "{spacing} m: {variance} m^2"
        );
    }
    let floored = |spacing| chain_variance_m2(config(), spacing);
    assert!(floored(0.1) > 4.0 * floored(0.5));
}

/// The loop residual after a solve of a 30-keyframe arc closed by `measurement` from 0 to 29.
fn loop_residual_after(measurement: Pose3d, sigma_rad: f64) -> f64 {
    let mut graph = PoseGraph::new(config());
    for (index, pose) in odometry_trajectory(30).into_iter().enumerate() {
        graph.add_keyframe(index, pose).unwrap();
    }
    let measurement = LoopMeasurement {
        candidate_to_query: measurement,
        sigma_m: 0.05,
        sigma_rad,
    };
    graph
        .add_loop_edges([(0, 29, measurement)])
        .unwrap()
        .loop_residual_after
}

/// Every node keeps `R·g` on the gravity manifold, so a loop measurement tilted in roll by
/// 3 sigma cannot be met. The tilt left at least 3 whitened sigmas on the loop, and the Cauchy
/// weight at or below 0.5, for good; now only the rotation about gravity is compared.
#[test]
fn a_tilted_loop_measurement_leaves_no_residual_on_the_gravity_manifold() {
    let sigma_rad = 0.01;
    let measured = truth(29).inverse().compose(&truth(0));
    let roll = SO3F64::exp(Vec3F64::new(3.0 * sigma_rad, 0.0, 0.0)).matrix();
    let tilted = Pose3d::new(roll * measured.rotation, measured.translation);
    let plain = loop_residual_after(measured, sigma_rad);
    let tilted = loop_residual_after(tilted, sigma_rad);
    assert!(
        (tilted - plain).abs() < 0.1,
        "tilted {tilted} against {plain} sigmas"
    );
}

/// The solved camera centres of the 30-keyframe arc closed from 0 to 29, with the world origin
/// moved `offset` away from it and moved back after.
fn arc_centres_solved_at(offset: Vec3F64) -> Vec<Vec3F64> {
    let shift = Pose3d::new(kornia_algebra::Mat3F64::IDENTITY, offset);
    let mut graph = PoseGraph::new(config());
    for (index, pose) in odometry_trajectory(30).into_iter().enumerate() {
        graph.add_keyframe(index, pose.compose(&shift)).unwrap();
    }
    let measurement = LoopMeasurement {
        candidate_to_query: truth(29).inverse().compose(&truth(0)),
        sigma_m: 0.05,
        sigma_rad: 0.01,
    };
    graph.add_loop_edges([(0, 29, measurement)]).unwrap();
    graph
        .nodes()
        .iter()
        .map(|node| node.pose.camera_centre() + offset)
        .collect()
}

/// The f32 residual `t_b - R_b R_a^T t_a` cancels at the scale of the camera centre, so a
/// graph 2 km from the world origin solved to other poses than the same graph at the origin.
/// The solve now runs about node 0's centre.
#[test]
fn a_graph_far_from_the_origin_solves_as_at_the_origin() {
    let near = arc_centres_solved_at(Vec3F64::ZERO);
    let far = arc_centres_solved_at(Vec3F64::new(2000.0, -1500.0, 30.0));
    let largest = near
        .iter()
        .zip(&far)
        .map(|(a, b)| (*a - *b).length())
        .fold(0.0, f64::max);
    assert!(largest < 1e-4, "solved centres differ by up to {largest} m");
}
