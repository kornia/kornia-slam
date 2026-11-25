//! T7: the two-camera flow frontend on synthetic shifted pairs.

use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, Vec3F64};
use kornia_image::{Image, ImageSize};

use crate::odometry::config::OdometryTuning;
use crate::odometry::frontend::FlowFrontend;
use crate::odometry::window::{Observation, StereoView};

const W: usize = 752;
const H: usize = 480;

/// A smooth random texture: values on an 8 px grid, bilinear between.
fn value(x: f64, y: f64) -> f64 {
    let cell = |i: i64, j: i64| {
        let mut h = (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ (j as u64).wrapping_mul(0xc2b2_ae3d_27d4_eb4f);
        h ^= h >> 29;
        h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        (h >> 40) as f64 / (1u64 << 24) as f64
    };
    let (gx, gy) = (x / 8.0, y / 8.0);
    let (i, j) = (gx.floor() as i64, gy.floor() as i64);
    let (fx, fy) = (gx - i as f64, gy - j as f64);
    let top = cell(i, j) * (1.0 - fx) + cell(i + 1, j) * fx;
    let bottom = cell(i, j + 1) * (1.0 - fx) + cell(i + 1, j + 1) * fx;
    top * (1.0 - fy) + bottom * fy
}

/// The texture seen shifted by `(dx, dy)`, with an optional square patch
/// `(x0, y0, side)` replaced by another texture.
fn image(dx: f64, dy: f64, occluded: Option<(usize, usize, usize)>) -> Image<u8, 1> {
    let data = (0..W * H)
        .map(|k| {
            let (x, y) = ((k % W) as f64, (k / W) as f64);
            let inside = occluded.is_some_and(|(x0, y0, s)| {
                (x0..x0 + s).contains(&(k % W)) && (y0..y0 + s).contains(&(k / W))
            });
            let v = if inside {
                value(y + 1000.0, x + 500.0)
            } else {
                value(x - dx, y - dy)
            };
            (30.0 + 200.0 * v).round() as u8
        })
        .collect();
    Image::new(
        ImageSize {
            width: W,
            height: H,
        },
        data,
    )
    .unwrap()
}

fn views() -> [StereoView; 2] {
    let camera = crate::odometry::synthetic::camera();
    [
        StereoView {
            camera: camera.clone(),
            camera_from_primary: Pose3d::IDENTITY,
        },
        StereoView {
            camera,
            camera_from_primary: Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(-0.11, 0.0, 0.0)),
        },
    ]
}

fn frontend() -> FlowFrontend {
    FlowFrontend::new(&OdometryTuning::default(), &views()).unwrap()
}

fn pairs(obs: &[Observation]) -> Vec<(Observation, Observation)> {
    let left: Vec<&Observation> = obs.iter().filter(|o| o.camera == 0).collect();
    obs.iter()
        .filter(|o| o.camera == 1)
        .filter_map(|r| {
            left.iter()
                .find(|l| l.landmark == r.landmark)
                .map(|l| (**l, *r))
        })
        .collect()
}

/// T7: seeded on the empty set, every new left corner is linked into a
/// right image shifted by 12 px, at that disparity within 0.05 px; refill
/// creates tracks in camera 0 only.
#[test]
fn a_stereo_link_finds_the_disparity() {
    let mut f = frontend();
    let obs = f
        .process(&image(0.0, 0.0, None), &image(-12.0, 0.0, None), None)
        .unwrap()
        .to_vec();
    let left = obs.iter().filter(|o| o.camera == 0).count();
    let links = pairs(&obs);
    assert!(left >= 100, "{left} left corners");
    assert!(
        links.len() as f64 >= 0.9 * left as f64,
        "{} links of {left}",
        links.len()
    );
    for (l, r) in &links {
        assert!((l.pixel[0] - r.pixel[0] - 12.0).abs() < 0.05, "{l:?} {r:?}");
        assert!((l.pixel[1] - r.pixel[1]).abs() < 0.05);
    }
    assert_eq!(f.counts().stereo_links as usize, links.len());
    // The next frame: tracks move by 2 px; every right track has its left.
    let next = f
        .process(&image(2.0, 0.0, None), &image(-10.0, 0.0, None), None)
        .unwrap()
        .to_vec();
    let next_pairs = pairs(&next);
    assert!(
        next_pairs.len() >= links.len() * 9 / 10,
        "{} of {}",
        next_pairs.len(),
        links.len()
    );
    assert!(next.iter().filter(|o| o.camera == 1).all(|r| {
        next.iter()
            .any(|l| l.camera == 0 && l.landmark == r.landmark)
    }));
}

/// T7: a patch occluded in the right image fails the round trip.
#[test]
fn an_occluded_patch_fails_the_round_trip() {
    let mut f = frontend();
    let clear = f
        .process(&image(0.0, 0.0, None), &image(-12.0, 0.0, None), None)
        .unwrap()
        .to_vec();
    let mut g = frontend();
    let occluded = g
        .process(
            &image(0.0, 0.0, None),
            &image(-12.0, 0.0, Some((200, 100, 200))),
            None,
        )
        .unwrap()
        .to_vec();
    let in_patch = |o: &Observation| {
        (212.0..388.0).contains(&o.pixel[0]) && (112.0..288.0).contains(&o.pixel[1])
    };
    let clear_links = pairs(&clear).iter().filter(|(l, _)| in_patch(l)).count();
    let occluded_links = pairs(&occluded).iter().filter(|(l, _)| in_patch(l)).count();
    assert!(clear_links >= 5, "{clear_links}");
    assert_eq!(
        occluded_links, 0,
        "{occluded_links} links into the occluded patch"
    );
    assert!(g.counts().stereo_link_failures as usize >= clear_links);
}

/// T7: a right image 3 px lower than the left (above 0.005 fy = 2.2 px)
/// keeps no link.
#[test]
fn the_row_gate_drops_off_row_links() {
    let mut f = frontend();
    let obs = f
        .process(&image(0.0, 0.0, None), &image(-12.0, 3.0, None), None)
        .unwrap()
        .to_vec();
    assert_eq!(pairs(&obs).len(), 0);
    assert!(f.counts().row_gate_rejects >= 100, "{:?}", f.counts());
}

/// T7: the same images give the same output.
#[test]
fn the_frontend_is_deterministic() {
    let run = || {
        let mut f = frontend();
        let mut all = Vec::new();
        for k in 0..4 {
            let dx = 1.5 * k as f64;
            all.extend(
                f.process(
                    &image(dx, 0.5 * k as f64, None),
                    &image(dx - 12.0, 0.5 * k as f64, None),
                    None,
                )
                .unwrap()
                .to_vec(),
            );
        }
        all
    };
    assert_eq!(run(), run());
}
