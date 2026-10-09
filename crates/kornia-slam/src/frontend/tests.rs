use kornia_sensors::imu::ImuMeasurement;

use super::*;

const FRONTEND_FIXTURE: &str = include_str!("../../tests/fixtures/frontend/frontend.txt");
const HILTI_FIXTURE: &str = include_str!("../../tests/fixtures/frontend/hilti.txt");

/// Deterministic texture of 4x4 blocks of noise. The right view is the left
/// view shifted by `disparity` pixels, so a point at `x` appears at `x - d`.
pub(crate) fn synthetic_pair(
    width: usize,
    height: usize,
    disparity: usize,
) -> (Image<u8, 1>, Image<u8, 1>) {
    let value = |x: usize, y: usize| -> u8 {
        let mut s = ((x / 4) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ ((y / 4) as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
        s ^= s >> 29;
        s = s.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        s ^= s >> 32;
        (s & 0xFF) as u8
    };
    let size = ImageSize { width, height };
    let left: Vec<u8> = (0..height)
        .flat_map(|y| (0..width).map(move |x| value(x, y)))
        .collect();
    let right: Vec<u8> = (0..height)
        .flat_map(|y| (0..width).map(move |x| value(x + disparity, y)))
        .collect();
    (
        Image::from_size_slice(size, &left).unwrap(),
        Image::from_size_slice(size, &right).unwrap(),
    )
}

fn fnv(bytes: impl IntoIterator<Item = u8>) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn f32_bytes(values: &[f32]) -> impl Iterator<Item = u8> + '_ {
    values.iter().flat_map(|v| v.to_bits().to_le_bytes())
}

fn fixture_value<'a>(fixture: &'a str, key: &str) -> &'a str {
    fixture
        .lines()
        .find_map(|line| line.strip_prefix(key)?.strip_prefix(' '))
        .unwrap_or_else(|| panic!("fixture has no {key}"))
}

fn pinhole(fx: f64, fy: f64, cx: f64, cy: f64) -> PinholeCamera {
    PinholeCamera {
        fx,
        fy,
        cx,
        cy,
        k1: 0.0,
        k2: 0.0,
        p1: 0.0,
        p2: 0.0,
    }
}

fn hilti_fisheye() -> FisheyeCamera {
    FisheyeCamera {
        fx: 461.64,
        fy: 459.72,
        cx: 732.95,
        cy: 720.54,
        k1: 0.0344,
        k2: -0.0216,
        k3: 0.0031,
        k4: -0.0005,
    }
}

fn hilti_mapping() -> FisheyeMapping {
    let fisheye = hilti_fisheye();
    FisheyeMapping {
        camera: pinhole(fisheye.fx, fisheye.fy, fisheye.cx, fisheye.cy),
        fisheye,
        min_bearing_z: MAX_INCIDENCE_DEG.to_radians().cos(),
    }
}

/// Features whose orientation, descriptor and octave all encode their index.
fn indexed_features(keypoints_xy: Vec<[f32; 2]>) -> OrbFeatures {
    let n = keypoints_xy.len();
    OrbFeatures {
        keypoints_xy,
        orientations: (0..n).map(|i| i as f32 * 0.01).collect(),
        descriptors: (0..n)
            .map(|i| {
                let mut d = [0u8; 32];
                d[..8].copy_from_slice(&(i as u64).to_le_bytes());
                d
            })
            .collect(),
        octaves: (0..n).map(|i| (i % 8) as u8).collect(),
    }
}

fn input<'a>(image: &'a Image<u8, 1>, right: Option<&'a Image<u8, 1>>) -> SensorFrame<'a> {
    const NO_SAMPLES: &[ImuMeasurement] = &[];
    SensorFrame {
        idx: 3,
        timestamp_sec: 0.1,
        image,
        right_image: right,
        imu_samples: NO_SAMPLES,
    }
}

fn stereo_frontend() -> OrbFrontend {
    let rig = SensorRig::new(pinhole(435.0, 435.0, 188.0, 120.0)).with_stereo_baseline(0.11);
    let detector = OrbDetector {
        n_keypoints: 1000,
        ..OrbDetector::default()
    };
    OrbFrontend::new(detector, &rig)
}

#[test]
fn stereo_preparation_matches_the_recorded_baseline() {
    let (left, right) = synthetic_pair(376, 240, 8);
    let mut frontend = stereo_frontend();
    let frame = frontend.prepare(&input(&left, Some(&right))).unwrap();
    let f = &frame.features;

    let hashes = [
        (
            "keypoints_xy",
            fnv(f32_bytes(&f.keypoints_xy.concat()).collect::<Vec<_>>()),
        ),
        (
            "orientations",
            fnv(f32_bytes(&f.orientations).collect::<Vec<_>>()),
        ),
        ("descriptors", fnv(f.descriptors.concat())),
        ("octaves", fnv(f.octaves.iter().copied())),
        (
            "u_right",
            fnv(f32_bytes(&frame.u_right).collect::<Vec<_>>()),
        ),
        ("depth", fnv(f32_bytes(&frame.depth).collect::<Vec<_>>())),
        ("colors", fnv(frame.keypoint_colors.concat())),
    ];
    assert_eq!(
        f.keypoints_xy.len().to_string(),
        fixture_value(FRONTEND_FIXTURE, "keypoints")
    );
    for (key, hash) in hashes {
        assert_eq!(
            format!("{hash:016x}"),
            fixture_value(FRONTEND_FIXTURE, key),
            "{key}"
        );
    }
    let matched = frontend.observation().stereo_matched.unwrap();
    assert_eq!(
        matched.to_string(),
        fixture_value(FRONTEND_FIXTURE, "stereo_matched")
    );
    assert_eq!(frame.idx, 3);
    assert_eq!(frame.image_size, left.size());
    assert_eq!(frontend.observation().keypoints_xy, f.keypoints_xy);
}

#[test]
fn fisheye_mapping_matches_the_recorded_baseline() {
    let grid: Vec<[f32; 2]> = (0..40)
        .flat_map(|j| (0..40).map(move |i| [i as f32 * 1471.0 / 39.0, j as f32 * 1439.0 / 39.0]))
        .collect();
    let mut features = indexed_features(grid);
    hilti_mapping().map_keypoints(&mut features);

    let hash = fnv(f32_bytes(&features.keypoints_xy.concat())
        .chain(f32_bytes(&features.orientations))
        .chain(features.descriptors.concat())
        .chain(features.octaves.iter().copied())
        .collect::<Vec<_>>());
    assert_eq!(
        features.keypoints_xy.len().to_string(),
        fixture_value(HILTI_FIXTURE, "kept")
    );
    assert_eq!(format!("{hash:016x}"), fixture_value(HILTI_FIXTURE, "hash"));
}

#[test]
fn fisheye_principal_point_is_a_fixed_point() {
    let fisheye = hilti_fisheye();
    let mut features = indexed_features(vec![[fisheye.cx as f32, fisheye.cy as f32]]);
    hilti_mapping().map_keypoints(&mut features);
    assert_eq!(features.keypoints_xy.len(), 1);
    let [u, v] = features.keypoints_xy[0];
    assert!((u as f64 - fisheye.cx).abs() < 1e-2);
    assert!((v as f64 - fisheye.cy).abs() < 1e-2);
}

#[test]
fn dropped_fisheye_keypoints_take_their_attributes_with_them() {
    let fisheye = hilti_fisheye();
    let centre = [fisheye.cx as f32, fisheye.cy as f32];
    // The corners lie beyond the incidence cap; the centre points do not.
    let mut features = indexed_features(vec![[1.0, 1.0], centre, [1471.0, 1.0], centre]);
    hilti_mapping().map_keypoints(&mut features);

    let kept_indices: Vec<u64> = features
        .descriptors
        .iter()
        .map(|d| u64::from_le_bytes(d[..8].try_into().unwrap()))
        .collect();
    assert_eq!(kept_indices, [1, 3]);
    assert_eq!(features.orientations, [0.01, 0.03]);
    assert_eq!(features.octaves, [1, 3]);
    assert_eq!(features.keypoints_xy.len(), 2);
}

#[test]
fn featureless_images_prepare_empty_frames() {
    let blank = Image::from_size_val(
        ImageSize {
            width: 376,
            height: 240,
        },
        128u8,
    )
    .unwrap();
    let mut frontend = stereo_frontend();
    let frame = frontend.prepare(&input(&blank, Some(&blank))).unwrap();
    assert!(frame.features.keypoints_xy.is_empty());
    assert!(frame.keypoint_colors.is_empty());
    assert!(frame.depth.is_empty());
    assert_eq!(frontend.observation().stereo_matched, Some(0));
}

/// No stereo matches is a valid result, distinct from missing right-image input.
#[test]
fn unrelated_stereo_views_prepare_without_matches() {
    let (left, _) = synthetic_pair(376, 240, 8);
    let blank = Image::from_size_val(left.size(), 128u8).unwrap();
    let mut frontend = stereo_frontend();
    let frame = frontend.prepare(&input(&left, Some(&blank))).unwrap();
    assert!(!frame.features.keypoints_xy.is_empty());
    assert_eq!(frame.depth.len(), frame.features.keypoints_xy.len());
    assert!(frame.depth.iter().all(|&d| d <= 0.0));
    assert_eq!(frontend.observation().stereo_matched, Some(0));
}

#[test]
fn fisheye_observation_keeps_raw_image_keypoints() {
    let (image, _) = synthetic_pair(1472, 1440, 0);
    let fisheye = hilti_fisheye();
    let rig = SensorRig::new(pinhole(fisheye.fx, fisheye.fy, fisheye.cx, fisheye.cy))
        .with_fisheye(fisheye);
    let mut frontend = OrbFrontend::new(OrbDetector::default(), &rig);
    let frame = frontend.prepare(&input(&image, None)).unwrap();

    let observed = &frontend.observation().keypoints_xy;
    assert!(observed.len() > frame.features.keypoints_xy.len());
    assert!(frontend.observation().stereo_matched.is_none());
}
