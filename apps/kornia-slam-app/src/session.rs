//! `--session-out`: records an online run for the `offline-ba` subcommand and
//! writes it as CBOR at shutdown; `offline-ba` reads it back, and
//! `session-slice` cuts a time span out of it.

use std::collections::{BTreeSet, HashSet};
use std::path::Path;

use kornia_image::ImageSize;
use kornia_sensors::imu::ImuMeasurement;
use kornia_slam::SlamSystem;
use kornia_slam::mapping::session::{
    ImuSample, SESSION_DUMP_VERSION, SessionDump, SessionFrame, SessionPose, session_rig,
};

use crate::trajectory::SmoothedTrajectory;

/// What the system does not keep: the IMU stream as it was fed, and the
/// primary image size.
pub struct SessionRecorder {
    imu: Vec<ImuSample>,
    imu_time_offset_ns: i64,
    image_size: Option<ImageSize>,
}

impl SessionRecorder {
    /// `imu_time_offset_ns` is the offset the source already applied to the
    /// samples, recorded for the report.
    pub fn new(imu_time_offset_ns: i64) -> Self {
        Self {
            imu: Vec::new(),
            imu_time_offset_ns,
            image_size: None,
        }
    }

    /// Records the samples fed with one frame; a sample at or before the last
    /// recorded one is skipped, so the stream stays ordered.
    pub fn record(&mut self, image_size: ImageSize, samples: &[ImuMeasurement]) {
        self.image_size.get_or_insert(image_size);
        for sample in samples {
            if self
                .imu
                .last()
                .is_some_and(|last| sample.timestamp <= last.t)
            {
                continue;
            }
            self.imu.push(ImuSample {
                t: sample.timestamp,
                gyro: sample.gyro.to_array(),
                accel: sample.accel.to_array(),
            });
        }
    }

    /// Writes the session; returns how many frames had no keyframe in the
    /// dump to stand on and were left out.
    pub fn write(
        self,
        path: &Path,
        system: &SlamSystem,
        smoothed: &SmoothedTrajectory,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let (keyframes, landmarks) = system.session_snapshot();
        let kept: HashSet<u64> = keyframes.iter().map(|kf| kf.kf_idx).collect();
        let frames: Vec<SessionFrame> = smoothed
            .anchored_frames(|kf| kept.contains(&(kf as u64)))
            .map(|(timestamp_sec, kf, frame_from_kf)| SessionFrame {
                timestamp_sec,
                reference_kf: kf as u64,
                frame_from_kf: SessionPose::from_pose(&frame_from_kf),
            })
            .collect();
        let unanchored = smoothed.frame_count() - frames.len();
        let image_size = self.image_size.unwrap_or(ImageSize {
            width: 0,
            height: 0,
        });
        let dump = SessionDump {
            version: SESSION_DUMP_VERSION,
            rig: session_rig(system.rig(), image_size, self.imu_time_offset_ns),
            keyframes,
            landmarks,
            frames,
            imu: self.imu,
            gravity_world: system.gravity_world().to_array(),
        };
        write_dump(path, &dump)?;
        Ok(unanchored)
    }
}

/// Writes `dump` as CBOR, creating the parent directory.
pub fn write_dump(path: &Path, dump: &SessionDump) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(dir) = path.parent().filter(|dir| !dir.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    let out = std::io::BufWriter::new(std::fs::File::create(path)?);
    ciborium::into_writer(dump, out)?;
    Ok(())
}

/// The keyframes of `dump` stamped in `[from_sec, to_sec]` (the dump's
/// clock), the landmarks they observe, the frames standing on them in the
/// span, and the IMU samples from the last keyframe before the span to its
/// end, so the first kept keyframe keeps its preintegration interval.
pub fn slice(dump: &SessionDump, from_sec: f64, to_sec: f64) -> SessionDump {
    let in_span = |t: f64| t >= from_sec && t <= to_sec;
    let keyframes: Vec<_> = dump
        .keyframes
        .iter()
        .filter(|kf| in_span(kf.timestamp_sec))
        .cloned()
        .collect();
    let kept: HashSet<u64> = keyframes.iter().map(|kf| kf.kf_idx).collect();
    let observed: BTreeSet<u32> = keyframes
        .iter()
        .flat_map(|kf| kf.observations.iter().map(|obs| obs.landmark))
        .collect();
    let imu_from = dump
        .keyframes
        .iter()
        .map(|kf| kf.timestamp_sec)
        .filter(|&t| t < from_sec)
        .reduce(f64::max)
        .unwrap_or(from_sec);
    SessionDump {
        version: dump.version,
        rig: dump.rig.clone(),
        keyframes,
        landmarks: dump
            .landmarks
            .iter()
            .filter(|lm| observed.contains(&lm.idx))
            .copied()
            .collect(),
        frames: dump
            .frames
            .iter()
            .filter(|f| kept.contains(&f.reference_kf) && in_span(f.timestamp_sec))
            .copied()
            .collect(),
        imu: dump
            .imu
            .iter()
            .filter(|s| s.t >= imu_from && s.t <= to_sec)
            .copied()
            .collect(),
        gravity_world: dump.gravity_world,
    }
}

/// Reads a session written by [`SessionRecorder::write`].
pub fn read(path: &Path) -> Result<SessionDump, Box<dyn std::error::Error>> {
    let input = std::io::BufReader::new(std::fs::File::open(path)?);
    Ok(ciborium::from_reader(input)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_slam::mapping::session::{
        SessionCamera, SessionKeyframe, SessionLandmark, SessionObservation, SessionRig,
    };

    /// Keyframe `k` at `k` s observes landmarks `k` and `k + 1`; IMU every
    /// 0.125 s, so the span ends fall on samples exactly.
    fn dump() -> SessionDump {
        let identity = SessionPose {
            rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
            translation: [0.0; 3],
        };
        let observation = |landmark| SessionObservation {
            landmark,
            camera: 0,
            octave: 0,
            u: 1.0,
            v: 2.0,
            depth: None,
        };
        SessionDump {
            version: SESSION_DUMP_VERSION,
            rig: SessionRig {
                cameras: vec![SessionCamera {
                    fx: 242.0,
                    fy: 242.0,
                    cx: 390.0,
                    cy: 292.0,
                    width: 780,
                    height: 584,
                    primary_from_camera: identity,
                }],
                camera_to_body: identity,
                imu_noise: [1.7e-4, 7.8e-4, 2.4e-4, 6.4e-3],
                imu_time_offset_ns: 0,
                stereo_baseline_m: None,
            },
            keyframes: (0..5u32)
                .map(|k| SessionKeyframe {
                    kf_idx: 10 * u64::from(k),
                    timestamp_sec: f64::from(k),
                    pose_cw: identity,
                    velocity_world: [0.0; 3],
                    bias_gyro: [0.0; 3],
                    bias_accel: [0.0; 3],
                    segment: 0,
                    inserted_while_lost: false,
                    anchor: false,
                    evicted: false,
                    observations: vec![observation(k), observation(k + 1)],
                })
                .collect(),
            landmarks: (0..6u32)
                .map(|idx| SessionLandmark {
                    idx,
                    position_world: [f64::from(idx), 0.0, 4.0],
                })
                .collect(),
            frames: (0..20u32)
                .map(|i| SessionFrame {
                    timestamp_sec: 0.25 * f64::from(i),
                    reference_kf: 10 * u64::from(i / 4),
                    frame_from_kf: identity,
                })
                .collect(),
            imu: (0..=32u32)
                .map(|i| ImuSample {
                    t: 0.125 * f64::from(i),
                    gyro: [0.0; 3],
                    accel: [0.0, 9.81, 0.0],
                })
                .collect(),
            gravity_world: [0.0, 9.81, 0.0],
        }
    }

    #[test]
    fn a_slice_keeps_its_span_and_survives_a_round_trip() {
        let sliced = slice(&dump(), 1.5, 3.0);
        let kfs: Vec<u64> = sliced.keyframes.iter().map(|kf| kf.kf_idx).collect();
        assert_eq!(kfs, [20, 30]);
        let landmarks: Vec<u32> = sliced.landmarks.iter().map(|lm| lm.idx).collect();
        assert_eq!(landmarks, [2, 3, 4]);
        // Frames 1.5-3.0 s standing on keyframes 20 and 30: 2.0-3.0 s.
        assert_eq!(sliced.frames.len(), 5);
        // IMU from keyframe 10 (1 s) to 3 s.
        assert_eq!(sliced.imu.len(), 17);
        assert_eq!(sliced.imu[0].t, 1.0);

        let path = std::env::temp_dir().join(format!("session-slice-{}.cbor", std::process::id()));
        write_dump(&path, &sliced).unwrap();
        let back = read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(back, sliced);
    }
}
