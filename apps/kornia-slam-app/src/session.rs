//! `--session-out`: records an online run for the `offline-ba` subcommand and
//! writes it as CBOR at shutdown; `offline-ba` reads it back.

use std::collections::HashSet;
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
        if let Some(dir) = path.parent().filter(|dir| !dir.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)?;
        }
        let out = std::io::BufWriter::new(std::fs::File::create(path)?);
        ciborium::into_writer(&dump, out)?;
        Ok(unanchored)
    }
}

/// Reads a session written by [`SessionRecorder::write`].
pub fn read(path: &Path) -> Result<SessionDump, Box<dyn std::error::Error>> {
    let input = std::io::BufReader::new(std::fs::File::open(path)?);
    Ok(ciborium::from_reader(input)?)
}
