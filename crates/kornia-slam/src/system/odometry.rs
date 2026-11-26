//! `odometry: SlidingWindow(( mode: Observe ))`: the sliding-window
//! estimator beside the pipeline, on the tracking thread, outside
//! `track_frame`. It reads the IMU samples and the rectified pair the system
//! is given and writes nothing the pipeline reads.

use kornia_image::Image;
use kornia_sensors::SensorRig;
use kornia_sensors::imu::{ImuMeasurement, SampleInterpolation};

use super::config::OrbSlamPipeline;
use crate::odometry::{OdometryCounters, OdometryOutput, SlidingWindowOdometry};
use crate::tracking::flow_pool::FlowPoolError;

pub(super) struct OdometryRunner {
    odometry: SlidingWindowOdometry,
    last: Option<OdometryOutput>,
}

impl OdometryRunner {
    /// The runner `pipeline.odometry` asks for; `None` for `MapTracking` or a
    /// rig without the IMU or the stereo baseline (validation refuses both).
    pub(super) fn build(
        pipeline: &OrbSlamPipeline,
        rig: &SensorRig,
        interpolation: SampleInterpolation,
    ) -> Result<Option<Self>, FlowPoolError> {
        let Some(config) = pipeline.odometry.sliding_window() else {
            return Ok(None);
        };
        Ok(
            SlidingWindowOdometry::new(config, &pipeline.tuning.odometry, rig, interpolation)?.map(
                |odometry| Self {
                    odometry,
                    last: None,
                },
            ),
        )
    }

    pub(super) fn push_imu(&mut self, samples: &[ImuMeasurement]) {
        self.odometry.push_imu(samples);
    }

    /// Runs the estimator on one pair; an error is counted and logged, never
    /// returned, and leaves no output for the frame.
    pub(super) fn observe(
        &mut self,
        timestamp_sec: f64,
        left: &Image<u8, 1>,
        right: &Image<u8, 1>,
    ) {
        match self.odometry.process_pair(timestamp_sec, left, right) {
            Ok(output) => self.last = Some(output),
            Err(error) => {
                self.last = None;
                self.odometry.count_error();
                debug_eprintln!("[odometry] frame {timestamp_sec:.6}: {error}");
            }
        }
    }

    pub(super) fn output(&self) -> Option<&OdometryOutput> {
        self.last.as_ref()
    }

    pub(super) fn counters(&self) -> &OdometryCounters {
        self.odometry.counters()
    }
}
