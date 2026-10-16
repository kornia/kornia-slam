//! Assembles tracker frames from independently arriving ROS 2 messages.

use std::collections::{BTreeMap, VecDeque};

use kornia_image::Image;
use kornia_sensors::imu::ImuMeasurement;

/// Unmatched views kept per side while waiting for their partner.
const PENDING_LIMIT: usize = 32;

/// Header stamp in nanoseconds; stereo views pair on exact equality.
pub type StampNs = i64;

pub fn stamp_ns(sec: i32, nanosec: u32) -> StampNs {
    i64::from(sec) * 1_000_000_000 + i64::from(nanosec)
}

pub fn seconds(stamp: StampNs) -> f64 {
    stamp as f64 * 1e-9
}

/// A camera frame ready for the tracker.
pub struct Frame {
    pub stamp: StampNs,
    pub image: Image<u8, 1>,
    pub right_image: Option<Image<u8, 1>>,
}

/// Pairs stereo views by stamp and queues frames for the tracker, dropping the
/// oldest once `capacity` frames wait. Frames not newer than the last queued
/// one (a restarted bag, a reordered delivery) are ignored.
pub struct FrameSync {
    stereo: bool,
    capacity: usize,
    left: BTreeMap<StampNs, Image<u8, 1>>,
    right: BTreeMap<StampNs, Image<u8, 1>>,
    ready: VecDeque<Frame>,
    last: Option<StampNs>,
    dropped: usize,
}

impl FrameSync {
    pub fn new(stereo: bool, capacity: usize) -> Self {
        Self {
            stereo,
            capacity,
            left: BTreeMap::new(),
            right: BTreeMap::new(),
            ready: VecDeque::with_capacity(capacity),
            last: None,
            dropped: 0,
        }
    }

    pub fn push_left(&mut self, stamp: StampNs, image: Image<u8, 1>) {
        if !self.stereo {
            self.queue(stamp, image, None);
        } else if let Some(right) = self.right.remove(&stamp) {
            self.queue(stamp, image, Some(right));
        } else {
            insert_pending(&mut self.left, stamp, image);
        }
    }

    pub fn push_right(&mut self, stamp: StampNs, image: Image<u8, 1>) {
        if !self.stereo {
            return;
        }
        match self.left.remove(&stamp) {
            Some(left) => self.queue(stamp, left, Some(image)),
            None => insert_pending(&mut self.right, stamp, image),
        }
    }

    pub fn pop(&mut self) -> Option<Frame> {
        self.ready.pop_front()
    }

    /// Frames dropped because the tracker fell behind.
    pub fn dropped(&self) -> usize {
        self.dropped
    }

    fn queue(&mut self, stamp: StampNs, image: Image<u8, 1>, right_image: Option<Image<u8, 1>>) {
        if self.last.is_some_and(|last| stamp <= last) {
            return;
        }
        self.last = Some(stamp);
        // A view older than a completed pair will never be matched.
        self.left = self.left.split_off(&stamp);
        self.right = self.right.split_off(&stamp);
        if self.ready.len() == self.capacity {
            self.ready.pop_front();
            self.dropped += 1;
        }
        self.ready.push_back(Frame {
            stamp,
            image,
            right_image,
        });
    }
}

fn insert_pending(
    pending: &mut BTreeMap<StampNs, Image<u8, 1>>,
    stamp: StampNs,
    image: Image<u8, 1>,
) {
    pending.insert(stamp, image);
    if pending.len() > PENDING_LIMIT {
        pending.pop_first();
    }
}

/// IMU samples in arrival order, handed out per camera frame.
#[derive(Default)]
pub struct ImuBuffer {
    samples: VecDeque<ImuMeasurement>,
}

impl ImuBuffer {
    /// Samples not newer than the last one are ignored.
    pub fn push(&mut self, sample: ImuMeasurement) {
        if self
            .samples
            .back()
            .is_none_or(|last| sample.timestamp > last.timestamp)
        {
            self.samples.push_back(sample);
        }
    }

    /// Whether a sample at or after `timestamp_sec` has arrived, so everything
    /// up to it is known.
    pub fn covers(&self, timestamp_sec: f64) -> bool {
        self.samples
            .back()
            .is_some_and(|last| last.timestamp >= timestamp_sec)
    }

    /// Removes and returns the samples up to and including `timestamp_sec`.
    pub fn drain_until(&mut self, timestamp_sec: f64) -> Vec<ImuMeasurement> {
        let end = self
            .samples
            .partition_point(|sample| sample.timestamp <= timestamp_sec);
        self.samples.drain(..end).collect()
    }
}

#[cfg(test)]
mod tests {
    use kornia_algebra::Vec3F64;
    use kornia_image::ImageSize;

    use super::*;

    fn image(value: u8) -> Image<u8, 1> {
        Image::from_size_val(
            ImageSize {
                width: 1,
                height: 1,
            },
            value,
        )
        .unwrap()
    }

    fn stamps(sync: &mut FrameSync) -> Vec<StampNs> {
        std::iter::from_fn(|| sync.pop()).map(|f| f.stamp).collect()
    }

    #[test]
    fn stereo_views_pair_on_their_stamp() {
        let mut sync = FrameSync::new(true, 4);
        sync.push_left(1, image(1));
        sync.push_right(2, image(20));
        assert!(sync.pop().is_none());

        sync.push_left(2, image(2));
        let frame = sync.pop().unwrap();
        assert_eq!(frame.stamp, 2);
        assert_eq!(frame.image.as_slice(), &[2]);
        assert_eq!(frame.right_image.unwrap().as_slice(), &[20]);

        // The unmatched left view at 1 is now stale.
        sync.push_right(1, image(10));
        assert!(sync.pop().is_none());
    }

    #[test]
    fn mono_ignores_right_views() {
        let mut sync = FrameSync::new(false, 4);
        sync.push_right(1, image(10));
        sync.push_left(1, image(1));
        assert!(sync.pop().unwrap().right_image.is_none());
    }

    #[test]
    fn a_full_queue_drops_the_oldest_frame() {
        let mut sync = FrameSync::new(false, 2);
        for stamp in 1..=3 {
            sync.push_left(stamp, image(0));
        }
        assert_eq!(stamps(&mut sync), [2, 3]);
        assert_eq!(sync.dropped(), 1);
    }

    #[test]
    fn frames_must_move_forward_in_time() {
        let mut sync = FrameSync::new(false, 4);
        sync.push_left(5, image(0));
        sync.push_left(5, image(0));
        sync.push_left(3, image(0));
        assert_eq!(stamps(&mut sync), [5]);
    }

    #[test]
    fn unmatched_views_are_bounded() {
        let mut sync = FrameSync::new(true, 4);
        for stamp in 0..2 * PENDING_LIMIT as StampNs {
            sync.push_left(stamp, image(0));
        }
        assert_eq!(sync.left.len(), PENDING_LIMIT);
    }

    fn sample(timestamp: f64) -> ImuMeasurement {
        ImuMeasurement {
            timestamp,
            gyro: Vec3F64::ZERO,
            accel: Vec3F64::ZERO,
        }
    }

    #[test]
    fn imu_samples_are_handed_out_up_to_the_frame() {
        let mut imu = ImuBuffer::default();
        assert!(!imu.covers(0.0));
        for t in [0.1, 0.2, 0.2, 0.15, 0.3] {
            imu.push(sample(t));
        }
        assert!(imu.covers(0.3));
        assert!(!imu.covers(0.31));

        let drained: Vec<f64> = imu.drain_until(0.2).iter().map(|s| s.timestamp).collect();
        assert_eq!(drained, [0.1, 0.2]);
        assert_eq!(imu.drain_until(0.25).len(), 0);
        assert_eq!(imu.drain_until(1.0).len(), 1);
    }

    #[test]
    fn stamps_are_nanoseconds() {
        assert_eq!(stamp_ns(2, 5), 2_000_000_005);
        assert_eq!(seconds(stamp_ns(12, 500_000_000)), 12.5);
    }
}
