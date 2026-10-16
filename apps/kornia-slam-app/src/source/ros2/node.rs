//! ROS 2 subscriber node as a [`FrameSource`], built on
//! [ros2_rust](https://github.com/ros2-rust/ros2_rust).
//!
//! The executor spins on its own thread. Callbacks convert each message and
//! hand it over a channel; [`FrameSync`] and [`ImuBuffer`] assemble frames on
//! the caller's thread.

use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use kornia_algebra::Vec3F64;
use kornia_image::Image;
use kornia_sensors::SensorRig;
use kornia_sensors::imu::ImuMeasurement;
use kornia_slam::{CameraSelection, SensorSelection};
use rclrs::{
    Context, CreateBasicExecutor, ExecutorCommands, InitOptions, IntoPrimitiveOptions, Node,
    RclrsError, SpinOptions, Subscription,
};
use ros_env::sensor_msgs::msg::{CameraInfo, Image as ImageMsg, Imu};

use super::Ros2Config;
use super::convert::{camera_from_projection, gray_image, stereo_baseline_m};
use super::sync::{Frame, FrameSync, ImuBuffer, StampNs, seconds, stamp_ns};
use crate::source::{FrameItem, FrameSource, SourceError};

/// How long a frame waits for the IMU sample that closes its interval.
const IMU_WAIT: Duration = Duration::from_millis(100);
const POLL: Duration = Duration::from_millis(100);
const IMAGE_DEPTH: u32 = 10;
const IMU_DEPTH: u32 = 400;

enum Message {
    Left(StampNs, Image<u8, 1>),
    Right(StampNs, Image<u8, 1>),
    Imu(ImuMeasurement),
    LeftProjection([f64; 12]),
    RightProjection([f64; 12]),
    Invalid(String),
}

#[derive(Clone, Copy)]
enum Side {
    Left,
    Right,
}

pub struct Ros2Source {
    inbox: Inbox,
    rig: SensorRig,
    node_name: String,
    max_frames: usize,
    yielded: usize,
    reported_drops: usize,
    _subscriptions: Subscriptions,
    commands: Arc<ExecutorCommands>,
    spinner: Option<JoinHandle<()>>,
}

struct Subscriptions {
    _left: Subscription<ImageMsg>,
    _left_info: Subscription<CameraInfo>,
    _right: Option<(Subscription<ImageMsg>, Subscription<CameraInfo>)>,
    _imu: Option<Subscription<Imu>>,
}

impl Ros2Source {
    /// Subscribes to the configured topics and waits for the calibration.
    pub fn open(config: &Ros2Config, sensors: SensorSelection) -> Result<Self, SourceError> {
        let stereo = sensors.cameras == CameraSelection::Stereo;
        let context = Context::new(std::iter::empty::<String>(), InitOptions::default())
            .map_err(SourceError::other)?;
        let mut executor = context.create_basic_executor();
        let node = executor
            .create_node(config.node_name.as_str())
            .map_err(SourceError::other)?;

        let (tx, rx) = mpsc::channel();
        let subscriptions = subscribe(&node, config, sensors, &tx).map_err(SourceError::other)?;
        drop(tx);

        let commands = Arc::clone(executor.commands());
        let spinner = std::thread::spawn(move || {
            for err in executor.spin(SpinOptions::default()) {
                eprintln!("[ros2] executor: {err}");
            }
        });

        let mut inbox = Inbox {
            rx,
            frames: FrameSync::new(stereo, config.queue),
            imu: sensors.imu.then(ImuBuffer::default),
        };
        let rig = inbox.wait_for_rig(config, stereo)?;
        let rig = match (&config.camera_to_imu, sensors.imu) {
            (Some(extrinsic), true) => rig.with_imu(extrinsic.pose()),
            _ => rig,
        };
        Ok(Self {
            inbox,
            rig,
            node_name: node.fully_qualified_name(),
            max_frames: config.max_frames,
            yielded: 0,
            reported_drops: 0,
            _subscriptions: subscriptions,
            commands,
            spinner: Some(spinner),
        })
    }

    pub fn summary(&self) -> String {
        format!("ROS 2: node {} subscribed", self.node_name)
    }

    fn report_drops(&mut self) {
        let dropped = self.inbox.frames.dropped();
        if dropped > self.reported_drops {
            eprintln!(
                "[ros2] tracker behind: dropped {} frame(s), {dropped} in total",
                dropped - self.reported_drops
            );
            self.reported_drops = dropped;
        }
    }
}

/// Messages from the callbacks, assembled into frames on the caller's thread.
struct Inbox {
    rx: Receiver<Message>,
    frames: FrameSync,
    imu: Option<ImuBuffer>,
}

impl Inbox {
    fn wait_for_rig(
        &mut self,
        config: &Ros2Config,
        stereo: bool,
    ) -> Result<SensorRig, SourceError> {
        let deadline = Instant::now() + Duration::from_secs_f64(config.wait_s);
        let (mut left, mut right) = (None, None);
        loop {
            match (left, right) {
                (Some(left), Some(right)) if stereo => {
                    return Ok(rectified_rig(&left)?.with_stereo_baseline(
                        stereo_baseline_m(&right).map_err(SourceError::other)?,
                    ));
                }
                (Some(left), _) if !stereo => return rectified_rig(&left),
                _ => {}
            }
            let timeout = deadline.saturating_duration_since(Instant::now());
            match self.rx.recv_timeout(timeout) {
                Ok(Message::LeftProjection(p)) => left = Some(p),
                Ok(Message::RightProjection(p)) => right = Some(p),
                Ok(message) => self.accept(message)?,
                Err(RecvTimeoutError::Timeout) => {
                    let missing = if left.is_none() {
                        &config.camera_info
                    } else {
                        &config.right_camera_info
                    };
                    return Err(SourceError::other(format!(
                        "no CameraInfo on `{missing}` within {} s",
                        config.wait_s
                    )));
                }
                Err(RecvTimeoutError::Disconnected) => return Err(disconnected()),
            }
        }
    }

    /// Calibration updates after startup are ignored.
    fn accept(&mut self, message: Message) -> Result<(), SourceError> {
        match message {
            Message::Left(stamp, image) => self.frames.push_left(stamp, image),
            Message::Right(stamp, image) => self.frames.push_right(stamp, image),
            Message::Imu(sample) => {
                if let Some(imu) = &mut self.imu {
                    imu.push(sample);
                }
            }
            Message::LeftProjection(_) | Message::RightProjection(_) => {}
            Message::Invalid(reason) => return Err(SourceError::other(reason)),
        }
        Ok(())
    }

    fn next_frame(&mut self) -> Result<Frame, SourceError> {
        loop {
            while let Ok(message) = self.rx.try_recv() {
                self.accept(message)?;
            }
            if let Some(frame) = self.frames.pop() {
                return Ok(frame);
            }
            match self.rx.recv_timeout(POLL) {
                Ok(message) => self.accept(message)?,
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => return Err(disconnected()),
            }
        }
    }

    /// Waits briefly for the IMU sample that closes the frame's interval, so
    /// late IMU messages are not split across frames.
    fn imu_until(&mut self, timestamp_sec: f64) -> Result<Vec<ImuMeasurement>, SourceError> {
        let deadline = Instant::now() + IMU_WAIT;
        while self
            .imu
            .as_ref()
            .is_some_and(|imu| !imu.covers(timestamp_sec))
        {
            let timeout = deadline.saturating_duration_since(Instant::now());
            match self.rx.recv_timeout(timeout) {
                Ok(message) => self.accept(message)?,
                Err(RecvTimeoutError::Timeout) => break,
                Err(RecvTimeoutError::Disconnected) => return Err(disconnected()),
            }
        }
        Ok(self
            .imu
            .as_mut()
            .map(|imu| imu.drain_until(timestamp_sec))
            .unwrap_or_default())
    }
}

impl FrameSource for Ros2Source {
    fn rig(&self) -> SensorRig {
        self.rig.clone()
    }

    fn n_frames_hint(&self) -> Option<usize> {
        (self.max_frames > 0).then_some(self.max_frames)
    }

    fn next_frame(&mut self) -> Result<Option<FrameItem>, SourceError> {
        if self.max_frames > 0 && self.yielded >= self.max_frames {
            return Ok(None);
        }
        let frame = self.inbox.next_frame()?;
        let timestamp_sec = seconds(frame.stamp);
        let imu_samples = self.inbox.imu_until(timestamp_sec)?;
        self.report_drops();
        let idx = self.yielded;
        self.yielded += 1;
        Ok(Some(FrameItem {
            idx,
            timestamp_sec,
            stamp_ns: Some(frame.stamp),
            image: frame.image,
            right_image: frame.right_image,
            imu_samples,
        }))
    }
}

impl Drop for Ros2Source {
    fn drop(&mut self) {
        self.commands.halt_spinning();
        if let Some(spinner) = self.spinner.take() {
            spinner.join().ok();
        }
    }
}

fn subscribe(
    node: &Node,
    config: &Ros2Config,
    sensors: SensorSelection,
    tx: &Sender<Message>,
) -> Result<Subscriptions, RclrsError> {
    let right = if sensors.cameras == CameraSelection::Stereo {
        Some((
            image_subscription(node, &config.right_image, Side::Right, tx.clone())?,
            info_subscription(node, &config.right_camera_info, Side::Right, tx.clone())?,
        ))
    } else {
        None
    };
    let imu = if sensors.imu {
        Some(imu_subscription(node, &config.imu, tx.clone())?)
    } else {
        None
    };
    Ok(Subscriptions {
        _left: image_subscription(node, &config.image, Side::Left, tx.clone())?,
        _left_info: info_subscription(node, &config.camera_info, Side::Left, tx.clone())?,
        _right: right,
        _imu: imu,
    })
}

fn image_subscription(
    node: &Node,
    topic: &str,
    side: Side,
    tx: Sender<Message>,
) -> Result<Subscription<ImageMsg>, RclrsError> {
    node.create_subscription(
        topic.best_effort().keep_last(IMAGE_DEPTH),
        move |msg: ImageMsg| {
            tx.send(image_message(&msg, side)).ok();
        },
    )
}

fn info_subscription(
    node: &Node,
    topic: &str,
    side: Side,
    tx: Sender<Message>,
) -> Result<Subscription<CameraInfo>, RclrsError> {
    node.create_subscription(topic.best_effort().keep_last(1), move |msg: CameraInfo| {
        let message = match side {
            Side::Left => Message::LeftProjection(msg.p),
            Side::Right => Message::RightProjection(msg.p),
        };
        tx.send(message).ok();
    })
}

fn imu_subscription(
    node: &Node,
    topic: &str,
    tx: Sender<Message>,
) -> Result<Subscription<Imu>, RclrsError> {
    node.create_subscription(topic.best_effort().keep_last(IMU_DEPTH), move |msg: Imu| {
        let stamp = stamp_ns(msg.header.stamp.sec, msg.header.stamp.nanosec);
        let (w, a) = (msg.angular_velocity, msg.linear_acceleration);
        tx.send(Message::Imu(ImuMeasurement {
            timestamp: seconds(stamp),
            gyro: Vec3F64::new(w.x, w.y, w.z),
            accel: Vec3F64::new(a.x, a.y, a.z),
        }))
        .ok();
    })
}

fn rectified_rig(left: &[f64; 12]) -> Result<SensorRig, SourceError> {
    Ok(SensorRig::new(
        camera_from_projection(left).map_err(SourceError::other)?,
    ))
}

fn image_message(msg: &ImageMsg, side: Side) -> Message {
    let stamp = stamp_ns(msg.header.stamp.sec, msg.header.stamp.nanosec);
    match gray_image(&msg.encoding, msg.width, msg.height, msg.step, &msg.data) {
        Ok(image) => match side {
            Side::Left => Message::Left(stamp, image),
            Side::Right => Message::Right(stamp, image),
        },
        Err(reason) => Message::Invalid(reason),
    }
}

fn disconnected() -> SourceError {
    SourceError::other("the ROS 2 executor stopped")
}
