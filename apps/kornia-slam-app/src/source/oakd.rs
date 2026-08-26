//! Live OAK-D mono / stereo camera as a [`FrameSource`].
//!
//! `open()` pulls GRAY8 frames from CamB only (placeholder intrinsics).
//! `open_stereo()` opens CamB + CamC and rectifies online using a
//! [`StereoCalib`](crate::datasets::StereoCalib) YAML, yielding metric
//! row-aligned pairs.

use std::path::Path;
use std::time::{Duration, Instant};

use depthai::camera::{CameraNode, CameraOutputConfig, OutputQueue};
use depthai::common::{CameraBoardSocket, ImageFrameType};
use depthai::{Device, Pipeline as DaiPipeline};
use kornia_3d::camera::PinholeCamera;
use kornia_image::{Image, ImageSize};

use super::{FrameItem, FrameSource, SourceError};
use crate::datasets::{StereoCalib, StereoRectifier};

/// Live OAK-D frame source. `right_queue`/`rectifier`/`stereo_bf` are populated
/// when the source was opened via `open_stereo`; in mono mode they're `None`.
pub struct OakdSource {
    // Order matters for drop: the queue must outlive the pipeline/device only
    // if depthai needs it that way. The crate stores everything by-value with
    // shared_ptr semantics on the C++ side, so we keep all handles alive in
    // this struct and let drop run in field-declared order (top-to-bottom).
    queue: OutputQueue,
    right_queue: Option<OutputQueue>,
    _pipeline: DaiPipeline,
    _device: Device,
    image_size: ImageSize,
    n_pixels: usize,
    camera: PinholeCamera,
    rectifier: Option<StereoRectifier>,
    stereo_bf: Option<f64>,
    max_frames: usize,
    cursor: usize,
    start: Instant,
}

impl OakdSource {
    /// Opens the device and starts a single GRAY8 stream from CamB.
    ///
    /// `max_frames == 0` ⇒ run until the caller stops the loop (Ctrl-C).
    pub fn open(width: u32, height: u32, fps: f32, max_frames: usize) -> Result<Self, SourceError> {
        eprintln!("[oakd] opening device…");
        let device = Device::new().map_err(SourceError::other)?;
        let platform = device.platform().map_err(SourceError::other)?;
        eprintln!("[oakd] platform: {platform:?}");

        let pipeline = DaiPipeline::new()
            .with_device(&device)
            .build()
            .map_err(SourceError::other)?;
        let cam = pipeline
            .create_with::<CameraNode, _>(CameraBoardSocket::CamB)
            .map_err(SourceError::other)?;
        let out = cam
            .request_output(CameraOutputConfig {
                size: (width, height),
                frame_type: Some(ImageFrameType::GRAY8),
                fps: Some(fps),
                ..Default::default()
            })
            .map_err(SourceError::other)?;
        let queue = out.create_queue(4, false).map_err(SourceError::other)?;
        pipeline.start().map_err(SourceError::other)?;

        let image_size = ImageSize {
            width: width as usize,
            height: height as usize,
        };
        let n_pixels = image_size.width * image_size.height;

        // Placeholder intrinsics: OAK-D Pro mono is 1280×800 native with
        // roughly fx≈fy≈880 px; principal point at image center after scale.
        let scale = width as f64 / 1280.0;
        let camera = PinholeCamera {
            fx: 880.0 * scale,
            fy: 880.0 * scale,
            cx: width as f64 * 0.5,
            cy: height as f64 * 0.5,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        };
        eprintln!(
            "[oakd] placeholder intrinsics: fx={:.2} fy={:.2} cx={:.2} cy={:.2}",
            camera.fx, camera.fy, camera.cx, camera.cy
        );

        Ok(Self {
            queue,
            right_queue: None,
            _pipeline: pipeline,
            _device: device,
            image_size,
            n_pixels,
            camera,
            rectifier: None,
            stereo_bf: None,
            max_frames,
            cursor: 0,
            start: Instant::now(),
        })
    }

    /// Opens CamB + CamC GRAY8 streams at the resolution recorded in the
    /// calibration YAML and rectifies each pair online. The rectified camera
    /// is the one exposed via `camera()`; `stereo_bf` carries the metric
    /// `f * baseline` for the depth formula.
    pub fn open_stereo(
        fps: f32,
        calib_path: &Path,
        max_frames: usize,
    ) -> Result<Self, SourceError> {
        let calib = StereoCalib::load(calib_path).map_err(SourceError::other)?;
        let (width, height) = (calib.width as u32, calib.height as u32);
        let rectifier = calib.rectifier().map_err(SourceError::other)?;

        eprintln!("[oakd] opening device (stereo)…");
        let device = Device::new().map_err(SourceError::other)?;
        let platform = device.platform().map_err(SourceError::other)?;
        eprintln!("[oakd] platform: {platform:?}");

        let pipeline = DaiPipeline::new()
            .with_device(&device)
            .build()
            .map_err(SourceError::other)?;

        let make_cam = |socket: CameraBoardSocket| -> Result<OutputQueue, SourceError> {
            let cam = pipeline
                .create_with::<CameraNode, _>(socket)
                .map_err(SourceError::other)?;
            let out = cam
                .request_output(CameraOutputConfig {
                    size: (width, height),
                    frame_type: Some(ImageFrameType::GRAY8),
                    fps: Some(fps),
                    ..Default::default()
                })
                .map_err(SourceError::other)?;
            out.create_queue(4, false).map_err(SourceError::other)
        };
        let left_queue = make_cam(CameraBoardSocket::CamB)?;
        let right_queue = make_cam(CameraBoardSocket::CamC)?;
        pipeline.start().map_err(SourceError::other)?;

        let image_size = ImageSize {
            width: calib.width,
            height: calib.height,
        };
        let n_pixels = image_size.width * image_size.height;
        let rect_cam = rectifier.rectified_camera();
        let stereo_bf = rectifier.bf();
        eprintln!(
            "[oakd] stereo: {}x{} @ {fps}fps  rectified fx={:.2} baseline={:.4}m bf={:.2}",
            width,
            height,
            rect_cam.fx,
            rectifier.baseline(),
            stereo_bf,
        );

        Ok(Self {
            queue: left_queue,
            right_queue: Some(right_queue),
            _pipeline: pipeline,
            _device: device,
            image_size,
            n_pixels,
            camera: rect_cam,
            rectifier: Some(rectifier),
            stereo_bf: Some(stereo_bf),
            max_frames,
            cursor: 0,
            start: Instant::now(),
        })
    }
}

impl FrameSource for OakdSource {
    fn camera(&self) -> PinholeCamera {
        self.camera.clone()
    }

    fn stereo_bf(&self) -> Option<f64> {
        self.stereo_bf
    }

    fn n_frames_hint(&self) -> Option<usize> {
        if self.max_frames > 0 {
            Some(self.max_frames)
        } else {
            None
        }
    }

    fn next_frame(&mut self) -> Result<Option<FrameItem>, SourceError> {
        if self.max_frames > 0 && self.cursor >= self.max_frames {
            return Ok(None);
        }
        let left_raw = pull_gray8(&mut self.queue, &self.image_size, self.n_pixels)?;
        // OAK-D CamB/CamC are hardware-synced; the queues advance in lockstep
        // at steady state, so consecutive items pair up.
        let (image, right_image) =
            if let (Some(rq), Some(rect)) = (self.right_queue.as_mut(), self.rectifier.as_ref()) {
                let right_raw = pull_gray8(rq, &self.image_size, self.n_pixels)?;
                (
                    rect.rectify_left(&left_raw).map_err(SourceError::other)?,
                    Some(rect.rectify_right(&right_raw).map_err(SourceError::other)?),
                )
            } else {
                (left_raw, None)
            };
        let idx = self.cursor;
        self.cursor += 1;
        let timestamp_sec = self.start.elapsed().as_secs_f64();
        Ok(Some(FrameItem {
            idx,
            timestamp_sec,
            image,
            right_image,
            imu_samples: Vec::new(),
        }))
    }
}

/// Pull one GRAY8 frame from `queue`, validating dimensions and payload size.
/// Spins (with a 1ms sleep) until a frame is available.
fn pull_gray8(
    queue: &mut OutputQueue,
    image_size: &ImageSize,
    n_pixels: usize,
) -> Result<Image<u8, 1>, SourceError> {
    loop {
        let frame_msg = match queue.try_next().map_err(SourceError::other)? {
            Some(f) => f,
            None => {
                std::thread::sleep(Duration::from_millis(1));
                continue;
            }
        };
        if frame_msg.width() as usize != image_size.width
            || frame_msg.height() as usize != image_size.height
        {
            eprintln!(
                "[oakd] unexpected frame size: {}x{} (want {}x{})",
                frame_msg.width(),
                frame_msg.height(),
                image_size.width,
                image_size.height,
            );
            continue;
        }
        let bytes = frame_msg.bytes();
        if bytes.len() != n_pixels {
            eprintln!(
                "[oakd] unexpected payload {} bytes (want {})",
                bytes.len(),
                n_pixels,
            );
            continue;
        }
        return Image::new(*image_size, bytes).map_err(SourceError::other);
    }
}
