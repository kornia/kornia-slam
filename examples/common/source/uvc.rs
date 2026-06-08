//! Live UVC camera as a [`FrameSource`].
//!
//! Uses [`nokhwa`] for cross-platform capture (V4L2 on Linux, AVFoundation on
//! macOS, MSMF on Windows). Any UVC-class device works — built-in laptop
//! webcams, USB cameras, CSI-to-UVC adapters on a Raspberry Pi, etc. Frames
//! are decoded to grayscale `Image<u8, 1>` and fed through the same
//! `process_frame` orchestrator as the EuRoC dataset.
//!
//! Intrinsics are supplied by the caller (CLI flags). They must match the
//! device's *resolution at capture time* — if you calibrated at 1280×720, ask
//! for that resolution here too.

use std::time::Instant;

use kornia_3d::camera::PinholeCamera;
use kornia_image::{Image, ImageSize};
use kornia_tensor::CpuAllocator;
use nokhwa::Camera;
use nokhwa::pixel_format::LumaFormat;
use nokhwa::utils::{
    ApiBackend, CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType,
};

/// Open the device at `index` with a closest-resolution request for the given
/// frame format. Returns the camera ready to stream (caller must `open_stream`).
fn try_open(index: u32, width: u32, height: u32, fmt: FrameFormat) -> Result<Camera, SourceError> {
    let target = CameraFormat::new_from(width, height, fmt, 30);
    let requested = RequestedFormat::new::<LumaFormat>(RequestedFormatType::Closest(target));
    Camera::with_backend(CameraIndex::Index(index), requested, ApiBackend::Auto)
        .map_err(SourceError::other)
}

use super::{FrameItem, FrameSource, SourceError};

/// Live UVC frame source.
pub struct UvcSource {
    camera_dev: Camera,
    image_size: ImageSize,
    camera: PinholeCamera,
    max_frames: usize,
    cursor: usize,
    start: Instant,
}

impl UvcSource {
    /// Open the UVC device at `index` and request the closest format to
    /// `width × height` (decoding will produce Y8 grayscale regardless of the
    /// device's native pixel format).
    ///
    /// `max_frames == 0` ⇒ run until the caller stops the loop (Ctrl-C).
    pub fn open(
        index: u32,
        width: u32,
        height: u32,
        camera: PinholeCamera,
        max_frames: usize,
    ) -> Result<Self, SourceError> {
        eprintln!("[uvc] opening camera index {index}…");

        let mut camera_dev =
            try_open(index, width, height, FrameFormat::MJPEG).or_else(|first_err| {
                eprintln!("[uvc] MJPEG open failed ({first_err}); trying YUYV…");
                try_open(index, width, height, FrameFormat::YUYV)
            })?;

        camera_dev.open_stream().map_err(SourceError::other)?;

        let actual = camera_dev.resolution();
        let image_size = ImageSize {
            width: actual.width() as usize,
            height: actual.height() as usize,
        };

        if actual.width() != width || actual.height() != height {
            eprintln!(
                "[uvc] requested {width}x{height}, device selected {}x{} — \
                 intrinsics must match the selected resolution",
                actual.width(),
                actual.height(),
            );
        }

        eprintln!(
            "[uvc] streaming {}x{} @ ~{} fps  fx={:.2} fy={:.2} cx={:.2} cy={:.2}",
            actual.width(),
            actual.height(),
            camera_dev.frame_rate(),
            camera.fx,
            camera.fy,
            camera.cx,
            camera.cy,
        );

        Ok(Self {
            camera_dev,
            image_size,
            camera,
            max_frames,
            cursor: 0,
            start: Instant::now(),
        })
    }
}

impl FrameSource for UvcSource {
    fn camera(&self) -> PinholeCamera {
        self.camera.clone()
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

        let buffer = self.camera_dev.frame().map_err(SourceError::other)?;
        let luma = buffer
            .decode_image::<LumaFormat>()
            .map_err(SourceError::other)?;
        let bytes = luma.into_raw();
        let expected = self.image_size.width * self.image_size.height;
        if bytes.len() != expected {
            return Err(SourceError::other(format!(
                "uvc decoded payload {} bytes (want {expected})",
                bytes.len(),
            )));
        }

        let image: Image<u8, 1, CpuAllocator> =
            Image::new(self.image_size, bytes, CpuAllocator).map_err(SourceError::other)?;
        let idx = self.cursor;
        self.cursor += 1;
        let timestamp_sec = self.start.elapsed().as_secs_f64();
        Ok(Some(FrameItem {
            idx,
            timestamp_sec,
            image,
            right_image: None,
            imu_samples: Vec::new(),
        }))
    }
}
