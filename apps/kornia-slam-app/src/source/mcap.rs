//! MCAP file as a [`FrameSource`].
//!
//! Reads a bubbaloop `mcap-recorder` session and yields grayscale frames from
//! one chosen channel. The default channel suffix is `mono_left` — already
//! grayscale at 640×400, no color conversion needed. Each MCAP message is a
//! CBOR-wrapped SDK envelope `{header, body:{width, height, encoding:"jpeg", data}}`;
//! we strip the envelope, JPEG-decode `body.data` via `kornia-io`, and (for
//! the color `compressed` channel) convert to luma8 via `kornia-imgproc`.
//!
//! Stereo (`open_stereo`): the OAK-D `mono_left`/`mono_right` channels are the
//! raw, *unrectified* camera streams, so we rectify online with a
//! [`StereoCalib`](crate::datasets::StereoCalib) (factory intrinsics +
//! distortion + extrinsics) before yielding the pair.

use std::fs;
use std::path::Path;

use ciborium::value::Value as CborValue;
use kornia_3d::camera::PinholeCamera;
use kornia_image::Image;
use kornia_imgproc::color::gray_from_rgb_u8;
use kornia_io::jpeg::{decode_image_jpeg_layout, decode_image_jpeg_mono8, decode_image_jpeg_rgb8};
use mcap::McapError;

use super::{FrameItem, FrameSource, SourceError};
use crate::datasets::StereoCalib;

/// A timestamped grayscale frame, `(log_time_sec, image)`.
type TimedImage = (f64, Image<u8, 1>);
/// A timestamped left/right pair, `(log_time_sec, left, right)`.
type TimedPair = (f64, Image<u8, 1>, Image<u8, 1>);

/// A pre-decoded grayscale frame queued for `next_frame`.
struct PreparedFrame {
    timestamp_sec: f64,
    image: Image<u8, 1>,
    right_image: Option<Image<u8, 1>>,
}

/// Offline MCAP frame source.
pub struct McapSource {
    frames: std::vec::IntoIter<PreparedFrame>,
    n_total: usize,
    cursor: usize,
    camera: PinholeCamera,
    stereo_bf: Option<f64>,
}

impl McapSource {
    /// Open `path`, pick the channel whose topic ends in `/<channel_suffix>`,
    /// JPEG-decode every message to luma8, and stash the result.
    ///
    /// `start_frame` and `max_frames` mirror `EurocSource`: skip the first
    /// `start_frame`, then yield up to `max_frames` (0 = until exhausted).
    pub fn open(
        path: &Path,
        channel_suffix: &str,
        start_frame: usize,
        max_frames: usize,
    ) -> Result<Self, SourceError> {
        let bytes = fs::read(path).map_err(SourceError::Io)?;
        let frames = read_channel(&bytes, channel_suffix)?;
        let (w, h) = (frames[0].1.width(), frames[0].1.height());
        let frames = trim_and_rebase(frames, start_frame, max_frames);
        let n_total = frames.len();

        let prepared: Vec<PreparedFrame> = frames
            .into_iter()
            .map(|(t, image)| PreparedFrame {
                timestamp_sec: t,
                image,
                right_image: None,
            })
            .collect();

        // Placeholder intrinsics. OAK-D mono is 1280×800 native @ fx≈fy≈880;
        // our recorded mono is 640×400 so scale=0.5. (For metric stereo use
        // `open_stereo` with a calibration file instead.)
        let scale = w as f64 / 1280.0;
        let camera = PinholeCamera {
            fx: 880.0 * scale,
            fy: 880.0 * scale,
            cx: w as f64 * 0.5,
            cy: h as f64 * 0.5,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        };

        eprintln!(
            "[mcap] {}: {n_total} frames @ {w}x{h} from /{channel_suffix} \
             (placeholder intrinsics fx={:.2} fy={:.2})",
            path.display(),
            camera.fx,
            camera.fy,
        );

        Ok(Self {
            frames: prepared.into_iter(),
            n_total,
            cursor: 0,
            camera,
            stereo_bf: None,
        })
    }

    /// Open a rectified stereo pair from the `left_suffix`/`right_suffix`
    /// channels, using `calib_path` (a [`StereoCalib`] YAML) to undistort and
    /// row-align each frame. Left/right messages are paired by nearest
    /// timestamp. Yields rectified pairs with metric `stereo_bf`.
    pub fn open_stereo(
        path: &Path,
        left_suffix: &str,
        right_suffix: &str,
        calib_path: &Path,
        start_frame: usize,
        max_frames: usize,
    ) -> Result<Self, SourceError> {
        let bytes = fs::read(path).map_err(SourceError::Io)?;
        let calib = StereoCalib::load(calib_path).map_err(SourceError::other)?;
        let rectifier = calib.rectifier().map_err(SourceError::other)?;
        let rect_cam = rectifier.rectified_camera();

        let left = read_channel(&bytes, left_suffix)?;
        let right = read_channel(&bytes, right_suffix)?;
        let (w, h) = (left[0].1.width(), left[0].1.height());

        // Pair each left frame with the nearest-in-time right frame.
        let right_ts: Vec<f64> = right.iter().map(|(t, _)| *t).collect();
        let mut paired: Vec<TimedPair> = Vec::with_capacity(left.len());
        for (lt, limg) in left {
            let j = nearest_index(&right_ts, lt);
            paired.push((lt, limg, right[j].1.clone()));
        }

        // Trim/rebase on the paired list, then rectify only the survivors.
        let start = start_frame.min(paired.len());
        let mut sel: Vec<_> = paired.into_iter().skip(start).collect();
        if max_frames > 0 && sel.len() > max_frames {
            sel.truncate(max_frames);
        }
        let t0 = sel.first().map(|f| f.0).unwrap_or(0.0);
        let n_total = sel.len();

        let prepared: Vec<PreparedFrame> = sel
            .into_iter()
            .map(|(t, limg, rimg)| {
                Ok(PreparedFrame {
                    timestamp_sec: t - t0,
                    image: rectifier.rectify_left(&limg).map_err(SourceError::other)?,
                    right_image: Some(rectifier.rectify_right(&rimg).map_err(SourceError::other)?),
                })
            })
            .collect::<Result<_, SourceError>>()?;

        eprintln!(
            "[mcap] {}: {n_total} stereo pairs @ {w}x{h} ({left_suffix} + {right_suffix}) \
             rectified fx={:.2} baseline={:.4}m bf={:.2}",
            path.display(),
            rect_cam.fx,
            rectifier.baseline(),
            rectifier.bf(),
        );

        Ok(Self {
            frames: prepared.into_iter(),
            n_total,
            cursor: 0,
            camera: rect_cam,
            stereo_bf: Some(rectifier.bf()),
        })
    }
}

impl FrameSource for McapSource {
    fn camera(&self) -> PinholeCamera {
        self.camera.clone()
    }

    fn stereo_bf(&self) -> Option<f64> {
        self.stereo_bf
    }

    fn n_frames_hint(&self) -> Option<usize> {
        Some(self.n_total)
    }

    fn next_frame(&mut self) -> Result<Option<FrameItem>, SourceError> {
        let Some(prepared) = self.frames.next() else {
            return Ok(None);
        };
        let idx = self.cursor;
        self.cursor += 1;
        Ok(Some(FrameItem {
            idx,
            timestamp_sec: prepared.timestamp_sec,
            image: prepared.image,
            right_image: prepared.right_image,
            imu_samples: Vec::new(),
        }))
    }
}

/// Decode every message on the channel ending in `/<channel_suffix>` to luma8,
/// returning `(log_time_sec, image)` in file order. Errors if no such channel
/// or if frame size changes mid-stream.
fn read_channel(bytes: &[u8], channel_suffix: &str) -> Result<Vec<TimedImage>, SourceError> {
    let want = format!("/{channel_suffix}");
    let mut out: Vec<TimedImage> = Vec::new();
    let mut first_size: Option<(usize, usize)> = None;
    for msg_result in mcap::MessageStream::new(bytes).map_err(map_mcap)? {
        let msg = msg_result.map_err(map_mcap)?;
        if !msg.channel.topic.ends_with(&want) {
            continue;
        }
        let envelope: CborValue =
            ciborium::from_reader(msg.data.as_ref()).map_err(SourceError::other)?;
        let body = unwrap_body(&envelope).ok_or_else(|| {
            SourceError::other(format!(
                "MCAP message on {} is not a {{header, body}} envelope",
                msg.channel.topic
            ))
        })?;
        let jpeg_bytes = extract_jpeg(body).ok_or_else(|| {
            SourceError::other(format!(
                "MCAP message on {} has no JPEG payload (expected encoding=jpeg + data)",
                msg.channel.topic
            ))
        })?;
        let image = decode_jpeg_to_luma(jpeg_bytes).map_err(SourceError::other)?;
        let (got_w, got_h) = (image.width(), image.height());
        match first_size {
            Some(prev) if prev != (got_w, got_h) => {
                return Err(SourceError::other(format!(
                    "frame size changed mid-stream: {prev:?} → {got_w}x{got_h}"
                )));
            }
            None => first_size = Some((got_w, got_h)),
            _ => {}
        }
        out.push((msg.log_time as f64 / 1e9, image));
    }
    if out.is_empty() {
        return Err(SourceError::other(format!(
            "no messages found on a channel ending with /{channel_suffix}"
        )));
    }
    Ok(out)
}

/// Skip `start`, cap at `max` (0 = keep all), and re-base timestamps to 0.
fn trim_and_rebase(frames: Vec<TimedImage>, start: usize, max: usize) -> Vec<TimedImage> {
    let start = start.min(frames.len());
    let mut sel: Vec<_> = frames.into_iter().skip(start).collect();
    if max > 0 && sel.len() > max {
        sel.truncate(max);
    }
    if let Some(t0) = sel.first().map(|f| f.0) {
        for f in sel.iter_mut() {
            f.0 -= t0;
        }
    }
    sel
}

/// Index of the timestamp in `ts` nearest to `t`.
fn nearest_index(ts: &[f64], t: f64) -> usize {
    ts.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| (*a - t).abs().total_cmp(&(*b - t).abs()))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Decode a JPEG payload to a luma8 image. Mono JPEGs are decoded directly;
/// RGB JPEGs (used by the `compressed` channel) are decoded then converted.
fn decode_jpeg_to_luma(
    jpeg_bytes: &[u8],
) -> Result<Image<u8, 1>, Box<dyn std::error::Error + Send + Sync>> {
    let layout = decode_image_jpeg_layout(jpeg_bytes)?;
    let size = layout.image_size;
    match layout.channels {
        1 => {
            let mut dst = Image::<u8, 1>::from_size_val(size, 0)?;
            decode_image_jpeg_mono8(jpeg_bytes, &mut dst)?;
            Ok(dst)
        }
        3 => {
            let mut rgb = Image::<u8, 3>::from_size_val(size, 0)?;
            decode_image_jpeg_rgb8(jpeg_bytes, &mut rgb)?;
            let mut gray = Image::<u8, 1>::from_size_val(size, 0)?;
            gray_from_rgb_u8(&rgb, &mut gray)?;
            Ok(gray)
        }
        n => Err(format!("unsupported JPEG channel count: {n} (want 1 or 3)").into()),
    }
}

/// `{header, body}` envelope → body. Returns None if the value isn't a map
/// containing a `body` key.
fn unwrap_body(value: &CborValue) -> Option<&CborValue> {
    if let CborValue::Map(entries) = value {
        for (k, v) in entries {
            if let CborValue::Text(s) = k
                && s == "body"
            {
                return Some(v);
            }
        }
    }
    None
}

/// Pull JPEG bytes out of either a flat `{encoding:"jpeg", data}` body or
/// the nested `{rgb:{encoding:"jpeg", data}}` shape from the `compressed`
/// channel.
fn extract_jpeg(body: &CborValue) -> Option<&[u8]> {
    let CborValue::Map(entries) = body else {
        return None;
    };
    let mut encoding: Option<&str> = None;
    let mut data: Option<&[u8]> = None;
    let mut rgb_sub: Option<&CborValue> = None;
    for (k, v) in entries {
        let CborValue::Text(key) = k else { continue };
        match key.as_str() {
            "encoding" => encoding = cbor_text(v),
            "data" => data = cbor_bytes(v),
            "rgb" => rgb_sub = Some(v),
            _ => {}
        }
    }
    if let (Some(enc), Some(d)) = (encoding, data)
        && enc == "jpeg"
    {
        return Some(d);
    }
    if let Some(rgb) = rgb_sub {
        return extract_jpeg(rgb);
    }
    None
}

fn cbor_text(v: &CborValue) -> Option<&str> {
    match v {
        CborValue::Text(s) => Some(s.as_str()),
        _ => None,
    }
}

fn cbor_bytes(v: &CborValue) -> Option<&[u8]> {
    match v {
        CborValue::Bytes(b) => Some(b.as_slice()),
        _ => None,
    }
}

fn map_mcap(err: McapError) -> SourceError {
    SourceError::other(err)
}
