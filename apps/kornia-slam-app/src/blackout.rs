//! `--blackout`: replaces every image inside a time interval with black, to
//! measure how the system coasts through a total visual outage. Frames are
//! still processed, with their IMU samples.

use crate::source::FrameItem;

pub struct Blackout {
    /// Seconds after the first frame.
    start_sec: f64,
    duration_sec: f64,
    first_timestamp_sec: Option<f64>,
}

impl Blackout {
    /// Parses `<start_s>:<duration_s>`, the start counted from the first frame.
    pub fn parse(spec: &str) -> Result<Self, String> {
        let parse = |text: &str| {
            text.parse::<f64>()
                .ok()
                .filter(|value| value.is_finite() && *value >= 0.0)
        };
        let (start, duration) = spec
            .split_once(':')
            .and_then(|(start, duration)| Some((parse(start)?, parse(duration)?)))
            .ok_or_else(|| format!("--blackout must be <start_s>:<duration_s>, got {spec:?}"))?;
        Ok(Self {
            start_sec: start,
            duration_sec: duration,
            first_timestamp_sec: None,
        })
    }

    /// Blacks out `item`'s images when it falls inside the interval; returns whether it did.
    pub fn apply(&mut self, item: &mut FrameItem) -> bool {
        let first = *self.first_timestamp_sec.get_or_insert(item.timestamp_sec);
        let t = item.timestamp_sec - first;
        if t < self.start_sec || t >= self.start_sec + self.duration_sec {
            return false;
        }
        let images = std::iter::once(&mut item.image)
            .chain(item.right_image.as_mut())
            .chain(item.rig_images.iter_mut());
        for image in images {
            image.as_slice_mut().fill(0);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::{Image, ImageSize};

    fn item(timestamp_sec: f64) -> FrameItem {
        let image = Image::new(
            ImageSize {
                width: 2,
                height: 1,
            },
            vec![7u8, 9],
        )
        .unwrap();
        FrameItem {
            idx: 0,
            timestamp_sec,
            right_image: Some(image.clone()),
            image,
            rig_images: Vec::new(),
            imu_samples: Vec::new(),
        }
    }

    #[test]
    fn blacks_out_only_inside_the_interval_from_the_first_frame() {
        let mut blackout = Blackout::parse("1:0.5").unwrap();
        let mut outside = [item(100.0), item(100.9), item(101.5)];
        let mut inside = [item(101.0), item(101.4)];
        assert!(!blackout.apply(&mut outside[0]));
        for frame in &mut inside {
            assert!(blackout.apply(frame));
            assert!(frame.image.as_slice().iter().all(|&p| p == 0));
            assert!(
                frame
                    .right_image
                    .as_ref()
                    .unwrap()
                    .as_slice()
                    .iter()
                    .all(|&p| p == 0)
            );
        }
        for frame in &mut outside[1..] {
            assert!(!blackout.apply(frame));
            assert_eq!(frame.image.as_slice(), &[7, 9]);
        }
    }

    #[test]
    fn rejects_a_malformed_interval() {
        for spec in ["60", "a:2", "60:-1", ":2"] {
            assert!(Blackout::parse(spec).is_err(), "{spec}");
        }
    }
}
