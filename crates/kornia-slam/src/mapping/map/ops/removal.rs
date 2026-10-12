//! Keyframe removal, the canonical operation behind keyframe culling.
//!
//! A keyframe is named by its features' landmark links (both sides), by the
//! reference of every landmark it anchors, by retired landmarks' stale
//! records, and by the IMU chain through it. Removal repairs all of them.
//! IMU edges are merged, never dropped: `(p, k)` and `(k, n)` become one
//! `(p, n)` re-preintegrated from their raw samples, which is exactly the edge
//! the map would hold had `k` never been a keyframe (ORB-SLAM3's
//! `IMU::Preintegrated::MergePrevious`). A keyframe whose edges cannot be
//! merged that way is refused before anything is written.

use super::correction::valid_preintegration;
use crate::mapping::map::{ImuFactor, Map};
use kornia_sensors::imu::{ImuMeasurement, PreintegratedImu};

/// Largest gap, seconds, between the end of `(p, k)` and the start of
/// `(k, n)` that still counts as one contiguous interval. Both are the stamp
/// of keyframe `k`, so anything but rounding is a different interval.
const IMU_JOIN_TOLERANCE_SEC: f64 = 1e-9;

/// What [`Map::remove_keyframe`] changed. `#[non_exhaustive]`, so a count
/// can be added without a semver break.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct KeyframeRemoval {
    pub keyframe_idx: usize,
    /// Landmark observation records dropped, live and retired.
    pub observations_removed: usize,
    /// Live landmarks retired because the keyframe was their last observer.
    pub landmarks_retired: usize,
    /// `(prev, next)` keyframes of the IMU edge that replaced the two through
    /// the removed keyframe.
    pub merged_imu_edge: Option<(usize, usize)>,
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum KeyframeRemovalError {
    #[error("keyframe {0} is not in the map")]
    UnknownKeyframe(usize),
    #[error(
        "keyframe {keyframe_idx} has {incoming} incoming and {outgoing} outgoing imu edges; \
         only one of each can be merged"
    )]
    UnmergeableImuChain {
        keyframe_idx: usize,
        incoming: usize,
        outgoing: usize,
    },
    #[error("imu edges through keyframe {keyframe_idx} do not meet: {t1} vs {t0}")]
    ImuIntervalGap {
        keyframe_idx: usize,
        t1: f64,
        t0: f64,
    },
    #[error("imu edges through keyframe {0} carry no raw samples to re-preintegrate")]
    MissingImuSamples(usize),
    #[error("merged imu edge {prev} -> {next} already exists")]
    DuplicateImuFactor { prev: usize, next: usize },
    #[error("re-preintegrating the imu edges through keyframe {0} gave an invalid result")]
    InvalidMergedImu(usize),
}

/// The merged edge and the positions, in `Map::imu_factors`, of the two it replaces.
struct ImuMerge {
    incoming: usize,
    outgoing: usize,
    factor: ImuFactor,
}

impl Map {
    /// Removes keyframe `keyframe_idx` and every reference to it.
    ///
    /// Each landmark it observed loses that record: one left with no observer
    /// is retired, one it anchored adopts the smallest remaining
    /// `(keyframe, feature)` as its reference, and the rest have their derived
    /// geometry refreshed. Retired landmarks drop their stale record too, so no
    /// record names a keyframe the map does not hold. An IMU edge on each side
    /// is merged into one; a keyframe at an end of the IMU chain, or whose
    /// edges cannot be re-preintegrated, is refused with the map untouched.
    ///
    /// Other keyframe ids keep their meaning, but keyframe slots after the
    /// removed one shift down, so a [`super::BaWindow`] must be selected and
    /// captured under the same lock, as it already is. A BA snapshot captured
    /// earlier still merges: its writes to the removed keyframe and to the two
    /// replaced IMU edges find no live target and are skipped.
    pub fn remove_keyframe(
        &mut self,
        keyframe_idx: usize,
    ) -> Result<KeyframeRemoval, KeyframeRemovalError> {
        let slot = self
            .keyframes
            .iter()
            .position(|kf| kf.frame.idx == keyframe_idx)
            .ok_or(KeyframeRemovalError::UnknownKeyframe(keyframe_idx))?;
        let merge = self.imu_merge_through(keyframe_idx)?;

        // ── no error exit past this point ──
        self.keyframes.remove(slot);

        let mut removal = KeyframeRemoval {
            keyframe_idx,
            observations_removed: 0,
            landmarks_retired: 0,
            merged_imu_edge: None,
        };
        let mut dirty = Vec::new();
        for mp_idx in 0..self.map_points.len() {
            let point = &mut self.map_points[mp_idx];
            if point.remove_observation(keyframe_idx).is_none() {
                continue;
            }
            removal.observations_removed += 1;
            if point.culled {
                continue;
            }
            if point.observations().is_empty() {
                self.retire_landmark(mp_idx);
                removal.landmarks_retired += 1;
                continue;
            }
            if point.keyframe_idx == keyframe_idx {
                self.adopt_reference(mp_idx);
            }
            dirty.push(mp_idx);
        }
        for mp_idx in dirty {
            self.refresh_landmark(mp_idx);
        }

        if let Some(ImuMerge {
            incoming,
            outgoing,
            factor,
        }) = merge
        {
            removal.merged_imu_edge = Some((factor.prev_kf_idx, factor.curr_kf_idx));
            self.imu_factors[incoming] = factor;
            self.imu_factors.remove(outgoing);
        }
        Ok(removal)
    }

    /// The edge replacing the IMU edges through `keyframe_idx`, validated but
    /// not yet written; `None` when no edge touches it.
    fn imu_merge_through(
        &self,
        keyframe_idx: usize,
    ) -> Result<Option<ImuMerge>, KeyframeRemovalError> {
        let positions = |pick: fn(&ImuFactor) -> usize| -> Vec<usize> {
            self.imu_factors
                .iter()
                .enumerate()
                .filter(|(_, f)| pick(f) == keyframe_idx)
                .map(|(pos, _)| pos)
                .collect()
        };
        let incoming = positions(|f| f.curr_kf_idx);
        let outgoing = positions(|f| f.prev_kf_idx);
        let (incoming, outgoing) = match (incoming.as_slice(), outgoing.as_slice()) {
            ([], []) => return Ok(None),
            (&[i], &[o]) => (i, o),
            (i, o) => {
                return Err(KeyframeRemovalError::UnmergeableImuChain {
                    keyframe_idx,
                    incoming: i.len(),
                    outgoing: o.len(),
                });
            }
        };
        let (before, after) = (&self.imu_factors[incoming], &self.imu_factors[outgoing]);
        let (prev, next) = (before.prev_kf_idx, after.curr_kf_idx);
        if prev == next {
            return Err(KeyframeRemovalError::UnmergeableImuChain {
                keyframe_idx,
                incoming: 1,
                outgoing: 1,
            });
        }
        if self
            .imu_factors
            .iter()
            .any(|f| f.prev_kf_idx == prev && f.curr_kf_idx == next)
        {
            return Err(KeyframeRemovalError::DuplicateImuFactor { prev, next });
        }
        if (before.t1 - after.t0).abs() > IMU_JOIN_TOLERANCE_SEC {
            return Err(KeyframeRemovalError::ImuIntervalGap {
                keyframe_idx,
                t1: before.t1,
                t0: after.t0,
            });
        }
        if before.raw_samples.is_empty() || after.raw_samples.is_empty() {
            return Err(KeyframeRemovalError::MissingImuSamples(keyframe_idx));
        }

        let mut raw_samples: Vec<ImuMeasurement> = before
            .raw_samples
            .iter()
            .chain(&after.raw_samples)
            .copied()
            .collect();
        raw_samples.sort_by(|a, b| a.timestamp.total_cmp(&b.timestamp));
        // A sample stamped exactly at `k` sits in both windows.
        raw_samples.dedup_by(|a, b| a.timestamp == b.timestamp);
        // Linearized at the newest estimate of the edge's from-keyframe bias,
        // the point BA's repropagation measures drift from.
        let bias = self
            .get_keyframe(prev)
            .map_or(before.preintegrated.bias, |kf| kf.imu_bias);
        let preintegrated = PreintegratedImu::from_measurements(
            bias,
            before.preintegrated.calib,
            &raw_samples,
            before.t0,
            after.t1,
        );
        if !(preintegrated.dt > 0.0 && valid_preintegration(&preintegrated)) {
            return Err(KeyframeRemovalError::InvalidMergedImu(keyframe_idx));
        }
        Ok(Some(ImuMerge {
            incoming,
            outgoing,
            factor: ImuFactor {
                prev_kf_idx: prev,
                curr_kf_idx: next,
                preintegrated,
                raw_samples,
                t0: before.t0,
                t1: after.t1,
            },
        }))
    }
}

#[cfg(test)]
#[path = "removal_tests.rs"]
mod tests;
