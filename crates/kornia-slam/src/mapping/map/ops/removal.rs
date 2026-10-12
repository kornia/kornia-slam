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
use kornia_sensors::imu::ImuMeasurement;

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
    #[error("still neighbour {neighbour} of keyframe {keyframe_idx} cannot take over its imu edge")]
    InvalidStillNeighbour {
        keyframe_idx: usize,
        neighbour: usize,
    },
}

/// Which side of a keyframe its still neighbour is on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StillSide {
    /// The neighbour is the earlier keyframe.
    Before,
    /// The neighbour is the later keyframe.
    After,
}

/// A keyframe linked to the one being removed by a stationary constraint held
/// outside the map: neither moved over the `span_s` seconds between them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StillNeighbour {
    pub keyframe_idx: usize,
    pub side: StillSide,
    pub span_s: f64,
}

/// How the IMU chain through a removed keyframe is repaired, validated but
/// not yet written.
enum ImuRepair {
    /// `incoming` and `outgoing` (positions in `Map::imu_factors`) become
    /// `factor`.
    Merge {
        incoming: usize,
        outgoing: usize,
        factor: ImuFactor,
    },
    /// The edge at `position` is replaced by `factor`, its endpoint moved to
    /// the still neighbour.
    Move { position: usize, factor: ImuFactor },
}

impl ImuRepair {
    fn edge(&self) -> (usize, usize) {
        let (Self::Merge { factor, .. } | Self::Move { factor, .. }) = self;
        (factor.prev_kf_idx, factor.curr_kf_idx)
    }
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
        let slot = self.keyframe_slot(keyframe_idx)?;
        let repair = self
            .imu_merge_through(keyframe_idx)?
            .map(|(incoming, outgoing, factor)| ImuRepair::Merge {
                incoming,
                outgoing,
                factor,
            });
        Ok(self.remove_validated(slot, keyframe_idx, repair))
    }

    /// Removes keyframe `keyframe_idx`, which sits at the end of a stationary
    /// chain: linked to `still` by a stationary constraint the caller holds
    /// (and drops), and to the rest of the IMU chain by exactly one edge on the
    /// other side. That edge moves to the still neighbour, keeping its
    /// `[t0, t1]` and samples, its bias walk widened by `still.span_s`
    /// ([`ImuFactor::bias_walk_extra_s`]): the neighbour has the removed
    /// keyframe's pose and zero velocity, so it is the edge the map would hold
    /// had the removed keyframe never been inserted. Landmarks are repaired as
    /// in [`Self::remove_keyframe`].
    ///
    /// Refused, with the map untouched, unless the keyframe has no IMU edge
    /// towards `still` and one away from it, and the neighbour is another
    /// keyframe in the map holding no IMU edge on that side, with a positive
    /// finite span.
    pub fn remove_keyframe_beside_still(
        &mut self,
        keyframe_idx: usize,
        still: StillNeighbour,
    ) -> Result<KeyframeRemoval, KeyframeRemovalError> {
        let slot = self.keyframe_slot(keyframe_idx)?;
        let (position, factor) = self.imu_move_to_still(keyframe_idx, still)?;
        Ok(self.remove_validated(
            slot,
            keyframe_idx,
            Some(ImuRepair::Move { position, factor }),
        ))
    }

    fn keyframe_slot(&self, keyframe_idx: usize) -> Result<usize, KeyframeRemovalError> {
        self.keyframes
            .iter()
            .position(|kf| kf.frame.idx == keyframe_idx)
            .ok_or(KeyframeRemovalError::UnknownKeyframe(keyframe_idx))
    }

    /// The writes of a removal whose IMU repair was validated; cannot fail.
    fn remove_validated(
        &mut self,
        slot: usize,
        keyframe_idx: usize,
        repair: Option<ImuRepair>,
    ) -> KeyframeRemoval {
        self.keyframes.remove(slot);

        let mut removal = KeyframeRemoval {
            keyframe_idx,
            observations_removed: 0,
            landmarks_retired: 0,
            merged_imu_edge: repair.as_ref().map(ImuRepair::edge),
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

        match repair {
            Some(ImuRepair::Merge {
                incoming,
                outgoing,
                factor,
            }) => {
                self.imu_factors[incoming] = factor;
                self.imu_factors.remove(outgoing);
            }
            Some(ImuRepair::Move { position, factor }) => self.imu_factors[position] = factor,
            None => {}
        }
        removal
    }

    /// The edge through `keyframe_idx` moved onto its still neighbour, and
    /// its position; validated but not yet written.
    fn imu_move_to_still(
        &self,
        keyframe_idx: usize,
        still: StillNeighbour,
    ) -> Result<(usize, ImuFactor), KeyframeRemovalError> {
        let invalid = KeyframeRemovalError::InvalidStillNeighbour {
            keyframe_idx,
            neighbour: still.keyframe_idx,
        };
        let neighbour = still.keyframe_idx;
        if neighbour == keyframe_idx
            || self.get_keyframe(neighbour).is_none()
            || !(still.span_s.is_finite() && still.span_s > 0.0)
        {
            return Err(invalid);
        }
        let positions = |pick: fn(&ImuFactor) -> usize, kf: usize| -> Vec<usize> {
            self.imu_factors
                .iter()
                .enumerate()
                .filter(|(_, f)| pick(f) == kf)
                .map(|(pos, _)| pos)
                .collect()
        };
        let incoming = positions(|f| f.curr_kf_idx, keyframe_idx);
        let outgoing = positions(|f| f.prev_kf_idx, keyframe_idx);
        // The still side holds no edge; the other exactly one.
        let (towards, away) = match still.side {
            StillSide::Before => (&incoming, &outgoing),
            StillSide::After => (&outgoing, &incoming),
        };
        let position = match (towards.as_slice(), away.as_slice()) {
            ([], &[position]) => position,
            _ => {
                return Err(KeyframeRemovalError::UnmergeableImuChain {
                    keyframe_idx,
                    incoming: incoming.len(),
                    outgoing: outgoing.len(),
                });
            }
        };
        let mut factor = self.imu_factors[position].clone();
        let neighbour_side = match still.side {
            StillSide::Before => {
                factor.prev_kf_idx = neighbour;
                positions(|f| f.prev_kf_idx, neighbour)
            }
            StillSide::After => {
                factor.curr_kf_idx = neighbour;
                positions(|f| f.curr_kf_idx, neighbour)
            }
        };
        if factor.prev_kf_idx == factor.curr_kf_idx || !neighbour_side.is_empty() {
            return Err(invalid);
        }
        factor.preintegrated.widen_bias_walk(still.span_s);
        factor.bias_walk_extra_s += still.span_s;
        Ok((position, factor))
    }

    /// The edge replacing the IMU edges through `keyframe_idx`, validated but
    /// not yet written; `None` when no edge touches it.
    #[allow(clippy::type_complexity)]
    fn imu_merge_through(
        &self,
        keyframe_idx: usize,
    ) -> Result<Option<(usize, usize, ImuFactor)>, KeyframeRemovalError> {
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
        // Both edges' still time stays outside the merged span, so its walk
        // carries over ([`ImuFactor::bias_walk_extra_s`]).
        let factor = ImuFactor {
            prev_kf_idx: prev,
            curr_kf_idx: next,
            preintegrated: before.preintegrated.clone(),
            raw_samples,
            t0: before.t0,
            t1: after.t1,
            bias_walk_extra_s: before.bias_walk_extra_s + after.bias_walk_extra_s,
        };
        let preintegrated = factor.repreintegrate(bias);
        if !(preintegrated.dt > 0.0 && valid_preintegration(&preintegrated)) {
            return Err(KeyframeRemovalError::InvalidMergedImu(keyframe_idx));
        }
        Ok(Some((
            incoming,
            outgoing,
            ImuFactor {
                preintegrated,
                ..factor
            },
        )))
    }
}

#[cfg(test)]
#[path = "removal_tests.rs"]
mod tests;
