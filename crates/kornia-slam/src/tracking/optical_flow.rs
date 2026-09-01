//! Persistent KLT track state and map-point associations.

use std::collections::{HashMap, HashSet};

use kornia_image::Image;
use kornia_imgproc::optical_flow_pyr_lk::{PyrLKError, PyrLKParams, calc_optical_flow_pyr_lk};
use thiserror::Error;

/// Stable identity for a feature track.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TrackId(u64);

impl TrackId {
    /// Returns the numeric identifier for logging or serialization.
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

/// One point tracked frame-to-frame by optical flow.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Track {
    id: TrackId,
    pixel: [f32; 2],
    map_point_idx: Option<usize>,
    age: u32,
}

impl Track {
    /// Stable identity across frames.
    pub fn id(&self) -> TrackId {
        self.id
    }

    /// Raw, distorted pixel position in the current frame.
    pub fn pixel(&self) -> [f32; 2] {
        self.pixel
    }

    /// Associated map-point index, if any.
    pub fn map_point_idx(&self) -> Option<usize> {
        self.map_point_idx
    }

    /// Number of consecutive frames represented by this track.
    pub fn age(&self) -> u32 {
        self.age
    }
}

/// A map point matched to a detected keypoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MapKeypointMatch {
    pub map_point_idx: usize,
    pub keypoint_idx: usize,
}

/// One accepted numerical-flow result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlowSurvivor {
    pub track_id: TrackId,
    pub pixel: [f32; 2],
    pub error: f32,
}

/// A one-to-one association between a track, map point, and keypoint.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeypointCorrespondence {
    pub track_id: TrackId,
    pub map_point_idx: usize,
    pub keypoint_idx: usize,
    pub distance_sq: f32,
}

/// Invalid input to a persistent track-state transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum TrackSetError {
    #[error("unknown track ID {0:?}")]
    UnknownTrackId(TrackId),
    #[error("duplicate track ID {0:?}")]
    DuplicateTrackId(TrackId),
    #[error("duplicate map-point index {0}")]
    DuplicateMapPoint(usize),
    #[error("duplicate keypoint index {0}")]
    DuplicateKeypoint(usize),
    #[error("invalid keypoint index {0}")]
    InvalidKeypointIndex(usize),
}

/// Persistent owner of track identity, age, and map-point associations.
#[derive(Debug, Default)]
pub struct TrackSet {
    next_id: u64,
    tracks: Vec<Track>,
}

impl TrackSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn tracks(&self) -> &[Track] {
        &self.tracks
    }

    pub fn len(&self) -> usize {
        self.tracks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tracks.is_empty()
    }

    /// Inserts a new, unassociated track and returns its identity.
    pub fn spawn(&mut self, pixel: [f32; 2]) -> TrackId {
        let id = self.allocate_id();
        self.tracks.push(Track {
            id,
            pixel,
            map_point_idx: None,
            age: 1,
        });
        id
    }

    /// Advances the active set to the supplied KLT survivors.
    ///
    /// Validation completes before state changes, so an error leaves the set
    /// unchanged.
    pub fn advance(&mut self, survivors: Vec<FlowSurvivor>) -> Result<(), TrackSetError> {
        let by_id: HashMap<TrackId, Track> =
            self.tracks.iter().map(|track| (track.id, *track)).collect();
        let mut seen = HashSet::with_capacity(survivors.len());
        for survivor in &survivors {
            if !by_id.contains_key(&survivor.track_id) {
                return Err(TrackSetError::UnknownTrackId(survivor.track_id));
            }
            if !seen.insert(survivor.track_id) {
                return Err(TrackSetError::DuplicateTrackId(survivor.track_id));
            }
        }

        self.tracks = survivors
            .into_iter()
            .map(|survivor| {
                let previous = by_id[&survivor.track_id];
                Track {
                    id: survivor.track_id,
                    pixel: survivor.pixel,
                    map_point_idx: previous.map_point_idx,
                    age: previous.age.saturating_add(1),
                }
            })
            .collect();
        Ok(())
    }

    /// Reconciles active tracks with detector/map matches for this frame.
    ///
    /// Existing map points retain identity and advance one frame. New map
    /// points receive fresh identities at age one. Unmatched tracks are
    /// removed. Validation is transactional.
    pub fn reconcile_from_matches(
        &mut self,
        matches: &[MapKeypointMatch],
        keypoints: &[[f32; 2]],
    ) -> Result<(), TrackSetError> {
        let mut seen_map_points = HashSet::with_capacity(matches.len());
        let mut seen_keypoints = HashSet::with_capacity(matches.len());
        for matched in matches {
            if matched.keypoint_idx >= keypoints.len() {
                return Err(TrackSetError::InvalidKeypointIndex(matched.keypoint_idx));
            }
            if !seen_map_points.insert(matched.map_point_idx) {
                return Err(TrackSetError::DuplicateMapPoint(matched.map_point_idx));
            }
            if !seen_keypoints.insert(matched.keypoint_idx) {
                return Err(TrackSetError::DuplicateKeypoint(matched.keypoint_idx));
            }
        }

        let old_by_map_point: HashMap<usize, Track> = self
            .tracks
            .iter()
            .filter_map(|track| track.map_point_idx.map(|idx| (idx, *track)))
            .collect();
        let new_count = matches
            .iter()
            .filter(|matched| !old_by_map_point.contains_key(&matched.map_point_idx))
            .count();
        let start_id = self.reserve_ids(new_count);
        let mut next_new_id = start_id;

        self.tracks = matches
            .iter()
            .map(|matched| {
                if let Some(previous) = old_by_map_point.get(&matched.map_point_idx) {
                    Track {
                        id: previous.id,
                        pixel: keypoints[matched.keypoint_idx],
                        map_point_idx: Some(matched.map_point_idx),
                        age: previous.age.saturating_add(1),
                    }
                } else {
                    let id = TrackId(next_new_id);
                    next_new_id += 1;
                    Track {
                        id,
                        pixel: keypoints[matched.keypoint_idx],
                        map_point_idx: Some(matched.map_point_idx),
                        age: 1,
                    }
                }
            })
            .collect();
        Ok(())
    }

    fn allocate_id(&mut self) -> TrackId {
        let id = TrackId(self.next_id);
        self.next_id = self
            .next_id
            .checked_add(1)
            .expect("track ID space exhausted");
        id
    }

    fn reserve_ids(&mut self, count: usize) -> u64 {
        let start = self.next_id;
        self.next_id = self
            .next_id
            .checked_add(count as u64)
            .expect("track ID space exhausted");
        start
    }
}

/// SLAM-specific acceptance thresholds for numerical KLT results.
#[derive(Debug, Clone, Copy)]
pub struct SurvivorFilterConfig {
    pub max_error: f32,
    pub max_movement_px: f32,
    pub border_margin_px: f32,
}

impl Default for SurvivorFilterConfig {
    fn default() -> Self {
        Self {
            max_error: 20.0,
            max_movement_px: 60.0,
            border_margin_px: 10.0,
        }
    }
}

/// Stateless numerical KLT executor.
#[derive(Debug, Clone)]
pub struct KltTracker {
    params: PyrLKParams,
    survivor_filter: SurvivorFilterConfig,
}

impl Default for KltTracker {
    fn default() -> Self {
        Self::new(PyrLKParams::default(), SurvivorFilterConfig::default())
    }
}

impl KltTracker {
    pub fn new(params: PyrLKParams, survivor_filter: SurvivorFilterConfig) -> Self {
        Self {
            params,
            survivor_filter,
        }
    }

    pub fn params(&self) -> &PyrLKParams {
        &self.params
    }

    pub fn survivor_filter(&self) -> SurvivorFilterConfig {
        self.survivor_filter
    }

    pub fn track(
        &self,
        tracks: &[Track],
        previous: &Image<u8, 1>,
        current: &Image<u8, 1>,
    ) -> Result<Vec<FlowSurvivor>, PyrLKError> {
        self.track_impl(tracks, None, previous, current)
    }

    pub fn track_with_initial(
        &self,
        tracks: &[Track],
        initial_pixels: &[[f32; 2]],
        previous: &Image<u8, 1>,
        current: &Image<u8, 1>,
    ) -> Result<Vec<FlowSurvivor>, PyrLKError> {
        if initial_pixels.len() != tracks.len() {
            return Err(PyrLKError::InitialFlowLengthMismatch {
                expected: tracks.len(),
                provided: initial_pixels.len(),
            });
        }
        self.track_impl(tracks, Some(initial_pixels), previous, current)
    }

    fn track_impl(
        &self,
        tracks: &[Track],
        initial_pixels: Option<&[[f32; 2]]>,
        previous: &Image<u8, 1>,
        current: &Image<u8, 1>,
    ) -> Result<Vec<FlowSurvivor>, PyrLKError> {
        if tracks.is_empty() {
            return Ok(Vec::new());
        }

        let previous_f32 = previous
            .cast::<f32>()
            .expect("u8 -> f32 image cast is always representable");
        let current_f32 = current
            .cast::<f32>()
            .expect("u8 -> f32 image cast is always representable");
        let previous_pixels: Vec<[f32; 2]> = tracks.iter().map(Track::pixel).collect();
        let mut params = self.params.clone();
        params.use_initial_flow = initial_pixels.is_some();
        let result = calc_optical_flow_pyr_lk(
            &previous_f32,
            &current_f32,
            &previous_pixels,
            initial_pixels,
            &params,
        )?;

        let width = current.width() as f32;
        let height = current.height() as f32;
        let mut survivors = Vec::with_capacity(tracks.len());
        for (index, track) in tracks.iter().enumerate() {
            let next_pt = result.next_pts[index];
            if Self::accepts(
                track,
                result.status[index],
                result.error[index],
                next_pt,
                [width, height],
                &self.survivor_filter,
            ) {
                survivors.push(FlowSurvivor {
                    track_id: track.id,
                    pixel: next_pt,
                    error: result.error[index],
                });
            }
        }
        Ok(survivors)
    }

    fn accepts(
        track: &Track,
        status: u8,
        error: f32,
        next_pixel: [f32; 2],
        image_size: [f32; 2],
        filter: &SurvivorFilterConfig,
    ) -> bool {
        if status == 0 || !error.is_finite() || error > filter.max_error {
            return false;
        }
        if !next_pixel[0].is_finite() || !next_pixel[1].is_finite() {
            return false;
        }
        let margin = filter.border_margin_px;
        if next_pixel[0] < margin
            || next_pixel[1] < margin
            || next_pixel[0] > image_size[0] - margin
            || next_pixel[1] > image_size[1] - margin
        {
            return false;
        }
        let dx = next_pixel[0] - track.pixel[0];
        let dy = next_pixel[1] - track.pixel[1];
        let max_movement_sq = filter.max_movement_px * filter.max_movement_px;
        dx * dx + dy * dy <= max_movement_sq
    }
}

/// Deterministically snaps associated survivors to unique detected keypoints.
///
/// Candidates are considered greedily by distance, then track ID and
/// keypoint index. This guarantees one-to-one output but does not promise a
/// globally maximum-cardinality assignment.
pub fn snap_unique(
    tracks: &TrackSet,
    survivors: &[FlowSurvivor],
    keypoints: &[[f32; 2]],
    radius_px: f32,
) -> Result<Vec<KeypointCorrespondence>, TrackSetError> {
    let by_id: HashMap<TrackId, &Track> = tracks
        .tracks
        .iter()
        .map(|track| (track.id, track))
        .collect();
    let mut seen_survivors = HashSet::with_capacity(survivors.len());
    for survivor in survivors {
        if !by_id.contains_key(&survivor.track_id) {
            return Err(TrackSetError::UnknownTrackId(survivor.track_id));
        }
        if !seen_survivors.insert(survivor.track_id) {
            return Err(TrackSetError::DuplicateTrackId(survivor.track_id));
        }
    }

    if radius_px < 0.0 || !radius_px.is_finite() {
        return Ok(Vec::new());
    }
    let radius_sq = radius_px * radius_px;
    let mut candidates = Vec::new();
    for survivor in survivors {
        let track = by_id[&survivor.track_id];
        let Some(map_point_idx) = track.map_point_idx else {
            continue;
        };
        if !survivor.pixel[0].is_finite() || !survivor.pixel[1].is_finite() {
            continue;
        }
        for (keypoint_idx, keypoint) in keypoints.iter().enumerate() {
            let dx = keypoint[0] - survivor.pixel[0];
            let dy = keypoint[1] - survivor.pixel[1];
            let distance_sq = dx * dx + dy * dy;
            if distance_sq.is_finite() && distance_sq <= radius_sq {
                candidates.push(KeypointCorrespondence {
                    track_id: survivor.track_id,
                    map_point_idx,
                    keypoint_idx,
                    distance_sq,
                });
            }
        }
    }

    candidates.sort_by(|left, right| {
        left.distance_sq
            .total_cmp(&right.distance_sq)
            .then_with(|| left.track_id.cmp(&right.track_id))
            .then_with(|| left.keypoint_idx.cmp(&right.keypoint_idx))
            .then_with(|| left.map_point_idx.cmp(&right.map_point_idx))
    });

    let mut used_tracks = HashSet::new();
    let mut used_map_points = HashSet::new();
    let mut used_keypoints = HashSet::new();
    let mut accepted = Vec::new();
    for candidate in candidates {
        if used_tracks.contains(&candidate.track_id)
            || used_map_points.contains(&candidate.map_point_idx)
            || used_keypoints.contains(&candidate.keypoint_idx)
        {
            continue;
        }
        used_tracks.insert(candidate.track_id);
        used_map_points.insert(candidate.map_point_idx);
        used_keypoints.insert(candidate.keypoint_idx);
        accepted.push(candidate);
    }
    Ok(accepted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::ImageSize;

    #[test]
    fn spawn_inserts_tracks_with_unique_ids() {
        let mut tracks = TrackSet::new();
        let a = tracks.spawn([1.0, 2.0]);
        let b = tracks.spawn([3.0, 4.0]);

        assert_ne!(a, b);
        assert_eq!(a.as_u64(), 0);
        assert_eq!(b.as_u64(), 1);
        assert_eq!(tracks.len(), 2);
        assert_eq!(tracks.tracks()[0].id(), a);
        assert_eq!(tracks.tracks()[0].pixel(), [1.0, 2.0]);
        assert_eq!(tracks.tracks()[0].map_point_idx(), None);
        assert_eq!(tracks.tracks()[0].age(), 1);
    }

    #[test]
    fn advance_replaces_tracks_and_increments_age_once() {
        let mut tracks = TrackSet::new();
        let kept = tracks.spawn([1.0, 2.0]);
        tracks.spawn([3.0, 4.0]);

        tracks
            .advance(vec![FlowSurvivor {
                track_id: kept,
                pixel: [2.0, 3.0],
                error: 0.5,
            }])
            .unwrap();

        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks.tracks()[0].id(), kept);
        assert_eq!(tracks.tracks()[0].pixel(), [2.0, 3.0]);
        assert_eq!(tracks.tracks()[0].age(), 2);
    }

    #[test]
    fn advance_rejects_unknown_and_duplicate_ids_transactionally() {
        let mut tracks = TrackSet::new();
        let id = tracks.spawn([1.0, 2.0]);
        let before = tracks.tracks().to_vec();

        let duplicate = FlowSurvivor {
            track_id: id,
            pixel: [2.0, 3.0],
            error: 0.5,
        };
        assert_eq!(
            tracks.advance(vec![duplicate, duplicate]),
            Err(TrackSetError::DuplicateTrackId(id))
        );
        assert_eq!(tracks.tracks(), before);

        let unknown = TrackId(u64::MAX);
        assert_eq!(
            tracks.advance(vec![FlowSurvivor {
                track_id: unknown,
                pixel: [2.0, 3.0],
                error: 0.5,
            }]),
            Err(TrackSetError::UnknownTrackId(unknown))
        );
        assert_eq!(tracks.tracks(), before);
    }

    #[test]
    fn reconcile_preserves_identity_and_validates_matches() {
        let mut tracks = TrackSet::new();
        tracks
            .reconcile_from_matches(
                &[MapKeypointMatch {
                    map_point_idx: 42,
                    keypoint_idx: 0,
                }],
                &[[1.0, 2.0]],
            )
            .unwrap();
        let id_42 = tracks.tracks()[0].id();

        tracks
            .reconcile_from_matches(
                &[
                    MapKeypointMatch {
                        map_point_idx: 42,
                        keypoint_idx: 0,
                    },
                    MapKeypointMatch {
                        map_point_idx: 7,
                        keypoint_idx: 1,
                    },
                ],
                &[[5.0, 6.0], [10.0, 11.0]],
            )
            .unwrap();

        let retained = tracks
            .tracks()
            .iter()
            .find(|track| track.map_point_idx() == Some(42))
            .unwrap();
        assert_eq!(retained.id(), id_42);
        assert_eq!(retained.age(), 2);
        assert_eq!(retained.pixel(), [5.0, 6.0]);

        let new = tracks
            .tracks()
            .iter()
            .find(|track| track.map_point_idx() == Some(7))
            .unwrap();
        assert_ne!(new.id(), id_42);
        assert_eq!(new.age(), 1);
        let new_id = new.id();
        let spawned_after_reconcile = tracks.spawn([20.0, 21.0]);
        assert_ne!(spawned_after_reconcile, id_42);
        assert_ne!(spawned_after_reconcile, new_id);

        let before = tracks.tracks().to_vec();
        assert_eq!(
            tracks.reconcile_from_matches(
                &[
                    MapKeypointMatch {
                        map_point_idx: 42,
                        keypoint_idx: 0,
                    },
                    MapKeypointMatch {
                        map_point_idx: 42,
                        keypoint_idx: 1,
                    },
                ],
                &[[0.0, 0.0], [1.0, 1.0]],
            ),
            Err(TrackSetError::DuplicateMapPoint(42))
        );
        assert_eq!(tracks.tracks(), before);

        assert_eq!(
            tracks.reconcile_from_matches(
                &[
                    MapKeypointMatch {
                        map_point_idx: 42,
                        keypoint_idx: 0,
                    },
                    MapKeypointMatch {
                        map_point_idx: 7,
                        keypoint_idx: 0,
                    },
                ],
                &[[0.0, 0.0]],
            ),
            Err(TrackSetError::DuplicateKeypoint(0))
        );
        assert_eq!(tracks.tracks(), before);

        assert_eq!(
            tracks.reconcile_from_matches(
                &[MapKeypointMatch {
                    map_point_idx: 42,
                    keypoint_idx: 3,
                }],
                &[[0.0, 0.0]],
            ),
            Err(TrackSetError::InvalidKeypointIndex(3))
        );
        assert_eq!(tracks.tracks(), before);
    }

    #[test]
    fn age_saturates_at_u32_max() {
        let id = TrackId(0);
        let mut tracks = TrackSet {
            next_id: 1,
            tracks: vec![Track {
                id,
                pixel: [1.0, 2.0],
                map_point_idx: None,
                age: u32::MAX,
            }],
        };

        tracks
            .advance(vec![FlowSurvivor {
                track_id: id,
                pixel: [2.0, 3.0],
                error: 0.0,
            }])
            .unwrap();
        assert_eq!(tracks.tracks()[0].age(), u32::MAX);
    }

    /// Builds a black image with a white square whose corner is trackable.
    fn square_image(origin: [usize; 2]) -> Image<u8, 1> {
        const SIZE: usize = 200;
        let mut data = vec![0u8; SIZE * SIZE];
        for y in origin[1]..(origin[1] + 40).min(SIZE) {
            for x in origin[0]..(origin[0] + 40).min(SIZE) {
                data[y * SIZE + x] = 255;
            }
        }
        Image::new(
            ImageSize {
                width: SIZE,
                height: SIZE,
            },
            data,
        )
        .unwrap()
    }

    #[test]
    fn track_follows_a_known_shift() {
        let prev_img = square_image([90, 90]);
        let next_img = square_image([93, 92]); // shifted by (+3, +2)

        let mut tracks = TrackSet::new();
        let id = tracks.spawn([90.0, 90.0]);

        let survivors = KltTracker::default()
            .track(tracks.tracks(), &prev_img, &next_img)
            .expect("KLT should succeed on a clean synthetic shift");

        assert_eq!(survivors.len(), 1, "the corner should track successfully");
        let got = survivors[0].pixel;
        assert!(
            (got[0] - 93.0).abs() < 1.5 && (got[1] - 92.0).abs() < 1.5,
            "expected tracked pixel near (93, 92), got {got:?}"
        );
        assert_eq!(survivors[0].track_id, id);
    }

    #[test]
    fn tracker_handles_empty_and_explicit_initial_flow() {
        let image = square_image([90, 90]);
        let params = PyrLKParams {
            use_initial_flow: true,
            ..PyrLKParams::default()
        };
        let tracker = KltTracker::new(params, SurvivorFilterConfig::default());
        assert!(tracker.track(&[], &image, &image).unwrap().is_empty());

        let mut tracks = TrackSet::new();
        tracks.spawn([90.0, 90.0]);
        let mismatch = tracker
            .track_with_initial(tracks.tracks(), &[], &image, &image)
            .unwrap_err();
        assert!(matches!(
            mismatch,
            PyrLKError::InitialFlowLengthMismatch {
                expected: 1,
                provided: 0
            }
        ));

        // `track` overrides a configured true flag because no initial pixels
        // were supplied through its type-level contract.
        assert_eq!(
            tracker
                .track(tracks.tracks(), &image, &image)
                .unwrap()
                .len(),
            1
        );

        let shifted = square_image([93, 92]);
        let survivors = tracker
            .track_with_initial(tracks.tracks(), &[[93.0, 92.0]], &image, &shifted)
            .unwrap();
        assert_eq!(survivors.len(), 1);
        assert!((survivors[0].pixel[0] - 93.0).abs() < 1.5);
        assert!((survivors[0].pixel[1] - 92.0).abs() < 1.5);
    }

    #[test]
    fn survivor_filters_reject_each_failure_mode() {
        let filter = SurvivorFilterConfig::default();
        let track = Track {
            id: TrackId(0),
            pixel: [50.0, 50.0],
            map_point_idx: None,
            age: 1,
        };

        assert!(!KltTracker::accepts(
            &track,
            0,
            0.0,
            [51.0, 51.0],
            [100.0, 100.0],
            &filter
        ));
        assert!(!KltTracker::accepts(
            &track,
            1,
            filter.max_error + 1.0,
            [51.0, 51.0],
            [100.0, 100.0],
            &filter,
        ));
        assert!(!KltTracker::accepts(
            &track,
            1,
            0.0,
            [track.pixel()[0] + filter.max_movement_px + 1.0, 50.0],
            [200.0, 200.0],
            &filter,
        ));
        assert!(!KltTracker::accepts(
            &track,
            1,
            0.0,
            [filter.border_margin_px - 1.0, 50.0],
            [100.0, 100.0],
            &filter,
        ));
        assert!(KltTracker::accepts(
            &track,
            1,
            0.0,
            [51.0, 51.0],
            [100.0, 100.0],
            &filter,
        ));
    }

    fn associated_tracks() -> TrackSet {
        let mut tracks = TrackSet::new();
        tracks
            .reconcile_from_matches(
                &[
                    MapKeypointMatch {
                        map_point_idx: 10,
                        keypoint_idx: 0,
                    },
                    MapKeypointMatch {
                        map_point_idx: 20,
                        keypoint_idx: 1,
                    },
                ],
                &[[10.0, 10.0], [12.0, 10.0]],
            )
            .unwrap();
        tracks
    }

    #[test]
    fn snapping_is_unique_and_deterministic() {
        let tracks = associated_tracks();
        let a = tracks.tracks()[0].id();
        let b = tracks.tracks()[1].id();
        let survivors = [
            FlowSurvivor {
                track_id: a,
                pixel: [10.0, 10.0],
                error: 0.0,
            },
            FlowSurvivor {
                track_id: b,
                pixel: [10.0, 10.0],
                error: 0.0,
            },
        ];

        let first = snap_unique(&tracks, &survivors, &[[10.0, 10.0]], 3.0).unwrap();
        let second = snap_unique(&tracks, &survivors, &[[10.0, 10.0]], 3.0).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].track_id, a.min(b));
        assert_eq!(first[0].keypoint_idx, 0);
    }

    #[test]
    fn snapping_uses_nearest_keypoint_and_omits_invalid_candidates() {
        let mut tracks = associated_tracks();
        let associated = tracks.tracks()[0].id();
        let unassociated = tracks.spawn([50.0, 50.0]);
        let survivors = [
            FlowSurvivor {
                track_id: associated,
                pixel: [11.0, 10.0],
                error: 0.0,
            },
            FlowSurvivor {
                track_id: unassociated,
                pixel: [50.0, 50.0],
                error: 0.0,
            },
        ];

        let matches = snap_unique(
            &tracks,
            &survivors,
            &[[9.0, 10.0], [11.25, 10.0], [100.0, 100.0]],
            3.0,
        )
        .unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].keypoint_idx, 1);
        assert_eq!(matches[0].distance_sq, 0.0625);

        let unknown = TrackId(u64::MAX);
        assert_eq!(
            snap_unique(
                &tracks,
                &[FlowSurvivor {
                    track_id: unknown,
                    pixel: [0.0, 0.0],
                    error: 0.0,
                }],
                &[],
                3.0,
            ),
            Err(TrackSetError::UnknownTrackId(unknown))
        );

        assert_eq!(
            snap_unique(&tracks, &[survivors[0], survivors[0]], &[], 3.0),
            Err(TrackSetError::DuplicateTrackId(associated))
        );
    }

    #[test]
    fn snapping_never_emits_a_duplicate_map_point() {
        let a = TrackId(0);
        let b = TrackId(1);
        let tracks = TrackSet {
            next_id: 2,
            tracks: vec![
                Track {
                    id: a,
                    pixel: [10.0, 10.0],
                    map_point_idx: Some(42),
                    age: 1,
                },
                Track {
                    id: b,
                    pixel: [20.0, 20.0],
                    map_point_idx: Some(42),
                    age: 1,
                },
            ],
        };
        let survivors = [
            FlowSurvivor {
                track_id: a,
                pixel: [10.0, 10.0],
                error: 0.0,
            },
            FlowSurvivor {
                track_id: b,
                pixel: [20.0, 20.0],
                error: 0.0,
            },
        ];

        let matches = snap_unique(&tracks, &survivors, &[[10.0, 10.0], [20.0, 20.0]], 1.0).unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].map_point_idx, 42);
    }
}
