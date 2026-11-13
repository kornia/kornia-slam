//! Persistent KLT track state and map-point associations.

use std::collections::{HashMap, HashSet};

use kornia_image::{Image, ImageSize};
use kornia_imgproc::filter::scharr_spatial_gradient_float;
use kornia_imgproc::optical_flow_pyr_lk::{
    LkPyramidU8, PyrLKError, PyrLKParams, PyrLKPrecomputed, PyrLKResult, PyrLkU8Error,
    build_lk_precomputed, calc_optical_flow_pyr_lk_u8, calc_optical_flow_pyr_lk_with_precomputed,
};
use kornia_imgproc::pyramid::pyrdown_f32;
use thiserror::Error;

use crate::tracking::timings::{TrackingTimings, timed};

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

/// A start pixel per track and a shallower search, for a frame whose motion
/// is predicted well.
#[derive(Debug, Clone, Copy)]
pub struct FlowPrediction<'a> {
    /// Raw pixel where each track is expected, in track order.
    pub initial_pixels: &'a [[f32; 2]],
    /// Deepest pyramid level searched; capped at the tracker's own.
    pub max_level: usize,
    /// Gauss-Newton iterations per level.
    pub max_iter: usize,
}

/// Stateless numerical KLT executor.
#[derive(Debug, Clone)]
pub struct KltTracker {
    params: PyrLKParams,
    survivor_filter: SurvivorFilterConfig,
    kernel: LkKernel,
}

/// The pyramidal LK implementation behind [`KltTracker::track_cached`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LkKernel {
    /// kornia-imgproc's f32 LK on f32 pyramids and Scharr gradients.
    #[default]
    F32,
    /// OpenCV's scheme: u8 pyramid, i16 Scharr, Q14 fixed-point windows.
    FixedPointU8,
}

/// A failed flow solve, from either kernel.
#[derive(Debug, Error)]
pub enum FlowError {
    #[error(transparent)]
    F32(#[from] PyrLKError),
    #[error(transparent)]
    FixedPoint(#[from] PyrLkU8Error),
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
            kernel: LkKernel::default(),
        }
    }

    /// Tracks with `kernel` in [`Self::track_cached`].
    pub fn with_kernel(mut self, kernel: LkKernel) -> Self {
        self.kernel = kernel;
        self
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
        self.track_impl(
            tracks,
            None,
            previous,
            current,
            &mut TrackingTimings::default(),
        )
    }

    /// Like [`Self::track`], adding the pyramid build and the flow solve to
    /// `timings`.
    pub fn track_timed(
        &self,
        tracks: &[Track],
        previous: &Image<u8, 1>,
        current: &Image<u8, 1>,
        timings: &mut TrackingTimings,
    ) -> Result<Vec<FlowSurvivor>, PyrLKError> {
        self.track_impl(tracks, None, previous, current, timings)
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
        self.track_impl(
            tracks,
            Some(initial_pixels),
            previous,
            current,
            &mut TrackingTimings::default(),
        )
    }

    fn track_impl(
        &self,
        tracks: &[Track],
        initial_pixels: Option<&[[f32; 2]]>,
        previous: &Image<u8, 1>,
        current: &Image<u8, 1>,
        timings: &mut TrackingTimings,
    ) -> Result<Vec<FlowSurvivor>, PyrLKError> {
        if tracks.is_empty() {
            return Ok(Vec::new());
        }

        // The same two calls `calc_optical_flow_pyr_lk` makes, split so the
        // build and the solve are timed apart.
        let precomputed = timed(&mut timings.klt_build, || {
            let previous_f32 = previous.cast::<f32>()?;
            let current_f32 = current.cast::<f32>()?;
            build_lk_precomputed(&previous_f32, &current_f32, self.params.max_level)
        })?;
        self.solve(
            tracks,
            initial_pixels,
            &precomputed,
            &self.params,
            current.size(),
            timings,
        )
    }

    /// Like [`Self::track_timed`], reusing the pyramid `cache` built for the
    /// previous call's `current` image when it is this call's `previous`.
    /// The flow is bit-identical to [`Self::track`]: the same kornia-imgproc
    /// calls run on the same inputs.
    ///
    /// With a `prediction`, LK starts each track at its predicted pixel and
    /// searches only the prediction's levels of the same pyramid.
    pub fn track_cached(
        &self,
        cache: &mut KltPyramidCache,
        tracks: &[Track],
        prediction: Option<FlowPrediction<'_>>,
        previous: &Image<u8, 1>,
        current: &Image<u8, 1>,
        timings: &mut TrackingTimings,
    ) -> Result<Vec<FlowSurvivor>, FlowError> {
        if tracks.is_empty() {
            return Ok(Vec::new());
        }
        check_same_size(previous, current)?;
        if self.kernel == LkKernel::FixedPointU8 {
            return self.track_cached_u8(cache, tracks, prediction, previous, current, timings);
        }
        let max_level = self.params.max_level;
        let (mut precomputed, next_image) = timed(&mut timings.klt_build, || {
            let (prev_pyr, grad_x_pyr, grad_y_pyr) = match cache.take_for(previous) {
                Some(cached) => cached.into_parts()?,
                None => LevelPyramid::build(previous, max_level)?.into_parts()?,
            };
            let next = LevelPyramid::build(current, max_level)?;
            let precomputed = PyrLKPrecomputed {
                prev_pyr,
                next_pyr: next.levels,
                grad_x_pyr,
                grad_y_pyr,
            };
            Ok::<_, PyrLKError>((precomputed, next.image))
        })?;
        let survivors = match prediction {
            None => self.solve(
                tracks,
                None,
                &precomputed,
                &self.params,
                current.size(),
                timings,
            ),
            Some(prediction) => {
                let levels = prediction.max_level.min(max_level) + 1;
                let params = PyrLKParams {
                    max_level: levels - 1,
                    max_iter: prediction.max_iter,
                    ..self.params.clone()
                };
                // The coarse levels stay in the cache for the next frame.
                let mut next_tail = precomputed.next_pyr.split_off(levels);
                precomputed.prev_pyr.truncate(levels);
                precomputed.grad_x_pyr.truncate(levels);
                precomputed.grad_y_pyr.truncate(levels);
                let survivors = self.solve(
                    tracks,
                    Some(prediction.initial_pixels),
                    &precomputed,
                    &params,
                    current.size(),
                    timings,
                );
                precomputed.next_pyr.append(&mut next_tail);
                survivors
            }
        };
        cache.entry = Some(LevelPyramid {
            image: next_image,
            levels: precomputed.next_pyr,
        });
        Ok(survivors?)
    }

    /// [`Self::track_cached`] on the fixed-point kernel. The current frame's
    /// pyramid is built with derivatives so it serves as the next previous;
    /// the two pyramids are recycled.
    fn track_cached_u8(
        &self,
        cache: &mut KltPyramidCache,
        tracks: &[Track],
        prediction: Option<FlowPrediction<'_>>,
        previous: &Image<u8, 1>,
        current: &Image<u8, 1>,
        timings: &mut TrackingTimings,
    ) -> Result<Vec<FlowSurvivor>, FlowError> {
        let (win, max_level) = (self.params.win_size, self.params.max_level);
        let (prev_pyr, next_pyr) = timed(&mut timings.klt_build, || {
            let prev_pyr = match cache.take_u8_for(previous) {
                Some(cached) => cached,
                None => LkPyramidU8::new(previous, win, max_level, true)?,
            };
            let next_pyr = match cache.u8_spare.take() {
                Some(mut spare) => {
                    spare.rebuild(current, win, max_level, true)?;
                    spare
                }
                None => LkPyramidU8::new(current, win, max_level, true)?,
            };
            Ok::<_, PyrLkU8Error>((prev_pyr, next_pyr))
        })?;
        let mut params = self.params.clone();
        if let Some(prediction) = &prediction {
            params.max_level = prediction.max_level.min(max_level);
            params.max_iter = prediction.max_iter;
        }
        params.use_initial_flow = prediction.is_some();
        let previous_pixels: Vec<[f32; 2]> = tracks.iter().map(Track::pixel).collect();
        let initial = prediction.map(|p| p.initial_pixels);
        let result = timed(&mut timings.klt_track, || {
            calc_optical_flow_pyr_lk_u8(&prev_pyr, &next_pyr, &previous_pixels, initial, &params)
        });
        cache.u8_spare = Some(prev_pyr);
        cache.u8_entry = Some((current.clone(), next_pyr));
        Ok(self.survivors(tracks, &result?, current.size()))
    }

    fn solve(
        &self,
        tracks: &[Track],
        initial_pixels: Option<&[[f32; 2]]>,
        precomputed: &PyrLKPrecomputed,
        params: &PyrLKParams,
        image_size: ImageSize,
        timings: &mut TrackingTimings,
    ) -> Result<Vec<FlowSurvivor>, PyrLKError> {
        let previous_pixels: Vec<[f32; 2]> = tracks.iter().map(Track::pixel).collect();
        let mut params = params.clone();
        params.use_initial_flow = initial_pixels.is_some();
        let result = timed(&mut timings.klt_track, || {
            calc_optical_flow_pyr_lk_with_precomputed(
                precomputed,
                &previous_pixels,
                initial_pixels,
                &params,
            )
        })?;
        Ok(self.survivors(tracks, &result, image_size))
    }

    fn survivors(
        &self,
        tracks: &[Track],
        result: &PyrLKResult,
        image_size: ImageSize,
    ) -> Vec<FlowSurvivor> {
        let width = image_size.width as f32;
        let height = image_size.height as f32;
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
        survivors
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

fn check_same_size(previous: &Image<u8, 1>, current: &Image<u8, 1>) -> Result<(), PyrLKError> {
    if previous.size() != current.size() {
        return Err(PyrLKError::ImageSizeMismatch {
            prev_width: previous.width(),
            prev_height: previous.height(),
            next_width: current.width(),
            next_height: current.height(),
        });
    }
    Ok(())
}

/// One image's LK pyramid, built exactly as `build_lk_precomputed` builds
/// each side, so a cached pyramid yields bit-identical flow.
struct LevelPyramid {
    image: Image<u8, 1>,
    levels: Vec<Image<f32, 1>>,
}

type PyramidParts = (Vec<Image<f32, 1>>, Vec<Image<f32, 1>>, Vec<Image<f32, 1>>);

impl LevelPyramid {
    fn build(image: &Image<u8, 1>, max_level: usize) -> Result<Self, PyrLKError> {
        let mut levels = Vec::with_capacity(max_level + 1);
        levels.push(image.cast::<f32>()?);
        for level in 1..=max_level {
            let src = &levels[level - 1];
            let size = ImageSize {
                width: src.width().div_ceil(2),
                height: src.height().div_ceil(2),
            };
            let mut down = Image::from_size_val(size, 0.0f32)?;
            pyrdown_f32(src, &mut down)?;
            levels.push(down);
        }
        Ok(Self {
            image: image.clone(),
            levels,
        })
    }

    /// Splits into `(levels, grad_x, grad_y)`. The gradients are computed
    /// only here: an image needs them only once it is the previous frame.
    fn into_parts(self) -> Result<PyramidParts, PyrLKError> {
        let mut grad_x = Vec::with_capacity(self.levels.len());
        let mut grad_y = Vec::with_capacity(self.levels.len());
        for level in &self.levels {
            let mut ix = Image::from_size_val(level.size(), 0.0f32)?;
            let mut iy = Image::from_size_val(level.size(), 0.0f32)?;
            scharr_spatial_gradient_float(level, &mut ix, &mut iy)?;
            grad_x.push(ix);
            grad_y.push(iy);
        }
        Ok((self.levels, grad_x, grad_y))
    }
}

/// The last tracked frame's LK pyramid, kept so the next frame does not
/// rebuild it as its previous image.
#[derive(Default)]
pub struct KltPyramidCache {
    entry: Option<LevelPyramid>,
    u8_entry: Option<(Image<u8, 1>, LkPyramidU8)>,
    u8_spare: Option<LkPyramidU8>,
}

impl std::fmt::Debug for KltPyramidCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KltPyramidCache")
            .field("cached", &self.entry.is_some())
            .finish()
    }
}

impl KltPyramidCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Takes the cached pyramid if it was built from exactly `image`'s
    /// pixels; a byte compare is cheap next to a pyramid and makes a stale
    /// cache impossible to use.
    fn take_for(&mut self, image: &Image<u8, 1>) -> Option<LevelPyramid> {
        let entry = self.entry.take()?;
        (entry.image.size() == image.size() && entry.image.as_slice() == image.as_slice())
            .then_some(entry)
    }

    /// Like [`Self::take_for`] for the fixed-point kernel's pyramid; a stale
    /// one is kept as the spare to rebuild into.
    fn take_u8_for(&mut self, image: &Image<u8, 1>) -> Option<LkPyramidU8> {
        let (source, pyramid) = self.u8_entry.take()?;
        if source.size() == image.size() && source.as_slice() == image.as_slice() {
            Some(pyramid)
        } else {
            self.u8_spare = Some(pyramid);
            None
        }
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

/// Advances the track set onto this frame's optical-flow survivors, clearing it
/// when flow failed or the survivors cannot be applied. A cleared set simply
/// re-seeds from the next successful frame.
pub(crate) fn carry_klt_survivors(track_set: &mut TrackSet, survivors: Option<Vec<FlowSurvivor>>) {
    if survivors.is_none_or(|survivors| track_set.advance(survivors).is_err()) {
        *track_set = TrackSet::new();
    }
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
    fn cached_pyramid_tracks_bit_identically() {
        let frames = [
            square_image([90, 90]),
            square_image([93, 92]),
            square_image([95, 95]),
            square_image([96, 97]),
        ];
        let tracker = KltTracker::default();
        let mut cache = KltPyramidCache::new();
        let mut timings = TrackingTimings::default();
        let mut tracks = TrackSet::new();
        tracks.spawn([90.0, 90.0]);
        tracks.spawn([129.0, 90.0]);

        // Frames 1->2 hit the cache; 0->2 after it misses and rebuilds.
        for (prev, next) in [(0, 1), (1, 2), (0, 2), (2, 3)] {
            let plain = tracker
                .track(tracks.tracks(), &frames[prev], &frames[next])
                .unwrap();
            let cached = tracker
                .track_cached(
                    &mut cache,
                    tracks.tracks(),
                    None,
                    &frames[prev],
                    &frames[next],
                    &mut timings,
                )
                .unwrap();
            assert_eq!(plain.len(), cached.len());
            for (a, b) in plain.iter().zip(&cached) {
                assert_eq!(a.track_id, b.track_id);
                assert_eq!(a.pixel[0].to_bits(), b.pixel[0].to_bits());
                assert_eq!(a.pixel[1].to_bits(), b.pixel[1].to_bits());
                assert_eq!(a.error.to_bits(), b.error.to_bits());
            }
        }
    }

    #[test]
    fn predicted_flow_lands_on_the_motion_and_keeps_the_cache_whole() {
        let frames = [
            square_image([90, 90]),
            square_image([93, 92]),
            square_image([95, 95]),
        ];
        let tracker = KltTracker::default();
        let mut cache = KltPyramidCache::new();
        let mut timings = TrackingTimings::default();
        let mut tracks = TrackSet::new();
        tracks.spawn([90.0, 90.0]);
        tracks.spawn([129.0, 90.0]);

        let initial = [[93.0, 92.0], [132.0, 92.0]];
        let prediction = FlowPrediction {
            initial_pixels: &initial,
            max_level: 1,
            max_iter: 10,
        };
        let predicted = tracker
            .track_cached(
                &mut cache,
                tracks.tracks(),
                Some(prediction),
                &frames[0],
                &frames[1],
                &mut timings,
            )
            .unwrap();
        let plain = tracker
            .track(tracks.tracks(), &frames[0], &frames[1])
            .unwrap();
        assert_eq!(predicted.len(), plain.len());
        for (a, b) in predicted.iter().zip(&plain) {
            assert!((a.pixel[0] - b.pixel[0]).abs() < 0.05);
            assert!((a.pixel[1] - b.pixel[1]).abs() < 0.05);
        }

        // The cached pyramid of frame 1 still has every level: a full
        // search from it matches an uncached one bit for bit.
        let cached = tracker
            .track_cached(
                &mut cache,
                tracks.tracks(),
                None,
                &frames[1],
                &frames[2],
                &mut timings,
            )
            .unwrap();
        let uncached = tracker
            .track(tracks.tracks(), &frames[1], &frames[2])
            .unwrap();
        assert_eq!(cached, uncached);
    }

    #[test]
    fn fixed_point_kernel_agrees_with_f32_and_caches_exactly() {
        let frames = [
            square_image([90, 90]),
            square_image([93, 92]),
            square_image([95, 95]),
        ];
        let f32_tracker = KltTracker::default();
        let u8_tracker = KltTracker::default().with_kernel(LkKernel::FixedPointU8);
        let mut tracks = TrackSet::new();
        tracks.spawn([90.0, 90.0]);
        tracks.spawn([129.0, 90.0]);
        let mut timings = TrackingTimings::default();

        let mut cache = KltPyramidCache::new();
        let mut cached = Vec::new();
        for (prev, next) in [(0, 1), (1, 2)] {
            let fixed = u8_tracker
                .track_cached(
                    &mut cache,
                    tracks.tracks(),
                    None,
                    &frames[prev],
                    &frames[next],
                    &mut timings,
                )
                .unwrap();
            let float = f32_tracker
                .track(tracks.tracks(), &frames[prev], &frames[next])
                .unwrap();
            assert_eq!(fixed.len(), float.len());
            for (a, b) in fixed.iter().zip(&float) {
                assert!((a.pixel[0] - b.pixel[0]).abs() < 0.1);
                assert!((a.pixel[1] - b.pixel[1]).abs() < 0.1);
            }
            cached.push(fixed);
        }
        // Frame 1 -> 2 above reused frame 1's pyramid; a cold cache agrees.
        let cold = u8_tracker
            .track_cached(
                &mut KltPyramidCache::new(),
                tracks.tracks(),
                None,
                &frames[1],
                &frames[2],
                &mut timings,
            )
            .unwrap();
        assert_eq!(cold, cached[1]);
    }

    #[test]
    fn timed_tracking_matches_untimed_bit_for_bit() {
        let previous = square_image([90, 90]);
        let current = square_image([93, 92]);
        let tracker = KltTracker::default();
        let mut tracks = TrackSet::new();
        tracks.spawn([90.0, 90.0]);
        tracks.spawn([129.0, 90.0]);

        let plain = tracker.track(tracks.tracks(), &previous, &current).unwrap();
        let mut timings = TrackingTimings::default();
        let timed = tracker
            .track_timed(tracks.tracks(), &previous, &current, &mut timings)
            .unwrap();

        assert_eq!(plain, timed);
        assert!(timings.klt_build > std::time::Duration::ZERO);
        assert!(timings.klt_track > std::time::Duration::ZERO);
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
    #[test]
    fn klt_tracks_survive_skipped_frame_and_clear_without_survivors() {
        let mut tracks = TrackSet::new();
        tracks
            .reconcile_from_matches(
                &[MapKeypointMatch {
                    map_point_idx: 42,
                    keypoint_idx: 0,
                }],
                &[[10.0, 20.0]],
            )
            .unwrap();
        let track_id = tracks.tracks()[0].id();

        carry_klt_survivors(
            &mut tracks,
            Some(vec![FlowSurvivor {
                track_id,
                pixel: [12.0, 21.0],
                error: 0.5,
            }]),
        );

        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks.tracks()[0].id(), track_id);
        assert_eq!(tracks.tracks()[0].map_point_idx(), Some(42));
        assert_eq!(tracks.tracks()[0].pixel(), [12.0, 21.0]);
        assert_eq!(tracks.tracks()[0].age(), 2);

        carry_klt_survivors(&mut tracks, None);
        assert!(tracks.is_empty());
    }
}
