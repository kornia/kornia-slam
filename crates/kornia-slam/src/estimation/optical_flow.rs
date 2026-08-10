//! Persistent per-point track identity for optical-flow tracking.
//!
//! Unlike [`super::map_projection`], which re-derives every correspondence
//! from scratch each frame (project map points via the candidate pose, match
//! by descriptor), optical-flow tracking carries a point's identity and
//! map-point association *forward* from frame to frame — established once,
//! then just followed, not re-matched every time. This module holds that
//! carried state; the KLT call itself and the merge logic that produces a
//! new [`TrackState`] each frame live elsewhere (see the tracker built on
//! top of `kornia_imgproc::optical_flow_pyr_lk`).

use std::collections::HashMap;

use kornia_image::Image;
use kornia_imgproc::optical_flow_pyr_lk::{
    BorderMode, PyrLKError, PyrLKParams, TermCriteria, calc_optical_flow_pyr_lk,
};

/// One point being tracked frame-to-frame via optical flow, independent of
/// how many detection cycles it has survived.
#[derive(Debug, Clone, Copy)]
pub struct Track {
    /// Stable identity across frames, for observability only — nothing in
    /// the tracking mechanism itself keys off this; carrying state forward
    /// frame-to-frame is positional (see [`TrackState::pixels`] and the KLT
    /// call it feeds).
    pub track_id: u64,
    /// Raw (distorted) pixel position in the frame this `Track` belongs to —
    /// same convention as `Frame::features.keypoints_xy`, since KLT tracks
    /// directly on the raw image (there's no full-image undistortion step
    /// anywhere in this codebase, only point-wise). Undistort on demand
    /// wherever undistorted coordinates are needed, exactly as
    /// `Frame::undistorted_xy` already does for detected keypoints.
    pub pixel: [f32; 2],
    /// Map point this track is associated with, if any. `None` until the
    /// track survives to become part of a keyframe.
    pub map_point_idx: Option<usize>,
    /// Consecutive frames this track has survived. 1 = just detected.
    pub age: u32,
}

/// The set of points currently being tracked, owned by the pipeline and
/// carried from frame to frame.
///
/// Rebuilt wholesale each frame via [`TrackState::set_tracks`]: KLT survivors
/// keep their `Track` (carried-forward `track_id`/`map_point_idx`,
/// incremented `age`), failed tracks are dropped, and replenished detections
/// are appended as new `Track`s minted via [`TrackState::new_track`].
#[derive(Debug, Default)]
pub struct TrackState {
    next_track_id: u64,
    tracks: Vec<Track>,
}

/// Tunable parameters for KLT tracking: the library's own LK math
/// (`win_size` through `border_mode`, matching `PyrLKParams` field-for-field
/// — kept flattened here rather than nested, at the cost of an explicit
/// conversion where `track()` builds the `PyrLKParams` it actually passes to
/// `calc_optical_flow_pyr_lk`) plus the three accept/reject gates the library
/// itself doesn't provide.
#[derive(Debug)]
pub struct OpticalFlowConfig {
    /// LK integration window size in pixels. The library currently only
    /// supports exactly `21` — kept as a field (not a constant) so
    /// `calc_optical_flow_pyr_lk`'s own validation is what enforces that,
    /// not a silent assumption baked in here.
    pub win_size: usize,
    /// Pyramid depth: more levels track larger displacements, at higher cost
    /// and less precision at the coarsest level.
    pub max_level: usize,
    /// Gauss-Newton iterations per pyramid level before giving up.
    pub max_iter: usize,
    /// Convergence threshold on the incremental flow update.
    pub epsilon: f32,
    /// Minimum structure-tensor eigenvalue for a patch to be considered
    /// trackable at all (rejects flat/aliased regions).
    pub min_eigen_threshold: f32,
    /// Seed the search from a provided initial guess instead of `prev_pts`'
    /// own position. Left `false` for now — the pose-informed hybrid this
    /// would enable is a later refinement, not needed for the first working
    /// version.
    pub use_initial_flow: bool,
    /// Iteration stopping rule (count / epsilon / both).
    pub term_criteria: TermCriteria,
    /// How the LK math itself samples pixels near the image edge during
    /// iteration. Separate from `border_margin_px` below, which governs
    /// whether a result close to the edge is *trusted* afterward.
    pub border_mode: BorderMode,

    /// Reject a track if KLT's own reported error exceeds this, even when
    /// `status == 1` — `status` only means "didn't fail outright," `error`
    /// is the actual residual.
    pub max_error: f32,
    /// Reject a track whose pixel displacement from the previous frame
    /// exceeds this — catches LK converging to a plausible-looking but
    /// wrong local optimum (e.g. aliasing onto a nearby similar patch).
    pub max_movement_px: f32,
    /// Reject a track landing within this many pixels of the image edge —
    /// stricter than `border_mode`, which only affects the LK math's
    /// internal sampling, not whether the final result is trustworthy.
    pub border_margin_px: f32,
}

impl Default for OpticalFlowConfig {
    fn default() -> Self {
        Self {
            win_size: 21,
            max_level: 3,
            max_iter: 30,
            epsilon: 0.01,
            min_eigen_threshold: 1e-4,
            use_initial_flow: false,
            term_criteria: TermCriteria::Both,
            border_mode: BorderMode::Clamp,
            max_error: 20.0,
            max_movement_px: 60.0,
            border_margin_px: 10.0,
        }
    }
}

impl OpticalFlowConfig {
    /// Packs the flattened LK-math fields back into the `PyrLKParams` shape
    /// `calc_optical_flow_pyr_lk` actually takes. Pure plumbing — nothing
    /// here is a decision, just matching field-for-field.
    fn to_pyr_lk_params(&self) -> PyrLKParams {
        PyrLKParams {
            win_size: self.win_size,
            max_level: self.max_level,
            max_iter: self.max_iter,
            epsilon: self.epsilon,
            min_eigen_threshold: self.min_eigen_threshold,
            use_initial_flow: self.use_initial_flow,
            term_criteria: self.term_criteria.clone(),
            border_mode: self.border_mode,
        }
    }
}

impl TrackState {
    /// Empty track state, as at startup or right after a re-bootstrap.
    pub fn new() -> Self {
        Self {
            next_track_id: 0,
            tracks: Vec::new(),
        }
    }

    /// Current tracks, in the same order [`TrackState::pixels`] hands to KLT.
    pub fn tracks(&self) -> &[Track] {
        &self.tracks
    }

    /// Number of points currently tracked.
    pub fn len(&self) -> usize {
        self.tracks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tracks.is_empty()
    }

    /// Pixel positions of all current tracks, in a form the KLT call
    /// consumes directly as `prev_pts`. Positional order must be preserved
    /// through that call — `status`/`next_pts` come back parallel to this —
    /// so whatever merges KLT's output back into a new `TrackState` can zip
    /// them against `tracks()` by index.
    pub fn pixels(&self) -> Vec<[f32; 2]> {
        self.tracks.iter().map(|t| t.pixel).collect()
    }

    /// Mints a fresh, uniquely-identified `Track` for a newly detected point
    /// (replenishment) — not yet associated with any map point.
    pub fn new_track(&mut self, pixel: [f32; 2]) -> Track {
        let track_id = self.next_track_id;
        self.next_track_id += 1;
        Track {
            track_id,
            pixel,
            map_point_idx: None,
            age: 1,
        }
    }

    /// Replaces the track list wholesale with the next frame's full set.
    /// Callers assemble `tracks` themselves — carried-forward survivors
    /// (via [`Track`]'s fields, incremented `age`) plus any freshly-minted
    /// tracks from [`TrackState::new_track`] — this just commits it.
    pub fn set_tracks(&mut self, tracks: Vec<Track>) {
        self.tracks = tracks;
    }

    /// Advances `self` one frame via pyramidal Lucas-Kanade: tracks every
    /// point in `self` from `prev_img` into `next_img`, and returns the
    /// survivors as a fresh `Vec<Track>` — carried-forward `track_id`/
    /// `map_point_idx`, `age` incremented, `pixel` updated to wherever KLT
    /// found it.
    ///
    /// Does **not** call [`TrackState::set_tracks`] itself, and does not
    /// replenish — this only tracks what's already in `self`. The caller
    /// decides how to merge these survivors with newly-detected points
    /// before committing the next frame's state.
    ///
    /// `prev_img`/`next_img` must be the raw (distorted) grayscale images
    /// for the frames `self` and the frame being tracked into, respectively
    /// — same pixel space as `Track::pixel`.
    pub fn track(
        &self,
        prev_img: &Image<u8, 1>,
        next_img: &Image<u8, 1>,
        config: &OpticalFlowConfig,
    ) -> Result<Vec<Track>, PyrLKError> {
        if self.is_empty() {
            return Ok(Vec::new());
        }

        // u8 -> f32 is always exactly representable (every u8 value fits a
        // f32 with no precision loss), so `cast`'s only failure mode
        // (genuine numeric conversion loss) can't happen here.
        let prev_f32 = prev_img
            .cast::<f32>()
            .expect("u8 -> f32 image cast is always representable");
        let next_f32 = next_img
            .cast::<f32>()
            .expect("u8 -> f32 image cast is always representable");

        let prev_pts = self.pixels();
        let result = calc_optical_flow_pyr_lk(
            &prev_f32,
            &next_f32,
            &prev_pts,
            None,
            &config.to_pyr_lk_params(),
        )?;

        let width = next_img.width() as f32;
        let height = next_img.height() as f32;
        let margin = config.border_margin_px;

        let mut survivors = Vec::with_capacity(self.tracks.len());
        for (i, track) in self.tracks.iter().enumerate() {
            if result.status[i] == 0 {
                continue; // KLT itself failed for this point.
            }
            if result.error[i] > config.max_error {
                continue;
            }
            let next_pt = result.next_pts[i];
            if next_pt[0] < margin
                || next_pt[1] < margin
                || next_pt[0] > width - margin
                || next_pt[1] > height - margin
            {
                continue; // Too close to the edge to trust.
            }
            let dx = next_pt[0] - track.pixel[0];
            let dy = next_pt[1] - track.pixel[1];
            if (dx * dx + dy * dy).sqrt() > config.max_movement_px {
                continue; // Implausible jump — likely aliased onto the wrong patch.
            }
            survivors.push(Track {
                track_id: track.track_id,
                pixel: next_pt,
                map_point_idx: track.map_point_idx,
                age: track.age + 1,
            });
        }
        Ok(survivors)
    }

    /// Rebuilds the track list from a frame's winning correspondence set —
    /// meant to be called once per successful `estimate_pose`, regardless
    /// of whether `matches` came from KLT-seeded correspondences (see
    /// [`klt_correspondences`]) or the existing projection-search /
    /// reference-keyframe fallback. That's deliberate: it's what lets
    /// tracking *recover* rather than being a one-way ratchet that can only
    /// ever lose tracks — any map point the fallback path finds (e.g. one
    /// KLT lost, or one newly visible) becomes a fresh, trackable point
    /// again starting next frame.
    ///
    /// Track identity (`track_id`/`age`) is preserved for any map point
    /// that was already being tracked; anything new mints a fresh `Track`.
    /// `keypoints` must be the current frame's raw (distorted) keypoints,
    /// indexed the same way `matches`' second element is.
    pub fn refresh_from_matches(&mut self, matches: &[(usize, usize)], keypoints: &[[f32; 2]]) {
        let mut old_by_mp: HashMap<usize, Track> = self
            .tracks
            .drain(..)
            .filter_map(|t| t.map_point_idx.map(|mp| (mp, t)))
            .collect();

        let mut new_tracks = Vec::with_capacity(matches.len());
        for &(mp_idx, kp_idx) in matches {
            let Some(&pixel) = keypoints.get(kp_idx) else {
                continue;
            };
            let track = match old_by_mp.remove(&mp_idx) {
                Some(old) => Track {
                    track_id: old.track_id,
                    pixel,
                    map_point_idx: Some(mp_idx),
                    age: old.age + 1,
                },
                None => {
                    let mut t = self.new_track(pixel);
                    t.map_point_idx = Some(mp_idx);
                    t
                }
            };
            new_tracks.push(track);
        }
        self.tracks = new_tracks;
    }
}

/// Nearest keypoint in `keypoints` to `predicted`, within `radius_px` —
/// brute-force since per-frame track counts are modest (hundreds, not
/// thousands); a spatial grid isn't justified yet.
fn nearest_keypoint(predicted: [f32; 2], keypoints: &[[f32; 2]], radius_px: f32) -> Option<usize> {
    let r2 = radius_px * radius_px;
    let mut best: Option<(usize, f32)> = None;
    for (i, &kp) in keypoints.iter().enumerate() {
        let dx = kp[0] - predicted[0];
        let dy = kp[1] - predicted[1];
        let d2 = dx * dx + dy * dy;
        if d2 > r2 {
            continue;
        }
        match best {
            Some((_, best_d2)) if d2 >= best_d2 => {}
            _ => best = Some((i, d2)),
        }
    }
    best.map(|(i, _)| i)
}

/// Snaps each survivor's predicted pixel onto the nearest freshly detected
/// keypoint in `curr_keypoints` (raw/distorted, same convention as
/// `Track::pixel`) within `snap_radius_px`, producing
/// `(map_point_idx, current_frame_keypoint_idx)` correspondences.
///
/// Split out from [`klt_correspondences`] so a caller can hold onto
/// `survivors` itself (e.g. to commit them back via
/// [`TrackState::set_tracks`] even on a frame whose correspondences don't
/// win — see that method's doc) instead of only getting the derived
/// correspondences.
pub fn snap_survivors(
    survivors: &[Track],
    curr_keypoints: &[[f32; 2]],
    snap_radius_px: f32,
) -> Vec<(usize, usize)> {
    let mut correspondences = Vec::new();
    for t in survivors {
        if let Some(mp_idx) = t.map_point_idx
            && let Some(kp_idx) = nearest_keypoint(t.pixel, curr_keypoints, snap_radius_px)
        {
            correspondences.push((mp_idx, kp_idx));
        }
    }
    correspondences
}

/// Produces `(map_point_idx, current_frame_keypoint_idx)` correspondences by
/// KLT-tracking `track_state`'s existing points from `prev_img` into
/// `next_img`, then snapping each survivor's predicted pixel onto the
/// nearest freshly detected keypoint in `curr_keypoints` (raw/distorted,
/// same convention as `Track::pixel`) within `snap_radius_px`.
///
/// Returns only correspondences, not an updated `TrackState` — the caller
/// rebuilds that from the *winning* `Estimate`'s matches afterward via
/// [`TrackState::refresh_from_matches`], not from this function's output
/// directly (see that method's doc for why). Callers that also need the raw
/// survivors (e.g. to keep KLT continuity alive across a losing frame)
/// should call [`TrackState::track`] and [`snap_survivors`] directly instead
/// of this convenience wrapper.
pub fn klt_correspondences(
    track_state: &TrackState,
    prev_img: &Image<u8, 1>,
    next_img: &Image<u8, 1>,
    curr_keypoints: &[[f32; 2]],
    config: &OpticalFlowConfig,
    snap_radius_px: f32,
) -> Result<Vec<(usize, usize)>, PyrLKError> {
    let survivors = track_state.track(prev_img, next_img, config)?;
    Ok(snap_survivors(&survivors, curr_keypoints, snap_radius_px))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::ImageSize;

    #[test]
    fn new_track_gets_unique_ids() {
        let mut state = TrackState::new();
        let a = state.new_track([1.0, 2.0]);
        let b = state.new_track([3.0, 4.0]);
        assert_ne!(a.track_id, b.track_id);
        assert_eq!(a.map_point_idx, None);
        assert_eq!(a.age, 1);
    }

    #[test]
    fn pixels_are_positional_with_tracks() {
        let mut state = TrackState::new();
        let t0 = state.new_track([1.0, 2.0]);
        let t1 = state.new_track([3.0, 4.0]);
        state.set_tracks(vec![t0, t1]);
        assert_eq!(state.pixels(), vec![[1.0, 2.0], [3.0, 4.0]]);
    }

    /// Builds a 200x200 black image with a white square, corner at `origin`,
    /// 40px on a side — the corner itself is the useful test feature (a
    /// real gradient in both x and y), not the square's flat interior.
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

        let mut state = TrackState::new();
        // Seed at the square's top-left corner in `prev` — a real corner
        // feature (gradient in both directions), not the flat interior.
        let t = state.new_track([90.0, 90.0]);
        state.set_tracks(vec![t]);

        let survivors = state
            .track(&prev_img, &next_img, &OpticalFlowConfig::default())
            .expect("KLT should succeed on a clean synthetic shift");

        assert_eq!(survivors.len(), 1, "the corner should track successfully");
        let got = survivors[0].pixel;
        assert!(
            (got[0] - 93.0).abs() < 1.5 && (got[1] - 92.0).abs() < 1.5,
            "expected tracked pixel near (93, 92), got {got:?}"
        );
        assert_eq!(survivors[0].track_id, t.track_id);
        assert_eq!(survivors[0].age, 2);
    }

    #[test]
    fn refresh_from_matches_preserves_identity_for_known_map_points() {
        let mut state = TrackState::new();
        let mut t0 = state.new_track([1.0, 2.0]);
        t0.map_point_idx = Some(42);
        state.set_tracks(vec![t0]);

        // Map point 42 matched again this frame (at a new pixel, as if KLT
        // + snapping had moved it), and a brand-new map point 7 shows up
        // via the fallback path.
        let keypoints = [[5.0, 6.0], [10.0, 11.0]];
        state.refresh_from_matches(&[(42, 0), (7, 1)], &keypoints);

        let tracks = state.tracks();
        assert_eq!(tracks.len(), 2);
        let mp42 = tracks.iter().find(|t| t.map_point_idx == Some(42)).unwrap();
        assert_eq!(mp42.track_id, t0.track_id, "identity must carry forward");
        assert_eq!(mp42.age, 2);
        assert_eq!(mp42.pixel, [5.0, 6.0]);

        let mp7 = tracks.iter().find(|t| t.map_point_idx == Some(7)).unwrap();
        assert_ne!(mp7.track_id, t0.track_id, "new map point gets a fresh id");
        assert_eq!(mp7.age, 1);
    }
}
