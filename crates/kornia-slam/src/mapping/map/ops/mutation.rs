//! Accepted changes to map contents and relationships.
//!
//! Canonical mutations maintain both sides of observation links and finalize
//! affected landmark metadata. Batch insertion validates before its first write.

use crate::map::{
    ImuFactor, Keyframe, Map, MapPoint, ORB_N_LEVELS, ORB_SCALE_FACTOR, ObservationKey,
};
use kornia_algebra::Vec3F64;
use std::collections::{HashMap, HashSet};

/// Result of replacing a duplicate landmark with a surviving landmark.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MapPointMergeResult {
    pub survivor: usize,
    pub replaced: usize,
    pub redirected_associations: usize,
}

impl Map {
    /// Wipes all keyframes and map points. Used to discard a failed bootstrap.
    pub fn clear_active(&mut self) {
        self.world_epoch = self.world_epoch.wrapping_add(1);
        self.keyframes.clear();
        self.map_points.clear();
        self.imu_factors.clear();
    }

    /// Recomputes the mean viewing direction and scale-invariance distance
    /// bounds for one map point from its observing keyframes (ORB-SLAM3's
    /// `MapPoint::UpdateNormalAndDepth`). No-op if the point is culled or its
    /// reference keyframe is missing.
    pub fn update_map_point_geometry(&mut self, mp_idx: usize, scale_factor: f64, n_levels: usize) {
        let (position, kf_indices, ref_kf_idx, ref_octave) = {
            let Some(mp) = self.map_points.get(mp_idx) else {
                return;
            };
            if mp.culled {
                return;
            }
            (
                mp.position,
                mp.observer_keyframes().collect::<Vec<_>>(),
                mp.keyframe_idx,
                mp.reference_octave,
            )
        };

        let mut normal = Vec3F64::ZERO;
        let mut n = 0usize;
        for kf_idx in &kf_indices {
            if let Some(kf) = self.get_keyframe(*kf_idx) {
                let cam_center = kf.frame.pose_world_to_cam.inverse().translation;
                let dir = position - cam_center;
                let len = dir.length();
                if len > 1e-9 {
                    normal += dir / len;
                    n += 1;
                }
            }
        }

        let Some(ref_kf) = self.get_keyframe(ref_kf_idx) else {
            return;
        };
        let ref_center = ref_kf.frame.pose_world_to_cam.inverse().translation;
        let dist = (position - ref_center).length();
        let level_scale = scale_factor.powi(ref_octave as i32);
        let max_dist = dist * level_scale;
        let min_dist = if n_levels > 0 {
            max_dist / scale_factor.powi(n_levels as i32 - 1)
        } else {
            max_dist
        };

        if let Some(mp) = self.map_points.get_mut(mp_idx) {
            if n > 0 {
                mp.mean_viewing_direction = normal / n as f64;
            }
            mp.max_distance = max_dist;
            mp.min_distance = min_dist;
        }
    }

    /// Update `n_visible` and `n_found` counters for map points.
    pub fn update_observation_counts(
        &mut self,
        visible: &HashSet<usize>,
        matched: &[(usize, usize)],
    ) {
        let matched_set: HashSet<usize> = matched.iter().map(|&(mp_idx, _)| mp_idx).collect();

        for &mp_idx in visible {
            if let Some(mp) = self.map_points.get_mut(mp_idx) {
                mp.n_visible = mp.n_visible.saturating_add(1);
                if matched_set.contains(&mp_idx) {
                    mp.n_found = mp.n_found.saturating_add(1);
                }
            }
        }
    }

    /// Replaces a duplicate map point and redirects its keyframe associations.
    ///
    /// The point with stronger observation support survives. Associations are
    /// deduplicated so a keyframe references the survivor from at most one slot.
    pub fn merge_map_points(&mut self, first: usize, second: usize) -> Option<MapPointMergeResult> {
        if first == second {
            return None;
        }
        let first_point = self.map_points.get(first)?;
        let second_point = self.map_points.get(second)?;
        if first_point.culled || second_point.culled {
            return None;
        }

        // Records already carry at most one entry per keyframe, so support is
        // simply the number of links.
        let support = |point: &MapPoint| point.observations().len();
        let first_support = support(first_point);
        let second_support = support(second_point);
        let second_is_stronger = second_support > first_support
            || (second_support == first_support && second_point.n_found > first_point.n_found)
            || (second_support == first_support
                && second_point.n_found == first_point.n_found
                && second < first);
        let (survivor, replaced) = if second_is_stronger {
            (second, first)
        } else {
            (first, second)
        };

        // Deferred finalization: redirect every link, then retire the duplicate
        // and refresh the survivor once. Going through `unlink_observation`
        // per link would retire the duplicate the moment its last observation
        // went, mid-way through a change that is not yet complete.
        let replaced_observations: Vec<_> = self.map_points[replaced].observations().to_vec();
        let mut redirected_associations = 0;
        for observation in replaced_observations {
            let key = observation.key;
            // Clear the duplicate's feature either way.
            if let Some(kf) = self.get_keyframe_mut(key.keyframe_idx)
                && kf.map_point(key.feature_idx) == Some(replaced)
            {
                kf.clear_map_point(key.feature_idx);
            }
            if self.map_points[survivor].is_observed_by(key.keyframe_idx) {
                // The survivor already has a feature in this keyframe; keep it
                // and its descriptor contribution.
                continue;
            }
            self.link_unchecked(key, survivor, observation.descriptor);
            redirected_associations += 1;
        }

        // Any remaining slot still pointing at the duplicate (a keyframe that
        // held it without a matching record) is cleared too.
        for keyframe in &mut self.keyframes {
            for slot in 0..keyframe.map_point_by_desc_idx.len() {
                if keyframe.map_point(slot) == Some(replaced) {
                    keyframe.clear_map_point(slot);
                }
            }
        }

        let replaced_visible = self.map_points[replaced].n_visible;
        let replaced_found = self.map_points[replaced].n_found;
        self.map_points[survivor].n_visible = self.map_points[survivor]
            .n_visible
            .saturating_add(replaced_visible);
        self.map_points[survivor].n_found = self.map_points[survivor]
            .n_found
            .saturating_add(replaced_found);
        self.retire_landmark(replaced);
        self.refresh_landmark(survivor);

        Some(MapPointMergeResult {
            survivor,
            replaced,
            redirected_associations,
        })
    }
}

// ── Canonical mutation API ────────────────────────────────────────────────

/// A landmark to create, with the observation that anchors it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LandmarkSeed {
    pub position: Vec3F64,
    pub color: [u8; 3],
    pub reference: ObservationKey,
}

/// Which landmark a link points at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LandmarkTarget {
    /// A landmark already in the map.
    Existing(usize),
    /// The nth entry of this request's `landmarks`, not a map id.
    New(usize),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObservationLink {
    pub observation: ObservationKey,
    pub landmark: LandmarkTarget,
}

/// One validated batch: keyframes, landmarks, observations and IMU factors
/// that are accepted or rejected together.
#[derive(Debug, Default)]
pub struct MapInsertion {
    pub keyframes: Vec<Keyframe>,
    pub landmarks: Vec<LandmarkSeed>,
    pub observations: Vec<ObservationLink>,
    pub imu_factors: Vec<ImuFactor>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct InsertionResult {
    pub keyframe_ids: Vec<usize>,
    /// Same order as the request's `landmarks`.
    pub landmark_ids: Vec<usize>,
    /// Includes each seed's implicit reference link.
    pub observations_added: usize,
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum MapMutationError {
    #[error("keyframe {0} is already in the map")]
    DuplicateKeyframe(usize),
    #[error("keyframe {0} is not in the map")]
    UnknownKeyframe(usize),
    #[error("landmark {0} is not in the map")]
    UnknownLandmark(usize),
    #[error("landmark {0} has been retired")]
    RetiredLandmark(usize),
    #[error("keyframe {keyframe_idx} has no feature {feature_idx}")]
    InvalidFeature {
        keyframe_idx: usize,
        feature_idx: usize,
    },
    #[error("inserted keyframe {0} must arrive with empty, correctly sized associations")]
    MalformedKeyframe(usize),
    #[error("feature {feature_idx} of keyframe {keyframe_idx} already holds landmark {holder}")]
    FeatureOccupied {
        keyframe_idx: usize,
        feature_idx: usize,
        holder: usize,
    },
    #[error("landmark {landmark} is already observed by keyframe {keyframe_idx}")]
    DuplicateObservation {
        landmark: usize,
        keyframe_idx: usize,
    },
    #[error("new-landmark target {0} is out of range for this request")]
    InvalidNewLandmark(usize),
    #[error("landmark position is not finite")]
    NonFinitePosition,
    #[error("imu factor connects keyframe {0} to itself")]
    SelfImuFactor(usize),
    #[error("imu factor {prev} -> {curr} already exists")]
    DuplicateImuFactor { prev: usize, curr: usize },
    #[error("imu factor {prev} -> {curr} has invalid interval [{t0}, {t1}]")]
    InvalidImuInterval {
        prev: usize,
        curr: usize,
        t0: f64,
        t1: f64,
    },
}

impl Map {
    /// Adds a detached keyframe. Rejects a duplicate id rather than replacing a
    /// stored entity, and requires empty associations sized to its descriptors.
    pub fn insert_keyframe(&mut self, keyframe: Keyframe) -> Result<usize, MapMutationError> {
        let idx = keyframe.frame.idx;
        if self.get_keyframe(idx).is_some() {
            return Err(MapMutationError::DuplicateKeyframe(idx));
        }
        Self::check_detached(&keyframe)?;
        self.keyframes.push(keyframe);
        Ok(idx)
    }

    fn check_detached(keyframe: &Keyframe) -> Result<(), MapMutationError> {
        let n = keyframe.frame.features.descriptors.len();
        if keyframe.map_point_by_desc_idx.len() != n
            || keyframe.map_point_by_desc_idx.iter().any(|s| s.is_some())
        {
            return Err(MapMutationError::MalformedKeyframe(keyframe.frame.idx));
        }
        Ok(())
    }

    /// Appends a landmark and establishes its reference observation. Retired
    /// slots are never reused, so existing ids keep their meaning.
    pub fn insert_landmark(&mut self, seed: LandmarkSeed) -> Result<usize, MapMutationError> {
        if !seed.position.length().is_finite() {
            return Err(MapMutationError::NonFinitePosition);
        }
        let (descriptor, octave) = self.feature_data(seed.reference)?;
        self.require_free_feature(seed.reference)?;
        let idx = self.map_points.len();
        self.map_points.push(MapPoint::new(
            seed.position,
            descriptor,
            octave,
            seed.color,
            seed.reference.keyframe_idx,
        ));
        self.link_unchecked(seed.reference, idx, descriptor);
        self.refresh_landmark(idx);
        Ok(idx)
    }

    /// Links a feature to a landmark, updating both sides.
    ///
    /// Returns `true` for a new link and `false` when this exact link already
    /// exists. A feature held by another landmark, or a landmark already seen
    /// by this keyframe through a different feature, is a conflict.
    pub fn link_observation(
        &mut self,
        keyframe_idx: usize,
        feature_idx: usize,
        landmark_idx: usize,
    ) -> Result<bool, MapMutationError> {
        let key = ObservationKey {
            keyframe_idx,
            feature_idx,
        };
        let (descriptor, _) = self.feature_data(key)?;
        self.require_active_landmark(landmark_idx)?;
        match self.check_link(key, landmark_idx)? {
            LinkVerdict::AlreadyLinked => Ok(false),
            LinkVerdict::New => {
                self.link_unchecked(key, landmark_idx, descriptor);
                self.refresh_landmark(landmark_idx);
                Ok(true)
            }
        }
    }

    /// Clears a feature's link, returning the landmark it held. Retires that
    /// landmark when this was its last observation.
    pub fn unlink_observation(
        &mut self,
        keyframe_idx: usize,
        feature_idx: usize,
    ) -> Result<Option<usize>, MapMutationError> {
        let key = ObservationKey {
            keyframe_idx,
            feature_idx,
        };
        self.feature_data(key)?;
        let Some(landmark_idx) = self
            .get_keyframe(keyframe_idx)
            .and_then(|kf| kf.map_point(feature_idx))
        else {
            return Ok(None);
        };
        if let Some(kf) = self.get_keyframe_mut(keyframe_idx) {
            kf.clear_map_point(feature_idx);
        }
        // Whether this was the reference decides if the reference moves at all.
        // Unlinking any other observation must leave the reference keyframe and
        // its octave alone, even though the viewing direction and distance
        // bounds legitimately change with the remaining observers.
        let was_reference = self
            .map_points
            .get(landmark_idx)
            .is_some_and(|mp| mp.keyframe_idx == keyframe_idx);
        if let Some(mp) = self.map_points.get_mut(landmark_idx) {
            mp.remove_observation(keyframe_idx);
            if mp.observations().is_empty() {
                self.retire_landmark(landmark_idx);
            } else {
                if was_reference {
                    self.adopt_reference(landmark_idx);
                }
                self.refresh_landmark(landmark_idx);
            }
        }
        Ok(Some(landmark_idx))
    }

    /// Retires a landmark and clears every feature that referenced it.
    ///
    /// Deletion is logical: the slot stays, so all other ids keep their
    /// meaning. Removing an already-retired landmark is a no-op.
    pub fn remove_landmark(&mut self, landmark_idx: usize) -> Result<bool, MapMutationError> {
        let Some(mp) = self.map_points.get(landmark_idx) else {
            return Err(MapMutationError::UnknownLandmark(landmark_idx));
        };
        if mp.culled {
            return Ok(false);
        }
        let keys: Vec<ObservationKey> = mp.observations().iter().map(|o| o.key).collect();
        for key in keys {
            if let Some(kf) = self.get_keyframe_mut(key.keyframe_idx)
                && kf.map_point(key.feature_idx) == Some(landmark_idx)
            {
                kf.clear_map_point(key.feature_idx);
            }
        }
        self.retire_landmark(landmark_idx);
        Ok(true)
    }

    /// Applies a whole batch, or none of it.
    ///
    /// Everything is validated against both the live map and the request's own
    /// claims before the first write, so a rejected request leaves entities,
    /// links, counters, factors and the world epoch untouched. This is
    /// validation-before-write, not rollback: it does not survive a panic.
    pub fn apply_insertion(
        &mut self,
        insertion: MapInsertion,
    ) -> Result<InsertionResult, MapMutationError> {
        let MapInsertion {
            keyframes,
            landmarks,
            observations,
            imu_factors,
        } = insertion;

        // 1. keyframes, reserving their identities for later claims.
        let mut reserved: HashSet<usize> = HashSet::new();
        for kf in &keyframes {
            let idx = kf.frame.idx;
            if self.get_keyframe(idx).is_some() || !reserved.insert(idx) {
                return Err(MapMutationError::DuplicateKeyframe(idx));
            }
            Self::check_detached(kf)?;
        }
        let feature_count = |this: &Self, key: ObservationKey| -> Option<usize> {
            keyframes
                .iter()
                .find(|kf| kf.frame.idx == key.keyframe_idx)
                .or_else(|| this.get_keyframe(key.keyframe_idx))
                .map(|kf| {
                    kf.frame
                        .features
                        .descriptors
                        .len()
                        .min(kf.frame.features.keypoints_xy.len())
                })
        };

        // 2. resolve landmark targets; each seed carries an implicit first link.
        let first_new_id = self.map_points.len();
        let mut claims: Vec<(ObservationKey, usize)> = Vec::new();
        for (i, seed) in landmarks.iter().enumerate() {
            if !seed.position.length().is_finite() {
                return Err(MapMutationError::NonFinitePosition);
            }
            claims.push((seed.reference, first_new_id + i));
        }
        for link in &observations {
            let landmark = match link.landmark {
                LandmarkTarget::Existing(id) => {
                    self.require_active_landmark(id)?;
                    id
                }
                LandmarkTarget::New(i) => {
                    if i >= landmarks.len() {
                        return Err(MapMutationError::InvalidNewLandmark(i));
                    }
                    first_new_id + i
                }
            };
            claims.push((link.observation, landmark));
        }

        // 3. validate every claim against the live map and against each other.
        // Remember the resolved target for each claimed feature, and the feature
        // for each landmark/keyframe pair, so an identical repeat is a no-op
        // while a competing target is still a conflict.
        let mut claimed_features: HashMap<ObservationKey, usize> = HashMap::new();
        let mut claimed_pairs: HashMap<(usize, usize), usize> = HashMap::new();
        let mut accepted: Vec<(ObservationKey, usize)> = Vec::new();
        for &(key, landmark) in &claims {
            let Some(n_features) = feature_count(self, key) else {
                return Err(MapMutationError::UnknownKeyframe(key.keyframe_idx));
            };
            if key.feature_idx >= n_features {
                return Err(MapMutationError::InvalidFeature {
                    keyframe_idx: key.keyframe_idx,
                    feature_idx: key.feature_idx,
                });
            }
            if !reserved.contains(&key.keyframe_idx) {
                match self.check_link(key, landmark)? {
                    // Already in the live map: nothing to publish.
                    LinkVerdict::AlreadyLinked => continue,
                    LinkVerdict::New => {}
                }
            }
            match claimed_features.get(&key) {
                // The same link proposed twice in one request.
                Some(&holder) if holder == landmark => continue,
                Some(&holder) => {
                    return Err(MapMutationError::FeatureOccupied {
                        keyframe_idx: key.keyframe_idx,
                        feature_idx: key.feature_idx,
                        holder,
                    });
                }
                None => {}
            }
            if let Some(&feature) = claimed_pairs.get(&(landmark, key.keyframe_idx))
                && feature != key.feature_idx
            {
                return Err(MapMutationError::DuplicateObservation {
                    landmark,
                    keyframe_idx: key.keyframe_idx,
                });
            }
            claimed_features.insert(key, landmark);
            claimed_pairs.insert((landmark, key.keyframe_idx), key.feature_idx);
            accepted.push((key, landmark));
        }

        // 4. IMU endpoints against the resulting keyframe set.
        let mut proposed_edges: HashSet<(usize, usize)> = HashSet::new();
        for factor in &imu_factors {
            let (prev, curr) = (factor.prev_kf_idx, factor.curr_kf_idx);
            if prev == curr {
                return Err(MapMutationError::SelfImuFactor(prev));
            }
            // A duplicate inside the request would double-count integrated
            // duration in the initialization readiness gate and hand the
            // optimizer the same constraint twice.
            if !proposed_edges.insert((prev, curr)) {
                return Err(MapMutationError::DuplicateImuFactor { prev, curr });
            }
            for endpoint in [prev, curr] {
                if !reserved.contains(&endpoint) && self.get_keyframe(endpoint).is_none() {
                    return Err(MapMutationError::UnknownKeyframe(endpoint));
                }
            }
            if !(factor.t0.is_finite() && factor.t1.is_finite() && factor.t0 < factor.t1) {
                return Err(MapMutationError::InvalidImuInterval {
                    prev,
                    curr,
                    t0: factor.t0,
                    t1: factor.t1,
                });
            }
            if self
                .imu_factors
                .iter()
                .any(|f| f.prev_kf_idx == prev && f.curr_kf_idx == curr)
            {
                return Err(MapMutationError::DuplicateImuFactor { prev, curr });
            }
        }

        // ── no ordinary error exit past this point ──
        let keyframe_ids: Vec<usize> = keyframes.iter().map(|kf| kf.frame.idx).collect();
        self.keyframes.extend(keyframes);

        let mut landmark_ids = Vec::with_capacity(landmarks.len());
        for seed in &landmarks {
            let (descriptor, octave) = self.feature_data(seed.reference).expect("validated above");
            let idx = self.map_points.len();
            self.map_points.push(MapPoint::new(
                seed.position,
                descriptor,
                octave,
                seed.color,
                seed.reference.keyframe_idx,
            ));
            landmark_ids.push(idx);
        }

        // `accepted` is the deduplicated set validation approved, in request
        // order; nothing here may fail or be skipped.
        let mut observations_added = 0usize;
        let mut dirty: HashSet<usize> = HashSet::new();
        for (key, landmark) in accepted {
            let (descriptor, _) = self
                .feature_data(key)
                .expect("every accepted claim was validated above");
            self.link_unchecked(key, landmark, descriptor);
            observations_added += 1;
            dirty.insert(landmark);
        }
        for landmark in dirty {
            self.refresh_landmark(landmark);
        }
        self.imu_factors.extend(imu_factors);

        Ok(InsertionResult {
            keyframe_ids,
            landmark_ids,
            observations_added,
        })
    }

    // ── shared primitives ────────────────────────────────────────────────

    fn feature_data(&self, key: ObservationKey) -> Result<([u8; 32], u8), MapMutationError> {
        let Some(kf) = self.get_keyframe(key.keyframe_idx) else {
            return Err(MapMutationError::UnknownKeyframe(key.keyframe_idx));
        };
        // Both arrays are required: a descriptor with no keypoint carries no
        // image measurement, so no geometric consumer can use the observation.
        // Missing octave data keeps its existing fallback.
        let (Some(&descriptor), true) = (
            kf.frame.features.descriptors.get(key.feature_idx),
            key.feature_idx < kf.frame.features.keypoints_xy.len(),
        ) else {
            return Err(MapMutationError::InvalidFeature {
                keyframe_idx: key.keyframe_idx,
                feature_idx: key.feature_idx,
            });
        };
        let octave = kf
            .frame
            .features
            .octaves
            .get(key.feature_idx)
            .copied()
            .unwrap_or(0);
        Ok((descriptor, octave))
    }

    fn require_active_landmark(&self, landmark_idx: usize) -> Result<(), MapMutationError> {
        match self.map_points.get(landmark_idx) {
            None => Err(MapMutationError::UnknownLandmark(landmark_idx)),
            Some(mp) if mp.culled => Err(MapMutationError::RetiredLandmark(landmark_idx)),
            Some(_) => Ok(()),
        }
    }

    fn require_free_feature(&self, key: ObservationKey) -> Result<(), MapMutationError> {
        if let Some(holder) = self
            .get_keyframe(key.keyframe_idx)
            .and_then(|kf| kf.map_point(key.feature_idx))
        {
            return Err(MapMutationError::FeatureOccupied {
                keyframe_idx: key.keyframe_idx,
                feature_idx: key.feature_idx,
                holder,
            });
        }
        Ok(())
    }

    fn check_link(
        &self,
        key: ObservationKey,
        landmark_idx: usize,
    ) -> Result<LinkVerdict, MapMutationError> {
        let held = self
            .get_keyframe(key.keyframe_idx)
            .and_then(|kf| kf.map_point(key.feature_idx));
        match held {
            Some(existing) if existing == landmark_idx => return Ok(LinkVerdict::AlreadyLinked),
            Some(holder) => {
                return Err(MapMutationError::FeatureOccupied {
                    keyframe_idx: key.keyframe_idx,
                    feature_idx: key.feature_idx,
                    holder,
                });
            }
            None => {}
        }
        if self
            .map_points
            .get(landmark_idx)
            .is_some_and(|mp| mp.is_observed_by(key.keyframe_idx))
        {
            return Err(MapMutationError::DuplicateObservation {
                landmark: landmark_idx,
                keyframe_idx: key.keyframe_idx,
            });
        }
        Ok(LinkVerdict::New)
    }

    fn link_unchecked(&mut self, key: ObservationKey, landmark_idx: usize, descriptor: [u8; 32]) {
        if let Some(kf) = self.get_keyframe_mut(key.keyframe_idx) {
            kf.associate_map_point(key.feature_idx, landmark_idx);
        }
        if let Some(mp) = self.map_points.get_mut(landmark_idx) {
            mp.add_observation(key, descriptor);
        }
    }

    /// Smallest `(keyframe_idx, feature_idx)` becomes the reference.
    fn adopt_reference(&mut self, landmark_idx: usize) {
        let Some(mp) = self.map_points.get(landmark_idx) else {
            return;
        };
        let Some(next) = mp
            .observations()
            .iter()
            .min_by_key(|o| (o.key.keyframe_idx, o.key.feature_idx))
            .map(|o| o.key)
        else {
            return;
        };
        let octave = self.feature_data(next).map(|(_, o)| o).unwrap_or(0);
        if let Some(mp) = self.map_points.get_mut(landmark_idx) {
            mp.keyframe_idx = next.keyframe_idx;
            mp.reference_octave = octave;
        }
    }

    /// Clears derived geometry so a retired landmark keeps nothing stale.
    fn retire_landmark(&mut self, landmark_idx: usize) {
        if let Some(mp) = self.map_points.get_mut(landmark_idx) {
            mp.clear_observations();
            mp.mark_culled();
            mp.mean_viewing_direction = Vec3F64::ZERO;
            mp.min_distance = 0.0;
            mp.max_distance = 0.0;
        }
    }

    /// One finalization pass for a landmark whose records changed: the
    /// representative descriptor, then the geometry derived from its
    /// observers. Batches call this once per dirty landmark rather than per
    /// link, so the O(n^2) descriptor selection does not repeat.
    fn refresh_landmark(&mut self, landmark_idx: usize) {
        if let Some(mp) = self.map_points.get_mut(landmark_idx) {
            mp.finalize_descriptor();
        }
        self.update_map_point_geometry(landmark_idx, ORB_SCALE_FACTOR, ORB_N_LEVELS);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LinkVerdict {
    New,
    AlreadyLinked,
}

#[cfg(test)]
#[path = "mutation_tests.rs"]
mod tests;
