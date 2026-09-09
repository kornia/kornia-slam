//! Accepted changes to map contents and relationships.
//!
//! Behaviour-preserving relocation: the existing split between keyframe
//! association and point-side registration is unchanged here.

use crate::map::{
    ImuFactor, Keyframe, Map, MapPoint, ORB_N_LEVELS, ORB_SCALE_FACTOR, ObservationKey,
};
use kornia_algebra::Vec3F64;
use kornia_sensors::imu::{ImuMeasurement, PreintegratedImu};
use std::collections::HashSet;

/// A triangulated point ready for map insertion: (position, descriptor, color, prev_desc_idx, curr_desc_idx).
pub type TriangulatedPoint = (Vec3F64, [u8; 32], [u8; 3], usize, usize);

/// Result of replacing a duplicate landmark with a surviving landmark.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MapPointMergeResult {
    pub survivor: usize,
    pub replaced: usize,
    pub redirected_associations: usize,
}

impl Map {
    /// Inserts or replaces a keyframe by frame index.
    pub fn upsert_keyframe(&mut self, keyframe: Keyframe) {
        if let Some(pos) = self
            .keyframes
            .iter()
            .position(|kf| kf.frame.idx == keyframe.frame.idx)
        {
            self.keyframes[pos] = keyframe;
        } else {
            self.keyframes.push(keyframe);
        }
    }

    /// Appends a map point and returns its index.
    pub fn push_map_point(&mut self, map_point: MapPoint) -> usize {
        let idx = self.map_points.len();
        self.map_points.push(map_point);
        idx
    }

    /// Records preintegrated IMU measurements between two consecutive keyframes.
    /// `raw_samples` (covering `[t0, t1]`) are retained for repropagation —
    /// see `ImuFactor::raw_samples`.
    pub fn add_imu_factor(
        &mut self,
        prev_kf_idx: usize,
        curr_kf_idx: usize,
        preintegrated: PreintegratedImu,
        raw_samples: Vec<ImuMeasurement>,
        t0: f64,
        t1: f64,
    ) {
        self.imu_factors.push(ImuFactor {
            prev_kf_idx,
            curr_kf_idx,
            preintegrated,
            raw_samples,
            t0,
            t1,
        });
    }

    /// Wipes all keyframes and map points. Used to discard a failed bootstrap.
    pub fn clear_active(&mut self) {
        self.world_epoch = self.world_epoch.wrapping_add(1);
        self.keyframes.clear();
        self.map_points.clear();
        self.imu_factors.clear();
    }

    /// Inserts triangulated 3D points as map points and associates them to
    /// keyframes.
    ///
    /// `curr_kf` becomes the reference keyframe for each new point's scale
    /// geometry (matches ORB-SLAM3, which creates points referenced to the
    /// newer keyframe). If `prev_kf` is provided, its observation is also
    /// recorded. Mean viewing direction and scale-invariance bounds are
    /// computed for every new point.
    pub fn add_triangulated_points(
        &mut self,
        prev_kf: Option<&mut Keyframe>,
        curr_kf: &mut Keyframe,
        points: &[TriangulatedPoint],
    ) -> usize {
        let first_mp_idx = self.map_points.len();
        let curr_kf_idx = curr_kf.frame.idx;
        for &(position, descriptor, color, _, curr_desc_idx) in points.iter() {
            let octave = curr_kf
                .frame
                .features
                .octaves
                .get(curr_desc_idx)
                .copied()
                .unwrap_or(0);
            let desc = curr_kf
                .frame
                .features
                .descriptors
                .get(curr_desc_idx)
                .copied()
                .unwrap_or(descriptor);
            let mp_idx =
                self.push_map_point(MapPoint::new(position, desc, octave, color, curr_kf_idx));
            self.map_points[mp_idx].add_observation(
                ObservationKey {
                    keyframe_idx: curr_kf_idx,
                    feature_idx: curr_desc_idx,
                },
                desc,
            );
            curr_kf.associate_map_point(curr_desc_idx, mp_idx);
        }
        if let Some(prev) = prev_kf {
            let prev_kf_idx = prev.frame.idx;
            for (i, &(_, _, _, prev_desc_idx, _)) in points.iter().enumerate() {
                prev.associate_map_point(prev_desc_idx, first_mp_idx + i);
                // Feed the prev KF's descriptor as a second observation so the
                // representative descriptor is computed from both viewpoints.
                if let Some(&prev_desc) = prev.frame.features.descriptors.get(prev_desc_idx)
                    && let Some(mp) = self.map_points.get_mut(first_mp_idx + i)
                {
                    mp.add_observation(
                        ObservationKey {
                            keyframe_idx: prev_kf_idx,
                            feature_idx: prev_desc_idx,
                        },
                        prev_desc,
                    );
                }
            }
        }
        for i in 0..points.len() {
            self.update_map_point_geometry(first_mp_idx + i, ORB_SCALE_FACTOR, ORB_N_LEVELS);
        }
        points.len()
    }

    /// Records that an existing map point was observed at `desc_idx` in
    /// `keyframe`, pushing the descriptor into the map point's observation
    /// list, refreshing the representative descriptor, and recomputing the
    /// scale geometry.
    pub fn register_observation(&mut self, mp_idx: usize, keyframe: &Keyframe, desc_idx: usize) {
        let Some(&descriptor) = keyframe.frame.features.descriptors.get(desc_idx) else {
            return;
        };
        let kf_idx = keyframe.frame.idx;
        if let Some(mp) = self.map_points.get_mut(mp_idx) {
            mp.add_observation(
                ObservationKey {
                    keyframe_idx: kf_idx,
                    feature_idx: desc_idx,
                },
                descriptor,
            );
        }
        self.update_map_point_geometry(mp_idx, ORB_SCALE_FACTOR, ORB_N_LEVELS);
    }

    /// [`Map::register_observation`] for a keyframe already stored in the
    /// map, addressed by frame index. Lets callers register observations
    /// while only holding `&mut Map` (no borrowed `Keyframe` clone needed).
    pub fn register_observation_at(&mut self, mp_idx: usize, kf_idx: usize, desc_idx: usize) {
        let Some(descriptor) = self
            .get_keyframe(kf_idx)
            .and_then(|kf| kf.frame.features.descriptors.get(desc_idx))
            .copied()
        else {
            return;
        };
        if let Some(mp) = self.map_points.get_mut(mp_idx) {
            mp.add_observation(
                ObservationKey {
                    keyframe_idx: kf_idx,
                    feature_idx: desc_idx,
                },
                descriptor,
            );
        }
        self.update_map_point_geometry(mp_idx, ORB_SCALE_FACTOR, ORB_N_LEVELS);
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
        if let Some(mp) = self.map_points.get_mut(landmark_idx) {
            mp.remove_observation(keyframe_idx);
            if mp.observations().is_empty() {
                self.retire_landmark(landmark_idx);
            } else {
                self.adopt_reference(landmark_idx);
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
                .map(|kf| kf.frame.features.descriptors.len())
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
        let mut claimed_features: HashSet<ObservationKey> = HashSet::new();
        let mut claimed_pairs: HashSet<(usize, usize)> = HashSet::new();
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
                    LinkVerdict::AlreadyLinked => continue,
                    LinkVerdict::New => {}
                }
            }
            if !claimed_features.insert(key) {
                return Err(MapMutationError::FeatureOccupied {
                    keyframe_idx: key.keyframe_idx,
                    feature_idx: key.feature_idx,
                    holder: landmark,
                });
            }
            if !claimed_pairs.insert((landmark, key.keyframe_idx)) {
                return Err(MapMutationError::DuplicateObservation {
                    landmark,
                    keyframe_idx: key.keyframe_idx,
                });
            }
        }

        // 4. IMU endpoints against the resulting keyframe set.
        for factor in &imu_factors {
            let (prev, curr) = (factor.prev_kf_idx, factor.curr_kf_idx);
            if prev == curr {
                return Err(MapMutationError::SelfImuFactor(prev));
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

        let mut observations_added = 0usize;
        let mut dirty: HashSet<usize> = HashSet::new();
        for (key, landmark) in claims {
            let Ok((descriptor, _)) = self.feature_data(key) else {
                continue;
            };
            if self.check_link(key, landmark) == Ok(LinkVerdict::AlreadyLinked) {
                continue;
            }
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
        let Some(&descriptor) = kf.frame.features.descriptors.get(key.feature_idx) else {
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

    fn refresh_landmark(&mut self, landmark_idx: usize) {
        self.update_map_point_geometry(landmark_idx, ORB_SCALE_FACTOR, ORB_N_LEVELS);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LinkVerdict {
    New,
    AlreadyLinked,
}

#[cfg(test)]
mod tests {
    use super::{
        ImuFactor, LandmarkSeed, LandmarkTarget, MapInsertion, MapMutationError, ObservationLink,
    };
    use crate::map::{
        Keyframe, Map, MapPoint, ORB_N_LEVELS, ORB_SCALE_FACTOR, ObservationKey,
        tests::{test_frame, test_frame_with_pose},
    };
    use kornia_3d::pose::Pose3d;
    use kornia_algebra::Vec3F64;
    use kornia_sensors::imu::PreintegratedImu;
    use std::collections::HashSet;

    #[test]
    fn upsert_keyframe_replaces_existing_idx() {
        let mut map = Map::new();

        map.upsert_keyframe(Keyframe::from_frame(test_frame(
            10,
            vec![[0u8; 32], [1u8; 32]],
        )));
        assert_eq!(map.keyframes().len(), 1);

        map.upsert_keyframe(Keyframe::from_frame(test_frame(10, vec![[2u8; 32]])));

        assert_eq!(map.keyframes().len(), 1);
        assert_eq!(
            map.get_keyframe(10)
                .expect("expected keyframe with idx 10")
                .frame
                .features
                .descriptors
                .len(),
            1
        );
    }

    #[test]
    fn push_map_point_returns_sequential_index() {
        let mut map = Map::new();

        let first_idx = map.push_map_point(MapPoint::new(
            Vec3F64::new(0.0, 0.0, 1.0),
            [0u8; 32],
            0,
            [0; 3],
            0,
        ));
        let second_idx = map.push_map_point(MapPoint::new(
            Vec3F64::new(1.0, 0.0, 1.0),
            [1u8; 32],
            0,
            [0; 3],
            0,
        ));

        assert_eq!(first_idx, 0);
        assert_eq!(second_idx, 1);
        assert_eq!(map.num_map_points(), 2);
    }

    #[test]
    fn merge_map_points_redirects_associations_and_deduplicates_observers() {
        let mut map = Map::new();
        for idx in 0..3 {
            map.upsert_keyframe(Keyframe::from_frame(test_frame(
                idx,
                vec![[idx as u8; 32], [10 + idx as u8; 32]],
            )));
        }

        let survivor = map.push_map_point(MapPoint::new(
            Vec3F64::new(0.0, 0.0, 5.0),
            [0; 32],
            0,
            [0; 3],
            0,
        ));
        map.map_points_mut()[survivor].add_observation(
            ObservationKey {
                keyframe_idx: 0,
                feature_idx: 0,
            },
            [0; 32],
        );
        map.map_points_mut()[survivor].add_observation(
            ObservationKey {
                keyframe_idx: 1,
                feature_idx: 0,
            },
            [1; 32],
        );
        map.map_points_mut()[survivor].n_visible = 7;
        map.map_points_mut()[survivor].n_found = 5;
        map.get_keyframe_mut(0)
            .unwrap()
            .associate_map_point(0, survivor);
        map.get_keyframe_mut(1)
            .unwrap()
            .associate_map_point(0, survivor);

        let replaced = map.push_map_point(MapPoint::new(
            Vec3F64::new(0.01, 0.0, 5.0),
            [2; 32],
            0,
            [0; 3],
            2,
        ));
        map.map_points_mut()[replaced].add_observation(
            ObservationKey {
                keyframe_idx: 2,
                feature_idx: 0,
            },
            [2; 32],
        );
        map.map_points_mut()[replaced].add_observation(
            ObservationKey {
                keyframe_idx: 1,
                feature_idx: 1,
            },
            [11; 32],
        );
        map.map_points_mut()[replaced].n_visible = 4;
        map.map_points_mut()[replaced].n_found = 3;
        map.get_keyframe_mut(2)
            .unwrap()
            .associate_map_point(0, replaced);
        map.get_keyframe_mut(1)
            .unwrap()
            .associate_map_point(1, replaced);

        let result = map.merge_map_points(survivor, replaced).unwrap();

        assert_eq!(result.survivor, survivor);
        assert_eq!(result.replaced, replaced);
        assert_eq!(result.redirected_associations, 1);
        assert!(map.map_points()[replaced].culled);
        assert_eq!(map.map_points()[survivor].n_visible, 11);
        assert_eq!(map.map_points()[survivor].n_found, 8);
        assert_eq!(
            map.map_points()[survivor]
                .observer_keyframes()
                .collect::<HashSet<_>>(),
            HashSet::from([0, 1, 2])
        );
        assert_eq!(map.get_keyframe(2).unwrap().map_point(0), Some(survivor));
        assert_eq!(map.get_keyframe(1).unwrap().map_point(0), Some(survivor));
        assert_eq!(map.get_keyframe(1).unwrap().map_point(1), None);
        for keyframe in map.keyframes() {
            assert_eq!(
                keyframe
                    .map_point_by_desc_idx
                    .iter()
                    .filter(|&&point| point == Some(survivor))
                    .count(),
                1
            );
        }
    }

    #[test]
    fn merge_map_points_rejects_invalid_or_culled_inputs() {
        let mut map = Map::new();
        map.upsert_keyframe(Keyframe::from_frame(test_frame(0, vec![[0; 32]])));
        let first = map.push_map_point(MapPoint::new(
            Vec3F64::new(0.0, 0.0, 5.0),
            [0; 32],
            0,
            [0; 3],
            0,
        ));
        let second = map.push_map_point(MapPoint::new(
            Vec3F64::new(0.0, 0.0, 5.0),
            [1; 32],
            0,
            [0; 3],
            0,
        ));

        assert!(map.merge_map_points(first, first).is_none());
        assert!(map.merge_map_points(first, usize::MAX).is_none());
        map.map_points_mut()[second].mark_culled();
        assert!(map.merge_map_points(first, second).is_none());
    }

    // ── Scale-invariance state (T1: deterministic, cross-checked vs ORB-SLAM3
    //    formulas; ORB-SLAM3 itself not needed since these are closed-form). ──

    /// T1b: after `update_map_point_geometry`,
    ///   max_distance == dist_to_ref_kf * scaleFactor^reference_octave
    ///   max_distance / min_distance == scaleFactor^(n_levels - 1)
    #[test]
    fn scale_geometry_distance_invariants() {
        let mut map = Map::new();
        // Reference keyframe 0 at the world origin (identity pose => camera
        // center at origin).
        map.upsert_keyframe(Keyframe::from_frame(test_frame(0, vec![[0u8; 32]])));

        // Point referenced to KF 0, keypoint detected at octave 2, world (0,0,5).
        let mp_idx = map.push_map_point(MapPoint::new(
            Vec3F64::new(0.0, 0.0, 5.0),
            [0u8; 32],
            2,
            [0; 3],
            0,
        ));
        map.update_map_point_geometry(mp_idx, ORB_SCALE_FACTOR, ORB_N_LEVELS);

        let mp = &map.map_points()[mp_idx];
        let expected_max = 5.0 * ORB_SCALE_FACTOR.powi(2);
        assert!((mp.max_distance - expected_max).abs() < 1e-9);
        assert!(
            (mp.max_distance / mp.min_distance - ORB_SCALE_FACTOR.powi(ORB_N_LEVELS as i32 - 1))
                .abs()
                < 1e-9
        );
        // Margined bounds carry the 0.8 / 1.2 factors.
        assert!((mp.min_distance_invariance() - 0.8 * mp.min_distance).abs() < 1e-12);
        assert!((mp.max_distance_invariance() - 1.2 * mp.max_distance).abs() < 1e-12);
    }

    /// T1c: mean viewing direction is the average of unit (point - cam_center)
    /// over observing keyframes; ‖normal‖ <= 1, and for a single forward-facing
    /// observation it's exactly (point - cam_center) normalized.
    #[test]
    fn mean_viewing_direction_averages_observations() {
        let mut map = Map::new();
        map.upsert_keyframe(Keyframe::from_frame(test_frame(0, vec![[0u8; 32]])));

        let mp_idx = map.push_map_point(MapPoint::new(
            Vec3F64::new(0.0, 0.0, 5.0),
            [0u8; 32],
            0,
            [0; 3],
            0,
        ));
        // The reference observation is explicit now: `MapPoint::new` no longer
        // fabricates an observer from its reference keyframe id alone.
        map.map_points_mut()[mp_idx].add_observation(
            ObservationKey {
                keyframe_idx: 0,
                feature_idx: 0,
            },
            [0u8; 32],
        );
        map.update_map_point_geometry(mp_idx, ORB_SCALE_FACTOR, ORB_N_LEVELS);

        // Single observation from the origin looking at (0,0,5): normal = +z.
        let n0 = map.map_points()[mp_idx].mean_viewing_direction;
        assert!(n0.x.abs() < 1e-9 && n0.y.abs() < 1e-9 && (n0.z - 1.0).abs() < 1e-9);

        // Add a second keyframe translated along +x by 1 (world->cam pose has
        // translation -1 along x, so camera center is at (1,0,0)).
        let kf1 = Keyframe::from_frame(test_frame_with_pose(
            1,
            vec![[0u8; 32]],
            Pose3d::new(
                kornia_algebra::Mat3F64::IDENTITY,
                Vec3F64::new(-1.0, 0.0, 0.0),
            ),
        ));
        map.register_observation(mp_idx, &kf1, 0);
        map.upsert_keyframe(kf1);
        map.update_map_point_geometry(mp_idx, ORB_SCALE_FACTOR, ORB_N_LEVELS);

        let n = map.map_points()[mp_idx].mean_viewing_direction;
        // Average of (0,0,1) and (point-(1,0,0)) normalized = (-1,0,5)/sqrt(26).
        let d2 = Vec3F64::new(-1.0, 0.0, 5.0);
        let d2n = d2 / d2.length();
        let expected = Vec3F64::new(
            (n0.x + d2n.x) / 2.0,
            (n0.y + d2n.y) / 2.0,
            (n0.z + d2n.z) / 2.0,
        );
        assert!((n.x - expected.x).abs() < 1e-9);
        assert!((n.y - expected.y).abs() < 1e-9);
        assert!((n.z - expected.z).abs() < 1e-9);
        assert!(n.length() <= 1.0 + 1e-12);
    }

    // ── canonical mutation API ───────────────────────────────────────────

    fn detached(idx: usize, n: usize) -> Keyframe {
        Keyframe::from_frame(test_frame(idx, vec![[idx as u8; 32]; n]))
    }

    fn seed(kf: usize, feature: usize, z: f64) -> LandmarkSeed {
        LandmarkSeed {
            position: Vec3F64::new(0.0, 0.0, z),
            color: [0; 3],
            reference: ObservationKey {
                keyframe_idx: kf,
                feature_idx: feature,
            },
        }
    }

    /// Every link is mirrored on both sides, and no feature or landmark is
    /// claimed twice within a keyframe.
    fn assert_map_consistent(map: &Map) {
        for kf in map.keyframes() {
            let mut seen: HashSet<usize> = HashSet::new();
            for (feature, slot) in kf.map_point_by_desc_idx.iter().enumerate() {
                let Some(mp_idx) = *slot else { continue };
                let mp = &map.map_points()[mp_idx];
                assert!(!mp.culled, "kf {} links retired {mp_idx}", kf.frame.idx);
                assert!(
                    mp.observations().iter().any(|o| o.key.keyframe_idx == kf.frame.idx
                        && o.key.feature_idx == feature),
                    "kf {} feature {feature} -> {mp_idx} has no matching record",
                    kf.frame.idx
                );
                assert!(
                    seen.insert(mp_idx),
                    "landmark {mp_idx} claimed twice in one keyframe"
                );
            }
        }
        for (mp_idx, mp) in map.map_points().iter().enumerate() {
            if mp.culled {
                assert!(
                    mp.observations().is_empty(),
                    "retired {mp_idx} kept records"
                );
                continue;
            }
            let mut kfs: HashSet<usize> = HashSet::new();
            for obs in mp.observations() {
                assert!(kfs.insert(obs.key.keyframe_idx), "duplicate observer");
                let kf = map
                    .get_keyframe(obs.key.keyframe_idx)
                    .expect("observer exists");
                assert_eq!(kf.map_point(obs.key.feature_idx), Some(mp_idx));
            }
        }
    }

    #[test]
    fn insert_keyframe_rejects_a_duplicate_id() {
        let mut map = Map::new();
        assert_eq!(map.insert_keyframe(detached(7, 2)).unwrap(), 7);
        assert_eq!(
            map.insert_keyframe(detached(7, 2)).unwrap_err(),
            MapMutationError::DuplicateKeyframe(7)
        );
        assert_eq!(map.keyframes().len(), 1);
    }

    #[test]
    fn insert_keyframe_requires_empty_associations() {
        let mut map = Map::new();
        let mut kf = detached(1, 2);
        kf.associate_map_point(0, 3);
        assert_eq!(
            map.insert_keyframe(kf).unwrap_err(),
            MapMutationError::MalformedKeyframe(1)
        );
    }

    #[test]
    fn landmark_ids_are_append_only_and_carry_a_reference_link() {
        let mut map = Map::new();
        map.insert_keyframe(detached(0, 3)).unwrap();
        let a = map.insert_landmark(seed(0, 0, 5.0)).unwrap();
        let b = map.insert_landmark(seed(0, 1, 6.0)).unwrap();
        assert_eq!((a, b), (0, 1));
        assert_eq!(map.map_points()[a].observations().len(), 1);
        assert_eq!(map.get_keyframe(0).unwrap().map_point(0), Some(a));

        map.remove_landmark(a).unwrap();
        let c = map.insert_landmark(seed(0, 2, 7.0)).unwrap();
        assert_eq!(c, 2, "a retired slot is never reused");
        assert_map_consistent(&map);
    }

    #[test]
    fn duplicate_link_is_a_noop() {
        let mut map = Map::new();
        map.insert_keyframe(detached(10, 1)).unwrap();
        let point = map.insert_landmark(seed(10, 0, 5.0)).unwrap();
        assert!(!map.link_observation(10, 0, point).unwrap());
        assert_eq!(map.get_keyframe(10).unwrap().map_point(0), Some(point));
        assert_eq!(map.map_points()[point].observations().len(), 1);
        assert_map_consistent(&map);
    }

    #[test]
    fn conflicting_links_are_refused() {
        let mut map = Map::new();
        map.insert_keyframe(detached(0, 3)).unwrap();
        let a = map.insert_landmark(seed(0, 0, 5.0)).unwrap();
        let b = map.insert_landmark(seed(0, 1, 6.0)).unwrap();

        // Feature 0 already holds `a`.
        assert_eq!(
            map.link_observation(0, 0, b).unwrap_err(),
            MapMutationError::FeatureOccupied {
                keyframe_idx: 0,
                feature_idx: 0,
                holder: a
            }
        );
        // `a` is already seen by keyframe 0 through another feature.
        assert_eq!(
            map.link_observation(0, 2, a).unwrap_err(),
            MapMutationError::DuplicateObservation {
                landmark: a,
                keyframe_idx: 0
            }
        );
        assert_map_consistent(&map);
    }

    #[test]
    fn unknown_and_retired_inputs_are_errors() {
        let mut map = Map::new();
        map.insert_keyframe(detached(0, 1)).unwrap();
        let a = map.insert_landmark(seed(0, 0, 5.0)).unwrap();
        map.insert_keyframe(detached(1, 1)).unwrap();

        assert_eq!(
            map.link_observation(9, 0, a).unwrap_err(),
            MapMutationError::UnknownKeyframe(9)
        );
        assert_eq!(
            map.link_observation(1, 5, a).unwrap_err(),
            MapMutationError::InvalidFeature {
                keyframe_idx: 1,
                feature_idx: 5
            }
        );
        assert_eq!(
            map.link_observation(1, 0, 99).unwrap_err(),
            MapMutationError::UnknownLandmark(99)
        );
        map.remove_landmark(a).unwrap();
        assert_eq!(
            map.link_observation(1, 0, a).unwrap_err(),
            MapMutationError::RetiredLandmark(a)
        );
    }

    #[test]
    fn a_two_keyframe_batch_publishes_atomically() {
        let mut map = Map::new();
        let result = map
            .apply_insertion(MapInsertion {
                keyframes: vec![detached(0, 2), detached(1, 2)],
                landmarks: vec![seed(0, 0, 5.0), seed(0, 1, 6.0)],
                observations: vec![
                    ObservationLink {
                        observation: ObservationKey {
                            keyframe_idx: 1,
                            feature_idx: 0,
                        },
                        landmark: LandmarkTarget::New(0),
                    },
                    ObservationLink {
                        observation: ObservationKey {
                            keyframe_idx: 1,
                            feature_idx: 1,
                        },
                        landmark: LandmarkTarget::New(1),
                    },
                ],
                imu_factors: Vec::new(),
            })
            .expect("valid batch");

        assert_eq!(result.keyframe_ids, vec![0, 1]);
        assert_eq!(result.landmark_ids, vec![0, 1]);
        assert_eq!(result.observations_added, 4);
        for mp in map.map_points() {
            assert_eq!(mp.observations().len(), 2);
            // Geometry is valid on return, not at some later refresh.
            assert!(mp.max_distance > 0.0);
        }
        assert_map_consistent(&map);
    }

    #[test]
    fn an_invalid_claim_at_the_end_leaves_the_map_untouched() {
        let mut map = Map::new();
        map.insert_keyframe(detached(0, 2)).unwrap();
        let existing = map.insert_landmark(seed(0, 0, 5.0)).unwrap();
        let before_points = map.num_map_points();
        let before_kfs = map.keyframes().len();

        let err = map
            .apply_insertion(MapInsertion {
                keyframes: vec![detached(1, 2)],
                landmarks: vec![seed(1, 0, 7.0)],
                observations: vec![ObservationLink {
                    observation: ObservationKey {
                        keyframe_idx: 0,
                        feature_idx: 0,
                    },
                    // Last claim in the request: feature 0 of keyframe 0 is
                    // already held by `existing`, so this conflicts.
                    landmark: LandmarkTarget::New(0),
                }],
                imu_factors: Vec::new(),
            })
            .unwrap_err();
        assert_eq!(
            err,
            MapMutationError::FeatureOccupied {
                keyframe_idx: 0,
                feature_idx: 0,
                holder: existing
            }
        );
        assert_eq!(map.num_map_points(), before_points);
        assert_eq!(map.keyframes().len(), before_kfs);
        assert!(map.get_keyframe(1).is_none());
        assert_map_consistent(&map);
    }

    #[test]
    fn an_out_of_range_new_target_is_refused() {
        let mut map = Map::new();
        let err = map
            .apply_insertion(MapInsertion {
                keyframes: vec![detached(0, 2)],
                landmarks: vec![seed(0, 0, 5.0)],
                observations: vec![ObservationLink {
                    observation: ObservationKey {
                        keyframe_idx: 0,
                        feature_idx: 1,
                    },
                    landmark: LandmarkTarget::New(4),
                }],
                imu_factors: Vec::new(),
            })
            .unwrap_err();
        assert_eq!(err, MapMutationError::InvalidNewLandmark(4));
        assert!(map.keyframes().is_empty());
        assert_eq!(map.num_map_points(), 0);
    }

    #[test]
    fn two_new_claims_on_one_feature_conflict() {
        let mut map = Map::new();
        let err = map
            .apply_insertion(MapInsertion {
                keyframes: vec![detached(0, 2)],
                landmarks: vec![seed(0, 0, 5.0), seed(0, 0, 6.0)],
                observations: Vec::new(),
                imu_factors: Vec::new(),
            })
            .unwrap_err();
        assert!(matches!(err, MapMutationError::FeatureOccupied { .. }));
        assert!(map.keyframes().is_empty());
    }

    #[test]
    fn unlinking_the_last_observation_retires_the_landmark() {
        let mut map = Map::new();
        map.insert_keyframe(detached(0, 2)).unwrap();
        map.insert_keyframe(detached(1, 2)).unwrap();
        let point = map.insert_landmark(seed(0, 0, 5.0)).unwrap();
        map.link_observation(1, 0, point).unwrap();

        assert_eq!(map.unlink_observation(0, 0).unwrap(), Some(point));
        assert!(!map.map_points()[point].culled);
        assert_eq!(
            map.map_points()[point].keyframe_idx,
            1,
            "reference moved to the surviving observation"
        );

        assert_eq!(map.unlink_observation(1, 0).unwrap(), Some(point));
        assert!(map.map_points()[point].culled);
        assert!(map.map_points()[point].observations().is_empty());
        assert_eq!(map.map_points()[point].max_distance, 0.0);
        assert_map_consistent(&map);
    }

    #[test]
    fn unlinking_an_empty_slot_is_a_noop() {
        let mut map = Map::new();
        map.insert_keyframe(detached(0, 2)).unwrap();
        assert_eq!(map.unlink_observation(0, 1).unwrap(), None);
        assert_eq!(
            map.unlink_observation(0, 9).unwrap_err(),
            MapMutationError::InvalidFeature {
                keyframe_idx: 0,
                feature_idx: 9
            }
        );
    }

    #[test]
    fn removing_a_landmark_clears_every_slot_and_repeats_are_noops() {
        let mut map = Map::new();
        map.insert_keyframe(detached(0, 2)).unwrap();
        map.insert_keyframe(detached(1, 2)).unwrap();
        let point = map.insert_landmark(seed(0, 0, 5.0)).unwrap();
        let other = map.insert_landmark(seed(0, 1, 6.0)).unwrap();
        map.link_observation(1, 0, point).unwrap();

        assert!(map.remove_landmark(point).unwrap());
        assert_eq!(map.get_keyframe(0).unwrap().map_point(0), None);
        assert_eq!(map.get_keyframe(1).unwrap().map_point(0), None);
        assert_eq!(map.get_keyframe(0).unwrap().map_point(1), Some(other));
        assert!(!map.remove_landmark(point).unwrap());
        assert_eq!(
            map.remove_landmark(99).unwrap_err(),
            MapMutationError::UnknownLandmark(99)
        );
        assert_map_consistent(&map);
    }

    #[test]
    fn an_invalid_imu_edge_rejects_the_batch() {
        let mut map = Map::new();
        map.insert_keyframe(detached(0, 1)).unwrap();
        let factor = |prev, curr, t0: f64, t1: f64| ImuFactor {
            prev_kf_idx: prev,
            curr_kf_idx: curr,
            preintegrated: PreintegratedImu::new(Default::default(), test_calib()),
            raw_samples: Vec::new(),
            t0,
            t1,
        };
        let mut request = MapInsertion {
            keyframes: vec![detached(1, 1)],
            imu_factors: vec![factor(0, 0, 0.0, 1.0)],
            ..Default::default()
        };
        assert_eq!(
            map.apply_insertion(request).unwrap_err(),
            MapMutationError::SelfImuFactor(0)
        );
        request = MapInsertion {
            keyframes: vec![detached(1, 1)],
            imu_factors: vec![factor(0, 9, 0.0, 1.0)],
            ..Default::default()
        };
        assert_eq!(
            map.apply_insertion(request).unwrap_err(),
            MapMutationError::UnknownKeyframe(9)
        );
        request = MapInsertion {
            keyframes: vec![detached(1, 1)],
            imu_factors: vec![factor(0, 1, 1.0, 1.0)],
            ..Default::default()
        };
        assert!(matches!(
            map.apply_insertion(request).unwrap_err(),
            MapMutationError::InvalidImuInterval { .. }
        ));
        assert!(
            map.get_keyframe(1).is_none(),
            "batch left no keyframe behind"
        );
    }

    fn test_calib() -> kornia_sensors::imu::ImuCalib {
        kornia_sensors::imu::ImuCalib {
            gyro_noise: 1.0e-4,
            accel_noise: 1.0e-3,
            gyro_bias_noise: 1.0e-5,
            accel_bias_noise: 1.0e-3,
        }
    }

    #[test]
    fn merge_redirects_disjoint_observers_and_keeps_counters() {
        let mut map = Map::new();
        for idx in 0..2 {
            map.insert_keyframe(detached(idx, 2)).unwrap();
        }
        let survivor = map.insert_landmark(seed(0, 0, 5.0)).unwrap();
        let duplicate = map.insert_landmark(seed(1, 0, 5.01)).unwrap();
        map.map_points_mut()[survivor].n_visible = 7;
        map.map_points_mut()[survivor].n_found = 5;
        map.map_points_mut()[duplicate].n_visible = 4;
        map.map_points_mut()[duplicate].n_found = 3;

        let result = map.merge_map_points(survivor, duplicate).unwrap();

        assert_eq!(result.survivor, survivor);
        assert_eq!(result.redirected_associations, 1);
        assert_eq!(map.map_points()[survivor].n_visible, 11);
        assert_eq!(map.map_points()[survivor].n_found, 8);
        assert!(map.map_points()[duplicate].culled);
        // The duplicate's observer now points at the survivor, on the same
        // feature, with a matching record.
        assert_eq!(map.get_keyframe(1).unwrap().map_point(0), Some(survivor));
        assert!(map.map_points()[survivor].is_observed_by(1));
        assert_map_consistent(&map);
    }

    /// Both landmarks seen in one keyframe through different features: the
    /// survivor keeps its own feature, the duplicate's is released.
    #[test]
    fn merge_keeps_the_survivors_feature_in_a_shared_keyframe() {
        let mut map = Map::new();
        map.insert_keyframe(detached(0, 2)).unwrap();
        let survivor = map.insert_landmark(seed(0, 0, 5.0)).unwrap();
        let duplicate = map.insert_landmark(seed(0, 1, 5.01)).unwrap();
        let survivor_descriptor = map.map_points()[survivor].descriptor;

        map.merge_map_points(survivor, duplicate).unwrap();

        assert_eq!(map.get_keyframe(0).unwrap().map_point(0), Some(survivor));
        assert_eq!(map.get_keyframe(0).unwrap().map_point(1), None);
        assert_eq!(map.map_points()[survivor].observations().len(), 1);
        assert_eq!(
            map.map_points()[survivor].observations()[0].key.feature_idx,
            0
        );
        assert_eq!(
            map.map_points()[survivor].descriptor,
            survivor_descriptor,
            "kept its retained feature's contribution"
        );
        assert_map_consistent(&map);
    }

    /// The weaker landmark is the one retired, and the reference follows the
    /// survivor's remaining observations.
    #[test]
    fn merge_picks_the_better_supported_survivor() {
        let mut map = Map::new();
        for idx in 0..3 {
            map.insert_keyframe(detached(idx, 2)).unwrap();
        }
        let weak = map.insert_landmark(seed(0, 0, 5.0)).unwrap();
        let strong = map.insert_landmark(seed(1, 0, 5.01)).unwrap();
        map.link_observation(2, 0, strong).unwrap();

        // Named weakest-first; support decides, not argument order.
        let result = map.merge_map_points(weak, strong).unwrap();

        assert_eq!(result.survivor, strong);
        assert_eq!(result.replaced, weak);
        assert!(map.map_points()[weak].culled);
        assert_eq!(map.map_points()[strong].observations().len(), 3);
        assert_map_consistent(&map);
    }
}
