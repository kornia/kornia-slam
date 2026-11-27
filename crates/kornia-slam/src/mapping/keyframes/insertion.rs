//! Keyframe insertion policy and the evidence measured by tracking.

/// Selects the motion-specific part of keyframe admission. The frame-gap and
/// weak-reference guards in [`KeyframePolicy`] always apply.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub enum KeyframeInsertionStrategy {
    /// Admit from frame spacing and reference-tracking evidence only.
    FrameBased,
    /// Also admit once rotation-compensated median parallax reaches this
    /// threshold in pixels.
    Parallax { min_parallax_px: f64 },
}

/// Keyframe insertion heuristics.
///
/// Defaults preserve the historical frame-count policy: at least every eight
/// frames, with no parallax computation. Inertial systems that remove the
/// frame bound still require a finite [`Self::max_interval_sec`] bound; that
/// safety rule is enforced by `SlamSystem::build`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct KeyframePolicy {
    /// Motion-specific admission selection.
    pub insertion: KeyframeInsertionStrategy,
    /// Minimum frame gap before allowing keyframe insertion.
    pub min_frames_between: usize,
    /// Force a keyframe if this frame gap is reached; `usize::MAX` disables
    /// the bound.
    pub max_frames_between: usize,
    /// Relative inlier ratio threshold (vs reference keyframe tracked map points).
    pub ref_ratio: f64,
    /// Force a keyframe once this many seconds have passed since the last
    /// one (outside `min_frames_between`). `SlamSystem` applies it from frame
    /// timestamps; [`Self::should_insert_with`] does not see time.
    pub max_interval_sec: Option<f64>,
}

impl Default for KeyframePolicy {
    fn default() -> Self {
        Self {
            insertion: KeyframeInsertionStrategy::FrameBased,
            min_frames_between: 3,
            max_frames_between: 8,
            ref_ratio: 0.6,
            max_interval_sec: None,
        }
    }
}

/// What the tracker measured on the current frame, for
/// [`KeyframePolicy::should_insert_with`]. `#[non_exhaustive]`: build it with
/// [`Self::new`] or from [`Default`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[non_exhaustive]
pub struct KeyframeEvidence {
    /// `Frame::idx` of the current frame.
    pub curr_idx: usize,
    /// `Frame::idx` of the last inserted keyframe, `None` before the first.
    pub last_keyframe_idx: Option<usize>,
    /// Map points the current frame tracked as inliers.
    pub tracked_inliers: usize,
    /// Map points the reference keyframe holds.
    pub n_ref_map_points: usize,
    /// Median rotation-compensated parallax of tracked features against the
    /// last keyframe, in pixels; `None` when it was not measured.
    pub median_parallax_px: Option<f64>,
}

impl KeyframeEvidence {
    /// Evidence without a parallax measurement; set it only for the
    /// [`KeyframeInsertionStrategy::Parallax`] strategy.
    pub fn new(
        curr_idx: usize,
        last_keyframe_idx: Option<usize>,
        tracked_inliers: usize,
        n_ref_map_points: usize,
    ) -> Self {
        Self {
            curr_idx,
            last_keyframe_idx,
            tracked_inliers,
            n_ref_map_points,
            median_parallax_px: None,
        }
    }

    /// Returns this evidence with its optional parallax measurement.
    pub fn with_median_parallax_px(self, median_parallax_px: Option<f64>) -> Self {
        Self {
            median_parallax_px,
            ..self
        }
    }
}

/// Fewest tracked inliers for which a low reference ratio still earns a
/// keyframe (ORB-SLAM3's `mnMatchesInliers > 15`).
const MIN_INLIERS_FOR_RATIO_INSERT: usize = 15;

impl KeyframePolicy {
    /// Returns the parallax threshold selected by this policy, if any.
    pub const fn parallax_threshold_px(&self) -> Option<f64> {
        match self.insertion {
            KeyframeInsertionStrategy::FrameBased => None,
            KeyframeInsertionStrategy::Parallax { min_parallax_px } => Some(min_parallax_px),
        }
    }

    /// Decides whether to add a keyframe without measuring parallax.
    pub fn should_insert(
        &self,
        curr_idx: usize,
        last_keyframe_idx: Option<usize>,
        tracked_inliers: usize,
        n_ref_map_points: usize,
    ) -> bool {
        self.should_insert_with(&KeyframeEvidence::new(
            curr_idx,
            last_keyframe_idx,
            tracked_inliers,
            n_ref_map_points,
        ))
    }

    /// Decides whether a new keyframe should be inserted.
    ///
    /// Order is intentionally stable: first keyframe, minimum gap, maximum
    /// gap, empty reference, weak reference, and optional parallax.
    pub fn should_insert_with(&self, evidence: &KeyframeEvidence) -> bool {
        let Some(last_kf_idx) = evidence.last_keyframe_idx else {
            return true;
        };

        let frames_since_last_kf = evidence.curr_idx.saturating_sub(last_kf_idx);
        if frames_since_last_kf < self.min_frames_between {
            return false;
        }
        if frames_since_last_kf >= self.max_frames_between {
            return true;
        }
        if evidence.n_ref_map_points == 0 {
            return true;
        }

        let weak_threshold = (evidence.n_ref_map_points as f64 * self.ref_ratio) as usize;
        if evidence.tracked_inliers >= MIN_INLIERS_FOR_RATIO_INSERT
            && evidence.tracked_inliers < weak_threshold
        {
            return true;
        }

        self.parallax_threshold_px()
            .zip(evidence.median_parallax_px)
            .is_some_and(|(threshold, parallax)| parallax >= threshold)
    }

    /// Whether `max_interval_sec` forces a keyframe `secs_since_last_kf`
    /// seconds and `frames_since_last_kf` frames after the last one.
    pub(crate) fn interval_elapsed(
        &self,
        frames_since_last_kf: usize,
        secs_since_last_kf: f64,
    ) -> bool {
        frames_since_last_kf >= self.min_frames_between
            && self
                .max_interval_sec
                .is_some_and(|max| secs_since_last_kf >= max)
    }
}

#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
struct KeyframePolicyCompat {
    insertion: InsertionField,
    min_frames_between: usize,
    max_frames_between: usize,
    ref_ratio: f64,
    min_parallax_px: LegacyParallaxField,
    max_interval_sec: Option<f64>,
}

#[cfg(feature = "serde")]
impl Default for KeyframePolicyCompat {
    fn default() -> Self {
        let policy = KeyframePolicy::default();
        Self {
            insertion: InsertionField::default(),
            min_frames_between: policy.min_frames_between,
            max_frames_between: policy.max_frames_between,
            ref_ratio: policy.ref_ratio,
            min_parallax_px: LegacyParallaxField::default(),
            max_interval_sec: policy.max_interval_sec,
        }
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for KeyframePolicy {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error as _;

        let raw = KeyframePolicyCompat::deserialize(deserializer)?;
        let insertion = match (raw.insertion, raw.min_parallax_px) {
            (InsertionField { present: true, .. }, LegacyParallaxField { present: true, .. }) => {
                return Err(D::Error::custom(
                    "keyframes.insertion and legacy keyframes.min_parallax_px are ambiguous",
                ));
            }
            (
                InsertionField {
                    value: Some(insertion),
                    ..
                },
                _,
            ) => insertion,
            (
                InsertionField { present: false, .. },
                LegacyParallaxField {
                    value: Some(min_parallax_px),
                    ..
                },
            ) => KeyframeInsertionStrategy::Parallax { min_parallax_px },
            (InsertionField { present: false, .. }, _) => KeyframeInsertionStrategy::FrameBased,
            _ => KeyframeInsertionStrategy::FrameBased,
        };

        Ok(Self {
            insertion,
            min_frames_between: raw.min_frames_between,
            max_frames_between: raw.max_frames_between,
            ref_ratio: raw.ref_ratio,
            max_interval_sec: raw.max_interval_sec,
        })
    }
}

#[cfg(feature = "serde")]
#[derive(Default)]
struct InsertionField {
    present: bool,
    value: Option<KeyframeInsertionStrategy>,
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for InsertionField {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(Self {
            present: true,
            value: Some(KeyframeInsertionStrategy::deserialize(deserializer)?),
        })
    }
}

#[cfg(feature = "serde")]
#[derive(Default)]
struct LegacyParallaxField {
    present: bool,
    value: Option<f64>,
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for LegacyParallaxField {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(Self {
            present: true,
            value: Option::<f64>::deserialize(deserializer)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence(frames_since: usize, inliers: usize, n_ref: usize) -> KeyframeEvidence {
        KeyframeEvidence {
            curr_idx: 100 + frames_since,
            last_keyframe_idx: Some(100),
            tracked_inliers: inliers,
            n_ref_map_points: n_ref,
            median_parallax_px: None,
        }
    }

    fn parallax_policy() -> KeyframePolicy {
        KeyframePolicy {
            insertion: KeyframeInsertionStrategy::Parallax {
                min_parallax_px: 10.0,
            },
            max_frames_between: usize::MAX,
            ..KeyframePolicy::default()
        }
    }

    #[test]
    fn default_is_frame_based_and_ignores_parallax() {
        let policy = KeyframePolicy::default();
        assert_eq!(policy.insertion, KeyframeInsertionStrategy::FrameBased);
        assert!(
            !policy.should_insert_with(&evidence(5, 80, 100).with_median_parallax_px(Some(1e6)))
        );
        assert!(policy.should_insert_with(&evidence(8, 80, 100)));
    }

    #[test]
    fn default_preserves_frame_and_ratio_boundaries() {
        let policy = KeyframePolicy::default();
        // The original frame-based policy admits weak tracking from the
        // minimum gap, but otherwise waits for the maximum gap. Keep the
        // equality and minimum-inlier boundaries explicit through the move.
        let cases = [
            (80, 100, 8),
            (60, 100, 8), // Exactly at ref_ratio is not weak.
            (59, 100, 3),
            (15, 30, 3), // Fewest inliers eligible for the weak-ratio rule.
            (14, 30, 8),
            (50, 0, 3),
        ];
        for frames in 0..12 {
            for (inliers, n_ref, first_insert) in cases {
                let expected = frames >= first_insert;
                let evidence = evidence(frames, inliers, n_ref);
                let case = format!("frames={frames} inliers={inliers} n_ref={n_ref}");
                assert_eq!(policy.should_insert_with(&evidence), expected, "{case}");
                assert_eq!(
                    policy.should_insert(
                        evidence.curr_idx,
                        evidence.last_keyframe_idx,
                        inliers,
                        n_ref,
                    ),
                    expected,
                    "{case}"
                );
                assert_eq!(
                    policy.should_insert_with(&evidence.with_median_parallax_px(Some(1e6))),
                    expected,
                    "{case}: FrameBased ignores parallax"
                );
            }
        }
    }

    #[test]
    fn parallax_respects_shared_guards_and_threshold() {
        let policy = parallax_policy();
        assert!(!policy.should_insert_with(&evidence(5, 80, 100)));
        assert!(
            !policy.should_insert_with(&evidence(2, 80, 100).with_median_parallax_px(Some(50.0)))
        );
        assert!(
            policy.should_insert_with(&evidence(5, 80, 100).with_median_parallax_px(Some(10.0)))
        );
        assert!(policy.should_insert_with(&evidence(5, 0, 0)));
    }

    #[test]
    fn parked_camera_waits_for_the_explicit_frame_bound() {
        let policy = parallax_policy();
        for frames in [8, 100, 10_000] {
            assert!(
                !policy.should_insert_with(
                    &evidence(frames, 80, 100).with_median_parallax_px(Some(0.3))
                )
            );
        }
        let bounded = KeyframePolicy {
            max_frames_between: 30,
            ..policy
        };
        assert!(
            !bounded.should_insert_with(&evidence(29, 80, 100).with_median_parallax_px(Some(0.3)))
        );
        assert!(
            bounded.should_insert_with(&evidence(30, 80, 100).with_median_parallax_px(Some(0.3)))
        );
    }

    #[test]
    fn weak_and_missing_references_keep_their_precedence_over_parallax() {
        let policy = parallax_policy();
        assert!(
            policy.should_insert_with(&evidence(5, 50, 100).with_median_parallax_px(Some(0.0)))
        );
        assert!(
            !policy.should_insert_with(&evidence(5, 10, 100).with_median_parallax_px(Some(0.0)))
        );
        assert!(policy.should_insert_with(&evidence(5, 0, 0).with_median_parallax_px(Some(0.0))));
        assert!(policy.should_insert_with(&KeyframeEvidence {
            last_keyframe_idx: None,
            ..evidence(0, 0, 0)
        }));
    }

    #[test]
    fn interval_retains_minimum_gap() {
        let policy = KeyframePolicy {
            max_interval_sec: Some(0.5),
            ..parallax_policy()
        };
        assert!(!policy.interval_elapsed(2, 10.0));
        assert!(policy.interval_elapsed(3, 0.5));
    }
}
