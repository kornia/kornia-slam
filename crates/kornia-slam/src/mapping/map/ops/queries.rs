//! Structural queries over stored relationships.
//!
//! Covisibility is a property of the stored observation graph, so the map owns
//! it. Search policy does not live here: callers choose the minimum weight and
//! any neighbour limit.

use crate::mapping::map::Map;
use std::collections::HashMap;

impl Map {
    /// Covisibility neighbours of `kf_idx` as `(frame_idx, weight)`, where
    /// weight counts the active map points both keyframes observe.
    ///
    /// Every neighbour with a positive weight is returned, sorted by descending
    /// weight and then descending frame index for determinism. An unknown
    /// keyframe yields an empty result.
    ///
    /// Derived on demand by inverting each landmark's observation records — no
    /// cached graph state, so it stays correct across culls and fuses.
    pub fn covisible_keyframes(&self, kf_idx: usize) -> Vec<(usize, usize)> {
        let Some(kf) = self.get_keyframe(kf_idx) else {
            return Vec::new();
        };

        let mut weights: HashMap<usize, usize> = HashMap::new();
        for mp_idx in kf.map_point_by_desc_idx.iter().flatten() {
            let Some(mp) = self.map_points().get(*mp_idx) else {
                continue;
            };
            if mp.culled {
                continue;
            }
            for obs_kf in mp.observer_keyframes() {
                if obs_kf != kf_idx {
                    *weights.entry(obs_kf).or_insert(0) += 1;
                }
            }
        }

        let mut connections: Vec<(usize, usize)> = weights.into_iter().collect();
        connections.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| b.0.cmp(&a.0)));
        connections
    }

    /// [`Self::covisible_keyframes`] restricted to neighbours with at least
    /// `min_weight` shared landmarks, keeping its order.
    ///
    /// Mirrors ORB-SLAM3's `KeyFrame::UpdateConnections`: if none reach
    /// `min_weight`, the single strongest neighbour is kept so an
    /// under-connected keyframe is never orphaned. Callers apply any neighbour
    /// limit afterwards.
    pub(crate) fn covisible_keyframes_min_weight(
        &self,
        kf_idx: usize,
        min_weight: usize,
    ) -> Vec<(usize, usize)> {
        let connections = self.covisible_keyframes(kf_idx);
        let strongest = connections.first().copied();
        let mut kept: Vec<(usize, usize)> = connections
            .into_iter()
            .filter(|&(_, w)| w >= min_weight)
            .collect();
        if kept.is_empty() {
            kept.extend(strongest);
        }
        kept
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mapping::map::{Keyframe, LandmarkSeed, ObservationKey, tests::test_frame};
    use kornia_algebra::Vec3F64;

    /// KF0 shares two points with KF1 and one with KF2.
    fn covis_map() -> Map {
        let mut map = Map::new();
        for idx in 0..3 {
            map.insert_keyframe(Keyframe::from_frame(test_frame(
                idx,
                vec![[idx as u8; 32]; 3],
            )))
            .unwrap();
        }
        for (slot, observers) in [(0usize, [0usize, 1]), (1, [0, 1]), (2, [0, 2])] {
            let mut observers = observers.into_iter();
            let reference = observers.next().expect("each landmark has an observer");
            let mp = map
                .insert_landmark(LandmarkSeed {
                    position: Vec3F64::new(0.0, 0.0, 1.0),
                    color: [0; 3],
                    reference: ObservationKey {
                        keyframe_idx: reference,
                        feature_idx: slot,
                    },
                })
                .unwrap();
            for kf in observers {
                map.link_observation(kf, slot, mp).unwrap();
            }
        }
        map
    }

    #[test]
    fn raw_covisibility_returns_every_positive_weight() {
        let map = covis_map();
        assert_eq!(map.covisible_keyframes(0), vec![(1, 2), (2, 1)]);
    }

    #[test]
    fn an_unknown_keyframe_has_no_neighbours() {
        assert!(covis_map().covisible_keyframes(99).is_empty());
    }

    #[test]
    fn min_weight_drops_weaker_neighbours() {
        let map = covis_map();
        assert_eq!(
            map.covisible_keyframes_min_weight(0, 1),
            vec![(1, 2), (2, 1)]
        );
        assert_eq!(map.covisible_keyframes_min_weight(0, 2), vec![(1, 2)]);
    }

    #[test]
    fn min_weight_keeps_the_strongest_when_none_pass() {
        let map = covis_map();
        assert_eq!(map.covisible_keyframes_min_weight(0, 15), vec![(1, 2)]);
        assert!(map.covisible_keyframes_min_weight(99, 15).is_empty());
    }
}
