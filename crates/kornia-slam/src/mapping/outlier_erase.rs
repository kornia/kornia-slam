//! Post-solve outlier erase of the local VI-BA (gap G56).
//!
//! Tracking links every match into a new keyframe, outliers included, so that
//! bundle adjustment decides (ORB-SLAM3 `Tracking.cc:2259-2267`). This is the
//! deciding half: after an accepted merge, each window observation whose
//! residual at the solution fails the solve's own 95 % χ² test is unlinked
//! (ORB-SLAM3 `Optimizer::LocalInertialBA`, `Optimizer.cc:2857-2902`), and a
//! landmark left with a weighted view count of 2 or less is retired
//! (`MapPoint.cc:168-201`). VI-BA only.

use crate::mapping::map::{Map, ObservationKey};
use crate::mapping::vi_ba_schur::ObservationResidual;

/// Whether local VI-BA unlinks the observations its solution rejects.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum OutlierErase {
    #[default]
    Off,
    ChiSquare(ChiSquareErase),
}

/// χ² thresholds on the whitened residual. The defaults are ORB-SLAM3's.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default, deny_unknown_fields))]
pub struct ChiSquareErase {
    /// Reprojection-only row (2 dof); 95 % is 5.991.
    pub reprojection_chi2: f64,
    /// Reprojection plus stereo depth row (3 dof); 95 % is 7.815.
    pub stereo_chi2: f64,
    /// Factor on `reprojection_chi2` for a landmark nearer than `close_range_m`.
    pub close_factor: f64,
    /// Range from the observing camera, metres.
    pub close_range_m: f64,
}

impl Default for ChiSquareErase {
    fn default() -> Self {
        Self {
            reprojection_chi2: 5.991,
            stereo_chi2: 7.815,
            close_factor: 1.5,
            close_range_m: 10.0,
        }
    }
}

impl OutlierErase {
    pub fn is_enabled(&self) -> bool {
        matches!(self, Self::ChiSquare(_))
    }

    /// Whether the observation is erased. Stereo rows get no close factor and
    /// no depth-sign test, as ORB-SLAM3's (`Optimizer.cc:2879`).
    pub fn is_outlier(&self, residual: &ObservationResidual) -> bool {
        let Self::ChiSquare(rule) = self else {
            return false;
        };
        if let Some(depth_chi2) = residual.depth_chi2 {
            return residual.reprojection_chi2 + depth_chi2 > rule.stereo_chi2;
        }
        if residual.depth_m <= 0.0 {
            return true;
        }
        let threshold = if residual.range_m < rule.close_range_m {
            rule.close_factor * rule.reprojection_chi2
        } else {
            rule.reprojection_chi2
        };
        residual.reprojection_chi2 > threshold
    }
}

/// A keyframe feature's link to a landmark, as the solve's snapshot held it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutlierLink {
    pub key: ObservationKey,
    pub landmark: usize,
    /// On the newest keyframe of the solve.
    pub newest: bool,
}

/// Observations one solve tested, counted at the solve.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ErasedTested {
    pub tested: usize,
    pub tested_newest: usize,
}

/// One observation of the solve, for [`select_outliers`].
pub(crate) struct EraseCandidate<'a> {
    pub link: OutlierLink,
    pub fixed_pose: bool,
    pub residual: Option<&'a ObservationResidual>,
    /// The write-back guard kept the landmark's old position: its residual
    /// is of a position the map does not take.
    pub guarded: bool,
}

/// What [`select_outliers`] decided, for the debug line.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct OutlierSelection {
    pub links: Vec<OutlierLink>,
    pub tested: ErasedTested,
    pub tested_fixed: usize,
    pub erased_newest: usize,
    pub erased_fixed: usize,
    pub depth_nonpositive: usize,
    pub guard_skipped: usize,
}

/// The candidates `rule` erases, and how many it tested.
pub(crate) fn select_outliers<'a>(
    rule: &OutlierErase,
    candidates: impl IntoIterator<Item = EraseCandidate<'a>>,
) -> OutlierSelection {
    let mut selection = OutlierSelection::default();
    for candidate in candidates {
        let Some(residual) = candidate.residual else {
            continue;
        };
        if candidate.guarded {
            selection.guard_skipped += 1;
            continue;
        }
        let newest = candidate.link.newest;
        selection.tested.tested += 1;
        selection.tested.tested_newest += usize::from(newest);
        selection.tested_fixed += usize::from(candidate.fixed_pose);
        selection.depth_nonpositive += usize::from(residual.depth_m <= 0.0);
        if rule.is_outlier(residual) {
            selection.erased_newest += usize::from(newest);
            selection.erased_fixed += usize::from(candidate.fixed_pose);
            selection.links.push(candidate.link);
        }
    }
    selection
}

/// What one erase did to the live map, with the solve's tested counts.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EraseReport {
    pub tested: usize,
    pub tested_newest: usize,
    pub unlinked: usize,
    pub unlinked_newest: usize,
    /// Landmarks retired by the erase: no view left, or a weighted count of 2
    /// or less.
    pub landmarks_retired: usize,
    /// Links the live map no longer holds as the solve saw them: keyframe
    /// gone, feature re-linked, or landmark culled since the snapshot.
    pub stale_links: usize,
}

/// Unlinks each `links` entry the live map still holds, then retires each
/// touched landmark left at a weighted view count of 2 or less, a stereo view
/// counting 2 (ORB-SLAM3 `MapPoint::EraseObservation`). Untouched landmarks
/// are not judged.
pub fn erase(map: &mut Map, links: &[OutlierLink]) -> EraseReport {
    let mut report = EraseReport::default();
    if links.is_empty() {
        return report;
    }
    let mut touched = Vec::new();
    for link in links {
        if !holds_link(map, link) {
            report.stale_links += 1;
            continue;
        }
        match map.unlink_observation(link.key.keyframe_idx, link.key.feature_idx) {
            Ok(Some(_)) => {
                report.unlinked += 1;
                report.unlinked_newest += usize::from(link.newest);
                touched.push(link.landmark);
            }
            Ok(None) | Err(_) => report.stale_links += 1,
        }
    }
    touched.sort_unstable();
    touched.dedup();
    for landmark in touched {
        let retired = match map.map_points().get(landmark) {
            Some(mp) if mp.culled => true,
            Some(_) if weighted_views(map, landmark) <= 2 => {
                map.remove_landmark(landmark).unwrap_or(false)
            }
            _ => false,
        };
        report.landmarks_retired += usize::from(retired);
    }
    report
}

/// The live feature still holds the snapshot's landmark, and it is live.
fn holds_link(map: &Map, link: &OutlierLink) -> bool {
    let held = map
        .get_keyframe(link.key.keyframe_idx)
        .and_then(|kf| kf.map_point(link.key.feature_idx));
    held == Some(link.landmark)
        && map
            .map_points()
            .get(link.landmark)
            .is_some_and(|mp| !mp.culled)
}

/// Views of a live landmark, a feature with stereo depth counting 2.
fn weighted_views(map: &Map, landmark: usize) -> usize {
    map.map_points().get(landmark).map_or(0, |mp| {
        mp.observations()
            .iter()
            .map(|obs| {
                let stereo = map
                    .get_keyframe(obs.key.keyframe_idx)
                    .is_some_and(|kf| kf.frame.stereo_depth(obs.key.feature_idx).is_some());
                if stereo { 2 } else { 1 }
            })
            .sum()
    })
}

#[cfg(test)]
#[path = "outlier_erase_tests.rs"]
mod tests;
