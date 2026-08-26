use super::VerifiedLoopEdge;

/// Temporal and map-neighbourhood consistency required before a verified loop
/// becomes a pose-graph edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoopEpisodeConfig {
    pub min_consistent_edges: usize,
    pub max_query_gap: usize,
    pub candidate_neighborhood_radius: usize,
}

impl Default for LoopEpisodeConfig {
    fn default() -> Self {
        Self {
            min_consistent_edges: 3,
            max_query_gap: 5,
            candidate_neighborhood_radius: 10,
        }
    }
}

/// Decision produced for one geometrically verified observation.
#[derive(Debug, Clone, PartialEq)]
pub enum LoopEpisodeDecision {
    Pending {
        hits: usize,
        required: usize,
    },
    Ready {
        representative: VerifiedLoopEdge,
        hits: usize,
    },
    Suppressed {
        representative_query_kf_idx: usize,
        representative_candidate_kf_idx: usize,
    },
}

#[derive(Debug, Clone)]
struct LoopEpisode {
    candidate_anchor_order: usize,
    last_query_order: usize,
    hits: usize,
    representative: VerifiedLoopEdge,
    ready: bool,
}

/// Collapses adjacent verified observations into one physical revisit episode.
#[derive(Debug, Clone)]
pub struct LoopEpisodeTracker {
    config: LoopEpisodeConfig,
    current: Option<LoopEpisode>,
}

impl LoopEpisodeTracker {
    pub fn new(mut config: LoopEpisodeConfig) -> Self {
        config.min_consistent_edges = config.min_consistent_edges.max(1);
        Self {
            config,
            current: None,
        }
    }

    pub fn observe(
        &mut self,
        query_order: usize,
        candidate_order: usize,
        edge: VerifiedLoopEdge,
    ) -> LoopEpisodeDecision {
        let compatible = self.current.as_ref().is_some_and(|episode| {
            query_order >= episode.last_query_order
                && query_order - episode.last_query_order <= self.config.max_query_gap
                && candidate_order.abs_diff(episode.candidate_anchor_order)
                    <= self.config.candidate_neighborhood_radius
        });
        if !compatible {
            self.current = Some(LoopEpisode {
                candidate_anchor_order: candidate_order,
                last_query_order: query_order,
                hits: 1,
                representative: edge,
                ready: self.config.min_consistent_edges == 1,
            });
            let episode = self.current.as_ref().unwrap();
            return if episode.ready {
                LoopEpisodeDecision::Ready {
                    representative: episode.representative.clone(),
                    hits: episode.hits,
                }
            } else {
                LoopEpisodeDecision::Pending {
                    hits: episode.hits,
                    required: self.config.min_consistent_edges,
                }
            };
        }

        let episode = self.current.as_mut().unwrap();
        episode.last_query_order = query_order;
        if episode.ready {
            return LoopEpisodeDecision::Suppressed {
                representative_query_kf_idx: episode.representative.query_kf_idx,
                representative_candidate_kf_idx: episode.representative.candidate_kf_idx,
            };
        }
        episode.hits += 1;
        if edge_quality_better(&edge, &episode.representative) {
            episode.representative = edge;
        }
        if episode.hits >= self.config.min_consistent_edges {
            episode.ready = true;
            LoopEpisodeDecision::Ready {
                representative: episode.representative.clone(),
                hits: episode.hits,
            }
        } else {
            LoopEpisodeDecision::Pending {
                hits: episode.hits,
                required: self.config.min_consistent_edges,
            }
        }
    }
}

fn edge_quality_better(candidate: &VerifiedLoopEdge, current: &VerifiedLoopEdge) -> bool {
    candidate
        .inliers
        .cmp(&current.inliers)
        .then_with(|| candidate.inlier_ratio.total_cmp(&current.inlier_ratio))
        .then_with(|| candidate.occupied_cells.cmp(&current.occupied_cells))
        .then_with(|| {
            current
                .reprojection_rmse_px
                .total_cmp(&candidate.reprojection_rmse_px)
        })
        .is_gt()
}
