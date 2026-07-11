//! Appearance-based place recognition: a bag-of-words keyframe database for
//! loop-closure and relocalization candidate retrieval.
//!
//! Port of ORB-SLAM3's `KeyFrameDatabase` (`extra/ORB_SLAM3/src/KeyFrameDatabase.cc`)
//! on top of the [`kornia_bow`] vocabulary. Each keyframe is summarized by a
//! sparse [`BoW`] vector (word id → TF-IDF weight) computed from its ORB
//! descriptors against a shared [`OrbVocabulary`]. An inverted index (word →
//! keyframes) makes "which past keyframes share words with this one" cheap, and
//! candidates are scored by [`BoW::l1_similarity`] exactly as DBoW2 does.
//!
//! Covisibility-group score accumulation (ORB-SLAM3's second pass over each
//! candidate's neighbours) is left to the loop-closing layer; this database
//! returns the per-keyframe shared-word-filtered, L1-scored candidates.

use std::collections::{HashMap, HashSet};

use kornia_bow::BoW;
use kornia_bow::orb_slam3::{OrbVocabulary, pack_orb_descriptor};

// Re-export so downstream crates can load and hold a vocabulary through
// `kornia_slam::place_recognition` without depending on `kornia-bow` directly.
pub use kornia_bow::BoW as BagOfWords;
pub use kornia_bow::orb_slam3::{OrbVocabulary as Vocabulary, load_orb_slam3_vocabulary};

/// Computes the bag-of-words vector for a set of ORB descriptors.
///
/// Descriptors are packed into the vocabulary's `Feature<u64, 4>` layout and
/// transformed into a sparse, L1-normalized [`BoW`]. Returns an empty `BoW`
/// when there are no descriptors (an unusable keyframe for place recognition).
pub fn compute_bow(vocab: &OrbVocabulary, descriptors: &[[u8; 32]]) -> BoW {
    if descriptors.is_empty() {
        return BoW(Vec::new());
    }
    let packed: Vec<_> = descriptors.iter().map(pack_orb_descriptor).collect();
    vocab.transform(&packed).unwrap_or(BoW(Vec::new()))
}

/// Inverted-index bag-of-words database over keyframes.
#[derive(Debug, Default)]
pub struct KeyFrameDatabase {
    /// word id → keyframe indices containing that word.
    inverted: HashMap<u32, Vec<usize>>,
    /// keyframe index → its BoW vector (retained so [`Self::erase`] can undo the
    /// inverted-index entries without the caller resupplying the vector).
    bows: HashMap<usize, BoW>,
}

/// A retrieved place-recognition candidate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candidate {
    /// Keyframe index (`Keyframe::frame.idx`) of the candidate.
    pub kf_idx: usize,
    /// Number of vocabulary words shared with the query.
    pub shared_words: usize,
    /// L1 bag-of-words similarity to the query, in `[0, 1]`.
    pub score: f32,
}

impl KeyFrameDatabase {
    /// Creates an empty database.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of keyframes currently indexed.
    pub fn len(&self) -> usize {
        self.bows.len()
    }

    /// Whether the database holds no keyframes.
    pub fn is_empty(&self) -> bool {
        self.bows.is_empty()
    }

    /// Adds a keyframe's BoW vector to the index. Re-adding the same index
    /// first removes the previous entry so the inverted lists stay consistent.
    pub fn add(&mut self, kf_idx: usize, bow: BoW) {
        if self.bows.contains_key(&kf_idx) {
            self.erase(kf_idx);
        }
        for &(word, _) in &bow.0 {
            self.inverted.entry(word).or_default().push(kf_idx);
        }
        self.bows.insert(kf_idx, bow);
    }

    /// Removes a keyframe from the index (e.g. after culling). No-op if absent.
    pub fn erase(&mut self, kf_idx: usize) {
        let Some(bow) = self.bows.remove(&kf_idx) else {
            return;
        };
        for &(word, _) in &bow.0 {
            if let Some(list) = self.inverted.get_mut(&word) {
                list.retain(|&k| k != kf_idx);
                if list.is_empty() {
                    self.inverted.remove(&word);
                }
            }
        }
    }

    /// The stored BoW vector for a keyframe, if indexed.
    pub fn bow(&self, kf_idx: usize) -> Option<&BoW> {
        self.bows.get(&kf_idx)
    }

    /// Retrieves place-recognition candidates for a query BoW, sorted by
    /// descending score. `exclude` skips the query's own neighbours; only
    /// keyframes scoring at least `min_score` are returned.
    ///
    /// Mirrors the first phase of ORB-SLAM3's `DetectLoopCandidates` /
    /// `DetectRelocalizationCandidates`; covisibility-group accumulation is the
    /// caller's responsibility.
    pub fn detect_candidates(
        &self,
        query: &BoW,
        exclude: &HashSet<usize>,
        min_score: f32,
    ) -> Vec<Candidate> {
        // Phase 1: shared-word histogram over the inverted index.
        let mut shared: HashMap<usize, usize> = HashMap::new();
        for &(word, _) in &query.0 {
            if let Some(list) = self.inverted.get(&word) {
                for &kf in list {
                    if exclude.contains(&kf) {
                        continue;
                    }
                    *shared.entry(kf).or_insert(0) += 1;
                }
            }
        }
        if shared.is_empty() {
            return Vec::new();
        }

        // Phase 2: keep keyframes with enough shared words to be discriminative.
        let max_shared = shared.values().copied().max().unwrap_or(0);
        let min_shared = (0.8 * max_shared as f32) as usize;

        // Phase 3: L1-score the survivors.
        let mut candidates: Vec<Candidate> = shared
            .into_iter()
            .filter(|&(_, n)| n > min_shared)
            .filter_map(|(kf, n)| {
                let bow = self.bows.get(&kf)?;
                let score = query.l1_similarity(bow);
                (score >= min_score).then_some(Candidate {
                    kf_idx: kf,
                    shared_words: n,
                    score,
                })
            })
            .collect();

        candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates
    }

    /// ORB-SLAM3 `LoopClosing::DetectLoop` retrieval: excludes the query
    /// keyframe and its covisible neighbours, and uses the lowest BoW
    /// similarity to those neighbours as the score floor, so only a genuine
    /// revisit (not the local neighbourhood) can match.
    pub fn detect_loop_candidates(
        &self,
        query_kf_idx: usize,
        query: &BoW,
        covisible: impl IntoIterator<Item = usize>,
    ) -> Vec<Candidate> {
        let mut exclude = HashSet::new();
        exclude.insert(query_kf_idx);
        let mut min_score = 1.0f32;
        for nb in covisible {
            exclude.insert(nb);
            if let Some(nb_bow) = self.bow(nb) {
                min_score = min_score.min(query.l1_similarity(nb_bow));
            }
        }
        self.detect_candidates(query, &exclude, min_score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bow(entries: &[(u32, f32)]) -> BoW {
        BoW(entries.to_vec())
    }

    #[test]
    fn add_erase_roundtrip() {
        let mut db = KeyFrameDatabase::new();
        db.add(1, bow(&[(10, 0.5), (20, 0.5)]));
        assert_eq!(db.len(), 1);
        db.erase(1);
        assert!(db.is_empty());
        // Inverted lists fully cleaned up.
        assert!(db.inverted.is_empty());
    }

    #[test]
    fn detect_prefers_more_similar_keyframe() {
        let mut db = KeyFrameDatabase::new();
        // KF 1 shares both words with the query and is near-identical.
        db.add(1, bow(&[(10, 0.5), (20, 0.5)]));
        // KF 2 shares one word, weaker overlap.
        db.add(2, bow(&[(10, 0.5), (99, 0.5)]));

        let query = bow(&[(10, 0.5), (20, 0.5)]);
        let cands = db.detect_candidates(&query, &HashSet::new(), 0.0);

        assert_eq!(cands.first().map(|c| c.kf_idx), Some(1));
        assert!(cands[0].score >= cands.last().unwrap().score);
    }

    #[test]
    fn detect_excludes_listed_keyframes() {
        let mut db = KeyFrameDatabase::new();
        db.add(1, bow(&[(10, 1.0)]));
        let query = bow(&[(10, 1.0)]);

        let exclude: HashSet<usize> = [1].into_iter().collect();
        assert!(db.detect_candidates(&query, &exclude, 0.0).is_empty());
    }

    #[test]
    fn detect_loop_excludes_query_and_neighbours() {
        let mut db = KeyFrameDatabase::new();
        db.add(1, bow(&[(10, 0.5), (20, 0.5)])); // covisible neighbour of the query
        db.add(2, bow(&[(10, 0.5), (20, 0.5)])); // a genuine revisit

        let query = bow(&[(10, 0.5), (20, 0.5)]);
        // kf 3 is the query; kf 1 is its covisible neighbour → both excluded.
        let cands = db.detect_loop_candidates(3, &query, [1]);

        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].kf_idx, 2);
    }

    #[test]
    fn detect_applies_min_score() {
        let mut db = KeyFrameDatabase::new();
        db.add(1, bow(&[(10, 1.0)]));
        let query = bow(&[(20, 1.0)]); // disjoint words → no shared, no candidate
        assert!(
            db.detect_candidates(&query, &HashSet::new(), 0.5)
                .is_empty()
        );
    }
}
