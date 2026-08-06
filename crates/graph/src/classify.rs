//! Automatic tag suggestion for new notes — no external model needed.
//!
//! A note's tags are predicted from its **semantic neighbors**: embed the new
//! note, find its k nearest already-tagged notes (reusing the vector layer's ANN
//! index), and sum each neighbor's similarity as a vote for that neighbor's
//! tags. A tag's `score` is the fraction of neighborhood similarity-mass that
//! carries it, so it lands in `[0, 1]` and is easy to threshold. This is kNN
//! multi-label classification: robust, incremental, and interpretable.

use meandb_vector::{HnswIndex, Metric};
use std::collections::HashMap;

/// A suggested tag with its confidence and how many neighbors carried it.
#[derive(Clone, Debug, PartialEq)]
pub struct TagSuggestion {
    pub tag: String,
    pub score: f32,
    pub votes: usize,
}

/// Knobs for tag suggestion.
#[derive(Clone, Debug)]
pub struct SuggestOpts {
    /// Neighbors to consult.
    pub k: usize,
    /// ANN search beam.
    pub ef: usize,
    /// At most this many tags.
    pub max_tags: usize,
    /// Minimum confidence (fraction of neighborhood mass) to suggest a tag.
    pub min_score: f32,
    /// A tag must appear in at least this many neighbors.
    pub min_votes: usize,
}

impl Default for SuggestOpts {
    fn default() -> Self {
        SuggestOpts {
            k: 10,
            ef: 64,
            max_tags: 5,
            min_score: 0.15,
            min_votes: 1,
        }
    }
}

/// Aggregate tag votes from `(tags, similarity_weight)` neighbors into ranked
/// suggestions. Shared by [`TagClassifier`] and `GraphIndex::suggest_tags`.
pub fn score_tags<'a>(
    neighbors: impl IntoIterator<Item = (&'a [String], f32)>,
    opts: &SuggestOpts,
) -> Vec<TagSuggestion> {
    let mut sum: HashMap<&str, f32> = HashMap::new();
    let mut votes: HashMap<&str, usize> = HashMap::new();
    let mut total = 0.0f32;
    for (tags, w) in neighbors {
        let w = w.max(0.0);
        total += w;
        for t in tags {
            *sum.entry(t.as_str()).or_insert(0.0) += w;
            *votes.entry(t.as_str()).or_insert(0) += 1;
        }
    }
    if total <= 0.0 {
        return Vec::new();
    }
    let mut out: Vec<TagSuggestion> = sum
        .iter()
        .map(|(&t, &s)| TagSuggestion {
            tag: t.to_string(),
            score: s / total,
            votes: votes[t],
        })
        .filter(|s| s.score >= opts.min_score && s.votes >= opts.min_votes)
        .collect();
    out.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.tag.cmp(&b.tag)));
    out.truncate(opts.max_tags);
    out
}

/// A standalone tag classifier over `(id, embedding, tags)` training notes.
/// Use it when you only have embeddings + tags (no live graph); otherwise
/// `GraphIndex::suggest_tags` reuses the graph's own index.
pub struct TagClassifier {
    metric: Metric,
    dim: usize,
    hnsw: Option<HnswIndex>,
    tags: HashMap<u64, Vec<String>>,
}

impl TagClassifier {
    pub fn new(metric: Metric) -> Self {
        TagClassifier {
            metric,
            dim: 0,
            hnsw: None,
            tags: HashMap::new(),
        }
    }

    /// Build from labeled notes.
    pub fn build(items: &[(u64, Vec<f32>, Vec<String>)], metric: Metric) -> Self {
        let mut c = TagClassifier::new(metric);
        for (id, emb, tags) in items {
            c.add(*id, emb, tags.clone());
        }
        c
    }

    pub fn len(&self) -> usize {
        self.tags.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tags.is_empty()
    }

    /// Add (or replace) a labeled note.
    pub fn add(&mut self, id: u64, embedding: &[f32], tags: Vec<String>) {
        if self.dim == 0 {
            self.dim = embedding.len();
            self.hnsw = Some(HnswIndex::new(self.dim, self.metric, 16, 200));
        }
        assert_eq!(embedding.len(), self.dim, "dimension mismatch");
        let h = self.hnsw.as_mut().unwrap();
        if self.tags.contains_key(&id) {
            h.remove(id);
        }
        h.add(id, embedding);
        self.tags.insert(id, tags);
    }

    /// Suggest tags for a new note's `embedding`.
    pub fn suggest(&self, embedding: &[f32], opts: &SuggestOpts) -> Vec<TagSuggestion> {
        let Some(h) = &self.hnsw else {
            return Vec::new();
        };
        let hits = h.search(embedding, opts.k, opts.ef);
        let nbrs = hits
            .iter()
            .filter_map(|hit| self.tags.get(&hit.id).map(|t| (t.as_slice(), hit.score)));
        score_tags(nbrs, opts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(topic: usize) -> Vec<f32> {
        let mut e = vec![0.0f32; 6];
        e[topic] = 1.0;
        e
    }

    #[test]
    fn suggests_neighbor_tags() {
        // Two topics: rust (0) and cooking (1).
        let items = vec![
            (1u64, v(0), vec!["rust".into(), "lang".into()]),
            (2, v(0), vec!["rust".into()]),
            (3, v(0), vec!["rust".into(), "async".into()]),
            (4, v(1), vec!["cooking".into()]),
            (5, v(1), vec!["cooking".into(), "recipe".into()]),
        ];
        let c = TagClassifier::build(&items, Metric::Cosine);
        let opts = SuggestOpts {
            k: 3,
            min_score: 0.3,
            ..Default::default()
        };
        // A new rust-ish note.
        let s = c.suggest(&v(0), &opts);
        assert_eq!(s[0].tag, "rust", "top tag should be rust: {s:?}");
        assert!(s.iter().all(|t| t.tag != "cooking"));

        // A new cooking-ish note.
        let s = c.suggest(&v(1), &opts);
        assert_eq!(s[0].tag, "cooking");
    }

    #[test]
    fn empty_and_thresholds() {
        let c = TagClassifier::new(Metric::Cosine);
        assert!(c.suggest(&v(0), &SuggestOpts::default()).is_empty());
        // High threshold suppresses minority tags.
        let items = vec![
            (1u64, v(0), vec!["a".into()]),
            (2, v(0), vec!["a".into()]),
            (3, v(0), vec!["a".into(), "rare".into()]),
        ];
        let c = TagClassifier::build(&items, Metric::Cosine);
        let s = c.suggest(
            &v(0),
            &SuggestOpts {
                k: 3,
                min_score: 0.5,
                ..Default::default()
            },
        );
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].tag, "a");
    }
}
