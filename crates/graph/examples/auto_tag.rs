//! Automatic classification / tagging of new notes from an already-tagged
//! knowledge base — no external model. A new note's tags are voted by its
//! semantic neighbors (kNN over embeddings).
//!
//!   cargo run -p meandb-graph --example auto_tag

use meandb_graph::{GraphIndex, NodeMeta, SuggestOpts, TagClassifier};
use meandb_vector::Metric;

/// A vector with mass on the given topic dimensions (+ a little jitter).
fn vec_for(topics: &[usize], seed: u64, dim: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut rnd = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 40) as f32 / (1u64 << 24) as f32 * 0.05
    };
    (0..dim)
        .map(|d| {
            let base = if topics.contains(&(d % 8)) { 1.0 } else { 0.0 };
            base + rnd()
        })
        .collect()
}

fn main() {
    let dim = 24;
    // A labeled knowledge base: each note sits on one topic and is tagged.
    let topics = [
        "rust", "async", "graph", "search", "wasm", "db", "ml", "web",
    ];
    let mut kb = TagClassifier::new(Metric::Cosine);
    for i in 0..40u64 {
        let t = (i % 8) as usize;
        // Some notes carry a second, correlated tag.
        let mut tags = vec![topics[t].to_string()];
        if t == 0 {
            tags.push("systems".into()); // rust notes also "systems"
        }
        kb.add(i, &vec_for(&[t], i, dim), tags);
    }
    println!("knowledge base: {} tagged notes\n", kb.len());

    let opts = SuggestOpts {
        k: 8,
        max_tags: 3,
        min_score: 0.2,
        ..Default::default()
    };

    // New incoming articles → suggested tags.
    let incoming = [
        ("a Rust systems article", vec_for(&[0], 999, dim)),
        ("a graph+search article", vec_for(&[2, 3], 998, dim)),
        ("a web article", vec_for(&[7], 997, dim)),
    ];
    for (desc, emb) in &incoming {
        let sugg = kb.suggest(emb, &opts);
        let shown: Vec<String> = sugg
            .iter()
            .map(|s| format!("{}={:.2}", s.tag, s.score))
            .collect();
        println!("{desc:<26} -> {}", shown.join(", "));
    }

    // The same, integrated into the live graph: classify + file in one call.
    println!("\nlive graph — auto-tag on insert:");
    let mut gi = GraphIndex::new(Metric::Cosine, 8);
    gi.min_weight = 0.3;
    for i in 0..40u64 {
        let t = (i % 8) as usize;
        gi.insert(
            i,
            &vec_for(&[t], i, dim),
            NodeMeta {
                title: format!("note {i}"),
                tags: vec![topics[t].to_string()],
            },
        );
    }
    let applied =
        gi.insert_auto_tagged(100, &vec_for(&[0], 42, dim), "new rust note".into(), &opts);
    println!("note 100 auto-tagged: {applied:?}");
}
