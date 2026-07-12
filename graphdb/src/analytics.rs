//! Small graph analytics for the view layer: out-degree (node size) and
//! community detection (node color). Both use only the public `GraphStore` API.

use crate::graph::GraphStore;
use std::collections::HashMap;

/// Out-degree per node index.
pub fn degrees(g: &GraphStore) -> Vec<usize> {
    g.ids().iter().map(|&id| g.degree(id)).collect()
}

/// Weighted, synchronous-ish **label propagation** communities (deterministic).
/// Returns a compacted community id per node index — useful for coloring an
/// Obsidian-style graph view. `iters` bounds passes (converges early if stable).
pub fn communities_label_propagation(g: &GraphStore, iters: usize) -> Vec<u32> {
    let n = g.len();
    if n == 0 {
        return Vec::new();
    }
    let ids = g.ids();
    let idx: HashMap<u64, usize> = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    let mut label: Vec<u32> = (0..n as u32).collect();

    for _ in 0..iters {
        let mut changed = false;
        for i in 0..n {
            let mut votes: HashMap<u32, f32> = HashMap::new();
            for nb in g.neighbors(ids[i]) {
                if let Some(&j) = idx.get(&nb.id) {
                    *votes.entry(label[j]).or_insert(0.0) += nb.weight.max(0.0);
                }
            }
            if votes.is_empty() {
                continue;
            }
            // Highest total weight wins; ties broken toward the smallest label
            // for determinism.
            let best = votes
                .iter()
                .max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(a.0)))
                .map(|(l, _)| *l)
                .unwrap();
            if best != label[i] {
                label[i] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    // Compact raw labels to 0..c in order of first appearance.
    let mut remap: HashMap<u32, u32> = HashMap::new();
    let mut next = 0u32;
    label
        .iter()
        .map(|&l| {
            *remap.entry(l).or_insert_with(|| {
                let c = next;
                next += 1;
                c
            })
        })
        .collect()
}
