//! Small graph analytics for the view layer: out-degree (node size) and
//! community detection (node color). Both use only the public `GraphStore` API.

use crate::graph::GraphStore;
use std::collections::HashMap;

/// Out-degree per node index.
pub fn degrees(g: &GraphStore) -> Vec<usize> {
    g.ids().iter().map(|&id| g.degree(id)).collect()
}

/// Weighted PageRank per node index (importance for ranking / node sizing).
/// `damping` is the usual teleport factor (~0.85); `iters` bounds the power
/// iteration. Dangling nodes (no out-edges) redistribute their mass uniformly.
pub fn pagerank(g: &GraphStore, iters: usize, damping: f32) -> Vec<f32> {
    let n = g.len();
    if n == 0 {
        return Vec::new();
    }
    let ids = g.ids();
    let idx: std::collections::HashMap<u64, usize> =
        ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    // Out-edges as (dst index, weight), plus each node's total out-weight.
    let out: Vec<Vec<(usize, f32)>> = ids
        .iter()
        .map(|&id| {
            g.neighbors(id)
                .into_iter()
                .filter_map(|nb| idx.get(&nb.id).map(|&j| (j, nb.weight.max(0.0))))
                .collect()
        })
        .collect();
    let out_sum: Vec<f32> = out
        .iter()
        .map(|es| es.iter().map(|&(_, w)| w).sum())
        .collect();

    let base = (1.0 - damping) / n as f32;
    let mut pr = vec![1.0f32 / n as f32; n];
    for _ in 0..iters {
        let mut next = vec![base; n];
        let mut dangling = 0.0f32;
        for i in 0..n {
            if out_sum[i] <= 0.0 {
                dangling += pr[i];
                continue;
            }
            for &(j, w) in &out[i] {
                next[j] += damping * pr[i] * (w / out_sum[i]);
            }
        }
        // Redistribute dangling mass uniformly.
        let spread = damping * dangling / n as f32;
        for v in next.iter_mut() {
            *v += spread;
        }
        pr = next;
    }
    pr
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GraphBuilder;
    use vectordb::Metric;

    fn clustered() -> Vec<(u64, Vec<f32>)> {
        let mut s: u64 = 0xA11CE_u64.wrapping_mul(0x9E37_79B9);
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32 * 0.05
        };
        let a = [1.0f32, 0.0, 0.0, 0.0];
        let b = [0.0f32, 0.0, 1.0, 0.0];
        (0..8u64)
            .map(|i| {
                let c = if i < 4 { &a } else { &b };
                (i, c.iter().map(|&x| x + rnd()).collect())
            })
            .collect()
    }

    #[test]
    fn pagerank_is_a_distribution() {
        let items = clustered();
        let g = GraphBuilder::new(Metric::Cosine, 3).build(&items, &[]);
        let pr = pagerank(&g, 40, 0.85);
        assert_eq!(pr.len(), g.len());
        let sum: f32 = pr.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-3,
            "pagerank should sum to 1, got {sum}"
        );
        assert!(pr.iter().all(|&x| x > 0.0 && x.is_finite()));
    }
}
