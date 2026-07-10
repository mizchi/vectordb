//! DiskANN (Subramanya et al., NeurIPS 2019) — minimal single-machine version.
//!
//! A single-layer **Vamana** proximity graph, designed so most of the data can
//! live on disk while only a compressed copy stays resident:
//!
//! - **RAM tier:** PQ codes of every vector, used for the approximate distances
//!   that steer graph traversal (a query lookup table + one add per hop).
//! - **"disk" tier:** the graph adjacency and the raw f32 vectors — here kept in
//!   owned `Vec`s (and reloadable via mmap), read only for the final rerank.
//!
//! Build uses the Vamana insertion pass with `RobustPrune`: greedily search
//! from the medoid to each point, then prune the visited set to `≤ R` diverse
//! out-edges (keep `c` and drop any `c'` it dominates, i.e. `α·d(c,c') ≤
//! d(p,c')`), adding back-edges and re-pruning over-full neighbors. The α > 1
//! slack retains long-range shortcuts so search reaches any point in few hops —
//! the whole point of a *single*-layer graph (fewer disk reads than HNSW).
//!
//! The graph geometry and PQ are built in **squared-L2 over processed space**
//! (vectors are normalized for cosine, so L2 ordering matches cosine); the
//! `metric` only sets the final exact rerank score. Dot is best-effort.

use crate::distance::{dot_f32, l2sq_f32};
use crate::index::{Hit, Metric};
use crate::pq::{adc_sum, build_lut, encode_vector, train_codebooks};
use std::collections::HashSet;
use std::io;
use std::path::Path;

/// A minimal DiskANN / Vamana index.
pub struct DiskAnnIndex {
    dim: usize,
    metric: Metric,
    count: usize,
    r: usize,          // max out-degree
    l_build: usize,    // build/insert beam width
    alpha: f32,        // RobustPrune slack
    entry: u32,        // medoid
    vectors: Vec<f32>, // count * dim, processed ("disk" tier)
    ids: Vec<u64>,
    graph: Vec<Vec<u32>>, // adjacency, degree <= r
    deleted: Vec<bool>,   // per-row tombstones (in-memory)
    deleted_count: usize,
    // PQ, RAM-resident, always L2 for traversal.
    m: usize,
    dsub: usize,
    ksub: usize,
    codebooks: Vec<f32>,
    codes: Vec<u8>,
}

/// Small deterministic xorshift for the insertion order + random init graph.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        let inv = 1.0 / norm;
        v.iter().map(|x| x * inv).collect()
    } else {
        v.to_vec()
    }
}

/// Squared-L2 between stored rows `a` and `b`.
#[inline]
fn l2_ab(vectors: &[f32], dim: usize, a: u32, b: u32) -> f32 {
    let a = a as usize;
    let b = b as usize;
    l2sq_f32(
        &vectors[a * dim..(a + 1) * dim],
        &vectors[b * dim..(b + 1) * dim],
    )
}

/// Greedy beam search over `graph` from `entry`, ranking by `dist(node)`
/// (smaller = closer). Returns (all expanded nodes, the best `l` as sorted
/// (dist,node) pairs).
fn greedy_search<F: Fn(u32) -> f32>(
    graph: &[Vec<u32>],
    entry: u32,
    l: usize,
    dist: F,
) -> (Vec<u32>, Vec<(f32, u32)>) {
    let mut list: Vec<(f32, u32)> = vec![(dist(entry), entry)];
    let mut inserted: HashSet<u32> = HashSet::from([entry]);
    let mut expanded: HashSet<u32> = HashSet::new();
    let mut visited_all: Vec<u32> = Vec::new();
    loop {
        // list is kept sorted ascending, so the first unexpanded entry is the
        // nearest candidate to expand next.
        let Some(pi) = list.iter().position(|&(_, n)| !expanded.contains(&n)) else {
            break;
        };
        let p = list[pi].1;
        expanded.insert(p);
        visited_all.push(p);
        for &nb in &graph[p as usize] {
            if inserted.insert(nb) {
                list.push((dist(nb), nb));
            }
        }
        list.sort_by(|a, b| a.0.total_cmp(&b.0));
        if list.len() > l {
            list.truncate(l);
        }
    }
    (visited_all, list)
}

/// RobustPrune: from the candidate pool (visited ∪ existing neighbors), keep up
/// to `r` diverse out-edges for `p`. Distances are squared-L2 in processed space.
fn robust_prune(
    vectors: &[f32],
    dim: usize,
    p: u32,
    pool: &[u32],
    alpha: f32,
    r: usize,
) -> Vec<u32> {
    let mut seen: HashSet<u32> = HashSet::new();
    let mut cand: Vec<(f32, u32)> = Vec::new();
    for &c in pool {
        if c != p && seen.insert(c) {
            cand.push((l2_ab(vectors, dim, p, c), c));
        }
    }
    cand.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut removed = vec![false; cand.len()];
    let mut result: Vec<u32> = Vec::with_capacity(r);
    for i in 0..cand.len() {
        if removed[i] {
            continue;
        }
        let (_, c) = cand[i];
        result.push(c);
        if result.len() >= r {
            break;
        }
        // Drop any farther candidate c' that `c` dominates: α·d(c,c') ≤ d(p,c').
        for j in (i + 1)..cand.len() {
            if removed[j] {
                continue;
            }
            let cj = cand[j].1;
            let d_p_cj = cand[j].0;
            let d_c_cj = l2_ab(vectors, dim, c, cj);
            if alpha * d_c_cj <= d_p_cj {
                removed[j] = true;
            }
        }
    }
    result
}

impl DiskAnnIndex {
    /// Build a DiskANN index. `r` bounds out-degree (~32–64), `l_build` is the
    /// build beam (~64–128), `alpha ≥ 1.0` sets the prune slack (~1.2). PQ with
    /// `m` subspaces × `ksub` centroids drives traversal; `m` must divide `dim`.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        vectors: &[(u64, Vec<f32>)],
        metric: Metric,
        r: usize,
        l_build: usize,
        alpha: f32,
        m: usize,
        ksub: usize,
    ) -> DiskAnnIndex {
        assert!(!vectors.is_empty(), "DiskAnnIndex::build: empty input");
        let dim = vectors[0].1.len();
        assert!(m > 0 && dim.is_multiple_of(m), "m must divide dim");
        assert!(ksub > 0 && ksub <= 256, "ksub must be in 1..=256");
        assert!(r >= 1 && alpha >= 1.0);
        let n = vectors.len();
        let dsub = dim / m;
        let ksub = ksub.min(n);
        let cosine = metric == Metric::Cosine;

        // Processed matrix + ids.
        let mut vecs = vec![0f32; n * dim];
        let mut ids = Vec::with_capacity(n);
        for (i, (id, v)) in vectors.iter().enumerate() {
            assert_eq!(v.len(), dim, "inconsistent dimension");
            ids.push(*id);
            if cosine {
                vecs[i * dim..(i + 1) * dim].copy_from_slice(&normalize(v));
            } else {
                vecs[i * dim..(i + 1) * dim].copy_from_slice(v);
            }
        }

        // Medoid = point nearest to the mean (the search entry point).
        let mut mean = vec![0f32; dim];
        for i in 0..n {
            for d in 0..dim {
                mean[d] += vecs[i * dim + d];
            }
        }
        for d in mean.iter_mut() {
            *d /= n as f32;
        }
        let mut entry = 0u32;
        let mut best = f32::INFINITY;
        for i in 0..n {
            let d = l2sq_f32(&vecs[i * dim..(i + 1) * dim], &mean);
            if d < best {
                best = d;
                entry = i as u32;
            }
        }

        // PQ codes (RAM-resident, L2) for traversal distances.
        let codebooks = train_codebooks(&vecs, n, dim, m, ksub, 15);
        let mut codes = vec![0u8; n * m];
        for i in 0..n {
            encode_vector(
                &codebooks,
                dim,
                m,
                ksub,
                &vecs[i * dim..(i + 1) * dim],
                &mut codes[i * m..(i + 1) * m],
            );
        }

        // Random R-regular init graph so greedy is connected from the start.
        let mut graph: Vec<Vec<u32>> = vec![Vec::new(); n];
        if n > 1 {
            let mut rng = Rng(0x1234_5678_9ABC_DEF1 ^ n as u64);
            for (i, nbrs) in graph.iter_mut().enumerate() {
                let deg = r.min(n - 1);
                let mut seen = HashSet::new();
                while nbrs.len() < deg {
                    let c = (rng.next() % n as u64) as u32;
                    if c as usize != i && seen.insert(c) {
                        nbrs.push(c);
                    }
                }
            }
        }

        // Vamana insertion pass in a random order.
        let mut order: Vec<u32> = (0..n as u32).collect();
        let mut rng = Rng(0xF00D_BABE_1234_5678 ^ n as u64);
        for i in (1..order.len()).rev() {
            let j = (rng.next() % (i as u64 + 1)) as usize;
            order.swap(i, j);
        }
        for &p in &order {
            let (visited, _) =
                greedy_search(&graph, entry, l_build, |node| l2_ab(&vecs, dim, node, p));
            let mut pool = visited;
            pool.extend_from_slice(&graph[p as usize]);
            graph[p as usize] = robust_prune(&vecs, dim, p, &pool, alpha, r);
            // Add back-edges; re-prune neighbors that overflow.
            let nbrs = graph[p as usize].clone();
            for j in nbrs {
                let jn = j as usize;
                if !graph[jn].contains(&p) {
                    graph[jn].push(p);
                    if graph[jn].len() > r {
                        let mut pool2 = graph[jn].clone();
                        pool2.push(p);
                        graph[jn] = robust_prune(&vecs, dim, j, &pool2, alpha, r);
                    }
                }
            }
        }

        DiskAnnIndex {
            dim,
            metric,
            count: n,
            r,
            l_build,
            alpha,
            entry,
            vectors: vecs,
            ids,
            graph,
            deleted: vec![false; n],
            deleted_count: 0,
            m,
            dsub,
            ksub,
            codebooks,
            codes,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn metric(&self) -> Metric {
        self.metric
    }
    pub fn len(&self) -> usize {
        self.count
    }
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
    /// Average out-degree of the graph (a rough index-size signal).
    pub fn avg_degree(&self) -> f32 {
        if self.count == 0 {
            return 0.0;
        }
        self.graph.iter().map(|g| g.len()).sum::<usize>() as f32 / self.count as f32
    }

    /// Number of live (non-tombstoned) vectors.
    pub fn live_len(&self) -> usize {
        self.count - self.deleted_count
    }

    /// Incrementally insert one vector (FreshDiskANN-style): append the point,
    /// PQ-encode it against the existing codebooks (no retrain), greedily search
    /// the current graph for its neighborhood, `RobustPrune` to its out-edges,
    /// and add pruned back-edges. O(l_build · R) — no full rebuild.
    pub fn insert(&mut self, id: u64, vector: &[f32]) {
        assert_eq!(vector.len(), self.dim, "dimension mismatch");
        let dim = self.dim;
        let processed = if self.metric == Metric::Cosine {
            normalize(vector)
        } else {
            vector.to_vec()
        };
        let node = self.count as u32;
        self.vectors.extend_from_slice(&processed);
        self.ids.push(id);
        self.deleted.push(false);
        self.graph.push(Vec::new());
        let mut code = vec![0u8; self.m];
        encode_vector(
            &self.codebooks,
            dim,
            self.m,
            self.ksub,
            &processed,
            &mut code,
        );
        self.codes.extend_from_slice(&code);
        self.count += 1;
        if self.count == 1 {
            self.entry = 0;
            return;
        }
        // Greedy search from the entry to the new point (exact L2), prune to
        // out-edges, and wire back-edges.
        let (visited, _) = greedy_search(&self.graph, self.entry, self.l_build, |x| {
            l2_ab(&self.vectors, dim, x, node)
        });
        self.graph[node as usize] =
            robust_prune(&self.vectors, dim, node, &visited, self.alpha, self.r);
        let nbrs = self.graph[node as usize].clone();
        for j in nbrs {
            let jn = j as usize;
            if !self.graph[jn].contains(&node) {
                self.graph[jn].push(node);
                if self.graph[jn].len() > self.r {
                    let pool = self.graph[jn].clone();
                    self.graph[jn] = robust_prune(&self.vectors, dim, j, &pool, self.alpha, self.r);
                }
            }
        }
    }

    /// Tombstone every row with external id `id` (excluded from results; the
    /// graph is still traversed through it). Returns how many were newly
    /// deleted. Call [`consolidate`](Self::consolidate) to physically remove
    /// them (and before `save`, since the format stores only live rows).
    pub fn remove(&mut self, id: u64) -> usize {
        let mut removed = 0;
        for i in 0..self.count {
            if self.ids[i] == id && !self.deleted[i] {
                self.deleted[i] = true;
                self.deleted_count += 1;
                removed += 1;
            }
        }
        removed
    }

    /// Physically drop tombstoned points by rebuilding the graph over the live
    /// set (fresh Vamana). Codebooks are retrained on the survivors.
    pub fn consolidate(&mut self) {
        if self.deleted_count == 0 {
            return;
        }
        let dim = self.dim;
        let live: Vec<(u64, Vec<f32>)> = (0..self.count)
            .filter(|&i| !self.deleted[i])
            .map(|i| (self.ids[i], self.vectors[i * dim..(i + 1) * dim].to_vec()))
            .collect();
        *self = DiskAnnIndex::build(
            &live,
            self.metric,
            self.r,
            self.l_build,
            self.alpha,
            self.m,
            self.ksub,
        );
    }

    fn process(&self, query: &[f32]) -> Vec<f32> {
        assert_eq!(query.len(), self.dim, "dimension mismatch");
        if self.metric == Metric::Cosine {
            normalize(query)
        } else {
            query.to_vec()
        }
    }

    /// Search for the `k` nearest neighbors. `l_search` is the traversal beam
    /// width (larger → higher recall, more work); the beam is reranked with
    /// exact f32 distances in the true metric.
    pub fn search(&self, query: &[f32], k: usize, l_search: usize) -> Vec<Hit> {
        if k == 0 || self.count == 0 {
            return Vec::new();
        }
        let processed = self.process(query);
        // Traversal distances via PQ ADC (always L2 geometry).
        let lut = build_lut(
            &self.codebooks,
            self.dim,
            self.m,
            self.ksub,
            &processed,
            true,
        );
        let l = l_search.max(k);
        let (_, beam) = greedy_search(&self.graph, self.entry, l, |node| {
            adc_sum(
                &lut,
                self.ksub,
                &self.codes[node as usize * self.m..(node as usize + 1) * self.m],
            )
        });

        // Exact rerank of the beam in the true metric.
        let hib = self.metric.higher_is_better();
        let mut scored: Vec<(f32, u32)> = beam
            .iter()
            .filter(|&&(_, node)| !self.deleted[node as usize])
            .map(|&(_, node)| {
                let v = &self.vectors[node as usize * self.dim..(node as usize + 1) * self.dim];
                let key = match self.metric {
                    Metric::L2 => l2sq_f32(v, &processed),
                    Metric::Dot | Metric::Cosine => -dot_f32(v, &processed),
                };
                (key, node)
            })
            .collect();
        scored.sort_by(|a, b| a.0.total_cmp(&b.0));
        scored
            .into_iter()
            .take(k)
            .map(|(key, node)| Hit {
                id: self.ids[node as usize],
                score: if hib { -key } else { key },
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Persistence: a DiskANN `.vecdb` file (magic "VECDBDA1").
// Header (64B) + graph_offsets u64 + graph_neighbors u32 + ids u64
// + codebooks f32 + codes u8 + raw f32 (each section 16B-aligned).
// ---------------------------------------------------------------------------

const DA_MAGIC: &[u8; 8] = b"VECDBDA1";
const DA_VERSION: u32 = 1;

#[inline]
fn align16(x: usize) -> usize {
    (x + 15) & !15
}

struct DaLayout {
    goff: usize,
    gnbr: usize,
    ids: usize,
    codebooks: usize,
    codes: usize,
    raw: usize,
    total: usize,
}

impl DiskAnnIndex {
    fn edge_count(&self) -> usize {
        self.graph.iter().map(|g| g.len()).sum()
    }

    fn layout(&self, edges: usize) -> DaLayout {
        let dim = self.dim;
        let count = self.count;
        let goff = align16(64);
        let gnbr = align16(goff + (count + 1) * 8);
        let ids = align16(gnbr + edges * 4);
        let codebooks = align16(ids + count * 8);
        let codes = align16(codebooks + self.m * self.ksub * self.dsub * 4);
        let raw = align16(codes + count * self.m);
        let total = align16(raw + count * dim * 4);
        DaLayout {
            goff,
            gnbr,
            ids,
            codebooks,
            codes,
            raw,
            total,
        }
    }

    /// Serialize the index to a DiskANN `.vecdb` file.
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let edges = self.edge_count();
        let l = self.layout(edges);
        let mut b = vec![0u8; l.total];
        b[0..8].copy_from_slice(DA_MAGIC);
        b[8..12].copy_from_slice(&DA_VERSION.to_le_bytes());
        b[12..16].copy_from_slice(&(self.metric as u32).to_le_bytes());
        b[16..20].copy_from_slice(&(self.dim as u32).to_le_bytes());
        b[20..24].copy_from_slice(&(self.count as u32).to_le_bytes());
        b[24..28].copy_from_slice(&(self.r as u32).to_le_bytes());
        b[28..32].copy_from_slice(&self.entry.to_le_bytes());
        b[32..36].copy_from_slice(&(self.m as u32).to_le_bytes());
        b[36..40].copy_from_slice(&(self.ksub as u32).to_le_bytes());
        b[40..44].copy_from_slice(&(self.l_build as u32).to_le_bytes());
        b[44..48].copy_from_slice(&self.alpha.to_le_bytes());

        // CSR graph.
        let mut cursor = l.gnbr;
        let mut acc = 0u64;
        for i in 0..self.count {
            let o = l.goff + i * 8;
            b[o..o + 8].copy_from_slice(&acc.to_le_bytes());
            for &nb in &self.graph[i] {
                b[cursor..cursor + 4].copy_from_slice(&nb.to_le_bytes());
                cursor += 4;
            }
            acc += self.graph[i].len() as u64;
        }
        b[l.goff + self.count * 8..l.goff + self.count * 8 + 8].copy_from_slice(&acc.to_le_bytes());

        for (i, &id) in self.ids.iter().enumerate() {
            b[l.ids + i * 8..l.ids + i * 8 + 8].copy_from_slice(&id.to_le_bytes());
        }
        put_f32s(&mut b, l.codebooks, &self.codebooks);
        b[l.codes..l.codes + self.codes.len()].copy_from_slice(&self.codes);
        put_f32s(&mut b, l.raw, &self.vectors);
        std::fs::write(path, &b)
    }

    /// Load a DiskANN `.vecdb` file.
    pub fn load(path: impl AsRef<Path>) -> io::Result<DiskAnnIndex> {
        let b = std::fs::read(path)?;
        let bad = |m: &str| io::Error::new(io::ErrorKind::InvalidData, format!("diskann: {m}"));
        if b.len() < 64 || &b[0..8] != DA_MAGIC {
            return Err(bad("bad magic"));
        }
        let u32_at = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        let u64_at = |o: usize| {
            let mut a = [0u8; 8];
            a.copy_from_slice(&b[o..o + 8]);
            u64::from_le_bytes(a)
        };
        if u32_at(8) != DA_VERSION {
            return Err(bad("unsupported version"));
        }
        let metric = Metric::from_u32(u32_at(12)).ok_or_else(|| bad("bad metric"))?;
        let dim = u32_at(16) as usize;
        let count = u32_at(20) as usize;
        let r = u32_at(24) as usize;
        let entry = u32_at(28);
        let m = u32_at(32) as usize;
        let ksub = u32_at(36) as usize;
        let l_build = u32_at(40) as usize;
        let alpha = f32::from_le_bytes([b[44], b[45], b[46], b[47]]);
        if dim == 0 || m == 0 || !dim.is_multiple_of(m) {
            return Err(bad("bad dim/m"));
        }
        let dsub = dim / m;
        let goff = align16(64);
        // Total edges = last offset entry.
        if b.len() < goff + (count + 1) * 8 {
            return Err(bad("truncated"));
        }
        let edges = u64_at(goff + count * 8) as usize;
        let mut idx = DiskAnnIndex {
            dim,
            metric,
            count,
            r,
            l_build,
            alpha,
            entry,
            vectors: Vec::new(),
            ids: Vec::new(),
            graph: Vec::new(),
            deleted: vec![false; count],
            deleted_count: 0,
            m,
            dsub,
            ksub,
            codebooks: Vec::new(),
            codes: Vec::new(),
        };
        let l = idx.layout(edges);
        if b.len() < l.total {
            return Err(bad("file truncated"));
        }
        // Rebuild adjacency from CSR.
        let mut graph = Vec::with_capacity(count);
        for i in 0..count {
            let start = u64_at(l.goff + i * 8) as usize;
            let end = u64_at(l.goff + (i + 1) * 8) as usize;
            let mut row = Vec::with_capacity(end - start);
            for e in start..end {
                row.push(u32_at(l.gnbr + e * 4));
            }
            graph.push(row);
        }
        idx.graph = graph;
        idx.ids = (0..count).map(|i| u64_at(l.ids + i * 8)).collect();
        idx.codebooks = get_f32s(&b, l.codebooks, m * ksub * dsub);
        idx.codes = b[l.codes..l.codes + count * m].to_vec();
        idx.vectors = get_f32s(&b, l.raw, count * dim);
        Ok(idx)
    }

    /// Open a DiskANN `.vecdb` file as a **disk-resident** index: the graph
    /// adjacency and raw f32 vectors stay in the mmap (read lazily, per hop /
    /// at rerank), while only the small PQ codes + codebooks are copied into
    /// RAM. This is the DiskANN operating mode — RAM footprint is ~`count * m`
    /// bytes regardless of vector dimension.
    pub fn open(path: impl AsRef<Path>) -> io::Result<MmapDiskAnn> {
        let file = std::fs::File::open(path)?;
        // SAFETY: read-only mmap of a regular file held open for the call.
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let bad = |m: &str| io::Error::new(io::ErrorKind::InvalidData, format!("diskann: {m}"));
        let b: &[u8] = &mmap;
        if b.len() < 64 || &b[0..8] != DA_MAGIC {
            return Err(bad("bad magic"));
        }
        let u32_at = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        if u32_at(8) != DA_VERSION {
            return Err(bad("unsupported version"));
        }
        let metric = Metric::from_u32(u32_at(12)).ok_or_else(|| bad("bad metric"))?;
        let dim = u32_at(16) as usize;
        let count = u32_at(20) as usize;
        let r = u32_at(24) as usize;
        let entry = u32_at(28);
        let m = u32_at(32) as usize;
        let ksub = u32_at(36) as usize;
        if dim == 0 || m == 0 || !dim.is_multiple_of(m) {
            return Err(bad("bad dim/m"));
        }
        let dsub = dim / m;
        let goff = align16(64);
        if b.len() < goff + (count + 1) * 8 {
            return Err(bad("truncated"));
        }
        let edges = read_u64(b, goff + count * 8) as usize;
        let gnbr = align16(goff + (count + 1) * 8);
        let ids_off = align16(gnbr + edges * 4);
        let codebooks_off = align16(ids_off + count * 8);
        let codes_off = align16(codebooks_off + m * ksub * dsub * 4);
        let raw_off = align16(codes_off + count * m);
        if b.len() < align16(raw_off + count * dim * 4) {
            return Err(bad("file truncated"));
        }
        // Resident (RAM) tier: ids, codebooks, PQ codes.
        let ids: Vec<u64> = (0..count).map(|i| read_u64(b, ids_off + i * 8)).collect();
        let codebooks = get_f32s(b, codebooks_off, m * ksub * dsub);
        let codes = b[codes_off..codes_off + count * m].to_vec();
        Ok(MmapDiskAnn {
            _mmap: mmap,
            dim,
            metric,
            count,
            entry,
            m,
            ksub,
            goff,
            gnbr,
            raw_off,
            ids,
            codebooks,
            codes,
            _r: r,
        })
    }
}

/// A disk-resident DiskANN index: graph + raw vectors live in the mmap, only
/// the PQ codes/codebooks are RAM-resident. See [`DiskAnnIndex::open`].
pub struct MmapDiskAnn {
    _mmap: memmap2::Mmap,
    dim: usize,
    metric: Metric,
    count: usize,
    entry: u32,
    m: usize,
    ksub: usize,
    goff: usize,    // CSR offsets section
    gnbr: usize,    // CSR neighbors section
    raw_off: usize, // raw f32 section
    ids: Vec<u64>,
    codebooks: Vec<f32>,
    codes: Vec<u8>,
    _r: usize,
}

impl MmapDiskAnn {
    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn metric(&self) -> Metric {
        self.metric
    }
    pub fn len(&self) -> usize {
        self.count
    }
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Out-neighbors of node `i`, read from the mmap'd CSR (models a page read).
    fn neighbors(&self, i: u32) -> Vec<u32> {
        let b: &[u8] = &self._mmap;
        let base = self.goff + i as usize * 8;
        let start = read_u64(b, base) as usize;
        let end = read_u64(b, base + 8) as usize;
        (start..end)
            .map(|e| read_u32(b, self.gnbr + e * 4))
            .collect()
    }

    /// Raw f32 vector of node `i`, read from the mmap (the "disk" read).
    fn raw_at(&self, i: u32) -> Vec<f32> {
        let b: &[u8] = &self._mmap;
        let off = self.raw_off + i as usize * self.dim * 4;
        (0..self.dim).map(|d| read_f32(b, off + d * 4)).collect()
    }

    /// Search for the `k` nearest neighbors, traversing the mmap'd graph with
    /// PQ-approximate distances and reranking the beam with exact f32 read from
    /// the map.
    pub fn search(&self, query: &[f32], k: usize, l_search: usize) -> Vec<Hit> {
        if k == 0 || self.count == 0 {
            return Vec::new();
        }
        let processed = if self.metric == Metric::Cosine {
            normalize(query)
        } else {
            query.to_vec()
        };
        let lut = build_lut(
            &self.codebooks,
            self.dim,
            self.m,
            self.ksub,
            &processed,
            true,
        );
        let adc = |node: u32| {
            let c = &self.codes[node as usize * self.m..(node as usize + 1) * self.m];
            adc_sum(&lut, self.ksub, c)
        };
        let l = l_search.max(k);

        // Greedy beam over the disk-resident graph.
        let mut list: Vec<(f32, u32)> = vec![(adc(self.entry), self.entry)];
        let mut inserted: HashSet<u32> = HashSet::from([self.entry]);
        let mut expanded: HashSet<u32> = HashSet::new();
        loop {
            let Some(pi) = list.iter().position(|&(_, n)| !expanded.contains(&n)) else {
                break;
            };
            let p = list[pi].1;
            expanded.insert(p);
            for nb in self.neighbors(p) {
                if inserted.insert(nb) {
                    list.push((adc(nb), nb));
                }
            }
            list.sort_by(|a, b| a.0.total_cmp(&b.0));
            if list.len() > l {
                list.truncate(l);
            }
        }

        // Exact rerank of the beam (reads raw vectors from the map).
        let hib = self.metric.higher_is_better();
        let mut scored: Vec<(f32, u32)> = list
            .iter()
            .map(|&(_, node)| {
                let v = self.raw_at(node);
                let key = match self.metric {
                    Metric::L2 => l2sq_f32(&v, &processed),
                    Metric::Dot | Metric::Cosine => -dot_f32(&v, &processed),
                };
                (key, node)
            })
            .collect();
        scored.sort_by(|a, b| a.0.total_cmp(&b.0));
        scored
            .into_iter()
            .take(k)
            .map(|(key, node)| Hit {
                id: self.ids[node as usize],
                score: if hib { -key } else { key },
            })
            .collect()
    }
}

#[inline]
fn read_u32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
#[inline]
fn read_u64(b: &[u8], o: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[o..o + 8]);
    u64::from_le_bytes(a)
}
#[inline]
fn read_f32(b: &[u8], o: usize) -> f32 {
    f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

fn put_f32s(b: &mut [u8], off: usize, data: &[f32]) {
    for (i, &x) in data.iter().enumerate() {
        b[off + i * 4..off + i * 4 + 4].copy_from_slice(&x.to_le_bytes());
    }
}
fn get_f32s(b: &[u8], off: usize, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let o = off + i * 4;
            f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
        })
        .collect()
}

#[cfg(feature = "parallel")]
impl DiskAnnIndex {
    /// Run many queries concurrently (one query per rayon task).
    pub fn search_batch(&self, queries: &[Vec<f32>], k: usize, l_search: usize) -> Vec<Vec<Hit>> {
        use rayon::prelude::*;
        queries
            .par_iter()
            .map(|q| self.search(q, k, l_search))
            .collect()
    }
}

#[cfg(feature = "parallel")]
impl MmapDiskAnn {
    /// Run many queries concurrently (one query per rayon task).
    pub fn search_batch(&self, queries: &[Vec<f32>], k: usize, l_search: usize) -> Vec<Vec<Hit>> {
        use rayon::prelude::*;
        queries
            .par_iter()
            .map(|q| self.search(q, k, l_search))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clustered(n: usize, dim: usize, ncenters: usize) -> Vec<(u64, Vec<f32>)> {
        let mut s: u64 = 0xDA15_ABCD_0001_0001;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        };
        let centers: Vec<Vec<f32>> = (0..ncenters)
            .map(|_| (0..dim).map(|_| next()).collect())
            .collect();
        (0..n)
            .map(|i| {
                let c = &centers[i % ncenters];
                (i as u64, c.iter().map(|&x| x + next() * 0.1).collect())
            })
            .collect()
    }

    #[test]
    fn diskann_high_recall_vs_exact() {
        let dim = 64;
        let items = clustered(3000, dim, 25);
        let idx = DiskAnnIndex::build(&items, Metric::L2, 32, 96, 1.2, 16, 256);
        assert!(idx.avg_degree() <= 32.0 + 0.01);
        let mut flat = crate::FlatIndex::new(dim, Metric::L2, true);
        for (id, v) in &items {
            flat.add(*id, v);
        }
        let mut hit = 0;
        let mut total = 0;
        for t in 0..40 {
            let q = &items[t * 17 % items.len()].1;
            let truth: std::collections::HashSet<u64> =
                flat.search_exact(q, 10).iter().map(|h| h.id).collect();
            hit += idx
                .search(q, 10, 96)
                .iter()
                .filter(|h| truth.contains(&h.id))
                .count();
            total += truth.len();
        }
        let recall = hit as f64 / total as f64;
        assert!(recall >= 0.95, "DiskANN recall too low: {recall}");
    }

    #[test]
    fn diskann_cosine_finds_self() {
        let dim = 48;
        let items = clustered(1500, dim, 30);
        let idx = DiskAnnIndex::build(&items, Metric::Cosine, 32, 96, 1.2, 12, 256);
        for t in 0..20 {
            let (id, v) = &items[t * 13 % items.len()];
            let hits = idx.search(v, 1, 64);
            assert_eq!(hits[0].id, *id);
        }
    }

    #[test]
    fn diskann_save_load_roundtrip() {
        let dim = 48;
        let items = clustered(1200, dim, 20);
        let idx = DiskAnnIndex::build(&items, Metric::L2, 24, 64, 1.2, 12, 128);
        let mut path = std::env::temp_dir();
        path.push("vecdb_diskann_test.vecdb");
        idx.save(&path).unwrap();
        let loaded = DiskAnnIndex::load(&path).unwrap();
        assert_eq!(loaded.len(), idx.len());
        for t in 0..15 {
            let q = &items[t * 29 % items.len()].1;
            let a: Vec<u64> = idx.search(q, 10, 64).iter().map(|h| h.id).collect();
            let b: Vec<u64> = loaded.search(q, 10, 64).iter().map(|h| h.id).collect();
            assert_eq!(a, b);
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn diskann_incremental_insert_matches_search() {
        let dim = 48;
        let items = clustered(1500, dim, 20);
        // Build from the first 1000, then stream in the remaining 500.
        let idx_full = DiskAnnIndex::build(&items, Metric::L2, 32, 96, 1.2, 12, 128);
        let mut idx = DiskAnnIndex::build(&items[..1000], Metric::L2, 32, 96, 1.2, 12, 128);
        for (id, v) in &items[1000..] {
            idx.insert(*id, v);
        }
        assert_eq!(idx.len(), items.len());
        // Recall of the incrementally-grown index vs exact.
        let mut flat = crate::FlatIndex::new(dim, Metric::L2, true);
        for (id, v) in &items {
            flat.add(*id, v);
        }
        let mut hit = 0;
        let mut total = 0;
        for t in 0..40 {
            let q = &items[t * 17 % items.len()].1;
            let truth: std::collections::HashSet<u64> =
                flat.search_exact(q, 10).iter().map(|h| h.id).collect();
            hit += idx
                .search(q, 10, 96)
                .iter()
                .filter(|h| truth.contains(&h.id))
                .count();
            total += truth.len();
        }
        assert!(hit as f64 / total as f64 >= 0.90, "insert recall too low");
        // Sanity: the fully-built index has the same node count.
        assert_eq!(idx_full.len(), idx.len());
    }

    #[test]
    fn diskann_remove_and_consolidate() {
        let dim = 48;
        let items = clustered(1500, dim, 20);
        let mut idx = DiskAnnIndex::build(&items, Metric::L2, 32, 96, 1.2, 12, 128);
        let n = items.len();
        // Tombstone all ids divisible by 5; they must never appear.
        let mut removed = 0;
        for (id, _) in &items {
            if id % 5 == 0 {
                removed += idx.remove(*id);
            }
        }
        assert_eq!(idx.live_len(), n - removed);
        for t in 0..30 {
            let q = &items[t * 13 % n].1;
            assert!(idx.search(q, 10, 96).iter().all(|h| h.id % 5 != 0));
        }
        // Consolidation physically drops them; results still respect it.
        idx.consolidate();
        assert_eq!(idx.len(), n - removed);
        assert_eq!(idx.live_len(), n - removed);
        for t in 0..30 {
            let q = &items[t * 11 % n].1;
            assert!(idx.search(q, 10, 96).iter().all(|h| h.id % 5 != 0));
        }
    }

    #[test]
    fn diskann_mmap_matches_in_memory() {
        let dim = 48;
        let items = clustered(1200, dim, 20);
        let idx = DiskAnnIndex::build(&items, Metric::L2, 24, 64, 1.2, 12, 128);
        let mut path = std::env::temp_dir();
        path.push("vecdb_diskann_mmap.vecdb");
        idx.save(&path).unwrap();
        // Disk-resident view: graph + raw stay in the map, PQ codes in RAM.
        let m = DiskAnnIndex::open(&path).unwrap();
        assert_eq!(m.len(), idx.len());
        assert_eq!(m.dim(), idx.dim());
        for t in 0..15 {
            let q = &items[t * 29 % items.len()].1;
            let a: Vec<u64> = idx.search(q, 10, 64).iter().map(|h| h.id).collect();
            let b: Vec<u64> = m.search(q, 10, 64).iter().map(|h| h.id).collect();
            assert_eq!(a, b);
        }
        std::fs::remove_file(&path).ok();
    }
}
