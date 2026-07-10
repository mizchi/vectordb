# vectordb

コンパクトで組み込み向けのベクトル検索データベース。**Rust**（本命実装）と
**MoonBit**（`v128` SIMD 版の試作）の二本立てで、同じ設計・同じ `.vecdb` ファイル形式を共有する。

- **11 種のインデックス** — Flat / IVF / HNSW / int8-HNSW / PQ / OPQ / IVF+PQ / Binary / RaBitQ / IVF+RaBitQ / DiskANN
- **3 メトリクス** — `L2`（二乗距離）/ `Dot`（内積）/ `Cosine`（挿入時に正規化）
- **SIMD 距離カーネル** — Rust=AVX2（実行時分岐 + スカラ fallback）/ MoonBit=`v128`
- **mmap 単一ファイル永続化**（`.vecdb`）— 読み込みはゼロコピー
- **メタデータフィルタ / ペイロード / ソフト削除**、**rayon 並列**、依存最小

> このページは**使い方ガイド**。アルゴリズム・設計の経緯・ファイル形式の詳細は
> [`DESIGN.md`](./DESIGN.md) を参照。

---

## 目次

1. [インストール](#インストール)
2. [クイックスタート](#クイックスタート)
3. [インデックスの選び方](#インデックスの選び方)
4. [使い方ガイド](#使い方ガイド)（保存/検索・フィルタ・ペイロード・削除・大規模・並列）
5. [インデックス・リファレンス](#インデックスリファレンス)
6. [CLI リファレンス](#cli-リファレンス)
7. [ファイル形式 `.vecdb`](#ファイル形式-vecdb)
8. [ベンチマーク（実データ SIFT10K）](#ベンチマーク実データ-sift10k)
9. [MoonBit 版](#moonbit-版)
10. [プロジェクト構成](#プロジェクト構成)

---

## インストール

Rust 実装は `rust/` にある。ワークスペースではなく単体クレート。

```bash
cd rust
cargo build --release           # ライブラリ + CLI (target/release/vecdb)
cargo test                      # ユニット/結合テスト + doctest
```

依存は最小（`memmap2`、既定で `rayon`）。スレッド並列が不要なら
`--no-default-features` で `rayon` ごと外せる（`search` / `search_exact` はそのまま動く）。

同じリポジトリの別クレートから使う場合:

```toml
[dependencies]
vectordb = { path = "../vectordb/rust" }
```

---

## クイックスタート

### ライブラリ（30 秒）

```rust
use vectordb::{FlatIndex, Metric, save, open};

// 1) 索引を作って追加
let mut idx = FlatIndex::new(128, Metric::Cosine, /*keep_raw=*/true);
idx.add(1, &embedding_a);
idx.add(2, &embedding_b);

// 2) 検索（k=10, oversample=4）。返り値は Vec<Hit { id, score }>
let hits = idx.search(&query, 10, 4);

// 3) 単一ファイルに保存 → mmap でゼロコピー読み込み
save(&idx, "index.vecdb")?;
let m = open("index.vecdb")?;
let hits = m.search(&query, 10, 4);
```

`score` はメトリクスに応じて「大きいほど良い」向きに正規化される（Cosine/Dot は内積、
L2 は負の二乗距離）。

### CLI（30 秒）

入力は CSV（1 行 = `id,v0,v1,...`）。索引種別は検索時にファイル先頭のマジックで**自動判定**。

```bash
V="cargo run --release --bin vecdb --"
$V build  vecs.csv index.vecdb --metric cosine     # Flat 索引を構築
$V info   index.vecdb                              # 種別・件数・次元などを表示
$V search index.vecdb query.csv -k 10              # 各行: query_index<TAB>rank<TAB>id<TAB>score
```

---

## インデックスの選び方

まず結論の早見表。**embedding は普通クラスタ構造を持つので、多くの用途では HNSW か IVF が既定候補**。
メモリが厳しければ量子化系、RAM に載らない規模なら DiskANN。

| 状況 / 要件 | おすすめ | 理由 |
|---|---|---|
| とりあえず正確に。件数が小さい（〜数万） | **Flat (int8+rerank)** | recall 1.0、実装が単純、省メモリ |
| 低レイテンシ・高スループット重視 | **HNSW** | 全方式で最良の recall/レイテンシ |
| HNSW を省メモリで | **int8-HNSW** | HNSW とほぼ同 recall/qps、ベクトル部 1/4 |
| 大規模でスキャンを削りたい | **IVF** | nprobe セルだけ走査。recall/qps のバランス良 |
| メモリ最優先（32x 圧縮） | **PQ / OPQ** | 16 B/ベクトル。rerank 併用で recall 1.0 |
| 大規模 × 省メモリの定番 | **IVF+PQ** | 残差 PQ で 16 B のまま高 recall |
| angular で 1-bit（8x 圧縮 vs int8） | **IVF+RaBitQ** | セル毎重心 + 不偏推定。oversample で recall 1.0 |
| **RAM に載らない大規模を SSD で** | **DiskANN** | グラフ+生は mmap、RAM は PQ コードのみ（次元非依存） |

補足:
- **素の binary（1-bit, グローバル）は非負値データで壊滅**（SIFT で recall 0.037）。angular で
  1-bit を使うなら **RaBitQ**（重心を引く）を選ぶ。
- 一様ランダムデータは IVF / PQ 系の最悪ケース。実 embedding のクラスタ構造が前提。
- `Binary` / `RaBitQ` / `IVF+RaBitQ` は現状 **Rust ライブラリ内のみ**（`.vecdb` 永続化・CLI は未対応）。
  永続化・CLI があるのは Flat / IVF / HNSW / int8-HNSW / PQ / OPQ / IVF+PQ / DiskANN。

---

## 使い方ガイド

### 構築 → 保存 → mmap 検索（基本形）

すべての索引は「構築 → `.vecdb` に保存 → mmap で読み戻して検索」という同じ流れ。
Flat は `save` / `open`（mmap）/ `load`（所有）、他の索引は型ごとの `save` / `load` を持つ。

```rust
use vectordb::{FlatIndex, Metric, save, open};

let mut idx = FlatIndex::new(dim, Metric::Cosine, true);
for (id, v) in &items { idx.add(*id, v); }
save(&idx, "index.vecdb")?;

let m = open("index.vecdb")?;              // mmap（ゼロコピー、大きな索引でも即座に開く）
let hits = m.search(&query, 10, 4);
```

`keep_raw=false`（CLI では `--compact`）にすると生 f32 を捨て int8 のみになる（rerank なし、
最小サイズ）。

### メトリクスの選択

| メトリクス | 用途 | 挙動 |
|---|---|---|
| `Metric::Cosine` | 正規化済み埋め込み・角度類似 | 挿入時にベクトルを正規化して内積 |
| `Metric::Dot` | 内積（MIPS） | 生の内積 |
| `Metric::L2` | ユークリッド距離 | 二乗 L2（小さいほど近い） |

### メタデータフィルタ

述語 `Fn(u64) -> bool` を渡すと、条件を満たす id だけを候補にして上位 k を返す
（**全インデックス対応**）。

```rust
let hits = flat.search_filter(&query, 10, 4, |id| id % 2 == 0);
let hits = ivf.search_filter(&query, 10, /*nprobe*/16, 4, |id| allow.contains(&id));
let hits = hnsw.search_filter(&query, 10, /*ef*/128, |id| id < 1000);
let hits = pq.search_filter(&query, 10, /*oversample*/8, |id| id % 2 == 0);
```

Flat/IVF/PQ 系は候補に入れる前に非該当 id をスキップ、HNSW 系はグラフ探索は続けつつ**結果への
採用**だけを絞る。選択率が低い（ヒットが少ない）ときは `ef_search` / `oversample` を大きめに。

### ペイロード（メタデータ添付）

各ベクトルに任意のバイト列を紐付けて保存できる（検索は id を返すので id→payload を引く形）。
`.vecdb` に**永続化**され、mmap でもブロブをゼロコピー参照する。

```rust
idx.add_with_payload(1, &emb, br#"{"title":"..."}"#);
let meta: Option<&[u8]> = idx.payload(1);
save(&idx, "index.vecdb")?;
let m = open("index.vecdb")?;               // mmap でも payload(1) が引ける
```

### 削除（ソフト削除 → compact）

`remove(id)` で論理削除（検索から除外）、物理削除で領域回収。**Flat / IVF / HNSW / DiskANN**
が対応（`live_len()` で生存数）。

```rust
idx.remove(42);            // tombstone（検索結果から消える。全索引共通）
idx.live_len();            // 生存件数
idx.compact();             // 物理削除（Flat/IVF/HNSW。HNSW は生存集合から再構築）
```

- **Flat / IVF / HNSW** は tombstone を `.vecdb` に**永続化**（削除が無ければ従来とバイト同一）。
  物理削除してから保存したい場合は `save()` 前に `compact()`。
- **DiskANN** は物理削除が `consolidate()`（生存集合で Vamana を再構築）。[DiskANN の項](#diskann--vamana)を参照。

### 大規模データ（RAM に載らない）— DiskANN + ストリーミング構築

DiskANN はグラフ隣接と生 f32 を mmap のまま検索し、RAM には PQ コードだけを置く
（フットプリントは**次元非依存**の `count·m` バイト）。さらに `build_streaming` は、
**構築時にも生ベクトルを全部 RAM に載せない**低メモリ構築を提供する。

```rust
use vectordb::{DiskAnnIndex, Metric};

// 生ベクトルを一度も全常駐させずに .vecdb を構築（items は所有イテレータ＝実ストリーム源でも可）
DiskAnnIndex::build_streaming(
    "index.diskann.vecdb", dim, Metric::L2,
    /*R=*/32, /*l_build=*/96, /*alpha=*/1.2, /*pq_m=*/16, /*ksub=*/256,
    /*sample=*/50_000, items.into_iter(),
)?;

let disk = DiskAnnIndex::open("index.diskann.vecdb")?;   // mmap 常駐
let hits = disk.search(&query, 10, /*l_search=*/64);
```

CLI からも `build-diskann --streaming` で CSV を 1 行ずつ遅延読みして構築するので、
生ベクトルは非常駐（40k×256 合成データで peak RSS 137 MiB → 26 MiB, 約 5.3x 削減）。
詳細は [DiskANN の項](#diskann--vamana) を参照。

### 並列（`parallel` フィーチャ, 既定 ON）

`rayon` で 2 種類の並列を提供する。

| API | 種類 | 4 コアでの効果（参考） |
|---|---|---|
| `search_batch` | クエリ間（1 クエリ / タスク） | **約 3.3〜3.8x**（ほぼ線形）。スループット向け |
| `search_parallel` | クエリ内（1 クエリのスキャンを分割） | 大規模 N のみ有効（50k=1.0x, 400k=約 2x） |

```rust
let results = idx.search_batch(&queries, 10, 4);   // Vec<Vec<Hit>>、コア分散
let hits    = idx.search_parallel(&query, 10, 4);  // 1 クエリを分割
```

`search_batch` は**全インデックスに実装**。各索引の `search` と同じ引数列に
`queries: &[Vec<f32>]` を渡す形（例: `pq.search_batch(&qs, 10, 8)`,
`ivfpq.search_batch(&qs, 10, 16, 16)`）。`search_parallel` は Flat / IVF が対象で、
`PAR_THRESHOLD`（=131072 件）未満は自動的にシリアルへフォールバックする。

`FlatIndex::add_batch(&[(id, vec)])` は正規化 + 量子化を rayon で並列化して一括追加する
（結果は入力順で決定的）。

---

## インデックス・リファレンス

各方式の「いつ使うか・主要パラメータ・最小コード」。数値は 50k×128（cosine, ローカル参考値）。

### Flat（総当り, `index.rs`）

int8 量子化（`scale = max|x|/127`, メモリ f32 比 1/4）で全件を SIMD スキャンし、
`k × oversample` 件を粗選別 → 生 f32 で **rerank** して上位 k。実データでも recall 1.0 の安全な既定。

```rust
let mut idx = FlatIndex::new(dim, Metric::Cosine, /*keep_raw=*/true);
idx.add(1, &embedding);
let hits = idx.search(&query, 10, /*oversample=*/4);
```

### IVF（転置インデックス, `ivf.rs`）

k-means で `nlist` セルに分割し、クエリに近い `nprobe` セルだけ走査。ベクトルはセル順（CSR）
で連続格納。セル内は Flat と同じ int8 + rerank。

```rust
let ivf = IvfIndex::build(dim, Metric::Cosine, /*nlist=*/256, &items, /*keep_raw=*/true, /*iters=*/12);
let hits = ivf.search(&query, 10, /*nprobe=*/4, /*oversample=*/8);
ivf.save("index.ivf.vecdb")?;              // VECDBIV1
```

| nprobe | ms/query | recall@10 | 対 flat |
|---|---|---|---|
| 1 | 0.026 | 0.97 | 26.5x |
| 4 | 0.045 | **1.00** | 14.9x |
| 8 | 0.061 | 1.00 | 11.0x |

### HNSW（多層グラフ, `hnsw.rs`）

Navigable Small World グラフ（Malkov & Yashunin）。上層から貪欲降下 → 層 0 で幅 `ef_search`
のビーム探索。**recall/レイテンシは全方式で最良**（エッジ分メモリを使う）。

```rust
let mut idx = HnswIndex::new(dim, Metric::Cosine, /*m=*/16, /*ef_construction=*/200);
idx.add(1, &embedding);
let hits = idx.search(&query, 10, /*ef_search=*/64);
idx.save("graph.hnsw.vecdb")?;             // VECDBHN1
```

| ef_search | ms/query | recall@10 |
|---|---|---|
| 32 | 0.049 | 0.994 |
| 64 | 0.069 | 0.9995 |
| 128 | 0.103 | 1.0000 |

### int8 グラフ HNSW（`hnsw_q.rs` の `HnswQIndex`）

HNSW と同じグラフだが、各ノードを int8 コード + per-vector scale/sqnorm で保持
（≈`dim+8` B vs f32 の `dim*4`、ベクトル部で約 1/4）。**構築・探索とも量子化空間**で行い、
`keep_raw` 時は最終ビームだけ f32 で rerank して recall を回復。

```rust
let mut idx = HnswQIndex::new(dim, Metric::L2, /*m=*/16, /*ef_construction=*/200, /*keep_raw=*/true);
idx.add(1, &embedding);
let hits = idx.search(&query, 10, /*ef_search=*/96);
idx.save("graph.hnswq.vecdb")?;            // VECDBHQ1
```

### Product Quantization（`pq.rs` の `PqIndex`）

`dim` を `m` 個のサブベクトルに分割し、サブ空間ごとに k-means コードブック（`ksub` 重心, `ksub<=256`）
で量子化。1 ベクトル = `m` バイト（128 次元 → `m=16` で 16 B = **32x 圧縮**）。検索は **ADC**
（クエリは全精度、`m*ksub` の距離テーブルを前計算し `m` 回の参照和で近似距離）。`keep_raw` で rerank 可。

```rust
let pq = PqIndex::build(&items, Metric::L2, /*m=*/16, /*ksub=*/256, /*iters=*/20, /*keep_raw=*/true);
let hits = pq.search(&query, 10, /*oversample=*/16);
pq.save("index.pq.vecdb")?;                // VECDBPQ1
```

### OPQ（回転付き PQ, `opq.rs` の `OpqIndex`）

PQ の前に**直交回転 `R` を学習**してサブ空間のエネルギーを均し量子化誤差を下げる（Ge et al. 2013）。
交互最適化：(1) 現在の回転で PQ 学習 + 再構成、(2) 直交 Procrustes 解で `R` 更新（SVD は依存無しの
自作 Jacobi 固有値分解）。`R` は直交なので距離を保存し、検索は「クエリを回転 → ADC」だけ。

```rust
let opq = OpqIndex::build(&items, Metric::L2, 16, 256, /*iters=*/20, /*opq_iters=*/4, /*keep_raw=*/true);
let hits = opq.search(&query, 10, 16);
opq.save("index.opq.vecdb")?;              // VECDBOP1
```

### IVF + PQ（`ivf_pq.rs` の `IvfPqIndex`）

粗い k-means でセル割当し、**残差 `x - 重心` を共有 PQ コードブックで量子化**（Faiss `IVFPQ`）。
残差は中心化されているので同じ PQ 予算でも精度が高い。`nprobe` セルを探索 → 残差クエリで ADC → rerank。

```rust
let idx = IvfPqIndex::build(&items, Metric::L2, /*nlist=*/256, 16, 256, /*iters=*/15, /*keep_raw=*/true);
let hits = idx.search(&query, 10, /*nprobe=*/16, /*oversample=*/16);
idx.save("index.ivfpq.vecdb")?;            // VECDBIP1
```

### Binary（1-bit）/ RaBitQ / IVF+RaBitQ（`bin_quant.rs`, `rabitq.rs`）

いずれも 1-bit/dim の angular 向け量子化（**f32 比 32x, int8 比 8x 圧縮**）。現状 **Rust ライブラリ内のみ**
（永続化・CLI 未対応）。

- **Binary**: 符号ビットを `u64` にパックしハミング距離で粗選別 → f32 rerank。非負値データでは符号が
  潰れて精度が出ないので用途を選ぶ。
- **RaBitQ**（Gao & Long, SIGMOD 2024）: 重心を引いてランダム回転 → 符号ビット + 補正で内積を**不偏推定**。
  クエリは 4-bit プレーン量子化 + popcount。
- **IVF+RaBitQ**: セル毎重心で残差を小さくすると RaBitQ 推定量が本領を発揮（論文の標準構成）。

```rust
let bin = BinaryIndex::build(dim, Metric::Cosine, &items, /*keep_raw=*/true);
let rq  = RabitqIndex::build(dim, Metric::Cosine, &items, /*keep_raw=*/true, /*seed=*/1);
let idx = IvfRabitqIndex::build(dim, Metric::Cosine, 256, &items, true, 12, /*seed=*/1);
let hits = idx.search(&query, 10, /*nprobe=*/16, /*oversample=*/32);
```

IVF+RaBitQ, 50k, nprobe=16, **1-bit コード 781 KiB（int8 の 1/8）**:

| oversample | ms/query | recall@10 |
|---|---|---|
| 16 | 0.58 | 0.986 |
| 32 | 0.62 | **1.0000** |

### DiskANN / Vamana

大規模・省メモリ向けの**単層グラフ**（Subramanya et al. 2019）。HNSW の階層を捨て、`RobustPrune`
（α>1 の枝刈りで遠距離ショートカットを残す）で少ホップ到達を狙う。**PQ コードは RAM 常駐**で
探索の近似距離に使い（1 ホップ = テーブル参照 1 回）、**生 f32 は "ディスク層"** として最終 rerank
でのみ読む。

```rust
// (items, metric, R=最大次数, l_build, alpha, pq_m, ksub)
let idx = DiskAnnIndex::build(&items, Metric::L2, 32, 96, 1.2, 16, 256);
idx.save("index.diskann.vecdb")?;          // VECDBDA1

// ディスク常駐モード: グラフ+生 f32 は mmap のまま、RAM は PQ コードのみ（次元非依存の count·m B）
let disk = DiskAnnIndex::open("index.diskann.vecdb")?;   // -> MmapDiskAnn
let hits = disk.search(&query, 10, /*l_search=*/64);
```

**省メモリ / ストリーミング構築（`build_streaming`）**: 生ベクトルを RAM に全部載せずに `.vecdb` を
生成する。Pass1 で処理済みベクトルを一時ファイルへ逐次書き出し（平均・id・有限サンプルのみ蓄積）
→ サンプルで PQ 学習 → Pass2 で逐次読み直して符号化 + medoid 決定（常駐は 1 本ずつ）→ グラフ構築は
**SDC（対称距離計算＝PQ セントロイド対の距離表）**で生ベクトル不要 → 生 f32 は一時ファイルから
ストリームコピー。ピーク常駐は `O(count·m + m·ksub² + edges)` で次元非依存。

グラフ幾何は PQ 近似（厳密 L2 の `build` より低品質）だが、探索時のビーム rerank は厳密 f32 なので
実効 recall は高い。実 SIFT10K での検証（`examples/dogfood_streaming.rs`, fvecs を遅延ストリーム）:

```
streaming(SDC) + mmap  L=64  recall@10=0.9970   # 生ベクトル非常駐
in-memory(exact-L2)    L=64  recall@10=0.9990   # 全 raw 常駐
```

L=128 で厳密構築と完全一致、L=32 で約 1% 差。省メモリの旨味は生行列 `n·dim·4` が SDC テーブル
`m·ksub²`（n 非依存）を上回る大規模で顕著（SIFT10K/128d のような小規模では両者が拮抗する）。

**逐次更新（FreshDiskANN 相当）**: 全再構築なしの `insert` / `remove` / `consolidate`。

```rust
idx.insert(id, &vector);   // 既存グラフへ Vamana 挿入（O(l_build·R)、PQ は再学習しない）
idx.remove(id);            // tombstone（検索から除外、グラフは通過して連結性維持）
idx.live_len();            // 生存件数
idx.consolidate();         // 削除点を物理削除して生存集合で再構築
```

---

## CLI リファレンス

入力 CSV は `id,v0,v1,...`（`#` 始まりと空行は無視）。検索時に索引種別を自動判定。

```bash
V="cargo run --release --bin vecdb --"

# 構築（--compact で生 f32 を捨て最小サイズ = rerank なし）
$V build         vecs.csv flat.vecdb  --metric cosine [--compact]
$V build-ivf     vecs.csv ivf.vecdb   --metric cosine --nlist 256 [--iters 12]
$V build-hnsw    vecs.csv hnsw.vecdb  --metric cosine -m 16 --ef-construction 200
$V build-hnsw-q  vecs.csv hnswq.vecdb --metric cosine -m 16 --ef-construction 200 [--compact]
$V build-pq      vecs.csv pq.vecdb    --metric cosine --pq-m 16 --ksub 256 [--iters 20] [--compact]
$V build-opq     vecs.csv opq.vecdb   --metric cosine --pq-m 16 --ksub 256 --opq-iters 4 [--compact]
$V build-ivfpq   vecs.csv ivfpq.vecdb --metric cosine --nlist 256 --pq-m 16 --ksub 256 [--compact]
$V build-diskann vecs.csv da.vecdb    --metric cosine -r 32 --l-build 96 --alpha 1.2 --pq-m 16 --ksub 256
$V build-diskann vecs.csv da.vecdb    --streaming --sample 50000       # 省メモリ構築（生を非常駐）

# 情報表示
$V info hnsw.vecdb

# 検索（索引種別に応じたチューニングフラグ）
$V search flat.vecdb  query.csv -k 10 --oversample 4
$V search ivf.vecdb   query.csv -k 10 --nprobe 16 --oversample 8
$V search hnsw.vecdb  query.csv -k 10 --ef 64
$V search pq.vecdb    query.csv -k 10 --oversample 8
$V search ivfpq.vecdb query.csv -k 10 --nprobe 16 --oversample 16
$V search da.vecdb    query.csv -k 10 --ef 64          # DiskANN は --ef = l_search、自動で mmap 経路
```

検索出力は 1 行 = `query_index<TAB>rank<TAB>id<TAB>score`。

| コマンド | 主要フラグ | メモ |
|---|---|---|
| `build` | `--metric` `--compact` | Flat（int8+rerank） |
| `build-ivf` | `--nlist` `--iters` | 0 or 未指定の nlist は ~√n |
| `build-hnsw` | `-m` `--ef-construction` | |
| `build-hnsw-q` | `-m` `--ef-construction` `--compact` | int8 グラフ |
| `build-pq` | `--pq-m` `--ksub` `--iters` `--compact` | pq-m は dim を割り切ること |
| `build-opq` | `--pq-m` `--ksub` `--opq-iters` `--compact` | |
| `build-ivfpq` | `--nlist` `--pq-m` `--ksub` `--compact` | |
| `build-diskann` | `-r` `--l-build` `--alpha` `--pq-m` `--ksub` `--streaming` `--sample` | |
| `search` | `-k` `--oversample` `--nprobe` `--ef` | 種別自動判定 |
| `info` | — | 種別・件数・次元・メトリクスなど |

---

## ファイル形式 `.vecdb`

- 単一ファイル。先頭 8 バイトのマジックで種別を判定。64B ヘッダ + 16B 整列セクション、リトルエンディアン。
- 読み込みは **mmap ゼロコピー**（`open` / 各索引の `load`）。

| マジック | 索引 | 永続化される主なもの |
|---|---|---|
| `VECDB1\0\0` | Flat | int8 コード（+ 任意で生 f32 / tombstone / payload） |
| `VECDBIV1` | IVF | centroids + セル CSR（+ tombstone） |
| `VECDBHN1` | HNSW | 多層グラフ（+ tombstone） |
| `VECDBHQ1` | int8-HNSW | int8 グラフ |
| `VECDBPQ1` | PQ | コードブック + PQ コード |
| `VECDBOP1` | OPQ | 回転 `R` + コードブック + コード |
| `VECDBIP1` | IVF+PQ | centroids + 残差 PQ コード |
| `VECDBDA1` | DiskANN | CSR グラフ + PQ コード + 生 f32 |

tombstone / payload の無い Flat 索引は従来と**バイト完全一致**（MoonBit 互換維持）。
形式の詳細レイアウトは [`DESIGN.md`](./DESIGN.md)。

---

## ベンチマーク（実データ SIFT10K）

**SIFT10K**（10,000×128 の SIFT 特徴、100 クエリ、厳密 100-NN 正解、metric=L2）での recall@10 /
スループット。`examples/eval.rs` で計測。

```bash
cd rust
curl -fsSL https://huggingface.co/datasets/vecdata/siftsmall/resolve/main/siftsmall.tar.gz | tar xz
cargo run --release --example eval -- siftsmall
```

| 方式 | recall@10 | ms/query | qps |
|---|---|---|---|
| flat exact f32 | 1.0000 | 0.26 | 3.8k |
| flat int8+rerank (o=8) | **1.0000** | 0.14 | 7.3k |
| binary 1-bit+rerank (o=32) | **0.0370** | 0.05 | 18.8k |
| RaBitQ flat+rerank (o=32) | **0.9990** | 0.31 | 3.2k |
| PQ m=16 (o=32) | **1.0000** | 0.21 | 4.8k |
| OPQ m=16 (o=32) | **1.0000** | 0.24 | 4.1k |
| IVF nprobe=16 (o=8) | 0.9910 | 0.051 | 19.4k |
| IVF+RaBitQ nprobe=16 (o=32) | 0.9910 | 0.29 | 3.5k |
| IVF+PQ nprobe=16 (o=16) | 0.9910 | 0.40 | 2.5k |
| HNSW efSearch=32 | 0.9940 | 0.031 | 31.9k |
| HNSW efSearch=128 | 1.0000 | 0.093 | 10.8k |
| HNSW-q(int8) efSearch=64 | 0.9970 | 0.042 | 23.9k |
| DiskANN L=64 | 0.9990 | 0.186 | 5.4k |
| DiskANN L=128 | 1.0000 | 0.366 | 2.7k |

読み取れること（実データならでは）:
- **int8+rerank は実データでも recall 1.0** — 省メモリの安全な既定。
- **素の binary は recall 0.037 で壊滅** — SIFT は非負値なので符号ビットが潰れる。**RaBitQ は重心を引くので
  0.999** と、naive binary に対する優位が実データで明確。
- **PQ / OPQ は 16 B/ベクトル（32x 圧縮）で rerank 併用 recall 1.0**。SIFT は次元バランスが良く OPQ の
  回転効果は小さいが、歪んだ分布では効く。
- **IVF+PQ は残差量子化で 16 B のまま nprobe=16 で 0.991**（大規模・省メモリの定番）。
- **HNSW が最高スループット**（recall 0.994 で 32k qps、flat の約 8 倍）。**int8-HNSW** はそれをベクトル部
  1/4 メモリでほぼ再現（0.997 で 23.9k qps）。
- **DiskANN は L=64 で recall 0.999**。in-memory 10k では HNSW が速い（PQ 近似 + rerank のオーバーヘッド分）
  ——DiskANN の真価は **RAM に載らない大規模を SSD で捌く**領域。

---

## MoonBit 版

`v128` SIMD で距離カーネルを書いた試作。Rust と同じ設計・同じ `.vecdb` 形式を共有し、Flat / IVF / HNSW /
int8-HNSW / PQ / IVF+PQ / OPQ と各量子化（int8 / binary / RaBitQ / IVF+RaBitQ）を移植済み。
フィルタ付き検索も全索引で使える。

```bash
cd moonbit
moon test              # 全索引/量子化のテスト（既定 = wasm）
moon run cmd/main      # デモ
moon run cmd/bench --target native --release   # 統合ベンチ（recall/ms/query）
```

```moonbit
let idx = @vectordb.FlatIndex::build(vectors, ids, @vectordb.Cosine, true)
let hits = idx.search(query, 10, 4)   // Array[Hit{ id, score }]

// .vecdb シリアライズ（Rust とバイト互換）
let bytes = idx.to_bytes()                    // FixedArray[Byte]
let restored = @vectordb.FlatIndex::from_bytes(bytes)

// フィルタ付き検索（全索引）
let hits = flat.search_filter(query, 10, 4, fn(id) { id % 2L == 0L })
```

### SIMD の効くターゲット

SIMD intrinsic はスカラ fallback を持つため `native` / `wasm` / `js` すべてで動くが、
**v128 が実ハードウェア SIMD に落ちるのは `wasm` ターゲットのみ**。そのため `moon.mod` の
`preferred_target = "wasm"` を既定にしている（native/tcc・clang・nightly llvm はいずれも v128 を
x86 SIMD へ落とさずスカラ実行する）。

### Rust vs MoonBit(wasm) — 50k×128, cosine, k=10, ms/query

| 方式 | Rust (AVX2) | MoonBit (wasm/v128) | 倍率 |
|---|---|---|---|
| int8 + rerank | 0.61 | 1.51 | 約 2.5x |
| exact f32 | 1.06 | 2.78 | 約 2.6x |
| recall@10 | 1.0 | 1.0 | 同 |

チューニングで MoonBit(wasm) は初版 **約 24ms → 1.5ms（約 16 倍）**、Rust 比 約 40 倍差 → **約 2.4 倍差**まで
短縮。効いた最適化は「全件ソート → 上限付き top-k ヒープ」「int8 内積 8→16 要素/反復」「スキャンループの
フィールドをローカル退避」。残差は主に SIMD 幅差（AVX2=256bit vs wasm v128=128bit）と wasm ランタイムの
オーバーヘッド。

### Rust ⇄ MoonBit の相互運用

MoonBit の `to_bytes` は Rust の `save` と**バイト単位で同一**（決定的な Flat 形式）。他の索引も同じマジックの
レイアウトを共有し相互に読めるが、k-means の乱数列が言語間で異なるため生成バイト列は一般に一致しない
（どちらが書いたファイルも相手の `from_bytes`/`load` で読める）。

```bash
cd rust    && cargo run --release --example dump  # 小索引を save→hex
cd moonbit && moon run cmd/dump --target wasm     # 同内容を to_bytes→hex（192 B 完全一致）
```

MoonBit の wasm ランタイムにファイルシステムは無いため、入出力はバイト列（`to_bytes`/`from_bytes`）で扱い、
ファイル I/O はホスト側が担当する。

---

## プロジェクト構成

### Rust（`rust/src`）

- `distance.rs` — f32/int8 距離（スカラ + AVX2, int8 は 32 要素/反復）
- `quantize.rs` — int8 スカラ量子化
- `index.rs` — Flat 検索 + rerank（`View` に集約し owned/mmap 共有）+ rayon 並列
- `hnsw.rs` / `hnsw_q.rs` — HNSW / int8 グラフ HNSW
- `ivf.rs` — IVF（k-means + nprobe）+ save/load + 並列
- `bin_quant.rs` / `rabitq.rs` — binary(1-bit) / RaBitQ + IVF+RaBitQ
- `pq.rs` / `opq.rs` / `ivf_pq.rs` — PQ（共有プリミティブ）/ OPQ / IVF+PQ
- `diskann.rs` — DiskANN/Vamana（単層グラフ + RobustPrune + PQ 常駐 + rerank + streaming build）
- `storage.rs` — `.vecdb` の save / mmap open / load（tombstone/payload 永続化含む）
- `bin/vecdb.rs` — CLI、`examples/` — bench / eval / dump / dogfood_streaming

### MoonBit（`moonbit/`）

`distance.mbt` / `quantize.mbt` / `index.mbt` / `ivf.mbt` / `hnsw.mbt` / `hnsw_q.mbt` /
`bin_quant.mbt` / `rabitq.mbt` / `ivf_rabitq.mbt` / `pq.mbt` / `ivf_pq.mbt` / `opq.mbt` /
`storage.mbt`（`.vecdb` 相互運用）。

設計の経緯・アルゴリズムの詳細・調査ノートは [`DESIGN.md`](./DESIGN.md)。

## ライセンス

MIT OR Apache-2.0
