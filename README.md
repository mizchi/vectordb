# vectordb

コンパクトなベクトル検索に特化した組み込み向けDBの試作。**Rust**（本命）と
**MoonBit**（`v128` SIMD の試作）の二本立てで、同じ設計・同じ考え方で実装している。

- 方式: **Flat（総当り） + int8 スカラ量子化 + rerank**
- SIMD: Rust=AVX2（`std::arch`、実行時分岐）/ MoonBit=`moonbitlang/core/v128`
- 永続化: **mmap 単一ファイル**（`.vecdb`, Rust側）
- 目標: 小さいコード・小さいメモリ・依存最小

調査（先行実装・アルゴリズム・量子化・最適化手法）と設計・ファイル形式は
[`DESIGN.md`](./DESIGN.md) を参照。

## 仕組み

1. ベクトルを int8 に量子化（`scale = max|x|/127`）。メモリは f32 比 1/4。
2. 検索は全件を int8 で SIMD スキャンし、`k × oversample` 件を粗選別。
3. 生 f32 を残していれば、その候補を厳密距離で **rerank** し上位 k を返す
   （最小構成では f32 を捨てて int8 のみ＝rerank なしにもできる）。

メトリクス: `L2`（二乗距離）/ `Dot`（内積）/ `Cosine`（挿入時に正規化）。

## Rust

```bash
cd rust
cargo test                 # ユニット/結合テスト + doctest
cargo run --release --example bench   # 速度・recall・メモリの目安

# CLI（CSV: 1行 = id,v0,v1,...）。索引種別は検索時にマジックで自動判定。
V="cargo run --release --bin vecdb --"
$V build        vecs.csv flat.vecdb --metric cosine         # Flat（int8+rerank）
$V build-ivf    vecs.csv ivf.vecdb  --metric cosine --nlist 256
$V build-hnsw   vecs.csv hnsw.vecdb --metric cosine -m 16 --ef-construction 200
$V build-hnsw-q vecs.csv hnswq.vecdb --metric cosine        # int8グラフHNSW（省メモリ）
$V build-pq     vecs.csv pq.vecdb    --metric cosine --pq-m 16 --ksub 256  # Product Quantization
$V build-opq    vecs.csv opq.vecdb   --metric cosine --pq-m 16 --opq-iters 4  # 回転付きPQ
$V build-ivfpq  vecs.csv ivfpq.vecdb --metric cosine --nlist 256 --pq-m 16    # IVF+PQ
$V build-diskann vecs.csv da.vecdb   --metric cosine -r 32 --alpha 1.2        # DiskANN/Vamana
$V info   hnsw.vecdb
$V search flat.vecdb  query.csv -k 10 --oversample 4
$V search ivf.vecdb   query.csv -k 10 --nprobe 16
$V search hnsw.vecdb  query.csv -k 10 --ef 64
$V search pq.vecdb    query.csv -k 10 --oversample 8
$V search ivfpq.vecdb query.csv -k 10 --nprobe 16 --oversample 16
$V search da.vecdb    query.csv -k 10 --ef 64   # --ef = l_search
# 最小サイズ（rerankなし）: build 系の各コマンドに --compact
```

ライブラリAPI:

```rust
use vectordb::{FlatIndex, Metric, save, open};

let mut idx = FlatIndex::new(128, Metric::Cosine, /*keep_raw=*/true);
idx.add(1, &embedding);
let hits = idx.search(&query, 10, /*oversample=*/4); // Vec<Hit{id, score}>

save(&idx, "idx.vecdb")?;          // 単一ファイルに保存
let m = open("idx.vecdb")?;        // mmap でゼロコピー読み込み
let hits = m.search(&query, 10, 4);

// 並列（parallel フィーチャ、既定ON。rayon）
let results = idx.search_batch(&queries, 10, 4);   // 複数クエリをコア分散（スループット）
let hits = idx.search_parallel(&query, 10, 4);     // 1クエリを分割（大規模Nのみ有効）
```

ベンチ例（50k×128, Cosine, ローカル参考値）: int8+rerank が exact の約 2〜3 倍速、
recall@10 ≒ 1.0、メモリは int8 コード 6MiB（f32 なら 24MiB）。

### 並列化（`parallel` フィーチャ, 既定ON）

`rayon` で2種類の並列を提供:

| API | 種類 | 4コアでの効果（参考） |
|---|---|---|
| `search_batch` | クエリ間（1クエリ/タスク） | **約3.3〜3.8x**（ほぼ線形）。スループット向け |
| `search_parallel` | クエリ内（1クエリのスキャンを分割） | 大規模 N のみ有効（50k=1.0x, 400k=約2x） |

`search_parallel` は `PAR_THRESHOLD`（=131072 件）未満では自動的にシリアルにフォール
バックする（小規模では fork/join オーバーヘッドが上回るため）。スレッド並列が不要なら
`--no-default-features` で `rayon` 依存ごと外せる（`search`/`search_exact` はそのまま利用可）。

`search_batch`（クエリ間並列）は**全インデックスに実装**（Flat / IVF / HNSW / int8-HNSW /
PQ / IVF+PQ / OPQ / RaBitQ / IVF+RaBitQ / binary / DiskANN）。各 index の `search` と同じ
引数列に `queries: &[Vec<f32>]` を渡す形（例: `pq.search_batch(&qs, 10, 8)`,
`ivfpq.search_batch(&qs, 10, 16, 16)`, `diskann.search_batch(&qs, 10, 64)`）。

### IVF（転置インデックス, `ivf.rs`）

大規模向けに、k-means でベクトルを `nlist` セルに分割し、検索時はクエリに近い
`nprobe` セルだけを走査する。全件スキャンを避けるので桁違いに速い（データにクラスタ
構造がある前提。実際の embedding は該当する）。セル内は Flat と同じ int8 スキャン +
f32 rerank。ベクトルはセル順（CSR）で格納し連続アクセス。

```rust
use vectordb::{IvfIndex, Metric};
let ivf = IvfIndex::build(dim, Metric::Cosine, /*nlist=*/256, &items, /*keep_raw=*/true, /*kmeans_iters=*/12);
let hits = ivf.search(&query, 10, /*nprobe=*/4, /*oversample=*/8);
ivf.save("index.ivf.vecdb")?;                 // 永続化（centroids/セル込み）
let ivf = IvfIndex::load("index.ivf.vecdb")?; // mmap で読み込み→所有
let hits = ivf.search_parallel(&query, 10, 32, 8);   // 大規模時セル走査を並列
let batch = ivf.search_batch(&queries, 10, 4, 8);    // クエリ間並列
```

ベンチ例（50k×128, クラスタ構造あり, 対 flat int8+rerank）:

| nprobe | ms/query | recall@10 | 高速化 |
|---|---|---|---|
| 1 | 0.026 | 0.97 | 26.5x |
| 4 | 0.045 | **1.00** | 14.9x |
| 8 | 0.061 | 1.00 | 11.0x |

（一様ランダムデータは IVF の最悪ケースで recall が出ない点に注意）

IVF は `save`/`load`（IVF 用 `.vecdb`, magic `VECDBIV1`, centroids + セル offsets 込み）、
`search_parallel`（セル走査を rayon 分割）、`search_batch`（クエリ間並列）に対応。

### HNSW（グラフ, `hnsw.rs`）

多層 Navigable Small World グラフ（Malkov & Yashunin）。上層から貪欲降下 →
層0で幅 `ef_search` のビーム探索。**recall/レイテンシは全方式で最良**（グラフの
エッジ分メモリを使う）。`m`（層あたりエッジ）, `ef_construction`（構築ビーム）,
`ef_search`（探索ビーム）。近傍選択はヒューリスティック（多様性重視）。

```rust
use vectordb::{HnswIndex, Metric};
let mut idx = HnswIndex::new(dim, Metric::Cosine, /*m=*/16, /*ef_construction=*/200);
idx.add(1, &embedding);
let hits = idx.search(&query, 10, /*ef_search=*/64);
idx.save("graph.hnsw.vecdb")?;              // グラフを永続化（magic VECDBHN1）
let idx = HnswIndex::load("graph.hnsw.vecdb")?;
```

`FlatIndex::add_batch(&[(id, vec)])` は正規化+量子化を rayon で並列化して一括追加する
（`parallel` フィーチャ時。結果は入力順に追記され決定的）。

ベンチ例（50k×128, cosine, 構築 ~8s）:

| ef_search | ms/query | recall@10 |
|---|---|---|
| 32 | 0.049 | 0.994 |
| 64 | 0.069 | 0.9995 |
| 128 | 0.103 | 1.0000 |

→ **recall 0.99 を 0.05 ms/query**（flat の ~12倍速）。最速レイテンシ。

### binary（1-bit）量子化, `bin_quant.rs`

各次元を符号ビットに落として `u64` にパック（**f32 比 32x, int8 比 8x 圧縮**）。
ハミング距離（`popcount(xor)`）で粗選別し、f32 で rerank。angular（cosine/dot）向け。

```rust
use vectordb::{BinaryIndex, Metric};
let bin = BinaryIndex::build(dim, Metric::Cosine, &items, /*keep_raw=*/true);
let hits = bin.search(&query, 10, /*oversample=*/16);
```

ベンチ例（50k×128, クラスタ構造）: `binary+rerank(o=16)` ≈ 0.46 ms/query, recall@10≈0.90,
コード 781 KiB（int8 6.25 MiB / f32 25 MiB）。oversample を上げると recall 向上。

### RaBitQ（1-bit + 不偏推定量）, `rabitq.rs`

RaBitQ（Gao & Long, SIGMOD 2024）: 重心を引いて**ランダム回転**をかけ、符号ビット
（1 bit/dim）+ 補正係数で内積を**誤差限界つき不偏推定**する。DB は 1-bit のまま、
クエリはビットプレーン量子化（`QUERY_BITS=4`）して**popcount で推定**するため、素の
符号×f32（40ms）から **約22x 高速（~1.8ms/query）** になっている。

```rust
use vectordb::{RabitqIndex, Metric};
let rq = RabitqIndex::build(dim, Metric::Cosine, &items, /*keep_raw=*/true, /*seed=*/1);
let hits = rq.search(&query, 10, /*oversample=*/16);
```

注意（正直な結果）: **グローバル重心の Flat 構成**では、タイトなクラスタ内で符号が
同一化しやすく、精度は素の sign-binary と同程度（本ベンチ 50k で o=16: RaBitQ 0.905 /
binary 0.900）。RaBitQ の精度優位は **IVF のセル毎重心と組む（IVF+RaBitQ）** ときに
顕著になる（下記）。

### IVF + RaBitQ（`rabitq.rs` の `IvfRabitqIndex`）

k-means の**セル毎重心**で残差を小さく分散させると、RaBitQ 推定量が精度を発揮する
（論文の標準構成）。DB は 1-bit/dim のまま。

```rust
use vectordb::{IvfRabitqIndex, Metric};
let idx = IvfRabitqIndex::build(dim, Metric::Cosine, 256, &items, true, 12, /*seed=*/1);
let hits = idx.search(&query, 10, /*nprobe=*/16, /*oversample=*/32);
```

ベンチ例（50k×128, クラスタ, nprobe=16, **1-bit コード 781 KiB = int8 の 1/8**）:

| oversample | ms/query | recall@10 |
|---|---|---|
| 8 | 0.56 | 0.88 |
| 16 | 0.58 | 0.986 |
| 32 | 0.62 | **1.0000** |

1-bit 推定量は int8 より粗いので recall は rerank 候補数（oversample）で決まる。
oversample を上げると **int8 の 1/8 のメモリで recall 1.0** に到達する。

### Product Quantization（`pq.rs` の `PqIndex`）

`dim` を `m` 個のサブベクトル（各 `dim/m` 次元）に分割し、サブ空間ごとに小さな
コードブック（k-means, `ksub` 重心）で独立に量子化。1ベクトルを `m` バイト（`ksub<=256`）
で表す（例: 128次元 f32 → `m=16` で 16 バイト = **32x 圧縮**）。検索は **ADC**
（非対称距離計算）: クエリは全精度のまま、`m*ksub` の距離テーブルを前計算し、各DB
ベクトルの近似距離を `m` 回のテーブル参照の総和で求める。`keep_raw` で f32 rerank も可能。

```rust
use vectordb::{PqIndex, Metric};
let pq = PqIndex::build(&items, Metric::L2, /*m=*/16, /*ksub=*/256, /*iters=*/20, /*keep_raw=*/true);
let hits = pq.search(&query, 10, /*oversample=*/16);
pq.save("index.pq.vecdb")?;                    // magic VECDBPQ1
```

### OPQ（回転付き PQ, `opq.rs` の `OpqIndex`）

PQ の前に**直交回転 `R` を学習**してサブ空間のエネルギーを均し、量子化誤差を下げる
（Ge et al. 2013）。`R` は非パラメトリック交互最適化で学習: (1) 現在の回転でPQ学習+再構成、
(2) 生データと再構成を整列させる直交 Procrustes 解で `R` を更新。手順(2)の SVD は依存無しの
自作 Jacobi 固有値分解で計算。`R` は直交なので距離を保存し、検索は「クエリを回転→ADC」だけ。

```rust
use vectordb::{OpqIndex, Metric};
let opq = OpqIndex::build(&items, Metric::L2, 16, 256, /*iters=*/20, /*opq_iters=*/4, true);
let hits = opq.search(&query, 10, 16);
opq.save("index.opq.vecdb")?;                  // magic VECDBOP1
```

### IVF + PQ（`ivf_pq.rs` の `IvfPqIndex`）

粗い k-means でセルに割り当て、**残差 `x - 重心` を共有 PQ コードブックで量子化**（Faiss の
`IVFPQ`）。残差は小さく中心化されているので、同じPQ予算でも生ベクトルより精度が高い。検索は
`nprobe` セルを探索し、残差クエリで ADC → rerank。

```rust
use vectordb::{IvfPqIndex, Metric};
let idx = IvfPqIndex::build(&items, Metric::L2, /*nlist=*/256, 16, 256, 15, true);
let hits = idx.search(&query, 10, /*nprobe=*/16, /*oversample=*/16);
idx.save("index.ivfpq.vecdb")?;                // magic VECDBIP1
```

### int8 グラフ HNSW（`hnsw_q.rs` の `HnswQIndex`）

HNSW と同じグラフだが、各ノードを int8 コード + per-vector scale/sqnorm で保持
（≈`dim+8` バイト vs f32 の `dim*4`、ベクトル部で約 1/4）。グラフの**構築・探索とも
量子化空間**で行い、`keep_raw` 時は最終ビームだけ f32 で rerank して recall を回復。

```rust
use vectordb::{HnswQIndex, Metric};
let mut idx = HnswQIndex::new(dim, Metric::L2, 16, 200, /*keep_raw=*/true);
idx.add(1, &embedding);
let hits = idx.search(&query, 10, /*ef_search=*/96);   // rerank付きで高recall
idx.save("graph.hnswq.vecdb")?;                        // magic VECDBHQ1
```

### DiskANN / Vamana（`diskann.rs` の `DiskAnnIndex`）

大規模・省メモリ向けの **単層グラフ**（Vamana, Subramanya et al. 2019）。HNSW の階層を捨て、
`RobustPrune`（α>1 の枝刈りで遠距離ショートカットを残す）で少ホップ到達を狙う。**PQ コードは
RAM 常駐**でグラフ探索の近似距離に使い（1ホップ=テーブル参照1回）、**生 f32 は "ディスク層"**
として最終 rerank でのみ読む。グラフ幾何と PQ は processed 空間の L2 に統一し、metric は最終
スコアだけに効く（cosine は正規化で L2 と等価）。

```rust
use vectordb::{DiskAnnIndex, Metric};
// (items, metric, R=最大次数, l_build, alpha, pq_m, ksub)
let idx = DiskAnnIndex::build(&items, Metric::L2, 32, 96, 1.2, 16, 256);
let hits = idx.search(&query, 10, /*l_search=*/64);
idx.save("index.diskann.vecdb")?;              // magic VECDBDA1

// ディスク常駐モード: グラフ隣接と生 f32 は mmap のまま（ホップ毎/rerank 時に遅延読み）、
// RAM には PQ コード＋コードブックだけ（≈ count*m バイト、次元に依らない）。
let disk = DiskAnnIndex::open("index.diskann.vecdb")?;   // -> MmapDiskAnn
let hits = disk.search(&query, 10, 64);
```

構築は in-memory、探索は **`open()` で mmap 常駐**（グラフ＋生ベクトルはマップから読み、PQ だけ RAM）。
これで DiskANN 本来の「SSD にグラフ＋生・RAM に圧縮コード」構成になる（RAM フットプリントは
次元非依存の `count*m` バイト）。SIFT10K では L=64 で recall 0.999（in-memory 10k では HNSW が
速い＝DiskANN の真価は RAM に載らない大規模を SSD で捌く領域、という素直な結果）。CLI の
`search` も DiskANN は自動で mmap 経路を使う。

**逐次更新（FreshDiskANN 相当）**: 全再構築なしの `insert` / `remove` / `consolidate`。

```rust
idx.insert(id, &vector);   // 既存グラフへ Vamana 挿入（O(l_build·R)、PQは再学習しない）
idx.remove(id);            // tombstone（検索から除外、グラフは通過して連結性維持）
idx.live_len();            // 生存件数
idx.consolidate();         // 削除点を物理削除して生存集合で再構築
```

構成:
- `distance.rs` — f32/int8 距離（スカラ + AVX2, int8 は 32要素/反復）
- `quantize.rs` — int8 スカラ量子化
- `index.rs` — Flat 検索 + rerank（`View` に集約し owned/mmap で共有）+ rayon 並列
- `hnsw.rs` — HNSW（多層グラフ, 近傍ヒューリスティック）
- `hnsw_q.rs` — int8 グラフ HNSW（省メモリ, 量子化空間探索 + f32 rerank）
- `ivf.rs` — IVF（k-means + nprobe 探索）+ save/load + 並列
- `bin_quant.rs` — binary(1-bit) 量子化 + ハミング + rerank
- `rabitq.rs` — RaBitQ(1-bit + 回転 + 不偏推定量) + IVF+RaBitQ
- `pq.rs` — Product Quantization（サブ空間分割 + ADC + rerank; 共有プリミティブ）
- `opq.rs` — OPQ（学習回転 + PQ; 自作 Jacobi 固有値分解）
- `ivf_pq.rs` — IVF+PQ（粗量子化 + 残差 PQ）
- `diskann.rs` — DiskANN/Vamana（単層グラフ + RobustPrune + PQ 常駐 + rerank）
- `storage.rs` — `.vecdb` の save / mmap open / load（tombstone/payload 永続化含む）

## MoonBit（試作）

`v128` SIMD で距離カーネルを書いた試作。int8 内積は `v128_load8x8_s` +
`i32x4_dot_i16x8_s`、f32 は `v128_load` + `f32x4_*`。SIMD intrinsic はスカラ
fallback を持つため `native` / `wasm` / `wasm-gc` / `js` すべてで動くが、
**v128 が実ハードウェア SIMD に落ちるのは `wasm` ターゲットのみ**（下表参照）。
そのため **既定ターゲットを `wasm` に設定**している（`moon.mod` の
`preferred_target = "wasm"`）。

```bash
cd moonbit
moon test              # 全インデックス/量子化のテスト（既定 = wasm）
moon run cmd/main      # デモ
moon run cmd/bench --target native --release   # 全索引の統合ベンチ（recall/ms/query）
# cmd/bench は Flat/IVF/HNSW/int8-HNSW/PQ/IVF+PQ/OPQ を横断計測。graph 索引の
# 構築が重いので統合ベンチは native 推奨（wasm は scan 系の SIMD 計測向き）。
```

### バックエンド別の速度メモ（int8 only / exact f32, ms/query, 参考値）

| 経路 | int8 | exact | 備考 |
|---|---|---|---|
| native（内蔵 tcc の `tcc -run`） | 最速 | ≈int8 | v128 は**スカラ相当**（int8≈exact） |
| native + clang（`MOON_CC=clang`） | ≈tcc | ≈tcc | tcc とほぼ同じ。v128 は依然スカラ |
| 新 native（`MOONBIT_NEW_NATIVE=1` + clang） | 遅い | 遅い | 動くが安定版では未最適化。v128 も実SIMD化されず |
| **wasm（v128 SIMD）** | 4.6 | 5.1 | **v128 が実効**（exact が wasm-gc スカラ比で約2倍速） |
| llvm（nightly） | 9.2 | 36.7 | nightly で有効化できるが**遅く、v128 を実SIMD化しない**（exact が wasm の約7倍） |

（数値は n=8000 での**チューニング前**の相対比較。どのバックエンドで v128 が実効するかを
見るためのもの。チューニング後の絶対値は下の「Rust vs MoonBit」を参照）

要点:
- MoonBit で**ハードウェア SIMD を実際に使えるのは現状 `--target wasm` のみ**。
  native（tcc / clang / 新backend）も、nightly の `llvm` バックエンドも、この環境では
  v128 を x86 SIMD へ落とさずスカラ実行する（llvm はむしろ最も遅い）。
- 絶対速度が最速なのは native（tcc-run）だが v128 はスカラなので、SIMD を
  効かせる本プロジェクトでは既定を `wasm` にしている。

### Rust vs MoonBit(wasm) — チューニング後（50k×128, cosine, k=10, ms/query）

| 方式 | Rust (AVX2) | MoonBit (wasm/v128) | 倍率 |
|---|---|---|---|
| int8 + rerank | 0.61 | **1.51** | 約 2.5x |
| int8 only | 0.57 | 1.33 | 約 2.3x |
| exact f32 | 1.06 | 2.78 | 約 2.6x |
| recall@10 | 1.0 | 1.0 | 同 |

（共有CPUのため絶対値は run ごとに変動。倍率は概ね 2〜3x で安定）

チューニングで MoonBit(wasm) は初版の **約24ms → 1.5ms（約16倍高速化）**。当初 Rust比
約40倍差だったのが **約2.4倍差**まで縮小した。効いた最適化:
1. 全件ソート → **上限付き top-k ヒープ**（1クエリで N=5万件をソートしていたのを廃止。最大の効き）。
2. int8 内積を **8→16要素/反復**（`v128_load` + `i16x8_extend_low/high` + `i32x4_dot_i16x8_s`）。
3. スキャンループの構造体フィールドをローカルへ退避。

残差（約2倍）は主に SIMD 幅の差（AVX2=256bit vs wasm v128=128bit）と wasm ランタイム
（`moonrun`）のオーバーヘッド。

nightly の導入と llvm の実行:

```bash
curl -fsSL https://cli.moonbitlang.com/install/unix.sh | bash -s nightly
cd ~/.moon/lib/core && moon bundle --target llvm   # core を llvm 向けに用意
cd -/path/to/moonbit && moon run cmd/bench --target llvm --release
```

```bash
# 参考: 各経路の測り方
moon run cmd/bench --target native --release                       # 既定(tcc-run)
MOON_CC=clang moon run cmd/bench --target native --release         # clangでビルド
MOONBIT_NEW_NATIVE=1 MOON_CC=clang moon run cmd/bench --target native --release  # 新native
moon run cmd/bench --target wasm   --release                       # v128 SIMD 実効
```

API:

```moonbit
let idx = @vectordb.FlatIndex::build(vectors, ids, @vectordb.Cosine, true)
let hits = idx.search(query, 10, 4)   // Array[Hit{ id, score }]

// .vecdb シリアライズ（Rust とバイト互換）
let bytes = idx.to_bytes()                    // FixedArray[Byte]
let restored = @vectordb.FlatIndex::from_bytes(bytes)
```

### Rust ⇄ MoonBit の相互運用（`.vecdb` バイト互換）

MoonBit の `to_bytes` は Rust の `save` と**バイト単位で同一の出力**を生成する
（`storage.mbt` が同じ 64B ヘッダ + 16B 整列セクションを実装）。検証:

```bash
cd rust    && cargo run --release --example dump  # 同一の小インデックスを save→hex
cd moonbit && moon run cmd/dump --target wasm     # 同じ内容を to_bytes→hex
# → 192 バイトが完全一致（Rust が書いた .vecdb を MoonBit が読め、その逆も可能）
```

MoonBit の wasm ランタイムにファイルシステムは無いため、`to_bytes`/`from_bytes` は
バイト列を扱う（ファイル入出力はホスト側が担当）。

構成: `distance.mbt`（SIMD 距離）/ `quantize.mbt`（int8 量子化）/
`index.mbt`（Flat + rerank）/ `ivf.mbt`（IVF: k-means++ + nprobe）/
`hnsw.mbt`（HNSW グラフ）/ `hnsw_q.mbt`（int8 グラフ HNSW）/
`bin_quant.mbt`（binary 1-bit）/ `rabitq.mbt`（RaBitQ: 回転 + 符号 + 不偏推定）/
`ivf_rabitq.mbt`（IVF+RaBitQ）/ `pq.mbt`（PQ + 共有プリミティブ）/
`ivf_pq.mbt`（IVF+PQ）/ `opq.mbt`（OPQ: 学習回転 + 自作 Jacobi 固有値分解）/
`storage.mbt`（`.vecdb` 相互運用）。
量子化は Rust と同等（int8 / binary / RaBitQ / IVF+RaBitQ / **PQ / IVF+PQ / OPQ**）を移植済み。

**PQ**（`VECDBPQ1`）/ **IVF+PQ**（`VECDBIP1`）/ **OPQ**（`VECDBOP1`）/
**int8 グラフ HNSW**（`VECDBHQ1`）も Rust 版と同設計:

```moonbit
let pq = @vectordb.PqIndex::build(vectors, ids, @vectordb.L2, 16, 256, 20, true)
let hits = pq.search(query, 10, 16)      // (k, oversample)

let ipq = @vectordb.IvfPqIndex::build(vectors, ids, @vectordb.L2, 256, 16, 256, 15, true)
let hits2 = ipq.search(query, 10, 16, 16)  // (k, nprobe, oversample)

let opq = @vectordb.OpqIndex::build(vectors, ids, @vectordb.L2, 16, 256, 20, 4, true)
let hits3 = opq.search(query, 10, 16)

let hq = @vectordb.HnswQIndex::new(dim, @vectordb.L2, 16, 200, true)
hq.add(1L, embedding)
let hits4 = hq.search(query, 10, 96)     // int8グラフ + f32 rerank
```

各インデックスは `to_bytes` / `from_bytes` で round-trip でき、Rust と同じマジックの
`.vecdb` レイアウトを共有する（k-means の乱数列が言語間で異なるため生成バイト列は
一般に一致しないが、フォーマットは相互に読める）。

HNSW も Rust 版と同設計:

```moonbit
let idx = @vectordb.HnswIndex::new(dim, @vectordb.Cosine, 16, 200)
idx.add(1L, embedding)
let hits = idx.search(query, 10, 64)   // (k, ef_search)
let bytes = idx.to_bytes()             // VECDBHN1 形式（Rust と同レイアウト）
let restored = @vectordb.HnswIndex::from_bytes(bytes)
```

IVF は Rust 版と同設計、`.vecdb`（`VECDBIV1`）永続化も対応:

```moonbit
let ivf = @vectordb.IvfIndex::build(vectors, ids, @vectordb.Cosine, 256, true, 12)
let hits = ivf.search(query, 10, 4, 8)   // (k, nprobe, oversample)
let bytes = ivf.to_bytes()
let restored = @vectordb.IvfIndex::from_bytes(bytes)
```

IVF 形式は Rust と**相互に読める**（同じ `VECDBIV1` レイアウト）。ただし k-means++
初期化の乱数列が言語間で異なるため、生成されるセル割当＝バイト列は一般に一致しない
（どちらが書いたファイルも相手の `from_bytes`/`load` で正しく読める）。バイト単位で
完全一致するのは決定的な Flat 形式のほう。

**フィルタ付き検索**も Rust と同等に**全インデックス**（Flat / IVF / HNSW / int8-HNSW /
PQ / IVF+PQ / OPQ）で使える。述語 `(Int64) -> Bool` を渡すと、条件を満たす id だけが
結果に入る:

```moonbit
let hits = flat.search_filter(query, 10, 4, fn(id) { id % 2L == 0L })
let hits = ivf.search_filter(query, 10, 16, 4, fn(id) { id % 3L == 0L })
let hits = hnsw.search_filter(query, 10, 128, fn(id) { id < 1000L })
let hits = pq.search_filter(query, 10, 8, fn(id) { id % 2L == 0L })
```

Flat/IVF/PQ 系は候補ヒープに入れる前に非該当 id をスキップ、HNSW はグラフ探索は続けつつ
結果ビーム `w` への採用だけを述語で絞る（選択率が低い場合は `ef_search` を大きめに）。
`search` は常に true の述語で `search_filter` に委譲している。

## フィルタ付き検索・削除

**メタデータフィルタ**: 述語 `Fn(u64) -> bool` を渡すと、条件を満たす id だけを候補に
入れて上位 k を返す（**全インデックス対応**: Flat / IVF / HNSW / int8-HNSW / PQ /
IVF+PQ / OPQ）。HNSW 系はグラフを通過はするが結果には通さないので、選択率が低い時は
`ef_search` を大きめに。

```rust
let hits = flat.search_filter(&query, 10, 4, |id| id % 2 == 0);
let hits = ivf.search_filter(&query, 10, /*nprobe*/16, 4, |id| allow.contains(&id));
let hits = hnsw.search_filter(&query, 10, /*ef*/128, |id| id < 1000);
let hits = pq.search_filter(&query, 10, /*oversample*/8, |id| id % 2 == 0);
let hits = ivfpq.search_filter(&query, 10, /*nprobe*/16, 16, |id| allow.contains(&id));
```

**ペイロード（メタデータ）**: `add_with_payload(id, vec, bytes)` で各ベクトルに任意の
バイト列を付与し、`payload(id)` で取得できる（検索は id を返すので id→payload を引く）。
`.vecdb` に**永続化**され、`load` / `open`（mmap）双方で復元される（mmap 版もブロブは
マップ上をゼロコピー参照）。

```rust
idx.add_with_payload(1, &emb, br#"{"title":"..."}"#);
let meta: Option<&[u8]> = idx.payload(1);
save(&idx, "idx.vecdb")?;                       // ペイロードも書き出される
let m = open("idx.vecdb")?;                     // mmap でも payload(1) が引ける
```

**ソフト削除（tombstone）**: `remove(id)` で論理削除（検索から除外）、`compact()` で
物理削除して領域回収。**Flat / IVF / HNSW** が対応（`live_len()` で生存数）。

- **Flat**: tombstone は `.vecdb` に**永続化**され、`load` / `open`（mmap）双方で削除
  状態のまま復元。tombstone も payload も無いインデックスは従来と**バイト完全一致**
  （MoonBit 互換維持）。
- **IVF**: `compact()` は再クラスタリング無しでセル毎 CSR を詰め直す。
- **HNSW**: 削除ノードは結果から除外しつつグラフ探索は通過（連結性維持）。`compact()`
  は生存ノードから**グラフを再構築**。
- **IVF / HNSW も tombstone を永続化**（フラグ制御の追加セクション; 削除が無ければ
  従来と同一レイアウト）。`save`/`load` で削除状態のまま復元される。物理削除したい場合は
  `save()` 前に `compact()`。

```rust
idx.remove(42);            // tombstone（検索から消える。Flat/IVF/HNSW 共通）
idx.live_len();            // 生存件数
idx.compact();             // 物理削除して詰める（HNSW は再構築）
```

## 実データ評価（ANN_SIFT10K）

実データ **SIFT10K**（10,000×128 の SIFT 特徴、100 クエリ、厳密 100-NN の正解付き、
metric=L2）での recall@10 / スループット。`examples/eval.rs` で計測。

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
- **int8+rerank は実データでも recall 1.0**。省メモリの安全な既定。
- **素の binary は recall 0.037 で壊滅**。SIFT は値が非負なので符号ビットが全て 1 に
  なり情報が消える。**RaBitQ は重心を引くので 0.999** — naive binary に対する RaBitQ の
  優位が実データで明確に出る。
- **PQ / OPQ は 16 バイト/ベクトル（32x 圧縮）で rerank 併用 recall 1.0**。ADC テーブルで
  スキャンは軽い（SIFT は次元間バランスが良く OPQ の回転効果は小さいが、歪んだ分布では効く）。
- **IVF+PQ は残差量子化で 16 バイト/ベクトルのまま nprobe=16 で 0.991**（大規模・省メモリ向けの定番）。
- **HNSW が最高スループット**（recall 0.994 で 32k qps、flat の ~8倍）。
- **int8 グラフ HNSW は f32 HNSW とほぼ同 recall/qps**（0.997 で 23.9k qps）を
  **ベクトル部 1/4 のメモリ**で達成。
- **IVF** は recall/qps のバランスが良い（0.991 で 19.4k qps）。
- **DiskANN は L=64 で recall 0.999**。in-memory 10k では HNSW の方が速い（PQ近似探索＋
  rerank のオーバーヘッド分）——DiskANN の真価は RAM に載らない大規模を SSD で捌く領域。

## 段階的な拡張

Flat を土台に、同じ距離カーネル・量子化の上へインデックス（IVF → HNSW → int8 グラフ
HNSW）と量子化（Binary / RaBitQ / IVF+RaBitQ / PQ）を積み上げた構成。いずれも
`.vecdb` 系フォーマットで save/load でき、CLI から種別自動判定で扱える。設計の経緯は
`DESIGN.md` を参照。

## ライセンス

MIT OR Apache-2.0
