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

# CLI（CSV: 1行 = id,v0,v1,...）
cargo run --release --bin vecdb -- build vecs.csv idx.vecdb --metric cosine
cargo run --release --bin vecdb -- info   idx.vecdb
cargo run --release --bin vecdb -- search idx.vecdb query.csv -k 10 --oversample 4
# 最小サイズ（int8のみ・rerankなし）: build に --compact
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
顕著になる — これが論文の標準構成で、次の自然な統合先。誤差限界つき推定量と高速
popcount スキャンはこのモジュールで実装済み。

構成:
- `distance.rs` — f32/int8 距離（スカラ + AVX2, int8 は 32要素/反復）
- `quantize.rs` — int8 スカラ量子化
- `index.rs` — Flat 検索 + rerank（`View` に集約し owned/mmap で共有）+ rayon 並列
- `ivf.rs` — IVF（k-means + nprobe 探索）+ save/load + 並列
- `bin_quant.rs` — binary(1-bit) 量子化 + ハミング + rerank
- `rabitq.rs` — RaBitQ(1-bit + 回転 + 不偏推定量, ビットプレーン popcount)
- `storage.rs` — `.vecdb` の save / mmap open / load

## MoonBit（試作）

`v128` SIMD で距離カーネルを書いた試作。int8 内積は `v128_load8x8_s` +
`i32x4_dot_i16x8_s`、f32 は `v128_load` + `f32x4_*`。SIMD intrinsic はスカラ
fallback を持つため `native` / `wasm` / `wasm-gc` / `js` すべてで動くが、
**v128 が実ハードウェア SIMD に落ちるのは `wasm` ターゲットのみ**（下表参照）。
そのため **既定ターゲットを `wasm` に設定**している（`moon.mod` の
`preferred_target = "wasm"`）。

```bash
cd moonbit
moon test              # 8 tests（既定 = wasm。距離・量子化・検索・recall）
moon run cmd/main      # デモ
moon run cmd/bench --release   # ベンチ（Rustと同条件, n=50k）
# 他バックエンドで動かす場合は --target native / wasm-gc / js を明示
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
`index.mbt`（Flat + rerank）/ `ivf.mbt`（IVF: k-means + nprobe）/
`storage.mbt`（`.vecdb` 相互運用）。

IVF は Rust 版と同設計:

```moonbit
let ivf = @vectordb.IvfIndex::build(vectors, ids, @vectordb.Cosine, 256, true, 12)
let hits = ivf.search(query, 10, 4, 8)   // (k, nprobe, oversample)
```

## 段階的な拡張

Flat を土台に、同じ距離カーネル・量子化の上へ IVF（セル分割）→ HNSW、量子化は
PQ / Binary / RaBitQ を追加していける設計。詳細は `DESIGN.md` の「段階的拡張」。

## ライセンス

MIT OR Apache-2.0
