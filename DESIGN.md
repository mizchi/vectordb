# vectordb — 設計と調査ノート

コンパクトなベクトル検索に特化した組み込み向けDB。Rust を本命、MoonBit
(`v128` SIMD) を試作として、同一の設計・同一のファイル形式で二本立てで実装する。

- 方式: **Flat（総当り） + int8 スカラ量子化 + rerank**
- 永続化: **mmap 単一ファイル**（`.vecdb`）
- 目標: 小さいコード・小さいメモリ・依存最小・数十万件まで実用の速度

---

## 1. 調査: 先行実装

| 実装 | 方式 | メモ |
|---|---|---|
| faiss (C++) | Flat/IVF/HNSW/PQ 全部入り | アルゴリズムの参照実装 |
| qdrant (Rust) | HNSW + 量子化 + 独自ストレージ | Rust製本番DBの標準。設計の教科書 |
| usearch (C++/Rust) | 単一ヘッダ HNSW | 「compact」の代表。依存最小・多言語FFI |
| instant-distance (Rust) | HNSW | 純Rust・軽量 |
| arroy (Rust) | LMDB + ランダム射影木 | mmap 永続化の好例（Meilisearch） |
| rabitq-rs (Rust) | IVF + RaBitQ | 最新量子化の参照 |
| lance/lancedb (Rust) | 列指向 + IVF_PQ/RaBitQ | ディスク常駐 |

要点:
- 「compact」路線 (usearch / instant-distance) は **依存を削り、距離計算の SIMD に
  全力**、インデックスは素直、という構成。本プロジェクトもこれに倣う。
- 量子化 + rerank が現在の定番パイプライン（粗選別→高精度再スコア）。

## 2. 調査: インデックス方式

- **Flat**: 厳密・最単純。SIMD があれば数万〜数十万件で実用。量子化との相性◎。
- **IVF**: k-means でセル分割 → nprobe 個の近傍セルのみ探索。量子化と併用が定番。
- **HNSW**: recall/速度は最良だが、グラフのエッジ分メモリが重く、コード量も多い。
  `M=16–32`, `efConstruction=200–500`, `efSearch≈2〜4×k`。
- **DiskANN/Vamana**: 単層グラフ + ディスク常駐。大規模・省メモリ向け。

→ 第一段階は **Flat**。IVF（セル分割）や HNSW は同じ距離カーネル/量子化の上に
後段で載せられるよう、`distance` と `quantize` を独立モジュールに切る。

## 3. 調査: 量子化（コンパクトさの要）

| 手法 | 圧縮 | recall | 速度 | 備考 |
|---|---|---|---|---|
| int8 スカラ | 4x | 高いまま | ~3.7x | 実装単純・無難な中間解（**採用**） |
| Product (PQ) | 〜64x | 中〜低 | 高 | サブベクトル分割 + コードブック, ADC |
| Binary (1bit) | 32x | rerank併用で回復 | ~25x | ハミング距離 |
| RaBitQ (SIGMOD'24/'25) | 32x | 理論誤差 O(1/√D) | 高 | 二値化の正統進化 |

第一段階は **int8 スカラ量子化**。将来 PQ / RaBitQ を `quantize` に追加できる形にする。

### int8 スカラ量子化（対称・per-vector）

各ベクトル `x`（次元 D）について:

```
scale = max(|x_i|) / 127                 # per-vector, f32
q_i   = round(x_i / scale)  ∈ [-127,127] # i8
```

- 内積 `x·y ≈ scale_x · scale_y · (q_x · q_y)`。`q_x·q_y` は int32 で SIMD 集計。
- L2: `||x-y||² = ||x||² + ||y||² − 2 x·y`。`sqnorm` を per-vector で保持して復元。
- Cosine: 挿入時に単位長へ正規化してから量子化 → cosine = dot。

## 4. 調査: 最適化手法

- **SIMD 距離計算**（最重要）
  - Rust: `std::arch` の AVX2（`is_x86_feature_detected!` で実行時分岐）+ スカラ fallback。
    nightly/外部crate 不要。
  - MoonBit: `moonbitlang/core/v128`。f32 は `v128_load`/`f32x4_mul`/`f32x4_add`、
    int8 は `v128_load8x8_s`（8バイト→i16x8 符号拡張ロード）+ `i32x4_dot_i16x8_s`
    （i16x8 の対毎内積→i32x4）。intrinsic はスカラ fallback 付きで全ターゲット動作。
- **メモリレイアウト**: ベクトルを連続配置（row-major、`count×dim`）。
- **rerank**: int8 で候補を `k·over` 件に粗選別 → 生 f32 で再スコアし上位 k。
- **並列化**: rayon（後段。まずは単スレッド）。
- **永続化**: mmap でゼロコピーロード。

## 5. アーキテクチャ

```
distance   距離カーネル（f32 / int8, scalar + SIMD）
quantize   int8 スカラ量子化（encode/decode, scale/sqnorm）
index      Flat インデックス（add / search / rerank）
storage    .vecdb 単一ファイル（save / load / mmap）
```

同じ責務分割を Rust と MoonBit の双方に持たせ、ファイル形式を共有する。

## 6. ファイル形式 `.vecdb`（little-endian, v1）

固定 64B ヘッダ + 各セクション（16B 境界にパディング）。

### ヘッダ（64 バイト）

| off | size | 型 | 内容 |
|---|---|---|---|
| 0  | 8 | bytes | magic `"VECDB1\0\0"` |
| 8  | 4 | u32 | version = 1 |
| 12 | 4 | u32 | metric (0=L2, 1=Dot, 2=Cosine) |
| 16 | 4 | u32 | dim |
| 20 | 4 | u32 | count |
| 24 | 4 | u32 | flags (bit0: raw f32, bit1: 削除ビット, bit2: ペイロード) |
| 28 | 4 | u32 | reserved |
| 32 | 32 | — | reserved（0 埋め） |

### セクション（この順、各先頭を 16B 境界へ整列）

1. `ids`    : `count × u64` — 外部ID
2. `scales` : `count × f32` — per-vector 量子化スケール
3. `sqnorms`: `count × f32` — 格納ベクトルの二乗ノルム（L2 復元・cosine 検証用）
4. `codes`  : `count × dim × i8` — int8 量子化コード（row-major）
5. `raw`    : `count × dim × f32` — 生ベクトル（flags bit0 のときのみ。rerank/厳密用）
6. `deleted`: `count × u8` — tombstone（flags bit1 のときのみ。0=生存, 1=削除）
7. `payload_offsets` : `(count+1) × u64` — 各ペイロードの CSR オフセット（flags bit2）
8. `payload_blob`    : `Σlen × u8` — 連結したペイロード本体（flags bit2）

`raw` を省くと最小サイズ（int8 のみ、rerank 不可）。含めると int8 スキャン +
f32 rerank の両立。削除ビット・ペイロードのセクションは**実際に tombstone /
payload を持つときのみ**書き出すので、どちらも無いインデックスは従来の v1 と
バイト完全一致。Rust と MoonBit で先頭セクションを同一に読み書きでき（MoonBit は
末尾の追加セクションを無視）、相互運用可能。

## 7. 検索パイプライン

```
query(f32)
 ├─ (cosine なら) 正規化
 ├─ int8 量子化（scale_q, sqnorm_q）
 ├─ 全件スキャン: metric ごとに int8 近似距離を計算（SIMD）
 │    Dot   : approx = scale_x·scale_q·(q_x·q_q)
 │    Cosine: 同上（正規化済み）
 │    L2    : sqnorm_x + sqnorm_q − 2·approx_dot
 ├─ 上位 k·over を候補に（min-heap / partial sort）
 └─ rerank: raw f32 があれば厳密距離で再スコア → 上位 k
```

## 8. 実装状況

当初「段階的拡張の余地」として挙げた項目は概ね実装済み。現状の全体像:

### インデックス（Rust / MoonBit 両実装）
- **Flat**: int8 スキャン + f32 rerank、上限付き top-k ヒープ。
- **IVF**: k-means（**k-means++ 初期化**）+ nprobe 探索、CSR 格納。
- **HNSW**: 多層グラフ、近傍ヒューリスティック、ef 探索。

### インデックス（Rust のみ）
- **DiskANN / Vamana**: 単層グラフ + `RobustPrune`（α枝刈り）、PQ 常駐で探索を誘導し
  生 f32 で rerank（`VECDBDA1`）。`open()` で **mmap 常駐探索**（グラフ＋生はマップ上、
  RAM は PQ コードのみ = 次元非依存の `count*m` バイト）。**逐次更新**（FreshDiskANN 相当）:
  `insert`（グラフへ Vamana 挿入）/ `remove`（tombstone）/ `consolidate`（生存集合で再構築）。
- **省メモリ / ストリーミング構築**（`build_streaming`）: 生ベクトルを RAM に全部載せずに
  `.vecdb` を直接生成する。**Pass1** で処理済みベクトルを一時ファイルへ逐次書き出し（平均・id・
  PQ 学習用の有限サンプルのみ蓄積）→ サンプルで PQ 学習 → **Pass2** で一時ファイルを逐次読み直して
  PQ 符号化＋medoid 決定（常駐は1本ずつ）→ グラフ構築は **SDC（対称距離計算: PQ セントロイド対の
  距離表）** で行い生ベクトル不要 → 最後にヘッダ/グラフ/コードを書き、生 f32 セクションは一時ファイルから
  **ストリームコピー**。ピーク常駐は `O(count*m + m*ksub² + edges)` で**次元非依存**。グラフ幾何は
  PQ 近似（`build` の厳密 L2 より低品質）だが、探索時のビーム rerank は厳密 f32 なので実効 recall は高い
  （SIFT 相当の合成データで recall ≥ 0.90）。CLI: `build-diskann --streaming [--sample N]`。

### 量子化（Rust）
- int8 スカラ / **binary(1-bit, Hamming)** / **RaBitQ(回転+符号+不偏推定)** /
  **IVF+RaBitQ**（セル毎重心） / **PQ(サブ空間分割 + ADC + rerank)** /
  **OPQ(学習回転 + PQ; 自作 Jacobi 固有値分解による直交 Procrustes)** /
  **IVF+PQ(粗量子化 + 残差 PQ; Faiss IVFPQ 相当)**。
  MoonBit も同等に **int8 / binary / RaBitQ / IVF+RaBitQ / PQ / IVF+PQ / OPQ** を移植済み。
- **int8 グラフ HNSW**（`HnswQIndex`）: ノードを int8 で保持し f32 の約1/4メモリ。
  グラフの構築・探索とも量子化空間で行い、`keep_raw` 時のみ最終ビームを f32 で rerank。
  MoonBit にも移植済み（`hnsw_q.mbt`, `VECDBHQ1`）。
- PQ の共有プリミティブ（`train_codebooks` / `encode_vector` / `build_lut` / `adc_sum`）を
  `pq.rs` に切り出し、OPQ・IVF+PQ から再利用。

### 運用機能（Rust）
- **フィルタ付き検索**（述語, Flat/IVF/HNSW/HnswQ）、**ソフト削除 + compact**
  （Flat/IVF/HNSW; HNSW は再構築 compact）、**upsert**、**バッチ挿入**（rayon 並列）、
  **ペイロード**（メタデータ）。
- **ペイロード / tombstone の永続化**: Flat `.vecdb` にフラグ付きの追加セクション
  （削除ビット・可変長ペイロード）を持たせ、save/load・mmap 双方で復元。tombstone/
  payload が無いインデックスは従来と**バイト完全一致**（MoonBit 互換を維持）。
- MoonBit も **フィルタ付き検索**（述語, Flat/IVF/HNSW）を移植済み。
- **rayon 並列**（クエリ内/バッチ）、**mmap 永続化**。
- 全索引が `.vecdb` 系フォーマットで **save/load**（Flat=`VECDB1`, IVF=`VECDBIV1`,
  HNSW=`VECDBHN1`）。CLI `vecdb` から build/search/info（種別自動判定）。

### 評価
- 合成クラスタ + 実データ **ANN_SIFT10K**（`examples/eval.rs`）で recall/QPS を実測。
  binary が SIFT(非負値)で崩れ RaBitQ が復元する、等の知見を README に記録。

### SIMD
- Rust: AVX2（f32 dot/l2, int8 32要素/反復）実行時分岐 + スカラ fallback。
- MoonBit: `v128`（`--target wasm` で実効。native/llvm はスカラ）。

## 9. 今後の余地

- MoonBit への 並列 / mmap 相当の移植（インデックス・量子化はほぼ全て移植済み）。
- フィルタ選択率に応じた探索の適応化、GPU/バッチ ADC の SIMD 最適化。
- DiskANN 構築の並列グラフ化（`build_streaming` で生ベクトルの非常駐化＝省メモリ構築は実装済み。SDC グラフ構築は現状シングルスレッド）。
- IVF の削除後リバランス、HNSW の逐次削除（現状 compact は再構築）。
