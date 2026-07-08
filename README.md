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
```

ベンチ例（50k×128, Cosine, ローカル参考値）: int8+rerank が exact の約 2〜3 倍速、
recall@10 ≒ 1.0、メモリは int8 コード 6MiB（f32 なら 24MiB）。

構成:
- `distance.rs` — f32/int8 距離（スカラ + AVX2）
- `quantize.rs` — int8 スカラ量子化
- `index.rs` — Flat 検索 + rerank（`View` に集約し owned/mmap で共有）
- `storage.rs` — `.vecdb` の save / mmap open / load

## MoonBit（試作）

`v128` SIMD で距離カーネルを書いた試作。int8 内積は `v128_load8x8_s` +
`i32x4_dot_i16x8_s`、f32 は `v128_load` + `f32x4_*`。SIMD intrinsic はスカラ
fallback を持つため、`native` / `wasm` / `wasm-gc` / `js` すべてで動く
（`native` / `wasm` で実際に SIMD 命令になる）。

```bash
cd moonbit
moon test --target native   # 8 tests（距離・量子化・検索・recall）
moon run cmd/main --target native   # デモ
```

API:

```moonbit
let idx = @vectordb.FlatIndex::build(vectors, ids, @vectordb.Cosine, true)
let hits = idx.search(query, 10, 4)   // Array[Hit{ id, score }]
```

構成: `distance.mbt`（SIMD 距離）/ `quantize.mbt`（int8 量子化）/
`index.mbt`（Flat + rerank）。

## 段階的な拡張

Flat を土台に、同じ距離カーネル・量子化の上へ IVF（セル分割）→ HNSW、量子化は
PQ / Binary / RaBitQ を追加していける設計。詳細は `DESIGN.md` の「段階的拡張」。

## ライセンス

MIT OR Apache-2.0
