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
moon test --target native            # 8 tests（距離・量子化・検索・recall）
moon run cmd/main  --target native   # デモ
moon run cmd/bench --target native --release   # ベンチ（Rustと同条件）
# SIMD の効き比較: --target wasm（v128有効） vs --target wasm-gc（スカラ）
```

### バックエンド別の速度メモ（int8 only / exact f32, ms/query, 参考値）

| 経路 | int8 | exact | 備考 |
|---|---|---|---|
| native（既定 = 内蔵 tcc の `tcc -run`） | 最速 | ≈int8 | v128 は**スカラ相当**（int8≈exact） |
| native + clang（`MOON_CC=clang`） | ≈tcc | ≈tcc | tcc とほぼ同じ。v128 は依然スカラ |
| 新 native（`MOONBIT_NEW_NATIVE=1` + clang） | 遅い | 遅い | 動くが安定版では未最適化。v128 も実SIMD化されず |
| **wasm（v128 SIMD）** | 4.6 | 5.1 | **v128 が実効**（exact が wasm-gc スカラ比で約2倍速） |
| llvm（nightly） | 9.2 | 36.7 | nightly で有効化できるが**遅く、v128 を実SIMD化しない**（exact が wasm の約7倍） |

（数値は n=8000, nq=60 の参考値, ms/query）

要点:
- MoonBit で**ハードウェア SIMD を実際に使えるのは現状 `--target wasm` のみ**。
  native（tcc / clang / 新backend）も、nightly の `llvm` バックエンドも、この環境では
  v128 を x86 SIMD へ落とさずスカラ実行する（llvm はむしろ最も遅い）。
- 絶対速度が最速なのは既定の native（tcc-run）だが v128 はスカラ。
- Rust(AVX2) は同条件で MoonBit のどのバックエンドより1桁以上速い。

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
```

構成: `distance.mbt`（SIMD 距離）/ `quantize.mbt`（int8 量子化）/
`index.mbt`（Flat + rerank）。

## 段階的な拡張

Flat を土台に、同じ距離カーネル・量子化の上へ IVF（セル分割）→ HNSW、量子化は
PQ / Binary / RaBitQ を追加していける設計。詳細は `DESIGN.md` の「段階的拡張」。

## ライセンス

MIT OR Apache-2.0
