# graphdb

[`vectordb`](../rust) の上に載る、コンパクトな**意味的ナレッジグラフ**。Obsidian のような
ナレッジベースのグラフは本質的に「埋め込みの kNN グラフ + 明示リンク」なので、重い部分
（ANN 検索・SIMD 類似度）は vectordb をそのまま再利用し、graphdb は**グラフ層だけ**を足す。

- **意味エッジ**: ノート埋め込みの kNN。weight = 類似度（vectordb の検索結果）
- **明示エッジ**: wikilink 等。weight = 定数、`kind = link`
- 両方に該当するエッジは `kind = both`

## vectordb から使い回しているもの

| graphdb の機能 | 再利用先 |
|---|---|
| エッジ weight（類似度） | `vectordb::distance`（cosine/dot, AVX2 SIMD） |
| kNN エッジ構築 | `vectordb::HnswIndex` |
| 単一ファイル形式・整列・byte API | `.vecdb` の作法（8B magic + 64B ヘッダ + 16B 整列 + `to_bytes`/`from_bytes`） |
| CLI 引数 / CSV パース | `vectordb::cli::{flag_value, has_flag, parse_flag, parse_metric, parse_csv}` |

graphdb 独自コードは「重み付き CSR エッジストア + グラフクエリ + 枝刈り + 解析」だけ。

## ライブラリ API

```rust
use graphdb::{GraphBuilder, analytics};
use vectordb::Metric;

// items: (id, embedding)、links: 明示リンク (src_id, dst_id)
let mut b = GraphBuilder::new(Metric::Cosine, /*k=*/8);
b.mutual = true;               // mutual-kNN（相互近傍のみ残す＝読みやすいグローバルビュー）
b.min_weight = 0.2;            // 弱いエッジを捨てる
let g = b.build(&items, &links);

// ノードのメタデータ（title + tags）。tags は interning される
use graphdb::NodeMeta;
g.set_metadata(vec![
    (0u64, NodeMeta { title: "Rust入門".into(), tags: vec!["rust".into(), "pinned".into()] }),
    (1,    NodeMeta { title: "HNSW".into(),     tags: vec!["graph".into()] }),
]);
let _ = g.title(0);              // Option<&str>
let _ = g.tags(0);               // Vec<&str>
let _ = g.nodes_with_tag("rust"); // Vec<u64>
let _ = g.tag_counts();          // Vec<(&str, usize)> 降順

// 関連記事（重み降順）。タグで絞ることも
let related = g.related(note_id, 10);          // Vec<Neighbor{ id, weight, kind }>
let allow: std::collections::HashSet<u64> = g.nodes_with_tag("rust").into_iter().collect();
let related_rust = g.related_filter(note_id, 10, |id| allow.contains(&id));

// ローカルグラフビュー（note 中心の k-hop 近傍サブグラフ、keep で絞り込み可）
let sub = g.neighborhood(note_id, /*depth=*/2, /*max_nodes=*/50, |_| true);

// グローバルグラフビュー用の JSON（nodes: degree+label+tags+community、edges: weight+kind）
let comms = analytics::communities_label_propagation(&g, 20);  // ノード色分け
println!("{}", g.export_json(Some(&comms)));

// 永続化（.graphdb 単一ファイル、mmap 相当の byte API）
g.save("kb.graphdb")?;
let g = graphdb::GraphStore::load("kb.graphdb")?;
```

## CLI

```bash
G="cargo run -p graphdb --bin graphdb --"
$G build-graph notes.csv kb.graphdb --metric cosine --k 8 --mutual \
               --links links.csv --meta meta.tsv
$G info         kb.graphdb
$G related      kb.graphdb 42 -k 10              # rank<TAB>id<TAB>weight<TAB>kind<TAB>title
$G related      kb.graphdb 42 -k 10 --tag rust   # タグで絞った関連記事
$G neighborhood kb.graphdb 42 --depth 2 --max 50 [--tag rust]
$G tags         kb.graphdb                       # count<TAB>tag（降順）
$G by-tag       kb.graphdb rust                  # そのタグの id<TAB>title
$G export       kb.graphdb --communities > graph.json
```

- `notes.csv`: `id,v0,v1,...`（埋め込み。vecdb と同じ形式）
- `links.csv`: `src_id,dst_id`（省略可）
- `meta.tsv`: `id<TAB>title<TAB>tag1,tag2,...`（title / tags は省略可）

デモ: `cargo run -p graphdb --example graph_view`（合成クラスタで build→related→JSON 出力）。

## `.graphdb` 形式（v2）

`GRAPHDB1` magic + 64B ヘッダ（version / metric / node_count / edge_count / directed / tag_count）+
16B 整列セクション（リトルエンディアン）:

1. edge CSR: node ids u64 / offsets u64 / dst u32 / weight f32 / kind u8
2. メタデータ: label offsets+blob（title, UTF-8）/ tag offsets + node-tag ids u32 / tag-name offsets+blob

## 現状の範囲と今後

- 実装済み: 意味 kNN 構築 / mutual-kNN・閾値枝刈り / 明示リンク統合 / related（タグフィルタ可）/
  neighborhood（タグフィルタ可）/ **ノードメタデータ（title + tags, 永続化）** / タグ検索
  （nodes_with_tag / by-tag / tag_counts）/ JSON エクスポート（label+tags+community）/
  degree・label-propagation コミュニティ / `.graphdb` 永続化 / CLI。
- 今後: インクリメンタル upsert（編集ノートの再埋め込み＋該当エッジのみ更新）、タグ間の
  共起グラフ、PageRank、mmap ゼロコピー読み、大規模時の `search_batch` 並列構築。

## ライセンス

MIT OR Apache-2.0
