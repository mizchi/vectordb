# meandb-graph

[`meandb-vector`](../vector) の上に載る、コンパクトな**意味的ナレッジグラフ**。Obsidian のような
ナレッジベースのグラフは本質的に「埋め込みの kNN グラフ + 明示リンク」なので、重い部分
（ANN 検索・SIMD 類似度）は vector layer をそのまま再利用し、graph layer は**グラフ層だけ**を足す。

- **意味エッジ**: ノート埋め込みの kNN。weight = 類似度（vector layer の検索結果）
- **明示エッジ**: wikilink 等。weight = 定数、`kind = link`
- 両方に該当するエッジは `kind = both`

## vector layer から使い回しているもの

| graph layer の機能 | 再利用先 |
|---|---|
| エッジ weight（類似度） | `meandb::vector::distance`（cosine/dot, AVX2 SIMD） |
| kNN エッジ構築 | `meandb::vector::HnswIndex` |
| 単一ファイル形式・整列・byte API | `.vecdb` の作法（8B magic + 64B ヘッダ + 16B 整列 + `to_bytes`/`from_bytes`） |
| CLI 引数 / CSV パース | `meandb::vector::cli::{flag_value, has_flag, parse_flag, parse_metric, parse_csv}` |

graph layer 独自コードは「重み付き CSR エッジストア + グラフクエリ + 枝刈り + 解析」だけ。

## Install

通常は `meandb` facade を使い、`meandb::graph` として import する:

```toml
[dependencies]
meandb = "0.1"
```

`meandb-graph` は vector engine への依存だけを持つ。両 layer は同じ
repository と結合テストを共有するが、version と公開 API 互換性は独立して管理する。
[`RELEASING.md`](../../RELEASING.md) を参照。

## ライブラリ API

```rust
use meandb::graph::{GraphBuilder, analytics};
use meandb::vector::Metric;

// items: (id, embedding)、links: 明示リンク (src_id, dst_id)
let mut b = GraphBuilder::new(Metric::Cosine, /*k=*/8);
b.mutual = true;               // mutual-kNN（相互近傍のみ残す＝読みやすいグローバルビュー）
b.min_weight = 0.2;            // 弱いエッジを捨てる
let g = b.build(&items, &links);

// ノードのメタデータ（title + tags）。tags は interning される
use meandb::graph::NodeMeta;
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
let g = meandb::graph::GraphStore::load("kb.graphdb")?;
```

### パターン検索（SPARQL の実用サブセット）

RDF/SPARQL 全体ではなく、ノード条件・有向辺・変数束縛・`LIMIT` の積集合を
評価する JSON Query AST を提供する。辺は RDF に平坦化せず、`semantic` /
`link` / `both` と重みをそのまま条件にできる。

```json
{
  "version": 1,
  "select": ["source", "target"],
  "clauses": [
    { "type": "node", "var": "source", "tags": ["spec"] },
    { "type": "edge", "from": "source", "to": "target", "kinds": ["link"] },
    { "type": "node", "var": "target", "tags": ["invariant"] }
  ],
  "limit": 20,
  "budget": {
    "max_intermediate_rows": 10000,
    "max_edges_scanned": 100000
  }
}
```

これは「`spec` タグのノードから明示リンクを辿り、`invariant` タグのノードを
返す」という意味になる。`both` 辺は `link` と `semantic` の両方に一致する。
結果は `[{"bindings":{"source":1,"target":2}}]` のような JSON 配列である。

Rust では型付き builder も使える。

```rust
use meandb::graph::{EdgeKind, EdgePattern, NodePattern, Query};

let query = Query::select(["source", "target"])
    .node(NodePattern::new("source").tag("spec"))
    .edge(EdgePattern::new("source", "target").kind(EdgeKind::Link))
    .node(NodePattern::new("target").tag("invariant"))
    .limit(20);
let rows = query.execute(&g)?;
```

CLI からは `meandb query kb.graphdb query.json` と実行する。クエリは左から
順に評価されるため、タグや ID で絞る `node` 条件を先に置くと効率がよい。
`version: 1` は AST 契約の版であり、未知の版は実行を拒否する。budget を省略した
場合も中間行 10,000、走査辺 100,000 の安全上限が適用される。

同じ AST を人が直接書けるテキスト DSL もある。

```text
FIND source, target
WHERE source.tag = "spec"
  AND source -[link, weight >= 0.8]-> target
  AND target.tag = "invariant"
LIMIT 20
```

構文は `FIND`、`WHERE`、`AND`、`LIMIT` と、`tag` / `title` / `id`
のノード条件、有向辺 `source -[kind]-> target` からなる。辺の kind は
`semantic`、`link`、`both` をカンマ区切りで指定でき、`weight >= N` は
任意で追加できる。コメントは `#` から行末まで。実行は
`meandb query-dsl kb.graphdb query.gql`。構文エラーは行・列付きで報告される。

`--explain` を末尾に付けると、通常の行配列の代わりに、各行を成立させたノードと辺
（edge kind と weight）、句ごとの入力/出力行数、総走査辺数を JSON で返す。
```sh
meandb query-dsl kb.graphdb query.gql --explain
```

### インクリメンタル更新（`GraphIndex`）

ノートの追加・編集・削除で**全再構築せず**、該当ノードのエッジだけ更新する live グラフ
（FreshDiskANN 相当）。編集は「そのノートだけ再埋め込み」して `insert`(upsert) するだけ。
意味エッジは相互（reciprocal）に張るので、`related` は両方向から新/編集ノートを反映する。

```rust
use meandb::graph::{GraphIndex, NodeMeta};
use meandb::vector::Metric;

let mut gi = GraphIndex::new(Metric::Cosine, 8);
gi.min_weight = 0.5;                       // 弱い（遠い）エッジは張らない
gi.insert(id, &embedding, NodeMeta { title, tags });  // 追加 / 上書き(=編集)
gi.link(a, b);                             // 明示リンク
gi.remove(old_id);                         // 削除（全エッジから除去）
let g = gi.freeze();                       // 問い合わせ/保存用の GraphStore へ
g.save("kb.graphdb")?;
```

注意: エッジは各ノードの**挿入時点の kNN** を反映するため、バッチ `GraphBuilder` と違い、
クラスタが揃う前に入ったノードは弱い遠エッジを拾い得る。`min_weight` を設定すればそれを防げる
（ナレッジベースでは低類似度の「関連」は不要なので実運用でも推奨）。デモ:
`cargo run -p meandb-graph --example live_update`。

### 自動タグ付け / 分類（`TagClassifier`）

新しい記事のタグを、**意味的近傍のタグの加重投票**で推定する（外部 ML 不要）。新記事を埋め込み、
既存のタグ付きノートの kNN を取り、類似度を各タグへの票として集計する。`score` は「近傍の類似度質量の
うちそのタグが占める割合」なので `[0,1]` に収まり閾値化しやすい（＝ kNN マルチラベル分類）。

```rust
use meandb::graph::{TagClassifier, SuggestOpts};
use meandb::vector::Metric;

// 既存の (id, 埋め込み, タグ) から分類器を構築
let clf = TagClassifier::build(&labeled, Metric::Cosine);
let opts = SuggestOpts { k: 8, max_tags: 3, min_score: 0.2, ..Default::default() };
for s in clf.suggest(&new_embedding, &opts) {
    println!("{} (conf {:.2}, {} votes)", s.tag, s.score, s.votes);
}
```

live グラフに統合した「届いた記事を分類して即ファイル」フロー:

```rust
let sugg = gi.suggest_tags(&emb, &opts);                    // グラフを変えずに提案だけ
let applied = gi.insert_auto_tagged(id, &emb, title, &opts); // 提案タグを付けて挿入
```

デモ: `cargo run -p meandb-graph --example auto_tag`。

### 重要度（PageRank）

`analytics::pagerank(&g, iters, damping)` で重み付き PageRank をノードごとに算出（グラフビューの
ノードサイズ・ランキング用）。`analytics::degrees` と `communities_label_propagation` と併用する。

## グラフビュー UI（`viewer/`）

`viewer/index.html` は依存ゼロ・単一ファイルの **Obsidian 風グラフビュー**（Canvas 力学配置）。
`export` した JSON をそのまま読み込む。コミュニティで色分け・次数でノードサイズ・意味エッジ/明示リンクを
描き分け、ノードクリックで関連ノート（重み付き）、タグ/コミュニティで絞り込み、検索、ドラッグ/ズーム/パン。
ライト/ダーク両対応。

```bash
meandb export kb.graphdb --communities > graph.json
# viewer/index.html をブラウザで開く → 「Load your export…」で graph.json を読み込む
# （サンプルデータを同梱しているので、そのまま開くだけでも動く）
```

## CLI

```bash
G="cargo run -p meandb-graph --bin meandb --"
$G build-graph notes.csv kb.graphdb --metric cosine --k 8 --mutual \
               --links links.csv --meta meta.tsv
$G info         kb.graphdb
$G related      kb.graphdb 42 -k 10              # rank<TAB>id<TAB>weight<TAB>kind<TAB>title
$G related      kb.graphdb 42 -k 10 --tag rust   # タグで絞った関連記事
$G neighborhood kb.graphdb 42 --depth 2 --max 50 [--tag rust]
$G tags         kb.graphdb                       # count<TAB>tag（降順）
$G by-tag       kb.graphdb rust                  # そのタグの id<TAB>title
$G export       kb.graphdb --communities > graph.json
# 新記事の自動タグ付け（vecs+meta で分類器を作り、query の各行にタグ提案）
$G suggest-tags vecs.csv meta.tsv new.csv --k 8 --min-score 0.2   # qid<TAB>tag<TAB>score<TAB>votes
```

- `notes.csv`: `id,v0,v1,...`（埋め込み。vecdb と同じ形式）
- `links.csv`: `src_id,dst_id`（省略可）
- `meta.tsv`: `id<TAB>title<TAB>tag1,tag2,...`（title / tags は省略可）

デモ: `cargo run -p meandb-graph --example graph_view`（合成クラスタで build→related→JSON 出力）。

## `.graphdb` 形式（v2）

`GRAPHDB1` magic + 64B ヘッダ（version / metric / node_count / edge_count / directed / tag_count）+
16B 整列セクション（リトルエンディアン）:

1. edge CSR: node ids u64 / offsets u64 / dst u32 / weight f32 / kind u8
2. メタデータ: label offsets+blob（title, UTF-8）/ tag offsets + node-tag ids u32 / tag-name offsets+blob

## 現状の範囲と今後

- 実装済み: 意味 kNN 構築 / mutual-kNN・閾値枝刈り / 明示リンク統合 / related（タグフィルタ可）/
  neighborhood（タグフィルタ可）/ **ノードメタデータ（title + tags, 永続化）** / タグ検索
  （nodes_with_tag / by-tag / tag_counts）/ JSON エクスポート（label+tags+community）/
  degree・label-propagation コミュニティ・**PageRank** / `.graphdb` 永続化 / CLI /
  **インクリメンタル更新（`GraphIndex`: upsert / edit / remove → freeze）** /
  **自動タグ付け（`TagClassifier` / `suggest_tags` / `insert_auto_tagged`）**。
- 今後: タグ間の共起グラフ、mmap ゼロコピー読み、大規模時の `search_batch` 並列構築、
  live グラフの永続化（現状 `GraphIndex` は再構築でロード）。

## ライセンス

MIT OR Apache-2.0
