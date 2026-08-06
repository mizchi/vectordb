//! A deliberately small, serializable graph-query language.
//!
//! This is not RDF or full SPARQL. It evaluates a conjunction of node and
//! directed-edge patterns against a closed GraphStore. Edges keep their
//! graph-layer-specific weight and EdgeKind semantics, so callers can query
//! semantic neighbors and explicit links without flattening them into RDF.

use crate::{EdgeKind, GraphStore};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// Default maximum number of rows returned by a query.
///
/// A graph pattern can expand to many rows, so callers must opt in explicitly
/// to a larger result set with Query::limit.
pub const DEFAULT_LIMIT: usize = 100;
/// Version emitted by the JSON and text DSL query contract.
pub const QUERY_VERSION: u32 = 1;
/// Default cap for bindings produced by any one query clause.
pub const DEFAULT_MAX_INTERMEDIATE_ROWS: usize = 10_000;
/// Default cap for inspected stored edges during a query.
pub const DEFAULT_MAX_EDGES_SCANNED: usize = 100_000;

/// Explicit resource limits for one query evaluation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueryBudget {
    pub max_intermediate_rows: usize,
    pub max_edges_scanned: usize,
}

impl Default for QueryBudget {
    fn default() -> Self {
        QueryBudget {
            max_intermediate_rows: DEFAULT_MAX_INTERMEDIATE_ROWS,
            max_edges_scanned: DEFAULT_MAX_EDGES_SCANNED,
        }
    }
}

/// A serializable conjunction of graph patterns.
///
/// select names the variables included in each output row. Clauses are
/// evaluated left-to-right; placing selective node patterns before an edge
/// pattern avoids scanning unrelated edges.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Query {
    #[serde(default = "default_query_version")]
    version: u32,
    select: Vec<String>,
    #[serde(default)]
    clauses: Vec<Clause>,
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default)]
    budget: QueryBudget,
}

fn default_query_version() -> u32 {
    QUERY_VERSION
}

fn default_limit() -> usize {
    DEFAULT_LIMIT
}

impl Query {
    /// Start a query with the variables to return.
    pub fn select(vars: impl IntoIterator<Item = impl Into<String>>) -> Query {
        Query {
            version: QUERY_VERSION,
            select: vars.into_iter().map(Into::into).collect(),
            clauses: Vec::new(),
            limit: DEFAULT_LIMIT,
            budget: QueryBudget::default(),
        }
    }

    /// Add a condition for a node variable.
    pub fn node(mut self, pattern: NodePattern) -> Query {
        self.clauses.push(Clause::Node(pattern));
        self
    }

    /// Add a condition for a stored directed edge from to.
    pub fn edge(mut self, pattern: EdgePattern) -> Query {
        self.clauses.push(Clause::Edge(pattern));
        self
    }

    /// Cap the number of distinct projected rows.
    pub fn limit(mut self, limit: usize) -> Query {
        self.limit = limit;
        self
    }

    /// Set resource limits for this query evaluation.
    pub fn budget(mut self, budget: QueryBudget) -> Query {
        self.budget = budget;
        self
    }

    /// Encode the query contract as JSON for CLIs and cross-language callers.
    pub fn to_json(&self) -> Result<String, QueryError> {
        serde_json::to_string(self).map_err(|err| QueryError::InvalidJson(err.to_string()))
    }

    /// Decode a JSON query contract.
    pub fn from_json(json: &str) -> Result<Query, QueryError> {
        serde_json::from_str(json).map_err(|err| QueryError::InvalidJson(err.to_string()))
    }

    /// Parse the small text DSL into this serializable query AST.
    pub fn from_dsl(text: &str) -> Result<Query, DslQueryError> {
        let tokens = tokenize(text)?;
        Parser::new(text, tokens).parse_query()
    }

    /// Run this query against a graph and return distinct projected bindings.
    pub fn execute(&self, graph: &GraphStore) -> Result<Vec<QueryRow>, QueryError> {
        Ok(self
            .execute_with_explain(graph)?
            .rows
            .into_iter()
            .map(|row| row.row)
            .collect())
    }

    /// Run a query and retain the matching node and edge evidence for every row.
    pub fn execute_with_explain(&self, graph: &GraphStore) -> Result<QueryExecution, QueryError> {
        self.validate()?;
        if self.limit == 0 {
            return Ok(QueryExecution {
                query_version: self.version,
                rows: Vec::new(),
                plan: QueryPlan::default(),
            });
        }

        let mut bindings = vec![Binding::new()];
        let mut state = ExecutionState::new(self.budget);
        let mut clauses = Vec::new();
        for (clause_index, clause) in self.clauses.iter().enumerate() {
            let input_rows = bindings.len();
            let edges_before = state.edges_scanned;
            bindings = match clause {
                Clause::Node(pattern) => apply_node(graph, bindings, pattern, &mut state)?,
                Clause::Edge(pattern) => apply_edge(graph, bindings, pattern, &mut state)?,
            };
            clauses.push(QueryClausePlan {
                clause_index,
                input_rows,
                output_rows: bindings.len(),
                edges_scanned: state.edges_scanned - edges_before,
            });
            if bindings.is_empty() {
                return Ok(QueryExecution {
                    query_version: self.version,
                    rows: Vec::new(),
                    plan: QueryPlan {
                        clauses,
                        edges_scanned: state.edges_scanned,
                        peak_intermediate_rows: state.peak_intermediate_rows,
                    },
                });
            }
        }

        let mut seen = BTreeSet::new();
        let mut rows = Vec::new();
        for binding in bindings {
            let mut projected = BTreeMap::new();
            for var in &self.select {
                let Some(&id) = binding.values.get(var) else {
                    return Err(QueryError::UnboundSelect(var.clone()));
                };
                projected.insert(var.clone(), id);
            }
            if seen.insert(projected.clone()) {
                rows.push(ExplainedQueryRow {
                    row: QueryRow {
                        bindings: projected,
                    },
                    evidence: binding.evidence,
                });
                if rows.len() == self.limit {
                    break;
                }
            }
        }
        Ok(QueryExecution {
            query_version: self.version,
            rows,
            plan: QueryPlan {
                clauses,
                edges_scanned: state.edges_scanned,
                peak_intermediate_rows: state.peak_intermediate_rows,
            },
        })
    }

    fn validate(&self) -> Result<(), QueryError> {
        if self.version != QUERY_VERSION {
            return Err(QueryError::UnsupportedVersion {
                expected: QUERY_VERSION,
                actual: self.version,
            });
        }
        if self.select.is_empty() {
            return Err(QueryError::EmptySelect);
        }
        for var in &self.select {
            validate_var(var)?;
        }
        for clause in &self.clauses {
            match clause {
                Clause::Node(pattern) => {
                    validate_var(&pattern.var)?;
                }
                Clause::Edge(pattern) => {
                    validate_var(&pattern.from)?;
                    validate_var(&pattern.to)?;
                    if pattern.min_weight.is_some_and(|weight| !weight.is_finite()) {
                        return Err(QueryError::InvalidWeight);
                    }
                }
            }
        }
        Ok(())
    }
}

fn validate_var(var: &str) -> Result<(), QueryError> {
    if var.trim().is_empty() {
        return Err(QueryError::InvalidVariable(var.to_string()));
    }
    Ok(())
}

/// A predicate over one node variable. All specified fields must match.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodePattern {
    pub var: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// All listed tags must be attached to the node.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
}

impl NodePattern {
    /// Match any node and bind it as var.
    pub fn new(var: impl Into<String>) -> NodePattern {
        NodePattern {
            var: var.into(),
            id: None,
            title: None,
            tags: Vec::new(),
        }
    }

    /// Require one external node id.
    pub fn id(mut self, id: u64) -> NodePattern {
        self.id = Some(id);
        self
    }

    /// Require an exact title.
    pub fn title(mut self, title: impl Into<String>) -> NodePattern {
        self.title = Some(title.into());
        self
    }

    /// Require a tag. Multiple calls are conjunctive.
    pub fn tag(mut self, tag: impl Into<String>) -> NodePattern {
        self.tags.push(tag.into());
        self
    }
}

/// A predicate over a stored directed edge, expressed as from to.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EdgePattern {
    pub from: String,
    pub to: String,
    /// Matches any edge when empty. Asking for link or semantic also matches an
    /// edge whose kind is both.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub kinds: Vec<EdgeKind>,
    /// Keep only edges whose raw graph weight is at least this value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_weight: Option<f32>,
}

impl EdgePattern {
    /// Match any stored edge from from to to.
    pub fn new(from: impl Into<String>, to: impl Into<String>) -> EdgePattern {
        EdgePattern {
            from: from.into(),
            to: to.into(),
            kinds: Vec::new(),
            min_weight: None,
        }
    }

    /// Require an edge kind.
    pub fn kind(mut self, kind: EdgeKind) -> EdgePattern {
        self.kinds.push(kind);
        self
    }

    /// Require a raw graph weight at least min_weight.
    pub fn min_weight(mut self, min_weight: f32) -> EdgePattern {
        self.min_weight = Some(min_weight);
        self
    }
}

/// One JSON clause. This is public so non-Rust callers have a stable AST shape.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Clause {
    Node(NodePattern),
    Edge(EdgePattern),
}

/// One projected row: variable name to external graph node id.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueryRow {
    pub bindings: BTreeMap<String, u64>,
}

impl QueryRow {
    /// Return a bound external node id for a selected variable.
    pub fn get(&self, var: &str) -> Option<u64> {
        self.bindings.get(var).copied()
    }
}

/// A row plus the node and edge facts that bound its variables.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExplainedQueryRow {
    pub row: QueryRow,
    pub evidence: Vec<QueryEvidence>,
}

/// One graph fact that contributed to a query result.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum QueryEvidence {
    Node {
        var: String,
        id: u64,
    },
    Edge {
        from_var: String,
        to_var: String,
        from: u64,
        to: u64,
        weight: f32,
        kind: EdgeKind,
    },
}

impl QueryEvidence {
    /// True when this entry records a matched graph edge.
    pub fn is_edge(&self) -> bool {
        matches!(self, QueryEvidence::Edge { .. })
    }
}

/// Detailed result of evaluating one query.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QueryExecution {
    pub query_version: u32,
    pub rows: Vec<ExplainedQueryRow>,
    pub plan: QueryPlan,
}

/// Work performed by the left-to-right query evaluator.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueryPlan {
    pub clauses: Vec<QueryClausePlan>,
    pub edges_scanned: usize,
    pub peak_intermediate_rows: usize,
}

/// Work performed while evaluating a single clause.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueryClausePlan {
    pub clause_index: usize,
    pub input_rows: usize,
    pub output_rows: usize,
    pub edges_scanned: usize,
}

/// Query construction, decoding, or execution failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QueryError {
    EmptySelect,
    InvalidVariable(String),
    InvalidWeight,
    UnboundSelect(String),
    InvalidJson(String),
    UnsupportedVersion { expected: u32, actual: u32 },
    IntermediateRowLimitExceeded { limit: usize },
    EdgeScanLimitExceeded { limit: usize },
}

impl fmt::Display for QueryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QueryError::EmptySelect => write!(f, "query must select at least one variable"),
            QueryError::InvalidVariable(var) => write!(f, "invalid empty variable name: {var:?}"),
            QueryError::InvalidWeight => write!(f, "edge min_weight must be finite"),
            QueryError::UnboundSelect(var) => {
                write!(f, "selected variable {var:?} is not bound by any clause")
            }
            QueryError::InvalidJson(err) => write!(f, "invalid query JSON: {err}"),
            QueryError::UnsupportedVersion { expected, actual } => {
                write!(f, "unsupported query version {actual}; expected {expected}")
            }
            QueryError::IntermediateRowLimitExceeded { limit } => {
                write!(f, "query intermediate row limit exceeded: {limit}")
            }
            QueryError::EdgeScanLimitExceeded { limit } => {
                write!(f, "query edge scan limit exceeded: {limit}")
            }
        }
    }
}

impl std::error::Error for QueryError {}

/// A syntax error in the text query DSL.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DslQueryError {
    pub line: usize,
    pub column: usize,
    pub message: String,
}

impl DslQueryError {
    fn at(text: &str, offset: usize, message: impl Into<String>) -> DslQueryError {
        let prefix = &text[..offset.min(text.len())];
        let line = prefix.bytes().filter(|&byte| byte == b'\n').count() + 1;
        let column = prefix
            .rsplit_once('\n')
            .map_or(prefix.chars().count() + 1, |(_, tail)| {
                tail.chars().count() + 1
            });
        DslQueryError {
            line,
            column,
            message: message.into(),
        }
    }
}

impl fmt::Display for DslQueryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}: {}", self.line, self.column, self.message)
    }
}

impl std::error::Error for DslQueryError {}

#[derive(Clone, Debug)]
struct Binding {
    values: BTreeMap<String, u64>,
    evidence: Vec<QueryEvidence>,
}

impl Binding {
    fn new() -> Binding {
        Binding {
            values: BTreeMap::new(),
            evidence: Vec::new(),
        }
    }

    fn node(mut self, var: &str, id: u64) -> Binding {
        self.values.insert(var.to_string(), id);
        self.evidence.push(QueryEvidence::Node {
            var: var.to_string(),
            id,
        });
        self
    }

    fn edge(
        mut self,
        from_var: &str,
        to_var: &str,
        from: u64,
        to: u64,
        weight: f32,
        kind: EdgeKind,
    ) -> Binding {
        self.evidence.push(QueryEvidence::Edge {
            from_var: from_var.to_string(),
            to_var: to_var.to_string(),
            from,
            to,
            weight,
            kind,
        });
        self
    }
}

struct ExecutionState {
    budget: QueryBudget,
    edges_scanned: usize,
    peak_intermediate_rows: usize,
}

impl ExecutionState {
    fn new(budget: QueryBudget) -> ExecutionState {
        ExecutionState {
            budget,
            edges_scanned: 0,
            peak_intermediate_rows: 1,
        }
    }

    fn inspect_edge(&mut self) -> Result<(), QueryError> {
        if self.edges_scanned >= self.budget.max_edges_scanned {
            return Err(QueryError::EdgeScanLimitExceeded {
                limit: self.budget.max_edges_scanned,
            });
        }
        self.edges_scanned += 1;
        Ok(())
    }

    fn push(&mut self, output: &mut Vec<Binding>, binding: Binding) -> Result<(), QueryError> {
        if output.len() >= self.budget.max_intermediate_rows {
            return Err(QueryError::IntermediateRowLimitExceeded {
                limit: self.budget.max_intermediate_rows,
            });
        }
        output.push(binding);
        self.peak_intermediate_rows = self.peak_intermediate_rows.max(output.len());
        Ok(())
    }
}

fn apply_node(
    graph: &GraphStore,
    input: Vec<Binding>,
    pattern: &NodePattern,
    state: &mut ExecutionState,
) -> Result<Vec<Binding>, QueryError> {
    let mut output = Vec::new();
    for binding in input {
        if let Some(&id) = binding.values.get(&pattern.var) {
            if node_matches(graph, id, pattern) {
                state.push(&mut output, binding.node(&pattern.var, id))?;
            }
            continue;
        }
        for &id in graph.ids() {
            if node_matches(graph, id, pattern) {
                state.push(&mut output, binding.clone().node(&pattern.var, id))?;
            }
        }
    }
    Ok(output)
}

fn node_matches(graph: &GraphStore, id: u64, pattern: &NodePattern) -> bool {
    pattern.id.is_none_or(|expected| expected == id)
        && pattern
            .title
            .as_deref()
            .is_none_or(|title| graph.title(id) == Some(title))
        && pattern.tags.iter().all(|tag| graph.has_tag(id, tag))
}

fn apply_edge(
    graph: &GraphStore,
    input: Vec<Binding>,
    pattern: &EdgePattern,
    state: &mut ExecutionState,
) -> Result<Vec<Binding>, QueryError> {
    let mut output = Vec::new();
    for binding in input {
        let from = binding.values.get(&pattern.from).copied();
        let to = binding.values.get(&pattern.to).copied();
        match (from, to) {
            (Some(from), Some(to)) => {
                for edge in graph.neighbors(from) {
                    state.inspect_edge()?;
                    if edge.id == to && edge_matches(edge.kind, edge.weight, pattern) {
                        state.push(
                            &mut output,
                            binding.edge(
                                &pattern.from,
                                &pattern.to,
                                from,
                                to,
                                edge.weight,
                                edge.kind,
                            ),
                        )?;
                        break;
                    }
                }
            }
            (Some(from), None) => {
                for edge in graph.neighbors(from) {
                    state.inspect_edge()?;
                    if edge_matches(edge.kind, edge.weight, pattern) {
                        state.push(
                            &mut output,
                            binding.clone().node(&pattern.to, edge.id).edge(
                                &pattern.from,
                                &pattern.to,
                                from,
                                edge.id,
                                edge.weight,
                                edge.kind,
                            ),
                        )?;
                    }
                }
            }
            (None, Some(to)) => {
                for &candidate_from in graph.ids() {
                    for edge in graph.neighbors(candidate_from) {
                        state.inspect_edge()?;
                        if edge.id == to && edge_matches(edge.kind, edge.weight, pattern) {
                            state.push(
                                &mut output,
                                binding.clone().node(&pattern.from, candidate_from).edge(
                                    &pattern.from,
                                    &pattern.to,
                                    candidate_from,
                                    to,
                                    edge.weight,
                                    edge.kind,
                                ),
                            )?;
                        }
                    }
                }
            }
            (None, None) => {
                if pattern.from == pattern.to {
                    for &id in graph.ids() {
                        for edge in graph.neighbors(id) {
                            state.inspect_edge()?;
                            if edge.id == id && edge_matches(edge.kind, edge.weight, pattern) {
                                state.push(
                                    &mut output,
                                    binding.clone().node(&pattern.from, id).edge(
                                        &pattern.from,
                                        &pattern.to,
                                        id,
                                        id,
                                        edge.weight,
                                        edge.kind,
                                    ),
                                )?;
                                break;
                            }
                        }
                    }
                    continue;
                }
                for &candidate_from in graph.ids() {
                    for edge in graph.neighbors(candidate_from) {
                        state.inspect_edge()?;
                        if edge_matches(edge.kind, edge.weight, pattern) {
                            state.push(
                                &mut output,
                                binding
                                    .clone()
                                    .node(&pattern.from, candidate_from)
                                    .node(&pattern.to, edge.id)
                                    .edge(
                                        &pattern.from,
                                        &pattern.to,
                                        candidate_from,
                                        edge.id,
                                        edge.weight,
                                        edge.kind,
                                    ),
                            )?;
                        }
                    }
                }
            }
        }
    }
    Ok(output)
}

fn edge_matches(kind: EdgeKind, weight: f32, pattern: &EdgePattern) -> bool {
    pattern.min_weight.is_none_or(|min| weight >= min)
        && (pattern.kinds.is_empty()
            || pattern.kinds.iter().copied().any(|wanted| {
                kind == wanted || (kind == EdgeKind::Both && wanted != EdgeKind::Both)
            }))
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum TokenKind {
    Word(String),
    String(String),
    Number(String),
    Comma,
    Dot,
    Eq,
    Ge,
    Minus,
    LBracket,
    RBracket,
    Greater,
    Eof,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Token {
    kind: TokenKind,
    offset: usize,
}

fn tokenize(text: &str) -> Result<Vec<Token>, DslQueryError> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let (offset, ch) = chars[i];
        if ch.is_whitespace() {
            i += 1;
            continue;
        }
        if ch == '#' {
            while i < chars.len() && chars[i].1 != '\n' {
                i += 1;
            }
            continue;
        }
        if ch.is_ascii_alphabetic() || ch == '_' {
            let start = offset;
            i += 1;
            while i < chars.len() && (chars[i].1.is_ascii_alphanumeric() || chars[i].1 == '_') {
                i += 1;
            }
            let end = chars.get(i).map_or(text.len(), |(at, _)| *at);
            tokens.push(Token {
                kind: TokenKind::Word(text[start..end].to_string()),
                offset,
            });
            continue;
        }
        if ch.is_ascii_digit() {
            let start = offset;
            i += 1;
            while i < chars.len() && chars[i].1.is_ascii_digit() {
                i += 1;
            }
            if i + 1 < chars.len() && chars[i].1 == '.' && chars[i + 1].1.is_ascii_digit() {
                i += 1;
                while i < chars.len() && chars[i].1.is_ascii_digit() {
                    i += 1;
                }
            }
            let end = chars.get(i).map_or(text.len(), |(at, _)| *at);
            tokens.push(Token {
                kind: TokenKind::Number(text[start..end].to_string()),
                offset,
            });
            continue;
        }
        if ch == '"' {
            i += 1;
            let mut value = String::new();
            let mut closed = false;
            while i < chars.len() {
                let (_, current) = chars[i];
                i += 1;
                match current {
                    '"' => {
                        closed = true;
                        break;
                    }
                    '\\' => {
                        let Some((escape_offset, escaped)) = chars.get(i).copied() else {
                            return Err(DslQueryError::at(text, offset, "unterminated string"));
                        };
                        i += 1;
                        match escaped {
                            '"' => value.push('"'),
                            '\\' => value.push('\\'),
                            'n' => value.push('\n'),
                            'r' => value.push('\r'),
                            't' => value.push('\t'),
                            _ => {
                                return Err(DslQueryError::at(
                                    text,
                                    escape_offset,
                                    "unsupported string escape",
                                ));
                            }
                        }
                    }
                    _ => value.push(current),
                }
            }
            if !closed {
                return Err(DslQueryError::at(text, offset, "unterminated string"));
            }
            tokens.push(Token {
                kind: TokenKind::String(value),
                offset,
            });
            continue;
        }

        let kind = match ch {
            ',' => TokenKind::Comma,
            '.' => TokenKind::Dot,
            '=' => TokenKind::Eq,
            '-' => TokenKind::Minus,
            '[' => TokenKind::LBracket,
            ']' => TokenKind::RBracket,
            '>' if chars.get(i + 1).is_some_and(|(_, next)| *next == '=') => {
                i += 1;
                TokenKind::Ge
            }
            '>' => TokenKind::Greater,
            _ => {
                return Err(DslQueryError::at(
                    text,
                    offset,
                    format!("unexpected character {ch:?}"),
                ));
            }
        };
        tokens.push(Token { kind, offset });
        i += 1;
    }
    tokens.push(Token {
        kind: TokenKind::Eof,
        offset: text.len(),
    });
    Ok(tokens)
}

struct Parser<'a> {
    text: &'a str,
    tokens: Vec<Token>,
    at: usize,
}

impl<'a> Parser<'a> {
    fn new(text: &'a str, tokens: Vec<Token>) -> Parser<'a> {
        Parser {
            text,
            tokens,
            at: 0,
        }
    }

    fn parse_query(&mut self) -> Result<Query, DslQueryError> {
        self.expect_keyword("FIND")?;
        let mut select = vec![self.expect_word("expected a selected variable")?];
        while self.consume_comma() {
            select.push(self.expect_word("expected a variable after comma")?);
        }

        self.expect_keyword("WHERE")?;
        let mut query = Query::select(select);
        loop {
            query = match self.parse_clause()? {
                Clause::Node(pattern) => query.node(pattern),
                Clause::Edge(pattern) => query.edge(pattern),
            };
            if !self.consume_keyword("AND") {
                break;
            }
        }

        self.expect_keyword("LIMIT")?;
        let limit = self.expect_usize()?;
        if !matches!(self.current().kind, TokenKind::Eof) {
            return Err(self.error_here("expected end of query"));
        }
        Ok(query.limit(limit))
    }

    fn parse_clause(&mut self) -> Result<Clause, DslQueryError> {
        let first = self.expect_word("expected a variable")?;
        if self.consume_dot() {
            let property = self.expect_word("expected node property after dot")?;
            self.expect_eq()?;
            let mut pattern = NodePattern::new(first);
            match property.to_ascii_lowercase().as_str() {
                "tag" => pattern = pattern.tag(self.expect_string()?),
                "title" => pattern = pattern.title(self.expect_string()?),
                "id" => pattern = pattern.id(self.expect_u64()?),
                _ => return Err(self.error_here("node property must be tag, title, or id")),
            }
            return Ok(Clause::Node(pattern));
        }

        self.expect_minus()?;
        self.expect_lbracket()?;
        let mut pattern = EdgePattern::new(first, "");
        if !self.consume_rbracket() {
            loop {
                if self.consume_keyword("weight") {
                    self.expect_ge()?;
                    pattern = pattern.min_weight(self.expect_f32()?);
                } else {
                    let kind = self.expect_word("expected edge kind or weight condition")?;
                    let kind = match kind.to_ascii_lowercase().as_str() {
                        "semantic" => EdgeKind::Semantic,
                        "link" => EdgeKind::Link,
                        "both" => EdgeKind::Both,
                        _ => {
                            return Err(
                                self.error_here("edge kind must be semantic, link, or both")
                            );
                        }
                    };
                    pattern = pattern.kind(kind);
                }
                if !self.consume_comma() {
                    self.expect_rbracket()?;
                    break;
                }
            }
        }
        self.expect_minus()?;
        self.expect_greater()?;
        pattern.to = self.expect_word("expected target variable after edge")?;
        Ok(Clause::Edge(pattern))
    }

    fn current(&self) -> &Token {
        &self.tokens[self.at]
    }

    fn bump(&mut self) {
        if !matches!(self.current().kind, TokenKind::Eof) {
            self.at += 1;
        }
    }

    fn error_here(&self, message: impl Into<String>) -> DslQueryError {
        DslQueryError::at(self.text, self.current().offset, message)
    }

    fn consume_keyword(&mut self, expected: &str) -> bool {
        let TokenKind::Word(actual) = &self.current().kind else {
            return false;
        };
        if !actual.eq_ignore_ascii_case(expected) {
            return false;
        }
        self.bump();
        true
    }

    fn expect_keyword(&mut self, expected: &str) -> Result<(), DslQueryError> {
        if self.consume_keyword(expected) {
            Ok(())
        } else {
            Err(self.error_here(format!("expected keyword {expected}")))
        }
    }

    fn expect_word(&mut self, message: &str) -> Result<String, DslQueryError> {
        let TokenKind::Word(word) = &self.current().kind else {
            return Err(self.error_here(message));
        };
        let word = word.clone();
        self.bump();
        Ok(word)
    }

    fn expect_string(&mut self) -> Result<String, DslQueryError> {
        let TokenKind::String(value) = &self.current().kind else {
            return Err(self.error_here("expected a quoted string"));
        };
        let value = value.clone();
        self.bump();
        Ok(value)
    }

    fn expect_u64(&mut self) -> Result<u64, DslQueryError> {
        let TokenKind::Number(value) = &self.current().kind else {
            return Err(self.error_here("expected an integer id"));
        };
        let value = value.clone();
        self.bump();
        value
            .parse()
            .map_err(|_| self.error_here("id must be a non-negative integer"))
    }

    fn expect_usize(&mut self) -> Result<usize, DslQueryError> {
        let TokenKind::Number(value) = &self.current().kind else {
            return Err(self.error_here("expected a non-negative LIMIT"));
        };
        let value = value.clone();
        self.bump();
        value
            .parse()
            .map_err(|_| self.error_here("LIMIT must be a non-negative integer"))
    }

    fn expect_f32(&mut self) -> Result<f32, DslQueryError> {
        let TokenKind::Number(value) = &self.current().kind else {
            return Err(self.error_here("expected a number"));
        };
        let value = value.clone();
        self.bump();
        let parsed: f32 = value
            .parse()
            .map_err(|_| self.error_here("expected a finite number"))?;
        if parsed.is_finite() {
            Ok(parsed)
        } else {
            Err(self.error_here("expected a finite number"))
        }
    }

    fn consume_comma(&mut self) -> bool {
        if matches!(self.current().kind, TokenKind::Comma) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn consume_dot(&mut self) -> bool {
        if matches!(self.current().kind, TokenKind::Dot) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn consume_rbracket(&mut self) -> bool {
        if matches!(self.current().kind, TokenKind::RBracket) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn expect_eq(&mut self) -> Result<(), DslQueryError> {
        self.expect_punctuation(TokenKind::Eq, "expected =")
    }

    fn expect_ge(&mut self) -> Result<(), DslQueryError> {
        self.expect_punctuation(TokenKind::Ge, "expected >=")
    }

    fn expect_minus(&mut self) -> Result<(), DslQueryError> {
        self.expect_punctuation(TokenKind::Minus, "expected -")
    }

    fn expect_lbracket(&mut self) -> Result<(), DslQueryError> {
        self.expect_punctuation(TokenKind::LBracket, "expected [")
    }

    fn expect_rbracket(&mut self) -> Result<(), DslQueryError> {
        self.expect_punctuation(TokenKind::RBracket, "expected ]")
    }

    fn expect_greater(&mut self) -> Result<(), DslQueryError> {
        self.expect_punctuation(TokenKind::Greater, "expected >")
    }

    fn expect_punctuation(
        &mut self,
        expected: TokenKind,
        message: &str,
    ) -> Result<(), DslQueryError> {
        if self.current().kind == expected {
            self.bump();
            Ok(())
        } else {
            Err(self.error_here(message))
        }
    }
}
