#!/usr/bin/env python3
"""
Pipeline script for equations:

1. Build a vocabulary / tokenizer either on:
    - the entire input CSV (global vocab), or
    - only on the training split (train-only vocab),
   depending on --vocab-scope.

   Modes:
    - char : character-level (includes spaces)
    - byte : byte-level (UTF-8 bytes as strings)
    - bpe  : real BPE (character-based, ignoring spaces as tokens by default)
    - hierarchical / hier : AST-based hierarchical tokenizer for equations

2. Tokenize the CSV with that vocab (add tokens + token_ids columns).

3. Split into train/val/test CSVs with token ids.

Expected base CSV columns (at least):
    id,
    full_equation,
    complexity

For BPE:
    - Training is done on the specified column.
    - With global vocab: across the entire CSV.
    - With train-only vocab: across the training split only.
    - Spaces are used only as separators; BPE operates inside each
      space-separated segment (spaces are not tokens) by default.
    - The vocab JSON will contain:
        {
          "token_to_id": {...},
          "merges": [["a", "b"], ["ab", "c"], ...]
        }

For hierarchical:
    - Uses a hand-written recursive-descent parser and serializer to
      produce a sequence of symbolic, structured tokens.
    - The vocab JSON will contain:
        {
          "token_to_id": {...},
          "special_tokens": {
              "pad_token": "<pad>",
              "unk_token": "<unk>",
              "bos_token": "<bos>",
              "eos_token": "<eos>"
          }
        }

Outputs:
    - Full tokenized CSV at --output
    - Train/val/test tokenized CSVs in --output-dir with names:
        tokenized_equations_train_<MODE>.csv
        tokenized_equations_val_<MODE>.csv
        tokenized_equations_test_<MODE>.csv

Additionally, this script now also writes a CSV with vocabulary stats:
    - size of the vocabulary JSON file in megabytes
    - time it took to build the vocabulary (seconds)
    - number of tokens in the vocabulary (for BPE: len(token_to_id), merges excluded)
"""

import argparse
import json
import os
import sys
from collections import Counter
from typing import List, Dict, Set, Iterable, Tuple, Optional, Any

import pandas as pd
import time
import csv


# ---------------- Required columns for basic validation ---------------- #

REQUIRED_COLS = {
    "id",
    "full_equation",
    "complexity",
}

# Unknown token used when vocab is trained only on the train split
UNK_TOKEN = "<unk>"


# ---------------- Basic tokenizers (char / byte) ---------------- #

def tokenize_char(text: str) -> List[str]:
    """Simple character-level tokenizer (includes spaces)."""
    return list(text)


def tokenize_byte(text: str) -> List[str]:
    """Byte-level tokenizer: returns a list of byte values as strings."""
    return [str(b) for b in text.encode("utf-8")]


TOKENIZER_MAP = {
    "char": tokenize_char,
    "byte": tokenize_byte,
    # "bpe" and "hierarchical" handled separately
}


# ---------------- Generic vocab helpers (for char/byte) ---------------- #

def build_vocab(tokenized_sequences: List[List[str]]) -> Dict[str, int]:
    """
    Build a vocabulary mapping token -> integer id, sorted by frequency
    (most frequent first), then lexicographically.
    """
    counter = Counter()
    for seq in tokenized_sequences:
        counter.update(seq)

    sorted_tokens = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))

    vocab: Dict[str, int] = {}
    for idx, (tok, _) in enumerate(sorted_tokens):
        vocab[tok] = idx  # id from 0 .. |V|-1
    return vocab


def tokens_to_ids(tokens: List[str], vocab: Dict[str, int], unk_token: str = None) -> List[int]:
    """
    Map tokens to ids using the given vocab.
    If unk_token is None, tokens not in vocab will raise a KeyError.
    """
    ids: List[int] = []
    for t in tokens:
        if t in vocab:
            ids.append(vocab[t])
        elif unk_token is not None and unk_token in vocab:
            ids.append(vocab[unk_token])
        else:
            raise KeyError(f"Token '{t}' not in vocabulary")
    return ids


# ---------------- Vocab stats helper ---------------- #

def write_vocab_stats_csv(vocab_path: str, build_time_sec: float, num_tokens: int) -> None:
    """
    Save a small CSV with metadata about the vocabulary:
    - path to the vocab JSON
    - vocab file size in megabytes
    - time it took to build the vocab (seconds)
    - number of tokens in the vocab (for BPE: only token_to_id size)
    """
    stats_path = os.path.splitext(vocab_path)[0] + "_stats.csv"

    stats_dir = os.path.dirname(stats_path)
    if stats_dir:
        os.makedirs(stats_dir, exist_ok=True)

    size_bytes = os.path.getsize(vocab_path)
    size_mb = size_bytes / (1024 * 1024)

    row = {
        "vocab_path": vocab_path,
        "vocab_file_size_mb": round(size_mb, 6),
        "vocab_build_time_sec": round(build_time_sec, 6),
        "vocab_num_tokens": num_tokens,
    }

    # Overwrite on each run
    with open(stats_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)

    print(f"Saved vocabulary stats CSV to: {stats_path}")


# ---------------- Real BPE helpers (char-level, ignore spaces) ---------------- #

def prepare_bpe_corpus(texts: Iterable[str], ignore_spaces: bool = True) -> List[List[str]]:
    """
    Convert raw texts into a corpus of symbol sequences for BPE training.

    If ignore_spaces=True:
        - Split each text on spaces.
        - Each non-empty segment becomes a list of characters.
        - Spaces themselves never appear as symbols.
    """
    corpus: List[List[str]] = []
    for text in texts:
        if not isinstance(text, str):
            text = str(text)
        if not text:
            continue

        if ignore_spaces:
            for segment in text.split():
                if segment:
                    corpus.append(list(segment))
        else:
            corpus.append(list(text))
    return corpus


def get_pair_stats(corpus: List[List[str]]) -> Counter:
    """
    Count frequency of adjacent symbol pairs over the entire corpus.
    """
    pair_stats: Counter = Counter()
    for symbols in corpus:
        if len(symbols) < 2:
            continue
        for a, b in zip(symbols, symbols[1:]):
            pair_stats[(a, b)] += 1
    return pair_stats


def merge_pair_in_sequence(sequence: List[str], pair: Tuple[str, str]) -> List[str]:
    """
    Merge all occurrences of 'pair' in a single symbol sequence.
    """
    if not sequence:
        return sequence

    a, b = pair
    merged: List[str] = []
    i = 0
    n = len(sequence)

    while i < n:
        if i < n - 1 and sequence[i] == a and sequence[i + 1] == b:
            merged.append(a + b)
            i += 2
        else:
            merged.append(sequence[i])
            i += 1

    return merged


def apply_merge_to_corpus(corpus: List[List[str]], pair: Tuple[str, str]) -> List[List[str]]:
    """
    Apply one merge step (pair) to the entire corpus.
    """
    return [merge_pair_in_sequence(seq, pair) for seq in corpus]


def learn_bpe(
    texts: List[str],
    num_merges: int = 1000,
    min_frequency: int = 2,
    ignore_spaces: bool = True,
) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
    """
    Learn BPE merges on the given texts.

    Args:
        texts: List of input strings.
        num_merges: Maximum number of merge operations.
        min_frequency: Stop if the best pair occurs fewer than this.
        ignore_spaces: If True, spaces are not tokens; BPE applies within
                       space-separated segments.

    Returns:
        merges: List of merged pairs in the order they were applied.
        token_to_id: Final token vocabulary mapping token -> int id.
    """
    corpus = prepare_bpe_corpus(texts, ignore_spaces=ignore_spaces)
    merges: List[Tuple[str, str]] = []

    for _ in range(num_merges):
        pair_stats = get_pair_stats(corpus)
        if not pair_stats:
            break

        best_pair, best_freq = max(pair_stats.items(), key=lambda x: x[1])
        if best_freq < min_frequency:
            break

        merges.append(best_pair)
        corpus = apply_merge_to_corpus(corpus, best_pair)

    # Build final vocab from the merged corpus
    tokens: Set[str] = set()
    for seq in corpus:
        tokens.update(seq)

    token_to_id: Dict[str, int] = {}
    for idx, tok in enumerate(sorted(tokens)):
        token_to_id[tok] = idx

    return merges, token_to_id


def bpe_encode(
    text: str,
    merges: List[Tuple[str, str]],
    ignore_spaces: bool = True,
) -> List[str]:
    """
    Encode a single string using learned BPE merges.

    - Start from characters (inside each space-separated segment if ignore_spaces=True).
    - Apply merges in order.
    - Concatenate tokens across segments (no space tokens).
    """
    if not isinstance(text, str):
        text = str(text)

    if ignore_spaces:
        segments = text.split()
    else:
        segments = [text]

    all_tokens: List[str] = []

    for segment in segments:
        if not segment:
            continue
        symbols = list(segment)
        for pair in merges:
            symbols = merge_pair_in_sequence(symbols, pair)
        all_tokens.extend(symbols)

    return all_tokens


# ---------------- Hierarchical tokenizer (AST-based) ---------------- #

import re as _re

class Node:
    def __init__(self, node_type: str, value: Optional[Any] = None,
                 children: Optional[List["Node"]] = None,
                 meta: Optional[Dict[str, Any]] = None):
        self.type = node_type  # e.g., 'ADD','MUL','POW','IDENT','NUMBER','FUNC','PAREN'
        self.value = value
        self.children = children or []
        self.meta = meta or {}

    def __repr__(self):
        if self.type in ("IDENT", "NUMBER"):
            return f"Node({self.type}:{self.value})"
        elif self.children:
            return f"Node({self.type}, children={len(self.children)}, meta={self.meta})"
        else:
            return f"Node({self.type})"


TOKEN_SPEC = [
    ("NUMBER",   r"\d+(?:\.\d+)?"),
    ("IDENT",    r"[A-Za-z][A-Za-z0-9_]*"),
    ("OP",       r"\+|\-|\*|/|\^|=|,|:"),
    ("LPAREN",   r"\("),
    ("RPAREN",   r"\)"),
    ("UNDERSCORE", r"_"),
    ("WS",       r"\s+"),
    ("UNKNOWN",  r"."),
]
MASTER_RE = _re.compile("|".join(f"(?P<{n}>{p})" for n, p in TOKEN_SPEC))


def lex(s: str) -> List[Tuple[str, str]]:
    """Return list of (TYPE, value) tokens. Skips whitespace."""
    tokens: List[Tuple[str, str]] = []
    for m in MASTER_RE.finditer(s):
        typ = m.lastgroup
        val = m.group()
        if typ == "WS":
            continue
        tokens.append((typ, val))
    return tokens


IMPLICIT_MUL_PATTERNS = [
    # number followed by identifier or '(' -> insert *
    (_re.compile(r"(\d)(\s*)(?=[A-Za-z(])"), r"\1*"),
    # identifier or ')' followed by identifier or number or '(' -> insert *
    (_re.compile(r"([A-Za-z0-9_\)])(\s*)(?=[A-Za-z0-9_\(])"), r"\1*"),
    # ) followed by ( -> )*(
    (_re.compile(r"(\))(\s*)(\()"), r")*("),
]


def normalize_implicit_mult(s: str) -> str:
    """Insert explicit '*' for common implicit multiplication cases."""
    out = s
    for pat, repl in IMPLICIT_MUL_PATTERNS:
        out = pat.sub(repl, out)
    # canonicalize unicode minus to ASCII hyphen
    out = out.replace("\u2212", "-")
    return out


class ParserError(Exception):
    pass


class Parser:
    def __init__(self, tokens: List[Tuple[str, str]]):
        self.tokens = tokens
        self.i = 0

    def peek(self) -> Tuple[Optional[str], Optional[str]]:
        if self.i < len(self.tokens):
            return self.tokens[self.i]
        return (None, None)

    def consume(self, expected_type: Optional[str] = None,
                expected_val: Optional[str] = None) -> Tuple[str, str]:
        if self.i >= len(self.tokens):
            raise ParserError("Unexpected EOF")
        tok = self.tokens[self.i]
        if expected_type and tok[0] != expected_type:
            raise ParserError(f"Expected type {expected_type} but got {tok}")
        if expected_val and tok[1] != expected_val:
            raise ParserError(f"Expected val {expected_val} but got {tok}")
        self.i += 1
        return tok

    def match(self, typ: str, val: Optional[str] = None) -> bool:
        t = self.peek()
        if t[0] == typ and (val is None or t[1] == val):
            return True
        return False

    # Grammar:
    # expression := add_expr ( '=' add_expr )?
    # add_expr := mul_expr ( ('+'|'-') mul_expr )*
    # mul_expr := pow_expr ( ('*'|'/') pow_expr )*
    # pow_expr := atom ( '^' pow_expr )?
    # atom := NUMBER | IDENT [function_call | '_' index] | '(' expression ')' | special constructs

    def parse(self) -> Node:
        node = self.parse_expression()
        return node

    def parse_expression(self) -> Node:
        left = self.parse_add()
        if self.match("OP", "="):
            self.consume("OP", "=")
            right = self.parse_add()
            return Node("ASSIGN", children=[left, Node("OP", "="), right])
        return left

    def parse_add(self) -> Node:
        node = self.parse_mul()
        children = [node]
        while self.match("OP") and self.peek()[1] in ("+", "-"):
            op = self.consume("OP")[1]
            right = self.parse_mul()
            children.append(Node("OP", value=op))
            children.append(right)
        if len(children) == 1:
            return children[0]
        return Node("ADD", children=children)

    def parse_mul(self) -> Node:
        node = self.parse_pow()
        children = [node]
        while self.match("OP") and self.peek()[1] in ("*", "/"):
            op = self.consume("OP")[1]
            right = self.parse_pow()
            children.append(Node("OP", value=op))
            children.append(right)
        if len(children) == 1:
            return children[0]
        return Node("MUL", children=children)

    def parse_pow(self) -> Node:
        left = self.parse_atom()
        if self.match("OP", "^"):
            self.consume("OP", "^")
            # right-associative
            # handle a corner: if exponent is missing or only '-', assume exponent -1
            if self.match("OP", "-"):
                # if '-' is followed by NUMBER/IDENT/LPAREN it's a unary negation for exponent
                # otherwise treat as exponent -1
                if self.i + 1 < len(self.tokens) and self.tokens[self.i + 1][0] in (
                    "NUMBER",
                    "IDENT",
                    "LPAREN",
                ):
                    # consume '-' and continue parse_pow to build unary negative exponent
                    self.consume("OP", "-")
                    if self.match("NUMBER"):
                        num = self.consume("NUMBER")[1]
                        exponent = Node("NUMBER", value="-" + num)
                    elif self.match("IDENT"):
                        ident = self.consume("IDENT")[1]
                        exponent = Node("IDENT", value="-" + ident)
                    elif self.match("LPAREN"):
                        self.consume("LPAREN")
                        sub = self.parse_add()
                        self.consume("RPAREN")
                        exponent = Node("PAREN", children=[sub], meta={"neg": True})
                    else:
                        exponent = Node("NUMBER", value="-1")
                else:
                    # ambiguous lone '-' after '^' => treat as -1
                    exponent = Node("NUMBER", value="-1")
            else:
                exponent = self.parse_pow()
            return Node("POW", children=[left, exponent])
        return left

    def parse_atom(self) -> Node:
        if self.match("NUMBER"):
            val = self.consume("NUMBER")[1]
            return Node("NUMBER", value=val)
        if self.match("IDENT"):
            name = self.consume("IDENT")[1]
            # special handling for 'int' and 'sum' (simple heuristics)
            if name == "int" and self.match("LPAREN"):
                # parse int(expr)dx pattern
                self.consume("LPAREN")
                integrand = self.parse_add()
                self.consume("RPAREN")
                # optional differential DX
                var = None
                # if next tokens form 'd' IDENT (e.g., dx)
                if (
                    self.match("IDENT")
                    and len(self.tokens[self.i][1]) >= 2
                    and self.tokens[self.i][1].startswith("d")
                ):
                    var_token = self.consume("IDENT")[1]
                    var = var_token[1:]
                return Node("INT", children=[integrand], meta={"d": var})

            if name == "sum":
                # try to parse limits: sum_n=0^N(...)
                var = None
                lower = None
                upper = None
                if self.match("UNDERSCORE"):
                    self.consume("UNDERSCORE")
                    if self.match("IDENT"):
                        var = self.consume("IDENT")[1]
                    # expect '='
                    if self.match("OP", "="):
                        self.consume("OP", "=")
                        if self.match("NUMBER"):
                            lower = self.consume("NUMBER")[1]
                        elif self.match("IDENT"):
                            lower = self.consume("IDENT")[1]
                if self.match("OP", "^"):
                    self.consume("OP", "^")
                    if self.match("IDENT"):
                        upper = self.consume("IDENT")[1]
                    elif self.match("NUMBER"):
                        upper = self.consume("NUMBER")[1]
                # argument in parentheses
                if self.match("LPAREN"):
                    self.consume("LPAREN")
                    body = self.parse_add()
                    self.consume("RPAREN")
                else:
                    body = Node("IDENT", value="")
                return Node("SUM", children=[body], meta={"var": var, "lower": lower, "upper": upper})

            # function call
            if self.match("LPAREN"):
                self.consume("LPAREN")
                args: List[Node] = []
                # parse comma-separated arguments
                if not self.match("RPAREN"):
                    args.append(self.parse_add())
                    while self.match("OP", ","):
                        self.consume("OP", ",")
                        args.append(self.parse_add())
                self.consume("RPAREN")
                return Node("FUNC", value=name, children=args)
            # indexed identifier x_y
            if self.match("UNDERSCORE"):
                self.consume("UNDERSCORE")
                if self.match("NUMBER"):
                    idx = self.consume("NUMBER")[1]
                elif self.match("IDENT"):
                    idx = self.consume("IDENT")[1]
                else:
                    idx = ""
                return Node("INDEX", value=name, children=[Node("INDEX_VAL", value=idx)])
            return Node("IDENT", value=name)

        if self.match("LPAREN"):
            self.consume("LPAREN")
            inner = self.parse_add()
            self.consume("RPAREN")
            return Node("PAREN", children=[inner])

        # fallback: unknown token -> consume and wrap
        if self.i < len(self.tokens):
            tok = self.consume()
            return Node("UNKNOWN", value=tok[1])
        raise ParserError("Unexpected end in parse_atom")


START_MARKER = "<{type}_START>"
END_MARKER = "<{type}_END>"


def serialize_ast(node: Node) -> List[str]:
    """Flatten AST to sequence of tokens with structural markers.

    Atom nodes produce 'IDENT:xxx' or 'NUMBER:yyy'. Operators are emitted as their symbol.
    Composite nodes produce <TYPE_START> ...children... <TYPE_END>.
    """
    out: List[str] = []
    t = node.type
    if t == "IDENT":
        out.append(f"IDENT:{node.value}")
        return out
    if t == "NUMBER":
        out.append(f"NUMBER:{node.value}")
        return out
    if t == "OP":
        out.append(node.value)
        return out
    if t == "UNKNOWN":
        out.append(f"UNKNOWN:{node.value}")
        return out

    # composite nodes
    out.append(START_MARKER.format(type=t))

    if t == "ASSIGN":
        # children [left, OP('='), right]
        for c in node.children:
            out.extend(serialize_ast(c))
    elif t in ("ADD", "MUL"):
        for c in node.children:
            out.extend(serialize_ast(c))
    elif t == "POW":
        # children [base, exponent]
        base, exp = node.children
        out.extend(serialize_ast(base))
        out.append("^")
        out.extend(serialize_ast(exp))
    elif t == "FUNC":
        # emit function name as marker including name for compactness
        out.append(f"FUNC_NAME:{node.value}")
        out.append("<ARG_START>")
        for idx, c in enumerate(node.children):
            out.extend(serialize_ast(c))
            if idx < len(node.children) - 1:
                out.append(",")
        out.append("<ARG_END>")
    elif t == "INT":
        # INT: children[0] integrand; meta 'd' for differential
        out.extend(serialize_ast(node.children[0]))
        if node.meta.get("d"):
            out.append(f"DIFF:{node.meta.get('d')}")
    elif t == "SUM":
        # meta contains var, lower, upper
        out.append("<LIMITS_START>")
        if node.meta.get("var"):
            out.append(f"VAR:{node.meta.get('var')}")
        if node.meta.get("lower"):
            out.append("=")
            out.append(f"LOWER:{node.meta.get('lower')}")
        if node.meta.get("upper"):
            out.append("^")
            out.append(f"UPPER:{node.meta.get('upper')}")
        out.append("<LIMITS_END>")
        out.append("<ARG_START>")
        out.extend(serialize_ast(node.children[0]))
        out.append("<ARG_END>")
    elif t == "PAREN":
        # just serialize inner
        out.extend(serialize_ast(node.children[0]))
    elif t == "INDEX":
        out.append(f"IDENT:{node.value}")
        out.append("<INDEX_START>")
        out.extend(serialize_ast(node.children[0]))
        out.append("<INDEX_END>")
    else:
        # generic fallback: serialize children
        for c in node.children:
            out.extend(serialize_ast(c))

    out.append(END_MARKER.format(type=t))
    return out


def hier_ast_parse(equation: str) -> Node:
    s = normalize_implicit_mult(equation.strip())
    toks = lex(s)
    p = Parser(toks)
    return p.parse()


def hier_tokenize(equation: str) -> List[str]:
    """
    Normalize, lex, parse and serialize to hierarchical token list.

    If the parser fails (e.g. due to unbalanced parentheses or other
    malformed input), fall back to a purely lexical representation so
    the pipeline never crashes. We still keep a structural marker
    <PARSE_ERROR> so a model can learn to treat these differently.
    """
    s = equation.strip()

    try:
        # Normal hierarchical path
        ast = hier_ast_parse(s)  # may raise ParserError
        tokens = ["<EXPR_START>"] + serialize_ast(ast) + ["<EXPR_END>"]
        return tokens
    except ParserError:
        # Fallback: lexical-only tokens with an explicit error marker.
        # This avoids blowing up and still yields a reasonable sequence.
        norm = normalize_implicit_mult(s)
        toks = lex(norm)  # list[(TYPE, value)]
        flat: List[str] = [f"{typ}:{val}" for typ, val in toks]

        # Note: <PARSE_ERROR> will be included in the vocab and mapped
        # to an id like any other token. Any unseen tokens at inference
        # time will go to <unk>.
        return ["<EXPR_START>", "<PARSE_ERROR>"] + flat + ["<EXPR_END>"]


def export_hier_vocabulary(equations: List[str]) -> Set[str]:
    vocab: Set[str] = set()
    for eq in equations:
        toks = hier_tokenize(eq)
        vocab.update(toks)
    return vocab


def build_hier_vocab(tokenized_sequences: List[List[str]]) -> Dict[str, int]:
    """
    Build a hierarchical vocabulary suitable for a Transformer model.

    - Adds special tokens first with fixed IDs:
        0: <pad>, 1: <unk>, 2: <bos>, 3: <eos>
    - Remaining tokens are sorted by frequency (desc), then lexicographically.
    """
    counter = Counter()
    for seq in tokenized_sequences:
        counter.update(seq)

    specials = ["<pad>", "<unk>", "<bos>", "<eos>"]
    vocab: Dict[str, int] = {tok: idx for idx, tok in enumerate(specials)}

    sorted_tokens = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    next_id = len(vocab)
    for tok, _ in sorted_tokens:
        if tok not in vocab:
            vocab[tok] = next_id
            next_id += 1

    return vocab


# ---------------- Split helpers ---------------- #

def validate_ratios(train: float, val: float, test: float) -> Tuple[float, float, float]:
    total = train + val + test
    if not (0.999 <= total <= 1.001):
        raise ValueError(
            f"train-ratio + val-ratio + test-ratio must sum to 1.0 (got {total:.4f})."
        )
    return train, val, test


# ---------------- Core processing: tokenize full CSV, then split (global vocab) ---------------- #

def tokenize_full_df(
    df: pd.DataFrame,
    column_name: str,
    mode: str,
    vocab_path: str,
    bpe_merges: int,
    bpe_min_frequency: int,
    bpe_keep_spaces: bool,
) -> pd.DataFrame:
    """
    Tokenize the entire DataFrame (df) on column_name using the given mode.
    - Builds vocab/BPE model on the entire df[column_name].
    - Adds columns:
        * <column_name>_tokens_<mode>
        * <column_name>_token_ids      (JSON list of ints)
    - Saves vocab/BPE/Hier JSON to vocab_path.
    - Also writes a *_stats.csv with vocab size, build time and file size.
    - Returns the augmented DataFrame.
    """
    if column_name not in df.columns:
        raise ValueError(
            f"Column '{column_name}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )

    texts = df[column_name].astype(str).tolist()

    if mode in ("char", "byte"):
        tokenizer_fn = TOKENIZER_MAP[mode]

        # Tokenize each sequence
        tokenized_sequences: List[List[str]] = []
        token_str_column: List[str] = []

        print(f"Tokenizing column '{column_name}' with mode '{mode}' on full dataset...")
        for text in texts:
            tokens = tokenizer_fn(text)
            tokenized_sequences.append(tokens)
            token_str_column.append("".join(tokens))

        df[f"{column_name}_tokens_{mode}"] = token_str_column

        # Build vocab over all tokens
        start_time = time.time()
        vocab = build_vocab(tokenized_sequences)
        build_time = time.time() - start_time
        num_tokens = len(vocab)

        # Map tokens to ids (stored as JSON strings for convenience)
        id_sequences: List[str] = []
        for tokens in tokenized_sequences:
            ids = tokens_to_ids(tokens, vocab)
            id_sequences.append(json.dumps(ids))

        df[f"{column_name}_token_ids"] = id_sequences

        # Ensure vocab directory exists
        vocab_dir = os.path.dirname(vocab_path)
        if vocab_dir:
            os.makedirs(vocab_dir, exist_ok=True)

        # Save vocab (just token->id mapping)
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        print(f"Saved {mode} vocabulary to: {vocab_path}")

        # Save vocab stats CSV
        write_vocab_stats_csv(vocab_path, build_time, num_tokens)

    elif mode == "bpe":
        ignore_spaces = not bpe_keep_spaces

        # Learn BPE merges and vocab on full texts
        print(
            f"Learning BPE on full dataset with num_merges={bpe_merges}, "
            f"min_frequency={bpe_min_frequency}, ignore_spaces={ignore_spaces}..."
        )
        start_time = time.time()
        merges, token_to_id = learn_bpe(
            texts,
            num_merges=bpe_merges,
            min_frequency=bpe_min_frequency,
            ignore_spaces=ignore_spaces,
        )
        build_time = time.time() - start_time
        num_tokens = len(token_to_id)

        print(f"Learned {len(merges)} merges.")
        print(f"Final BPE vocab size: {num_tokens} tokens.")

        # Save BPE model (vocab + merges)
        bpe_model = {
            "token_to_id": token_to_id,
            "merges": merges,
        }

        vocab_dir = os.path.dirname(vocab_path)
        if vocab_dir:
            os.makedirs(vocab_dir, exist_ok=True)

        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(bpe_model, f, ensure_ascii=False, indent=2)
        print(f"Saved BPE model (vocab + merges) to: {vocab_path}")

        # Save vocab stats CSV (tokens only from token_to_id, merges excluded)
        write_vocab_stats_csv(vocab_path, build_time, num_tokens)

        # Encode each equation with BPE
        print("Encoding equations with BPE on full dataset...")
        token_str_column: List[str] = []
        id_sequences: List[str] = []

        for text in texts:
            tokens = bpe_encode(text, merges, ignore_spaces=ignore_spaces)
            token_str_column.append("".join(tokens))
            ids = [token_to_id[tok] for tok in tokens]
            id_sequences.append(json.dumps(ids))

        df[f"{column_name}_tokens_bpe"] = token_str_column
        df[f"{column_name}_token_ids"] = id_sequences

    elif mode in ("hier", "hierarchical"):
        # Hierarchical tokenizer: AST-based tokens
        print(f"Tokenizing column '{column_name}' with hierarchical mode on full dataset...")
        tokenized_sequences: List[List[str]] = []
        token_str_column: List[str] = []

        for text in texts:
            tokens = hier_tokenize(text)
            tokenized_sequences.append(tokens)
            # join with spaces for readability
            token_str_column.append(" ".join(tokens))

        df[f"{column_name}_tokens_hier"] = token_str_column

        # Build hierarchical vocab with special tokens
        start_time = time.time()
        vocab = build_hier_vocab(tokenized_sequences)
        build_time = time.time() - start_time
        num_tokens = len(vocab)

        # Map tokens to ids with unknown fallback
        id_sequences: List[str] = []
        for tokens in tokenized_sequences:
            ids = tokens_to_ids(tokens, vocab, unk_token=UNK_TOKEN)
            id_sequences.append(json.dumps(ids))

        df[f"{column_name}_token_ids"] = id_sequences

        # Ensure vocab directory exists
        vocab_dir = os.path.dirname(vocab_path)
        if vocab_dir:
            os.makedirs(vocab_dir, exist_ok=True)

        # Save hierarchical vocab with special token metadata
        vocab_json = {
            "token_to_id": vocab,
            "special_tokens": {
                "pad_token": "<pad>",
                "unk_token": "<unk>",
                "bos_token": "<bos>",
                "eos_token": "<eos>",
            },
        }
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab_json, f, ensure_ascii=False, indent=2)
        print(f"Saved hierarchical vocabulary to: {vocab_path}")

        # Save vocab stats CSV
        write_vocab_stats_csv(vocab_path, build_time, num_tokens)

    else:
        raise ValueError(f"Unknown mode '{mode}'. Expected one of: char, byte, bpe, hier/hierarchical.")

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tokenize equations from CSV (char/byte/BPE/hierarchical), build a vocabulary either on "
            "the full dataset or only on the training split, then split into "
            "train/val/test tokenized CSVs."
        )
    )

    # Original tokenizer args
    parser.add_argument(
        "--input",
        default="data pipeline/equations.csv",
        help="Path to input CSV file (default: data pipeline/equations.csv).",
    )
    parser.add_argument(
        "--output",
        default="data pipeline/tokenized_equations.csv",
        help="Path to output FULL tokenized CSV (default: data pipeline/tokenized_equations.csv).",
    )
    parser.add_argument(
        "--column",
        default="full_equation",
        help='Name of the column containing equations (default: "full_equation").',
    )
    parser.add_argument(
        "--mode",
        choices=["char", "byte", "bpe", "hier", "hierarchical"],
        default="bpe",
        help="Tokenization mode: char | byte | bpe | hier/hierarchical (default: bpe).",
    )
    parser.add_argument(
        "--vocab",
        default="data pipeline/vocab.json",
        help="Path to save the vocabulary / BPE model JSON (default: data pipeline/vocab.json).",
    )
    parser.add_argument(
        "--bpe-merges",
        type=int,
        default=1000,
        help="Number of BPE merge operations to learn (default: 1000).",
    )
    parser.add_argument(
        "--bpe-min-frequency",
        type=int,
        default=2,
        help="Minimum pair frequency to merge during BPE training (default: 2).",
    )
    parser.add_argument(
        "--bpe-keep-spaces",
        action="store_true",
        help=(
            "If set, spaces are kept as characters and BPE is applied over the full text, "
            "including spaces. By default, spaces are not tokens and are only used as segment separators."
        ),
    )
    parser.add_argument(
        "--vocab-scope",
        choices=["full", "train"],
        default="full",
        help=(
            "Where to learn the vocabulary/tokenizer from: "
            "'full' = entire dataset (default), "
            "'train' = training split only (safe <unk> fallback for unseen tokens)."
        ),
    )

    # Original split args
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where tokenized train/val/test CSVs will be saved.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of data used for training (default: 0.8).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of data used for validation (default: 0.1).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction of data used for testing (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling before splitting (default: 42).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Normalize mode alias
    if args.mode == "hier":
        mode_normalized = "hierarchical"
    else:
        mode_normalized = args.mode

    # Validate split ratios
    train_ratio, val_ratio, test_ratio = validate_ratios(
        args.train_ratio, args.val_ratio, args.test_ratio
    )

    # Load input CSV
    print(f"Reading base CSV from: {args.input}")
    df = pd.read_csv(args.input)

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        print(f"Error: missing expected columns: {missing}", file=sys.stderr)
        print(f"Available columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    if args.vocab_scope == "full":
        # ---------------- Global vocabulary: old behavior ---------------- #
        # Tokenize full DataFrame and build vocab/BPE model on the entire CSV
        df = tokenize_full_df(
            df=df,
            column_name=args.column,
            mode=mode_normalized,
            vocab_path=args.vocab,
            bpe_merges=args.bpe_merges,
            bpe_min_frequency=args.bpe_min_frequency,
            bpe_keep_spaces=args.bpe_keep_spaces,
        )

        # Save full tokenized CSV
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"Saved FULL tokenized CSV to: {args.output}")

        # Shuffle and split tokenized DataFrame
        df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

        n = len(df)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val  # remainder

        train_df = df.iloc[:n_train]
        val_df = df.iloc[n_train: n_train + n_val]
        test_df = df.iloc[n_train + n_val:]

        # Create output directory for splits
        os.makedirs(args.output_dir, exist_ok=True)

        mode_suffix = mode_normalized  # e.g. bpe, char, byte, hierarchical

        train_path = os.path.join(args.output_dir, f"tokenized_train_{mode_suffix}.csv")
        val_path = os.path.join(args.output_dir, f"tokenized_val_{mode_suffix}.csv")
        test_path = os.path.join(args.output_dir, f"tokenized_test_{mode_suffix}.csv")

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(f"Total rows: {n}")
        print(f"Train: {len(train_df)} -> {train_path}")
        print(f"Val:   {len(val_df)} -> {val_path}")
        print(f"Test:  {len(test_df)} -> {test_path}")

    else:
        # ---------------- Train-only vocabulary ---------------- #
        print("Using train-only vocabulary: splitting raw data before tokenizer training...")

        # Shuffle and split the *raw* DataFrame
        df_shuffled = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

        n = len(df_shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val  # remainder

        train_df = df_shuffled.iloc[:n_train].copy()
        val_df = df_shuffled.iloc[n_train: n_train + n_val].copy()
        test_df = df_shuffled.iloc[n_train + n_val:].copy()

        print(f"Total rows: {n}")
        print(f"Train rows (for vocab + model): {len(train_df)}")
        print(f"Val rows:  {len(val_df)}")
        print(f"Test rows: {len(test_df)}")

        if args.column not in df_shuffled.columns:
            raise ValueError(
                f"Column '{args.column}' not found in CSV. "
                f"Available columns: {list(df_shuffled.columns)}"
            )

        train_texts = train_df[args.column].astype(str).tolist()

        if mode_normalized in ("char", "byte"):
            tokenizer_fn = TOKENIZER_MAP[mode_normalized]

            # Train vocab on TRAIN split
            tokenized_sequences: List[List[str]] = []
            for text in train_texts:
                tokens = tokenizer_fn(text)
                tokenized_sequences.append(tokens)

            start_time = time.time()
            vocab = build_vocab(tokenized_sequences)

            # Add unknown token for safety on val/test
            if UNK_TOKEN not in vocab:
                vocab[UNK_TOKEN] = len(vocab)

            build_time = time.time() - start_time
            num_tokens = len(vocab)

            vocab_dir = os.path.dirname(args.vocab)
            if vocab_dir:
                os.makedirs(vocab_dir, exist_ok=True)
            with open(args.vocab, "w", encoding="utf-8") as f:
                json.dump(vocab, f, ensure_ascii=False, indent=2)
            print(f"Saved {mode_normalized} vocabulary (train-only) to: {args.vocab}")

            # Save vocab stats CSV
            write_vocab_stats_csv(args.vocab, build_time, num_tokens)

            def tokenize_split(df_split: pd.DataFrame) -> pd.DataFrame:
                texts = df_split[args.column].astype(str).tolist()
                token_str_column: List[str] = []
                id_sequences: List[str] = []
                for text in texts:
                    tokens = tokenizer_fn(text)
                    token_str_column.append("".join(tokens))
                    ids = tokens_to_ids(tokens, vocab, unk_token=UNK_TOKEN)
                    id_sequences.append(json.dumps(ids))
                df_split[f"{args.column}_tokens_{mode_normalized}"] = token_str_column
                df_split[f"{args.column}_token_ids"] = id_sequences
                return df_split

            train_df = tokenize_split(train_df)
            val_df = tokenize_split(val_df)
            test_df = tokenize_split(test_df)

        elif mode_normalized == "bpe":
            ignore_spaces = not args.bpe_keep_spaces
            print(
                f"Learning BPE on TRAIN split only with num_merges={args.bpe_merges}, "
                f"min_frequency={args.bpe_min_frequency}, ignore_spaces={ignore_spaces}..."
            )
            start_time = time.time()
            merges, token_to_id = learn_bpe(
                train_texts,
                num_merges=args.bpe_merges,
                min_frequency=args.bpe_min_frequency,
                ignore_spaces=ignore_spaces,
            )

            # Add unknown token for safety on val/test
            if UNK_TOKEN not in token_to_id:
                token_to_id[UNK_TOKEN] = len(token_to_id)

            build_time = time.time() - start_time
            num_tokens = len(token_to_id)

            print(f"Learned {len(merges)} merges.")
            print(f"Final BPE vocab size (train-only): {num_tokens} tokens.")

            bpe_model = {
                "token_to_id": token_to_id,
                "merges": merges,
            }

            vocab_dir = os.path.dirname(args.vocab)
            if vocab_dir:
                os.makedirs(vocab_dir, exist_ok=True)
            with open(args.vocab, "w", encoding="utf-8") as f:
                json.dump(bpe_model, f, ensure_ascii=False, indent=2)
            print(f"Saved BPE model (train-only vocab + merges) to: {args.vocab}")

            # Save vocab stats CSV
            write_vocab_stats_csv(args.vocab, build_time, num_tokens)

            def tokenize_split_bpe(df_split: pd.DataFrame) -> pd.DataFrame:
                texts = df_split[args.column].astype(str).tolist()
                token_str_column: List[str] = []
                id_sequences: List[str] = []
                for text in texts:
                    tokens = bpe_encode(text, merges, ignore_spaces=ignore_spaces)
                    token_str_column.append("".join(tokens))
                    ids: List[int] = []
                    for tok in tokens:
                        if tok in token_to_id:
                            ids.append(token_to_id[tok])
                        else:
                            ids.append(token_to_id[UNK_TOKEN])
                    id_sequences.append(json.dumps(ids))
                df_split[f"{args.column}_tokens_bpe"] = token_str_column
                df_split[f"{args.column}_token_ids"] = id_sequences
                return df_split

            train_df = tokenize_split_bpe(train_df)
            val_df = tokenize_split_bpe(val_df)
            test_df = tokenize_split_bpe(test_df)

        elif mode_normalized == "hierarchical":
            print(
                "Learning hierarchical vocabulary on TRAIN split only "
                "(AST-based tokenizer with Transformer-friendly vocab)..."
            )
            # Tokenize TRAIN split to build vocab
            train_tokenized_sequences: List[List[str]] = []
            for text in train_texts:
                train_tokenized_sequences.append(hier_tokenize(text))

            start_time = time.time()
            vocab = build_hier_vocab(train_tokenized_sequences)
            build_time = time.time() - start_time
            num_tokens = len(vocab)

            vocab_dir = os.path.dirname(args.vocab)
            if vocab_dir:
                os.makedirs(vocab_dir, exist_ok=True)

            vocab_json = {
                "token_to_id": vocab,
                "special_tokens": {
                    "pad_token": "<pad>",
                    "unk_token": "<unk>",
                    "bos_token": "<bos>",
                    "eos_token": "<eos>",
                },
            }
            with open(args.vocab, "w", encoding="utf-8") as f:
                json.dump(vocab_json, f, ensure_ascii=False, indent=2)
            print(f"Saved hierarchical vocabulary (train-only) to: {args.vocab}")

            # Save vocab stats CSV
            write_vocab_stats_csv(args.vocab, build_time, num_tokens)

            def tokenize_split_hier(df_split: pd.DataFrame) -> pd.DataFrame:
                texts = df_split[args.column].astype(str).tolist()
                token_str_column: List[str] = []
                id_sequences: List[str] = []
                for text in texts:
                    tokens = hier_tokenize(text)
                    token_str_column.append(" ".join(tokens))
                    ids = tokens_to_ids(tokens, vocab, unk_token=UNK_TOKEN)
                    id_sequences.append(json.dumps(ids))
                df_split[f"{args.column}_tokens_hier"] = token_str_column
                df_split[f"{args.column}_token_ids"] = id_sequences
                return df_split

            train_df = tokenize_split_hier(train_df)
            val_df = tokenize_split_hier(val_df)
            test_df = tokenize_split_hier(test_df)

        else:
            raise ValueError(f"Unknown mode '{mode_normalized}'. Expected one of: char, byte, bpe, hierarchical.")

        # Create output directory for splits
        os.makedirs(args.output_dir, exist_ok=True)

        mode_suffix = mode_normalized  # e.g. bpe, char, byte, hierarchical

        train_path = os.path.join(args.output_dir, f"tokenized_train_{mode_suffix}.csv")
        val_path = os.path.join(args.output_dir, f"tokenized_val_{mode_suffix}.csv")
        test_path = os.path.join(args.output_dir, f"tokenized_test_{mode_suffix}.csv")

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(f"Train: {len(train_df)} -> {train_path}")
        print(f"Val:   {len(val_df)} -> {val_path}")
        print(f"Test:  {len(test_df)} -> {test_path}")

        # Save full tokenized CSV as concatenation of splits
        full_output_dir = os.path.dirname(args.output)
        if full_output_dir:
            os.makedirs(full_output_dir, exist_ok=True)

        full_df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
        full_df.to_csv(args.output, index=False)
        print(f"Saved FULL tokenized CSV (train/val/test concatenated) to: {args.output}")


if __name__ == "__main__":
    main()
