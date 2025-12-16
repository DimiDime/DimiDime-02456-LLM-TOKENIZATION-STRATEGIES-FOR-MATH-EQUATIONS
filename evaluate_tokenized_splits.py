#!/usr/bin/env python3
"""
token_stats.py

Compute statistics for tokenized splits stored in CSVs like:
  tokenized_train_bpe.csv
  tokenized_train_char.csv
  tokenized_train_hierarchical.csv
(and val/test variants)

Expected columns:
  - full_equation
  - full_equation_token_ids   (stringified list of ints)
Optional:
  - id
  - complexity

Typical pipeline usage:
  python token_stats.py \
    --input-dir "HPC pipeline/data/splits" \
    --output-prefix "HPC pipeline/data/tokenization/token_stats" \
    --chunksize 200000 --sample-size 200000 --top-k 20
"""

from __future__ import annotations
import argparse
import csv
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
import pandas as pd


FILE_RE = re.compile(r"tokenized_(?P<split>train|val|test)_(?P<tok>[a-zA-Z0-9]+)\.csv$")

def safe_len(x) -> int:
    if x is None:
        return 0
    if isinstance(x, float) and math.isnan(x):
        return 0
    return len(x)

def parse_token_ids(val) -> List[int]:
    """
    CSV stores lists like: "[656, 502, 275]" (valid JSON)
    """
    if val is None:
        return []
    if isinstance(val, float) and math.isnan(val):
        return []
    if isinstance(val, list):
        return [int(v) for v in val]

    s = str(val).strip()
    if not s:
        return []

    try:
        out = json.loads(s)
        if isinstance(out, list):
            return [int(v) for v in out]
    except Exception:
        pass

    # fallback
    s = s.strip("[] \t\n\r")
    if not s:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    ids: List[int] = []
    for p in parts:
        try:
            ids.append(int(p))
        except Exception:
            continue
    return ids


@dataclass
class OnlineMoments:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min_v: float = float("inf")
    max_v: float = float("-inf")

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2
        if x < self.min_v:
            self.min_v = x
        if x > self.max_v:
            self.max_v = x

    @property
    def var(self) -> float:
        return self.m2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.var)

    def finalize(self) -> Dict[str, float]:
        return {
            "n": self.n,
            "mean": float(self.mean) if self.n else 0.0,
            "std": float(self.std) if self.n else 0.0,
            "min": float(self.min_v) if self.n else 0.0,
            "max": float(self.max_v) if self.n else 0.0,
        }


class ReservoirSampler:
    """Reservoir sampling to estimate quantiles without storing all values."""
    def __init__(self, k: int, seed: int = 0) -> None:
        self.k = max(0, int(k))
        self.rng = random.Random(seed)
        self.n_seen = 0
        self.sample: List[float] = []

    def update(self, x: float) -> None:
        if self.k == 0:
            self.n_seen += 1
            return
        self.n_seen += 1
        if len(self.sample) < self.k:
            self.sample.append(x)
            return
        j = self.rng.randint(1, self.n_seen)
        if j <= self.k:
            self.sample[j - 1] = x

    def quantiles(self, qs: Iterable[float]) -> Dict[str, float]:
        if not self.sample:
            return {f"p{int(q * 100)}": 0.0 for q in qs}
        xs = sorted(self.sample)
        out: Dict[str, float] = {}
        for q in qs:
            q = float(q)
            if q <= 0:
                out[f"p{int(q * 100)}"] = float(xs[0])
                continue
            if q >= 1:
                out[f"p{int(q * 100)}"] = float(xs[-1])
                continue
            idx = int(round(q * (len(xs) - 1)))
            out[f"p{int(q * 100)}"] = float(xs[idx])
        return out


@dataclass
class TopKItem:
    token_len: int
    eq_len: int
    row_id: Optional[int]
    equation_preview: str


class TopKLongest:
    """Keep top-K by token length (K is small)."""
    def __init__(self, k: int) -> None:
        self.k = max(0, int(k))
        self.items: List[TopKItem] = []

    def update(self, token_len: int, eq_len: int, row_id: Optional[int], equation: str) -> None:
        if self.k == 0:
            return
        preview = (equation or "").replace("\n", " ")[:200]
        item = TopKItem(token_len=token_len, eq_len=eq_len, row_id=row_id, equation_preview=preview)

        inserted = False
        for i, it in enumerate(self.items):
            if token_len > it.token_len:
                self.items.insert(i, item)
                inserted = True
                break
        if not inserted:
            self.items.append(item)
        if len(self.items) > self.k:
            self.items = self.items[: self.k]


@dataclass
class RunningCorrelation:
    """Online correlation between X and Y using running sums."""
    n: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_x2: float = 0.0
    sum_y2: float = 0.0
    sum_xy: float = 0.0

    def update(self, x: float, y: float) -> None:
        self.n += 1
        self.sum_x += x
        self.sum_y += y
        self.sum_x2 += x * x
        self.sum_y2 += y * y
        self.sum_xy += x * y

    def corr(self) -> float:
        if self.n < 2:
            return 0.0
        num = self.n * self.sum_xy - self.sum_x * self.sum_y
        den_x = self.n * self.sum_x2 - self.sum_x * self.sum_x
        den_y = self.n * self.sum_y2 - self.sum_y * self.sum_y
        den = math.sqrt(max(den_x, 0.0) * max(den_y, 0.0))
        if den == 0.0:
            return 0.0
        return float(num / den)


def compute_stats_for_file(
    path: Path,
    chunksize: int,
    sample_size: int,
    top_k: int,
    seed: int,
) -> Dict:
    tok_m = OnlineMoments()
    eq_m = OnlineMoments()
    uniq_tok_per_eq_m = OnlineMoments()

    tok_sampler = ReservoirSampler(sample_size, seed=seed)
    eq_sampler = ReservoirSampler(sample_size, seed=seed + 1)

    vocab: set = set()
    topk = TopKLongest(top_k)

    corr_tok_eq = RunningCorrelation()
    corr_tok_complexity = RunningCorrelation()

    complexity_counts: Dict[int, int] = {}
    complexity_tok_sum: Dict[int, int] = {}

    for chunk in pd.read_csv(path, chunksize=chunksize):
        if "full_equation" not in chunk.columns or "full_equation_token_ids" not in chunk.columns:
            raise ValueError(
                f"{path.name} missing required columns. "
                f"Need 'full_equation' and 'full_equation_token_ids'. Found: {list(chunk.columns)}"
            )

        ids_col = chunk["id"] if "id" in chunk.columns else None
        eq_col = chunk["full_equation"]
        tokid_col = chunk["full_equation_token_ids"]
        comp_col = chunk["complexity"] if "complexity" in chunk.columns else None

        for i in range(len(chunk)):
            eq = eq_col.iat[i]
            row_id: Optional[int] = None
            if ids_col is not None:
                try:
                    row_id = int(ids_col.iat[i])
                except Exception:
                    row_id = None

            token_ids = parse_token_ids(tokid_col.iat[i])
            tlen = len(token_ids)
            elen = safe_len(eq)

            tok_m.update(float(tlen))
            eq_m.update(float(elen))
            uniq_tok_per_eq_m.update(float(len(set(token_ids)) if token_ids else 0))

            tok_sampler.update(float(tlen))
            eq_sampler.update(float(elen))

            topk.update(tlen, elen, row_id, str(eq) if eq is not None else "")

            for tid in token_ids:
                vocab.add(tid)

            corr_tok_eq.update(float(tlen), float(elen))
            if comp_col is not None:
                try:
                    c = int(comp_col.iat[i])
                    corr_tok_complexity.update(float(tlen), float(c))
                    complexity_counts[c] = complexity_counts.get(c, 0) + 1
                    complexity_tok_sum[c] = complexity_tok_sum.get(c, 0) + tlen
                except Exception:
                    pass

    tok_q = tok_sampler.quantiles([0.5, 0.9, 0.99])
    eq_q = eq_sampler.quantiles([0.5, 0.9, 0.99])

    avg_chars_per_token = (eq_m.mean / tok_m.mean) if tok_m.mean > 0 else 0.0

    comp_summary = []
    for c in sorted(complexity_counts.keys()):
        n = complexity_counts[c]
        s = complexity_tok_sum.get(c, 0)
        comp_summary.append({"complexity": c, "n": n, "avg_tokens": (s / n) if n else 0.0})

    return {
        "file": path.name,
        "path": str(path),
        "n_rows": tok_m.n,
        "token_count": {**tok_m.finalize(), **tok_q},
        "equation_chars": {**eq_m.finalize(), **eq_q},
        "avg_chars_per_token": float(avg_chars_per_token),
        "vocab_size": int(len(vocab)),
        "unique_tokens_per_equation": uniq_tok_per_eq_m.finalize(),
        "corr_token_vs_eq_chars": float(corr_tok_eq.corr()),
        "corr_token_vs_complexity": float(corr_tok_complexity.corr()) if corr_tok_complexity.n else 0.0,
        "by_complexity": comp_summary,
        "top_k_longest_by_tokens": [asdict(x) for x in topk.items],
    }


def discover_files(input_dir: Path) -> List[Path]:
    out: List[Path] = []
    for p in sorted(input_dir.iterdir()):
        if p.is_file() and FILE_RE.match(p.name):
            out.append(p)
    return out


def attach_split_tokenizer(stats: Dict) -> None:
    m = FILE_RE.match(Path(stats["file"]).name)
    if not m:
        stats["split"] = ""
        stats["tokenizer"] = ""
        return
    stats["split"] = m.group("split")
    stats["tokenizer"] = m.group("tok")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute stats for tokenized split CSVs.")
    ap.add_argument("--input-dir", type=str, default=None,
                    help="Directory containing tokenized_{train|val|test}_*.csv files.")
    ap.add_argument("--files", nargs="*", default=None,
                    help="Explicit list of CSV files (overrides --input-dir scanning).")

    ap.add_argument("--output-prefix", type=str, required=True,
                    help="Prefix for outputs. Writes <prefix>.json and <prefix>.csv")
    ap.add_argument("--write-topk-csv", action="store_true",
                    help="Also write <prefix>_topk.csv with the top-k examples per file.")

    ap.add_argument("--chunksize", type=int, default=200_000, help="CSV rows per chunk")
    ap.add_argument("--sample-size", type=int, default=200_000,
                    help="Reservoir sample size for quantiles (0 disables)")
    ap.add_argument("--top-k", type=int, default=10, help="Keep top-K longest sequences by token count")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    ap.add_argument("--strict", action="store_true",
                    help="Fail if any discovered file doesn't match naming pattern.")
    args = ap.parse_args()

    out_prefix = Path(args.output_prefix)
    ensure_parent(out_prefix.with_suffix(".json"))
    ensure_parent(out_prefix.with_suffix(".csv"))
    if args.write_topk_csv:
        ensure_parent(Path(str(out_prefix) + "_topk.csv"))

    # Decide which files to process
    files: List[Path] = []
    if args.files:
        files = [Path(f) for f in args.files]
    else:
        if not args.input_dir:
            print("[ERROR] Provide --input-dir or --files", file=sys.stderr)
            return 2
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            print(f"[ERROR] input-dir not found: {input_dir}", file=sys.stderr)
            return 2
        files = discover_files(input_dir)

    if not files:
        print("[ERROR] No input files found.", file=sys.stderr)
        return 2

    # Optional strict checks
    if args.strict:
        for f in files:
            if not FILE_RE.match(f.name):
                print(f"[ERROR] File name does not match expected pattern: {f.name}", file=sys.stderr)
                return 2

    # Compute stats
    results: List[Dict] = []
    for f in files:
        if not f.exists():
            print(f"[ERROR] Missing file: {f}", file=sys.stderr)
            return 2

        print("=" * 70)
        print(f"token_stats :: processing: {f}")
        print("=" * 70)

        try:
            stats = compute_stats_for_file(
                path=f,
                chunksize=args.chunksize,
                sample_size=args.sample_size,
                top_k=args.top_k,
                seed=args.seed,
            )
            attach_split_tokenizer(stats)
            results.append(stats)
        except Exception as e:
            print(f"[ERROR] Failed on {f}: {e}", file=sys.stderr)
            return 1

    # Write JSON (full detail)
    out_json = out_prefix.with_suffix(".json")
    with out_json.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
    print(f"[OK] Wrote {out_json}")

    # Write flattened CSV summary
    out_csv = out_prefix.with_suffix(".csv")
    fieldnames = [
        "split", "tokenizer", "file", "path", "n_rows",
        "avg_tokens", "std_tokens", "min_tokens", "max_tokens", "p50_tokens", "p90_tokens", "p99_tokens",
        "avg_eq_chars", "std_eq_chars", "min_eq_chars", "max_eq_chars", "p50_eq_chars", "p90_eq_chars", "p99_eq_chars",
        "avg_chars_per_token",
        "vocab_size",
        "avg_unique_tokens_per_eq",
        "corr_token_vs_eq_chars",
        "corr_token_vs_complexity",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            tc = r["token_count"]
            ec = r["equation_chars"]
            w.writerow({
                "split": r.get("split", ""),
                "tokenizer": r.get("tokenizer", ""),
                "file": r["file"],
                "path": r["path"],
                "n_rows": r["n_rows"],

                "avg_tokens": tc["mean"],
                "std_tokens": tc["std"],
                "min_tokens": tc["min"],
                "max_tokens": tc["max"],
                "p50_tokens": tc.get("p50", 0.0),
                "p90_tokens": tc.get("p90", 0.0),
                "p99_tokens": tc.get("p99", 0.0),

                "avg_eq_chars": ec["mean"],
                "std_eq_chars": ec["std"],
                "min_eq_chars": ec["min"],
                "max_eq_chars": ec["max"],
                "p50_eq_chars": ec.get("p50", 0.0),
                "p90_eq_chars": ec.get("p90", 0.0),
                "p99_eq_chars": ec.get("p99", 0.0),

                "avg_chars_per_token": r["avg_chars_per_token"],
                "vocab_size": r["vocab_size"],
                "avg_unique_tokens_per_eq": r["unique_tokens_per_equation"]["mean"],
                "corr_token_vs_eq_chars": r["corr_token_vs_eq_chars"],
                "corr_token_vs_complexity": r["corr_token_vs_complexity"],
            })
    print(f"[OK] Wrote {out_csv}")

    # Optional top-k CSV
    if args.write_topk_csv:
        out_topk = Path(str(out_prefix) + "_topk.csv")
        with out_topk.open("w", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=[
                "split", "tokenizer", "file", "row_id", "token_len", "eq_len", "equation_preview"
            ])
            w.writeheader()
            for r in results:
                for item in r["top_k_longest_by_tokens"]:
                    w.writerow({
                        "split": r.get("split", ""),
                        "tokenizer": r.get("tokenizer", ""),
                        "file": r["file"],
                        "row_id": item.get("row_id"),
                        "token_len": item.get("token_len"),
                        "eq_len": item.get("eq_len"),
                        "equation_preview": item.get("equation_preview", ""),
                    })
        print(f"[OK] Wrote {out_topk}")

    # Print concise summary (pipeline logs)
    print("\nSummary:")
    for r in results:
        print(
            f"- {r.get('split',''):>5} / {r.get('tokenizer',''):<12} "
            f"rows={r['n_rows']:,}  "
            f"avg_tokens={r['token_count']['mean']:.2f}  "
            f"avg_eq_chars={r['equation_chars']['mean']:.2f}  "
            f"vocab={r['vocab_size']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
