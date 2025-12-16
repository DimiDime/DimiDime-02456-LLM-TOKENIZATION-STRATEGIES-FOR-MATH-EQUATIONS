#!/usr/bin/env python
import argparse
import ast
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from functools import partial

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel
import psutil  # for CPU memory logging
from torch.cuda.amp import autocast, GradScaler


# =========================
# Seeding
# =========================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        # For speed: allow cuDNN to pick fastest algorithms (non-deterministic)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


# =========================
# Dataset
# =========================

class PreTokenizedEquationDataset(Dataset):
    """
    Dataset for pre-tokenized equations.

    Expected CSV columns:
      - ids_column: string representation of a Python list of ints
                    e.g. "[49, 5, 2, 29, ...]"
      - text_column: original equation string (used for char_count / BPC)

    We store:
      - token_ids: List[int]
      - char_count: int (len of original text)
    """

    def __init__(
        self,
        csv_path: str,
        ids_column: str,
        text_column: str,
    ):
        self.examples: List[Dict[str, Any]] = []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if ids_column not in reader.fieldnames:
                raise ValueError(
                    f"Column '{ids_column}' not found in {csv_path}. "
                    f"Available columns: {reader.fieldnames}"
                )
            if text_column not in reader.fieldnames:
                raise ValueError(
                    f"Column '{text_column}' not found in {csv_path}. "
                    f"Available columns: {reader.fieldnames}"
                )

            for row in reader:
                ids_str = row[ids_column]
                if ids_str is None or ids_str.strip() == "":
                    continue

                # Parse "[1, 2, 3]" -> [1, 2, 3]
                try:
                    token_ids = ast.literal_eval(ids_str)
                except Exception as e:
                    raise ValueError(
                        f"Failed to parse token ids from value '{ids_str}' in file {csv_path}"
                    ) from e

                if not isinstance(token_ids, list) or len(token_ids) == 0:
                    continue

                # ensure ints
                token_ids = [int(x) for x in token_ids]

                text = row[text_column] or ""
                text = text.strip()
                char_count = len(text)

                self.examples.append(
                    {
                        "token_ids": token_ids,
                        "char_count": char_count,
                    }
                )

        if len(self.examples) == 0:
            raise ValueError(f"No valid examples loaded from {csv_path}.")

        print(
            f"Loaded {len(self.examples)} examples from {csv_path} "
            f"(ids_column='{ids_column}', text_column='{text_column}')"
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "token_ids": ex["token_ids"],
            "char_count": ex["char_count"],
        }

def collate_batch(
    batch: List[Dict[str, Any]],
    max_length: int,
    pad_id: int,
) -> Dict[str, torch.Tensor]:
    """
    Collate function that:
      - truncates sequences to max_length
      - pads with pad_id up to max_length
      - builds attention_mask and labels
    """
    batch_size = len(batch)
    input_ids = torch.full((batch_size, max_length), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
    labels = torch.full((batch_size, max_length), -100, dtype=torch.long)
    char_counts = torch.zeros(batch_size, dtype=torch.long)

    for i, ex in enumerate(batch):
        seq: List[int] = ex["token_ids"]
        L = min(len(seq), max_length)
        if L > 0:
            input_ids[i, :L] = torch.tensor(seq[:L], dtype=torch.long)
            attention_mask[i, :L] = 1
            labels[i, :L] = input_ids[i, :L]
        char_counts[i] = int(ex["char_count"])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "char_count": char_counts,
    }


# =========================
# Model builder
# =========================

@dataclass
class ModelSizeConfig:
    n_layer: int
    n_head: int
    n_embd: int


MODEL_PRESETS = {
    # good for labtops
    "tiny": ModelSizeConfig(n_layer=4, n_head=4, n_embd=256),
    # a bit larger
    "small": ModelSizeConfig(n_layer=6, n_head=6, n_embd=384),
    # HPC-style, closer to GPT-2 small
    "medium": ModelSizeConfig(n_layer=12, n_head=12, n_embd=768),
}


def build_model(
    vocab_size: int,
    pad_id: int,
    max_length: int,
    model_size: str,
    n_layer: Optional[int] = None,
    n_head: Optional[int] = None,
    n_embd: Optional[int] = None,
) -> GPT2LMHeadModel:
    if model_size not in MODEL_PRESETS and model_size != "custom":
        raise ValueError(
            f"Unknown model_size '{model_size}'. "
            f"Choose from {list(MODEL_PRESETS.keys()) + ['custom']}"
        )

    if model_size == "custom":
        if n_layer is None or n_head is None or n_embd is None:
            raise ValueError(
                "For model_size='custom', you must specify --n-layer, --n-head, and --n-embd."
            )
        cfg = ModelSizeConfig(n_layer=n_layer, n_head=n_head, n_embd=n_embd)
    else:
        cfg = MODEL_PRESETS[model_size]

    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=max_length,
        n_ctx=max_length,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        pad_token_id=pad_id,
        bos_token_id=None,
        eos_token_id=None,
    )

    model = GPT2LMHeadModel(config)
    return model


# =========================
# Training & evaluation
# =========================

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    epoch_idx: int,
    log_interval: int = 100,
    scaler: Optional[GradScaler] = None,
):
    """
    Train for a single epoch.

    Uses:
      - Cross-entropy loss (HF GPT2LMHeadModel + labels)
      - Next-token prediction (causal LM)

    Logs:
      - loss (nats/token)
      - perplexity
      - tokens/sec
      - CPU memory (MB)
      - GPU memory (MB, if CUDA)
      - wall-clock time (s)
    """
    model.train()
    total_nats = 0.0
    total_tokens = 0
    start_time = time.time()

    # reset GPU memory tracking (for per-epoch peak)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None and device.type == "cuda":
            # Mixed precision training
            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss  # average nats/token in this batch

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Full precision fallback (CPU / MPS / no CUDA)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        num_tokens = attention_mask.sum().item()
        total_tokens += num_tokens
        total_nats += loss.item() * num_tokens

        if (step + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            avg_loss = total_nats / total_tokens
            ppl = math.exp(avg_loss)
            tokens_per_sec = total_tokens / max(elapsed, 1e-6)
            print(
                f"[Epoch {epoch_idx}] Step {step+1}/{len(dataloader)} "
                f"loss={avg_loss:.4f} ppl={ppl:.2f} tokens/sec={tokens_per_sec:.1f}"
            )

    epoch_time = time.time() - start_time
    avg_loss = total_nats / total_tokens
    ppl = math.exp(avg_loss)
    tokens_per_sec = total_tokens / max(epoch_time, 1e-6)

    # CPU memory
    process = psutil.Process(os.getpid())
    cpu_mem_mb = process.memory_info().rss / (1024 ** 2)

    # GPU memory (if CUDA)
    gpu_mem_mb = None
    if torch.cuda.is_available():
        gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print(
        f"[Epoch {epoch_idx}] Epoch done | "
        f"loss={avg_loss:.4f} ppl={ppl:.2f} "
        f"tokens/sec={tokens_per_sec:.1f} "
        f"wall_clock={epoch_time:.1f}s "
        f"cpu_mem={cpu_mem_mb:.1f}MB"
        + (f" gpu_mem={gpu_mem_mb:.1f}MB" if gpu_mem_mb is not None else "")
    )

    return avg_loss, ppl, tokens_per_sec, epoch_time, cpu_mem_mb, gpu_mem_mb


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_nats = 0.0
    total_tokens = 0
    total_chars = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        char_counts = batch["char_count"].to(device, non_blocking=True)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss  # average nats/token

        num_tokens = attention_mask.sum().item()
        total_tokens += num_tokens
        total_nats += loss.item() * num_tokens
        total_chars += char_counts.sum().item()

    avg_loss = total_nats / total_tokens
    ppl = math.exp(avg_loss)

    bpc = (total_nats / math.log(2)) / max(total_chars, 1)

    return avg_loss, ppl, bpc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a GPT-2-style Transformer LM on pre-tokenized equations."
    )

    # Data
    parser.add_argument("--train-path", type=str, required=True,
                        help="Path to train CSV file.")
    parser.add_argument("--val-path", type=str, required=True,
                        help="Path to validation CSV file.")
    parser.add_argument("--test-path", type=str, default=None,
                        help="Path to test CSV file (optional).")
    parser.add_argument("--ids-column", type=str, default="full_equation_token_ids",
                        help="CSV column with token id lists.")
    parser.add_argument("--text-column", type=str, default="full_equation",
                        help="CSV column with original equation text (for BPC).")

    # Vocab / pad
    parser.add_argument("--vocab-json", type=str, required=True,
                        help="Path to vocab JSON.")
    parser.add_argument("--pad-id", type=int, default=None,
                        help=(
                            "Pad token id. "
                            "If not set, we infer vocab size and use the next id for padding."
                        ))

    # Model size
    parser.add_argument("--model-size", type=str, default="tiny",
                        choices=list(MODEL_PRESETS.keys()) + ["custom"],
                        help="Model size preset or 'custom'.")
    parser.add_argument("--n-layer", type=int, default=None,
                        help="Number of layers (if model-size='custom').")
    parser.add_argument("--n-head", type=int, default=None,
                        help="Number of attention heads (if model-size='custom').")
    parser.add_argument("--n-embd", type=int, default=None,
                        help="Embedding/hidden size (if model-size='custom').")

    # Training
    parser.add_argument("--max-length", type=int, default=128,
                        help="Max sequence length in tokens (for truncation/padding).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--log-interval", type=int, default=100)

    # DataLoader / system
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Number of worker processes for DataLoader (default: 0).")

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="checkpoints_transformer",
                        help="Where to save checkpoints and metrics.")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device: 'cpu', 'cuda', or 'mps'. Default: auto-detect.")

    # Tag for CSV & best-model filename (e.g. bpe, char, hierarchical)
    parser.add_argument("--tokenizer-type", type=str, default="generic",
                        help="Short tag like 'bpe', 'char', 'hierarchical' used in filenames.")

    return parser.parse_args()


def main():
    args = parse_args()

    # Seed
    set_seed(args.seed)

    # Device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    print(f"Using device: {device}")

    # --------- Vocab loading ---------
    with open(args.vocab_json, "r", encoding="utf-8") as f:
        vocab_raw = json.load(f)

    base_vocab_size = None

    # Case 1: our BPE format: {"token_to_id": {...}, "merges": [...]}
    if isinstance(vocab_raw, dict) and "token_to_id" in vocab_raw and isinstance(vocab_raw["token_to_id"], dict):
        token_to_id = vocab_raw["token_to_id"]
        if len(token_to_id) == 0:
            raise ValueError("Empty 'token_to_id' vocab in vocab JSON.")
        base_vocab_size = max(token_to_id.values()) + 1
        print(f"Detected 'token_to_id' vocab with size={base_vocab_size}")

    # Case 2: classic {"stoi": {...}}
    elif isinstance(vocab_raw, dict) and "stoi" in vocab_raw and isinstance(vocab_raw["stoi"], dict):
        stoi = vocab_raw["stoi"]
        if len(stoi) == 0:
            raise ValueError("Empty 'stoi' vocab in vocab JSON.")
        base_vocab_size = max(stoi.values()) + 1
        print(f"Detected 'stoi' vocab with size={base_vocab_size}")

    # Case 3: flat {"token": id, ...}
    elif isinstance(vocab_raw, dict) and len(vocab_raw) > 0 and all(isinstance(v, int) for v in vocab_raw.values()):
        base_vocab_size = max(vocab_raw.values()) + 1
        print(f"Detected flat token->id vocab with size={base_vocab_size}")

    # Fallback
    else:
        base_vocab_size = len(vocab_raw)
        print(f"Fallback vocab size from len(vocab_raw) = {base_vocab_size}")

    if base_vocab_size <= 0:
        raise ValueError("Inferred base_vocab_size <= 0 from vocab JSON, something is wrong.")

    if args.pad_id is None:
        pad_id = base_vocab_size  # add one new id for padding
        vocab_size = base_vocab_size + 1
        print(
            f"No pad_id provided. Using pad_id={pad_id} and "
            f"vocab_size={vocab_size} (base={base_vocab_size})."
        )
    else:
        pad_id = args.pad_id
        vocab_size = max(base_vocab_size, pad_id + 1)
        print(
            f"Using provided pad_id={pad_id}. "
            f"Base vocab size={base_vocab_size}, effective vocab_size={vocab_size}."
        )

    # Datasets
    print("Loading datasets...")
    train_ds = PreTokenizedEquationDataset(
        csv_path=args.train_path,
        ids_column=args.ids_column,
        text_column=args.text_column,
    )
    val_ds = PreTokenizedEquationDataset(
        csv_path=args.val_path,
        ids_column=args.ids_column,
        text_column=args.text_column,
    )
    test_ds = None
    if args.test_path:
        test_ds = PreTokenizedEquationDataset(
            csv_path=args.test_path,
            ids_column=args.ids_column,
            text_column=args.text_column,
        )

    # DataLoaders (with picklable collate + multiprocessing)
    collate_fn = partial(
        collate_batch,
        max_length=args.max_length,
        pad_id=pad_id,
    )
    pin_memory = (device.type == "cuda")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )

    # Model
    print("Building model...")
    model = build_model(
        vocab_size=vocab_size,
        pad_id=pad_id,
        max_length=args.max_length,
        model_size=args.model_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # AMP scaler (CUDA only)
    scaler = GradScaler() if device.type == "cuda" else None

    # Output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Metrics CSV setup
    metrics_fieldnames = [
        "epoch",
        "train_loss",
        "train_ppl",
        "train_tokens_per_sec",
        "train_epoch_time_sec",
        "train_cpu_mem_mb",
        "train_gpu_mem_mb",
        "val_loss",
        "val_ppl",
        "val_bpc",
    ]
    metrics_rows: List[Dict[str, Any]] = []

    best_val_loss = float("inf")
    best_model_path = os.path.join(
        args.output_dir,
        f"best_transformer_{args.tokenizer_type}.pth",
    )

    # Training loop
    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_loss, train_ppl, train_tps, train_time, cpu_mem_mb, gpu_mem_mb = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch_idx=epoch,
            log_interval=args.log_interval,
            scaler=scaler,
        )

        val_loss, val_ppl, val_bpc = evaluate(model, val_loader, device)
        print(
            f"[Validation] loss={val_loss:.4f} ppl={val_ppl:.2f} "
            f"bpc={val_bpc:.4f}"
        )

        # Record metrics for this epoch
        metrics_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_ppl": train_ppl,
                "train_tokens_per_sec": train_tps,
                "train_epoch_time_sec": train_time,
                "train_cpu_mem_mb": cpu_mem_mb,
                "train_gpu_mem_mb": gpu_mem_mb if gpu_mem_mb is not None else "",
                "val_loss": val_loss,
                "val_ppl": val_ppl,
                "val_bpc": val_bpc,
            }
        )

        # Track best model and save ONLY the best as .pth
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                },
                best_model_path,
            )
            print(f"[Checkpoint] New best model saved to {best_model_path}")

    # Final test evaluation using the best validation checkpoint
    if test_loader is not None:
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"\n[Checkpoint] Loaded best model from {best_model_path} "
            f"(val_loss={checkpoint.get('val_loss', float('nan')):.4f})"
        )

        test_loss, test_ppl, test_bpc = evaluate(model, test_loader, device)
        print(
            f"\n[Test] loss={test_loss:.4f} ppl={test_ppl:.2f} "
            f"bpc={test_bpc:.4f}"
        )

    # Save metrics CSV
    csv_filename = f"transformer_{args.tokenizer_type}.csv"
    csv_path = os.path.join(args.output_dir, csv_filename)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=metrics_fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)
    print(f"[Metrics] Saved training metrics to {csv_path}")
    print(f"[Checkpoint] Best model is stored at {best_model_path}")


if __name__ == "__main__":
    main()
