#!/usr/bin/env python
"""LSTM language model training with pre-tokenized equations.

Similar structure to transformer.py, training a next-token prediction model
with cross-entropy loss and reporting loss, tokens/second, memory usage, and 
wall-clock time per epoch.
"""

import argparse
import ast
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import psutil 


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
        # Deterministic = True: nice for reproducibility, but slower.
        # More speed can be achieved by setting deterministic=False & benchmark=True.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =========================
# Dataset
# =========================

class PreTokenizedEquationDataset(Dataset):
    """
    Dataset for pre-tokenized equations.

    Expected CSV columns:
      - ids_column: string representation of a Python list of ints
                    e.g. "[49, 5, 2, 29, ...]"
      - text_column: original equation string (used for char_count)

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


def create_collate_fn(max_length: int, pad_id: int):
    """
    Returns a collate_fn that:
      - truncates sequences to max_length
      - pads with pad_id up to max_length
      - builds attention_mask and labels for LSTM
    """

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
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
                # NEXT-TOKEN PREDICTION:
                # labels = input_ids, each position predicts the next token.
                labels[i, :L] = input_ids[i, :L]
            char_counts[i] = int(ex["char_count"])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "char_count": char_counts,
        }

    return collate


# =========================
# Model
# =========================

@dataclass
class LSTMSizePreset:
    """Preset describing an LSTM architecture size."""
    embedding_dim: int
    hidden_size: int
    num_layers: int


LSTM_MODEL_PRESETS: Dict[str, LSTMSizePreset] = {
    # Close to the previous defaults
    "tiny": LSTMSizePreset(embedding_dim=256, hidden_size=256, num_layers=2),
    # A step up in capacity
    "small": LSTMSizePreset(embedding_dim=320, hidden_size=512, num_layers=3),
    # Heavier option for HPC runs
    "medium": LSTMSizePreset(embedding_dim=512, hidden_size=1024, num_layers=4),
}


@dataclass
class LSTMConfig:
    """Configuration for LSTM model."""
    vocab_size: int
    hidden_size: int = 256
    num_layers: int = 2
    embedding_dim: int = 256


class LSTMLanguageModel(nn.Module):
    """LSTM-based language model for next-token prediction."""

    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for LSTM language model.

        Args:
            input_ids: [batch_size, seq_len]

        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        x = self.embedding(input_ids)  # [batch, seq, embedding_dim]
        x, _ = self.lstm(x)           # [batch, seq, hidden_size]
        logits = self.head(x)         # [batch, seq, vocab_size]
        return logits


def resolve_lstm_config(args, vocab_size: int) -> LSTMConfig:
    """Map CLI args into an LSTMConfig, supporting presets and custom values."""
    if args.model_size == "custom":
        return LSTMConfig(
            vocab_size=vocab_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            embedding_dim=args.embedding_dim,
        )

    if args.model_size not in LSTM_MODEL_PRESETS:
        raise ValueError(
            f"Unknown model_size '{args.model_size}'. Choose from {list(LSTM_MODEL_PRESETS.keys()) + ['custom']}"
        )

    preset = LSTM_MODEL_PRESETS[args.model_size]
    return LSTMConfig(
        vocab_size=vocab_size,
        hidden_size=preset.hidden_size,
        num_layers=preset.num_layers,
        embedding_dim=preset.embedding_dim,
    )


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
):
    """
    Train for a single epoch.

    Uses:
      - Cross-entropy loss
      - Next-token prediction

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
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # reset GPU memory tracking (for per-epoch peak)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad()

        # Model outputs logits, we compute next-token cross-entropy
        logits = model(input_ids)  # [batch, seq, vocab]
        loss = criterion(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

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
    """
    Evaluate model on validation/test set.

    Returns:
      - avg_loss (nats/token)
      - perplexity
      - bits-per-character
    """
    model.eval()
    total_nats = 0.0
    total_tokens = 0
    total_chars = 0
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        char_counts = batch["char_count"].to(device, non_blocking=True)

        logits = model(input_ids)
        loss = criterion(
           logits.view(-1, logits.size(-1)),
           labels.view(-1),
        )

        num_tokens = attention_mask.sum().item()
        total_tokens += num_tokens
        total_nats += loss.item() * num_tokens
        total_chars += char_counts.sum().item()

    avg_loss = total_nats / total_tokens
    ppl = math.exp(avg_loss)

    bpc = (total_nats / math.log(2)) / max(total_chars, 1)

    return avg_loss, ppl, bpc


# =========================
# CLI helpers
# =========================

def infer_tokenization_method(train_path: str, vocab_path: str) -> str:
    """
    Infer tokenization method from file paths.
    Returns 'bpe', 'char', 'hierarchical', or 'unknown'.
    """
    train_lower = train_path.lower()
    vocab_lower = vocab_path.lower()
    
    if 'hierarchical' in train_lower or 'hierarchical' in vocab_lower:
        return 'hierarchical'
    elif 'bpe' in train_lower or 'bpe' in vocab_lower:
        return 'bpe'
    elif 'char' in train_lower or 'char' in vocab_lower:
        return 'char'
    else:
        return 'unknown'


def save_epoch_metrics(csv_path: str, epoch_data: List[Dict[str, Any]]):
    """
    Save epoch metrics to CSV file, overwriting if it exists.
    
    Args:
        csv_path: Path to the CSV file to write
        epoch_data: List of dictionaries containing epoch metrics
    """
    if not epoch_data:
        return
    
    # Define CSV columns
    fieldnames = [
        'epoch',
        'train_loss',
        'train_ppl',
        'train_tokens_per_sec',
        'train_epoch_time_sec',
        'train_cpu_mem_mb',
        'train_gpu_mem_mb',
        'val_loss',
        'val_ppl',
        'val_bpc',
        'test_loss',
        'test_ppl',
        'test_bpc'
    ]
    
    # Write CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(epoch_data)
    
    print(f"Epoch metrics saved to: {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an LSTM language model on pre-tokenized equations."
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

    # Model architecture
    parser.add_argument("--model-size", type=str, default="custom",
                        choices=list(LSTM_MODEL_PRESETS.keys()) + ["custom"],
                        help="Preset size (tiny/small/medium) or 'custom' to use the manual flags below.")
    parser.add_argument("--hidden-size", type=int, default=256,
                        help="Hidden size of LSTM.")
    parser.add_argument("--embedding-dim", type=int, default=256,
                        help="Embedding dimension.")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="Number of LSTM layers.")

    # Training
    parser.add_argument("--max-length", type=int, default=128,
                        help="Max sequence length in tokens (for truncation/padding).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--log-interval", type=int, default=100)

    # DataLoader / multiprocessing
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Number of worker processes for DataLoader (default: 0).")

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="checkpoints_lstm",
                        help="Where to save checkpoints.")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device: 'cpu', 'cuda', or 'mps'. Default: auto-detect.")

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

    # DataLoaders (multiprocessing via num_workers)
    collate_fn = create_collate_fn(max_length=args.max_length, pad_id=pad_id)
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
    config = resolve_lstm_config(args, vocab_size)
    print(
        f"Using LSTM model_size={args.model_size} "
        f"(embedding_dim={config.embedding_dim}, hidden_size={config.hidden_size}, "
        f"num_layers={config.num_layers})"
    )
    model = LSTMLanguageModel(config)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Infer tokenization method and set up CSV logging
    tokenization_method = infer_tokenization_method(args.train_path, args.vocab_json)
    csv_filename = f"LSTM_{tokenization_method}_epochs.csv"
    csv_path = os.path.join(args.output_dir, csv_filename)
    epoch_metrics = []  # Store all epoch data
    
    print(f"Detected tokenization method: {tokenization_method}")
    print(f"Epoch metrics will be saved to: {csv_path}")

    # Training loop
    best_val_loss = float("inf")
    best_model_path = os.path.join(args.output_dir, "best_model.pth")

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_loss, train_ppl, train_tps, train_time, cpu_mem_mb, gpu_mem_mb = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch_idx=epoch,
            log_interval=args.log_interval,
        )

        val_loss, val_ppl, val_bpc = evaluate(model, val_loader, device)
        print(
            f"[Validation] loss={val_loss:.4f} ppl={val_ppl:.2f} "
            f"bpc={val_bpc:.4f}"
        )

        # Record epoch metrics
        epoch_metrics.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_ppl': train_ppl,
            'train_tokens_per_sec': train_tps,
            'train_epoch_time_sec': train_time,
            'train_cpu_mem_mb': cpu_mem_mb,
            'train_gpu_mem_mb': gpu_mem_mb if gpu_mem_mb is not None else '',
            'val_loss': val_loss,
            'val_ppl': val_ppl,
            'val_bpc': val_bpc,
            'test_loss': '',
            'test_ppl': '',
            'test_bpc': ''
        })
        
        # Save metrics to CSV after each epoch (overwrite)
        save_epoch_metrics(csv_path, epoch_metrics)

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

    # Final test evaluation
    if test_loader is not None:
        test_loss, test_ppl, test_bpc = evaluate(model, test_loader, device)
        print(
            f"\n[Test] loss={test_loss:.4f} ppl={test_ppl:.2f} "
            f"bpc={test_bpc:.4f}"
        )

if __name__ == "__main__":
    main()
