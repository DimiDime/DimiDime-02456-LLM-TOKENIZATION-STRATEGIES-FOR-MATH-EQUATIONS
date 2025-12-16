#!/usr/bin/env python
"""
Evaluation script for comparing LSTM and Transformer models across different tokenization schemes.

Evaluates model size, performance (perplexity, loss, BPC), and inference speed on physics equations.
Supports three tokenization schemes: BPE, Character, and Hierarchical.
"""

import argparse
import ast
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel


# =========================
# Constants & Config
# =========================

TOKENIZATION_METHODS = ["bpe", "char", "hierarchical"]
MODEL_TYPES = ["lstm", "transformer"]


# =========================
# Dataset for inference
# =========================

class InferenceEquationDataset(Dataset):
    """Dataset for evaluation on raw equations (needs tokenization on-the-fly)."""
    
    def __init__(self, csv_path: str):
        """
        Load equations from CSV.
        Expected column: 'Equation'
        """
        df = pd.read_csv(csv_path)
        if "Equation" not in df.columns:
            raise ValueError(f"CSV must have 'Equation' column. Found: {df.columns.tolist()}")
        
        # Clean and filter
        self.equations = [
            eq.strip() for eq in df["Equation"].tolist() 
            if isinstance(eq, str) and eq.strip()
        ]
        print(f"Loaded {len(self.equations)} equations from {csv_path}")
    
    def __len__(self):
        return len(self.equations)
    
    def __getitem__(self, idx):
        return {"equation": self.equations[idx]}


class PreTokenizedEquationDataset(Dataset):
    """Dataset for pre-tokenized equations."""
    
    def __init__(self, csv_path: str, ids_column: str = "full_equation_token_ids", 
                 text_column: str = "full_equation"):
        self.examples = []
        
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if ids_column not in row or text_column not in row:
                    continue
                
                token_str = row[ids_column].strip()
                if not token_str or token_str == "[]":
                    continue
                
                try:
                    token_ids = ast.literal_eval(token_str)
                    token_ids = [int(x) for x in token_ids]
                    text = row[text_column].strip() if row[text_column] else ""
                    char_count = len(text)
                    
                    self.examples.append({
                        "token_ids": token_ids,
                        "char_count": char_count,
                        "equation": text,
                    })
                except (ValueError, SyntaxError):
                    continue
        
        if len(self.examples) == 0:
            raise ValueError(f"No valid examples loaded from {csv_path}")
        
        print(f"Loaded {len(self.examples)} examples from {csv_path}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "token_ids": ex["token_ids"],
            "char_count": ex["char_count"],
            "equation": ex["equation"],
        }


# =========================
# Tokenization helpers
# =========================

def tokenize_char(text: str) -> List[str]:
    """Character-level tokenization."""
    return list(text)


def merge_pair_in_sequence(sequence: List[str], pair: Tuple[str, str]) -> List[str]:
    """Merge all occurrences of a pair in sequence."""
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


def bpe_encode(text: str, merges: List[Tuple[str, str]], ignore_spaces: bool = True) -> List[str]:
    """BPE encoding using learned merges."""
    if ignore_spaces:
        segments = text.split()
    else:
        segments = [text]
    
    all_tokens: List[str] = []
    for segment in segments:
        if segment:
            sequence = list(segment)
            for pair in merges:
                sequence = merge_pair_in_sequence(sequence, pair)
            all_tokens.extend(sequence)
    
    return all_tokens


def tokens_to_ids(tokens: List[str], vocab: Dict[str, int], unk_token: Optional[str] = None) -> List[int]:
    """Convert tokens to IDs using vocabulary."""
    ids: List[int] = []
    for t in tokens:
        if t in vocab:
            ids.append(vocab[t])
        elif unk_token and unk_token in vocab:
            ids.append(vocab[unk_token])
        else:
            # Skip unknown tokens if no unk_token provided
            pass
    return ids


def build_physics_examples(
    physics_csv: str,
    tokenizer_fn,
    vocab: Dict[str, int],
    merges: Optional[List[Tuple[str, str]]] = None,
    max_length: int = 128,
) -> List[Dict[str, Any]]:
    """
    Build a list of examples from the physics CSV:
    each example is {"token_ids": List[int], "char_count": int}.
    No padding is applied here – collate_fn will handle it.
    """
    df = pd.read_csv(physics_csv)
    if "Equation" not in df.columns:
        raise ValueError("Physics CSV must have 'Equation' column")

    equations = [
        eq.strip() for eq in df["Equation"].tolist()
        if isinstance(eq, str) and eq.strip()
    ]

    examples: List[Dict[str, Any]] = []
    for eq in equations:
        if tokenizer_fn.__name__ == "bpe_encode":
            tokens = tokenizer_fn(eq, merges)
        else:
            tokens = tokenizer_fn(eq)

        token_ids = tokens_to_ids(tokens, vocab, unk_token="<unk>")
        if not token_ids:
            continue

        # Truncate to max_length
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]

        examples.append({
            "token_ids": token_ids,
            "char_count": len(eq),
        })

    if not examples:
        raise ValueError(f"No valid equations found in {physics_csv}")

    return examples


# =========================
# Model loading functions
# =========================

class LSTMConfig:
    """Configuration for LSTM model."""
    def __init__(self, vocab_size: int, hidden_size: int = 256, 
                 num_layers: int = 2, embedding_dim: int = 256):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim

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
        x = self.embedding(input_ids)
        x, _ = self.lstm(x)
        logits = self.head(x)
        return logits


def load_vocab(vocab_path: str) -> Tuple[Dict[str, int], Optional[List[List[str]]]]:
    """Load vocabulary from JSON file."""
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_raw = json.load(f)
    
    # Handle different vocab formats
    if isinstance(vocab_raw, dict) and "token_to_id" in vocab_raw:
        token_to_id = vocab_raw["token_to_id"]
        merges = vocab_raw.get("merges", None)
        if merges:
            merges = [tuple(m) for m in merges]
        return token_to_id, merges
    elif isinstance(vocab_raw, dict) and "stoi" in vocab_raw:
        return vocab_raw["stoi"], None
    elif isinstance(vocab_raw, dict) and all(isinstance(v, int) for v in vocab_raw.values()):
        return vocab_raw, None
    else:
        raise ValueError(f"Unrecognized vocab format in {vocab_path}")


def load_lstm_checkpoint(checkpoint_path: str, vocab_size: int, device: str) -> LSTMLanguageModel:
    """Load LSTM model from checkpoint, inferring its architecture from the weights."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint["model_state_dict"]

    # Infer dimensions from the state_dict
    emb_weight = state_dict["embedding.weight"]          # (vocab_size, embedding_dim)
    ckpt_vocab_size, embedding_dim = emb_weight.shape

    weight_ih_l0 = state_dict["lstm.weight_ih_l0"]       # (4 * hidden_size, embedding_dim)
    hidden_size = weight_ih_l0.shape[0] // 4

    # Count how many layers exist: weight_ih_l{0,1,...}
    num_layers = len([k for k in state_dict.keys() if k.startswith("lstm.weight_ih_l")])

    config = LSTMConfig(
        vocab_size=ckpt_vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        embedding_dim=embedding_dim,
    )

    model = LSTMLanguageModel(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model

def load_transformer_checkpoint(checkpoint_path: str, vocab_size: int, max_length: int, 
                               model_size: str, device: str) -> GPT2LMHeadModel:
    """Load Transformer model from checkpoint, reconstructing GPT2Config from the weights."""
    from transformers import GPT2Config

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint["model_state_dict"]

    if "config" in checkpoint:
        # If you saved a config during training, prefer that
        config = GPT2Config(**checkpoint["config"])
    else:
        # Infer basic dimensions from weights
        wte = state_dict["transformer.wte.weight"]  # (vocab_size, n_embd)
        vocab_size_ckpt, n_embd = wte.shape

        wpe = state_dict["transformer.wpe.weight"]  # (n_positions, n_embd)
        n_positions = wpe.shape[0]

        # Number of layers: transformer.h.{0..N-1}.*
        layer_ids = [
            int(k.split(".")[2])
            for k in state_dict.keys()
            if k.startswith("transformer.h.")
        ]
        n_layer = max(layer_ids) + 1 if layer_ids else 0

        # Infer a reasonable n_head (must divide n_embd).
        possible_heads = [h for h in range(1, n_embd + 1) if n_embd % h == 0]
        n_head = 12 if 12 in possible_heads else possible_heads[-1]

        config = GPT2Config(
            vocab_size=vocab_size_ckpt,
            n_positions=n_positions,
            n_ctx=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            pad_token_id=vocab_size_ckpt - 1,  # assume last index is pad
            bos_token_id=None,
            eos_token_id=None,
        )

    model = GPT2LMHeadModel(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model_path: str) -> float:
    """Get model file size in MB."""
    return os.path.getsize(model_path) / (1024 ** 2)


# =========================
# Evaluation functions
# =========================

@torch.no_grad()
def evaluate_lstm(model: nn.Module, dataloader: DataLoader, device: str, pad_id: int) -> Dict[str, float]:
    """
    Evaluate LSTM model on dataset with proper next-token prediction and pad masking.
    """
    model.eval()
    total_nats = 0.0
    total_tokens = 0
    total_chars = 0
    num_batches = 0

    # We'll sum the loss manually over valid positions
    criterion = nn.CrossEntropyLoss(reduction="sum")

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)            # [B, T]
        attention_mask = batch["attention_mask"].to(device)  # [B, T]

        # Need at least 2 tokens to do next-token prediction
        if input_ids.size(1) < 2:
            continue

        # Shift for next-token prediction: predict x_t from x_<t
        input_ids_model = input_ids[:, :-1]   # [B, T-1]
        labels = input_ids[:, 1:]            # [B, T-1]
        mask = attention_mask[:, 1:]         # [B, T-1], 1 for real tokens

        logits = model(input_ids_model)      # [B, T-1, V]
        logits = logits[:, :labels.size(1), :]  # safety trim

        vocab_size = logits.size(-1)
        logits_flat = logits.reshape(-1, vocab_size)
        labels_flat = labels.reshape(-1)
        mask_flat = mask.reshape(-1).bool()

        if mask_flat.sum().item() == 0:
            continue

        loss = criterion(logits_flat[mask_flat], labels_flat[mask_flat])
        num_tokens = int(mask_flat.sum().item())

        total_tokens += num_tokens
        total_nats += loss.item()
        total_chars += sum(batch["char_count"])
        num_batches += 1

    if total_tokens == 0:
        return {"loss": float("inf"), "perplexity": float("inf"), "bpc": float("inf")}

    avg_loss = total_nats / total_tokens
    perplexity = math.exp(avg_loss)
    bpc = (total_nats / math.log(2)) / max(total_chars, 1)

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "bpc": bpc,
        "num_batches": num_batches,
        "total_tokens": total_tokens,
    }


@torch.no_grad()
def evaluate_transformer(model: GPT2LMHeadModel, dataloader: DataLoader, device: str) -> Dict[str, float]:
    """
    Evaluate Transformer model on dataset with proper next-token prediction and pad masking.
    """
    model.eval()
    total_nats = 0.0
    total_tokens = 0
    total_chars = 0
    num_batches = 0

    criterion = nn.CrossEntropyLoss(reduction="sum")

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)            # [B, T]
        attention_mask = batch["attention_mask"].to(device)  # [B, T]

        if input_ids.size(1) < 2:
            continue

        # Forward once to get logits
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits                              # [B, T, V]

        # Shift for next-token prediction
        logits = logits[:, :-1, :]                          # predict next token
        labels = input_ids[:, 1:]                           # target tokens
        mask = attention_mask[:, 1:]                        # only real tokens

        vocab_size = logits.size(-1)
        logits_flat = logits.reshape(-1, vocab_size)
        labels_flat = labels.reshape(-1)
        mask_flat = mask.reshape(-1).bool()

        if mask_flat.sum().item() == 0:
            continue

        loss = criterion(logits_flat[mask_flat], labels_flat[mask_flat])
        num_tokens = int(mask_flat.sum().item())

        total_tokens += num_tokens
        total_nats += loss.item()
        total_chars += sum(batch["char_count"])
        num_batches += 1

    if total_tokens == 0:
        return {"loss": float("inf"), "perplexity": float("inf"), "bpc": float("inf")}

    avg_loss = total_nats / total_tokens
    perplexity = math.exp(avg_loss)
    bpc = (total_nats / math.log(2)) / max(total_chars, 1)

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "bpc": bpc,
        "num_batches": num_batches,
        "total_tokens": total_tokens,
    }


@torch.no_grad()
def measure_inference_speed(model: nn.Module, dataloader: DataLoader, device: str, 
                           model_type: str, pad_id: Optional[int] = None) -> float:
    """Measure inference speed in tokens per second."""
    total_tokens = 0
    total_time = 0.0
    
    for batch in dataloader:
        start_time = time.time()
        
        if model_type == "lstm":
            input_ids = batch["input_ids"].to(device)
            _ = model(input_ids)
            total_tokens += input_ids.numel()
        else:  # transformer
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            _ = model(input_ids=input_ids)
            total_tokens += input_ids.numel()
        
        total_time += time.time() - start_time
    
    tokens_per_sec = total_tokens / max(total_time, 1e-6)
    return tokens_per_sec


# (Old prepare_physics_dataset kept for reference; no longer used by evaluate_models)
def prepare_physics_dataset(physics_csv: str, tokenizer_fn, vocab: Dict[str, int], 
                           merges: Optional[List[Tuple[str, str]]] = None,
                           max_length: int = 128, pad_id: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare physics dataset for evaluation.
    Returns: (input_ids, attention_mask, char_counts)
    """
    df = pd.read_csv(physics_csv)
    if "Equation" not in df.columns:
        raise ValueError(f"Physics CSV must have 'Equation' column")
    
    equations = [eq.strip() for eq in df["Equation"].tolist() if isinstance(eq, str) and eq.strip()]
    
    batch_input_ids = []
    batch_attention_mask = []
    batch_char_counts = []
    
    for eq in equations:
        if tokenizer_fn.__name__ == "bpe_encode":
            tokens = tokenizer_fn(eq, merges)
        else:
            tokens = tokenizer_fn(eq)
        
        token_ids = tokens_to_ids(tokens, vocab, unk_token="<unk>")
        
        # Truncate or pad
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        
        seq_len = len(token_ids)
        input_seq = token_ids + [pad_id] * (max_length - seq_len)
        attention_seq = [1] * seq_len + [0] * (max_length - seq_len)
        
        batch_input_ids.append(input_seq)
        batch_attention_mask.append(attention_seq)
        batch_char_counts.append(len(eq))
    
    return (torch.tensor(batch_input_ids, dtype=torch.long),
            torch.tensor(batch_attention_mask, dtype=torch.long),
            torch.tensor(batch_char_counts, dtype=torch.long))


def find_checkpoint_for_config(base_dir: str, model_type: str, tokenization: str, size: str) -> Optional[str]:
    """
    Find checkpoint for a given (model_type, tokenization, size).
    Returns checkpoint_path or None.
    
    Directory structure from pipeline:
    - LSTM:      HPC pipeline/runs_10k/lstm_{tokenization}_{size}/best_model.pth
    - Transformer: HPC pipeline/runs_10k/transformer_{tokenization}_{size}/*.pth (best_transformer_*.pth)
    """
    if model_type == "lstm":
        checkpoint_dir = os.path.join(base_dir, f"lstm_{tokenization}_{size}")
        checkpoint_file = os.path.join(checkpoint_dir, "best_model.pth")
        if os.path.exists(checkpoint_file):
            return checkpoint_file
    else:  # transformer
        checkpoint_dir = os.path.join(base_dir, f"transformer_{tokenization}_{size}")
        if os.path.exists(checkpoint_dir):
            # Look for best_transformer_*.pth
            for filename in os.listdir(checkpoint_dir):
                if filename.startswith("best_transformer_") and filename.endswith(".pth"):
                    return os.path.join(checkpoint_dir, filename)
    return None


def evaluate_models(physics_csv: str, checkpoint_base_dir: str, vocab_base_dir: str, 
                   output_csv: str, device: str = "cuda") -> None:
    """
    Evaluate all model configurations on physics equations.
    Saves results to CSV with columns: 
      model_type, tokenization, model_size, loss, perplexity, bpc, 
      model_size_mb, num_parameters, tokens_per_sec, checkpoint_path
    """
    
    results: List[Dict[str, Any]] = []
    
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)

    model_sizes = ["medium", "tiny"]
    
    for model_type in MODEL_TYPES:
        for tokenization in TOKENIZATION_METHODS:
            for model_size in model_sizes:
                print(f"\n{'='*70}")
                print(f"Evaluating {model_type.upper()} ({model_size}) with {tokenization.upper()} tokenization")
                print(f"{'='*70}")
                
                # Find checkpoint for this exact config
                checkpoint_path = find_checkpoint_for_config(
                    checkpoint_base_dir, model_type, tokenization, model_size
                )
                if not checkpoint_path:
                    print(f"⚠ Checkpoint not found for {model_type} + {tokenization} + {model_size}, skipping...")
                    continue
                
                # Load vocabulary
                vocab_path = os.path.join(vocab_base_dir, f"vocab_{tokenization}.json")
                if not os.path.exists(vocab_path):
                    print(f"⚠ Vocab not found at {vocab_path}, skipping...")
                    continue
                
                try:
                    vocab, merges = load_vocab(vocab_path)
                except Exception as e:
                    print(f"✗ Error loading vocab: {e}")
                    continue
                
                vocab_size = max(vocab.values()) + 1 if vocab else 0
                pad_id = vocab_size  # assume pad is index vocab_size (one beyond max vocab id)
                
                # Build dataset examples (variable-length)
                try:
                    if tokenization == "bpe":
                        tokenizer_fn = bpe_encode
                        examples = build_physics_examples(
                            physics_csv, tokenizer_fn, vocab, merges=merges, max_length=128
                        )
                    else:  # char or hierarchical
                        tokenizer_fn = tokenize_char
                        examples = build_physics_examples(
                            physics_csv, tokenizer_fn, vocab, merges=None, max_length=128
                        )
                    
                    print(f"✓ Dataset prepared: {len(examples)} equations")
                except Exception as e:
                    print(f"✗ Error preparing dataset: {e}")
                    continue
                
                # Load model
                try:
                    if model_type == "lstm":
                        model = load_lstm_checkpoint(checkpoint_path, vocab_size + 1, device)
                    else:  # transformer
                        model = load_transformer_checkpoint(
                            checkpoint_path, vocab_size + 1, max_length=128, 
                            model_size=model_size, device=device
                        )
                    
                    num_params = count_parameters(model)
                    model_size_mb = get_model_size_mb(checkpoint_path) if os.path.isfile(checkpoint_path) else 0
                    print(f"✓ Model loaded | Params: {num_params:,} | Size: {model_size_mb:.2f} MB")
                except Exception as e:
                    print(f"✗ Error loading model: {e}")
                    continue
                
                # Collate function: pads sequences & builds masks
                def collate_fn(batch):
                    max_len = max(len(item["token_ids"]) for item in batch)
                    batch_input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
                    batch_attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
                    batch_token_ids: List[List[int]] = []
                    batch_char_counts: List[int] = []
                    
                    for i, item in enumerate(batch):
                        seq = item["token_ids"]
                        L = min(len(seq), max_len)
                        if L > 0:
                            batch_input_ids[i, :L] = torch.tensor(seq[:L], dtype=torch.long)
                            batch_attention_mask[i, :L] = 1
                        batch_token_ids.append(seq)
                        batch_char_counts.append(item["char_count"])
                    
                    return {
                        "input_ids": batch_input_ids,
                        "attention_mask": batch_attention_mask,
                        "token_ids": batch_token_ids,
                        "char_count": batch_char_counts,
                    }
                
                dataloader = DataLoader(examples, batch_size=32, collate_fn=collate_fn)
                
                # Evaluate
                try:
                    if model_type == "lstm":
                        metrics = evaluate_lstm(model, dataloader, device, pad_id)
                    else:
                        metrics = evaluate_transformer(model, dataloader, device)
                    
                    print(f"✓ Evaluation complete")
                    print(f"  Loss: {metrics['loss']:.4f}")
                    print(f"  Perplexity: {metrics['perplexity']:.2f}")
                    print(f"  BPC: {metrics['bpc']:.4f}")
                except Exception as e:
                    print(f"✗ Error during evaluation: {e}")
                    continue
                
                # Measure inference speed
                try:
                    if model_type == "lstm":
                        tokens_per_sec = measure_inference_speed(model, dataloader, device, model_type, pad_id)
                    else:
                        tokens_per_sec = measure_inference_speed(model, dataloader, device, model_type)
                    print(f"  Inference: {tokens_per_sec:.1f} tokens/sec")
                except Exception as e:
                    print(f"⚠ Could not measure inference speed: {e}")
                    tokens_per_sec = 0.0
                
                # Store results
                results.append({
                    "model_type": model_type,
                    "tokenization": tokenization,
                    "model_size": model_size,
                    "loss": float(metrics["loss"]),
                    "perplexity": float(metrics["perplexity"]),
                    "bpc": float(metrics["bpc"]),
                    "model_size_mb": model_size_mb,
                    "num_parameters": num_params,
                    "tokens_per_sec": tokens_per_sec,
                    "checkpoint_path": checkpoint_path,
                })
    
    # Save results to CSV
    if results:
        df_results = pd.DataFrame(results)
        df_results.to_csv(output_csv, index=False)
        print(f"\n✓ Evaluation results saved to: {output_csv}")
        print(f"\n{df_results.to_string()}")
    else:
        print("\n✗ No results to save")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate LSTM and Transformer models on physics equations."
    )
    
    parser.add_argument("--physics-csv", type=str, required=True,
                       help="Path to physics equations CSV file (must have 'Equation' column).")
    parser.add_argument("--checkpoint-base-dir", type=str, required=True,
                       help="Base directory containing model checkpoints.")
    parser.add_argument("--vocab-base-dir", type=str, required=True,
                       help="Directory containing vocabulary JSON files.")
    parser.add_argument("--output-csv", type=str, default="model_eval_results.csv",
                       help="Path to save evaluation results CSV.")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda, cpu, mps). Default: auto-detect.")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Device selection
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    print(f"Physics CSV: {args.physics_csv}")
    print(f"Checkpoint base dir: {args.checkpoint_base_dir}")
    print(f"Vocab base dir: {args.vocab_base_dir}")
    
    evaluate_models(
        physics_csv=args.physics_csv,
        checkpoint_base_dir=args.checkpoint_base_dir,
        vocab_base_dir=args.vocab_base_dir,
        output_csv=args.output_csv,
        device=device
    )


if __name__ == "__main__":
    main()
