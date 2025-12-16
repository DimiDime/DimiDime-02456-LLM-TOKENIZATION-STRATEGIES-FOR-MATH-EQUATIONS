#!/usr/bin/env python3
import subprocess
import sys
import argparse
from pathlib import Path
from typing import List, Dict

ROOT_DIR = Path(__file__).resolve().parent
SEED = 42
MODE = "char"
MODE2 = "bpe"
MODE3 = "hierarchical"
EPOCH = 20
TRANSFORMER_SIZE = ["medium","tiny"]
LSTM_SIZE = ["medium","tiny"]

base = Path("pipeline")

dirs = [
    base / "data",
    base / "data" / "tokenization",
    base / "data" / "splits",
    base / "data" / "vocabulary",
    base / "runs",
]

for d in dirs:
    d.mkdir(parents=True, exist_ok=True)
    print(f"Created: {d}")



DATA = [
    {
        "name": "generate_equations",
        "script": "synthetic.py",
        "args": [
            "--num-equations", "10000",
            "--min-complexity", "1",
            "--max-complexity", "5",
            "--output", "pipeline/data/equations.csv",
            "--seed", f"{SEED}",
            "--num-preview", "3",
            "--no-preview"
        ],
    },
    {
        "name": "evaluate_dataset",
        "script": "evaluate_dataset.py",
        "args": [],
    }
]


TOKENIZATION_STEP = [
    {
        "name": "tokenization_and_split_" + f"{MODE}",
        "script": "tokenization.py",
        "args": [
            "--mode", f"{MODE}",                          
            "--input", "pipeline/data/equations.csv",
            "--output", "pipeline/data/tokenization/tokenized_" + f"{MODE}.csv",
            "--column", "full_equation",
            "--vocab", "pipeline/data/vocabulary/vocab_" + f"{MODE}.json",


            "--vocab-scope", "train",

  
            "--output-dir", "pipeline/data/splits",
            "--train-ratio", "0.8",
            "--val-ratio", "0.1",
            "--test-ratio", "0.1",
            "--seed", f"{SEED}",
        ],
    },
    {
        "name": "tokenization_and_split_" + f"{MODE2}",
        "script": "tokenization.py",
        "args": [
            "--mode", f"{MODE2}",                          
            "--input", "pipeline/data/equations.csv",
            "--output", "pipeline/data/tokenization/tokenized_" + f"{MODE2}.csv",
            "--column", "full_equation",
            "--vocab", "pipeline/data/vocabulary/vocab_" + f"{MODE2}.json",

            "--vocab-scope", "train",

            # BPE-specific args
            "--bpe-merges", "800",
            "--bpe-min-frequency", "2",

            "--output-dir", "pipeline/data/splits",
            "--train-ratio", "0.8",
            "--val-ratio", "0.1",
            "--test-ratio", "0.1",
            "--seed", f"{SEED}",
        ],
    },

    {
        "name": "tokenization_and_split" + f"{MODE3}",
        "script": "tokenization.py",
        "args": [
            "--mode", f"{MODE3}",
            "--input", "pipeline/data/equations.csv",
            "--output", "pipeline/data/tokenization/tokenized_" + f"{MODE3}.csv",
            "--column", "full_equation",
            "--vocab", "pipeline/data/vocabulary/vocab_" + f"{MODE3}.json",

            "--output-dir", "pipeline/data/splits",
            "--train-ratio", "0.8",
            "--val-ratio", "0.1",
            "--test-ratio", "0.1",
            "--seed", f"{SEED}",
        ],
    },
]
TOKEN_STATS_STEP = [
    {
        "name": "token_stats_all",
        "script": "evaluate_tokenized_splits.py",
        "args": [
            "--input-dir", "pipeline/data/splits",
            "--output-prefix", "pipeline/data/tokenization/token_stats",
            "--chunksize", "200000",
            "--sample-size", "200000",
            "--top-k", "20",
            "--seed", f"{SEED}",
            "--write-topk-csv",
        ],
    }
]

TRAIN_TRANSFORMER_STEP =[ {
    "name": "train_transformer_char_medium",
    "script": "transformer.py",
    "args": [
        "--train-path", "pipeline/data/splits/tokenized_train_char.csv",
        "--val-path",   "pipeline/data/splits/tokenized_val_char.csv",
        "--test-path",  "pipeline/data/splits/tokenized_test_char.csv",

        "--ids-column",  "full_equation_token_ids",
        "--text-column", "full_equation",

        "--vocab-json", "pipeline/data/vocabulary/vocab_char.json",

        "--model-size", f"{TRANSFORMER_SIZE[0]}",
        "--max-length", "128",
        "--batch-size", "32",
        "--epochs", f"{EPOCH}",
        "--learning-rate", "1e-4",
        "--seed", f"{SEED}",
        "--output-dir", "pipeline/runs/transformer_char_"+f"{TRANSFORMER_SIZE[0]}",
        "--num-workers", "0", 
        "--tokenizer-type", "char",
    ],
}, {
    "name": "train_transformer_bpe_medium",
    "script": "transformer.py",
    "args": [
        "--train-path", "pipeline/data/splits/tokenized_train_bpe.csv",
        "--val-path",   "pipeline/data/splits/tokenized_val_bpe.csv",
        "--test-path",  "pipeline/data/splits/tokenized_test_bpe.csv",

        "--ids-column",  "full_equation_token_ids",
        "--text-column", "full_equation",

        "--vocab-json", "pipeline/data/vocabulary/vocab_bpe.json",

        "--model-size", f"{TRANSFORMER_SIZE[0]}",
        "--max-length", "128",
        "--batch-size", "32",
        "--epochs", f"{EPOCH}",
        "--learning-rate", "1e-4",
        "--seed", f"{SEED}",
        "--output-dir", "pipeline/runs/transformer_bpe_"+f"{TRANSFORMER_SIZE[0]}",
        "--num-workers", "0", 
        "--tokenizer-type", "bpe",
    ],
},
{
    "name": "train_transformer_hierarchical_medium",
    "script": "transformer.py",
    "args": [
        "--train-path", "pipeline/data/splits/tokenized_train_hierarchical.csv",
        "--val-path",   "pipeline/data/splits/tokenized_val_hierarchical.csv",
        "--test-path",  "pipeline/data/splits/tokenized_test_hierarchical.csv",

        "--ids-column",  "full_equation_token_ids",
        "--text-column", "full_equation",

        "--vocab-json", "pipeline/data/vocabulary/vocab_hierarchical.json",

        "--model-size", f"{TRANSFORMER_SIZE[0]}",
        "--max-length", "128",
        "--batch-size", "32",
        "--epochs", f"{EPOCH}",
        "--learning-rate", "1e-4",
        "--seed", f"{SEED}",
        "--output-dir", "pipeline/runs/transformer_hierarchical_"+f"{TRANSFORMER_SIZE[0]}",
        "--num-workers", "0", 
        "--tokenizer-type", "hierarchical",
    ],
},
 {
    "name": "train_transformer_char_tiny",
    "script": "transformer.py",
    "args": [
        "--train-path", "pipeline/data/splits/tokenized_train_char.csv",
        "--val-path",   "pipeline/data/splits/tokenized_val_char.csv",
        "--test-path",  "pipeline/data/splits/tokenized_test_char.csv",

        "--ids-column",  "full_equation_token_ids",
        "--text-column", "full_equation",

        "--vocab-json", "pipeline/data/vocabulary/vocab_char.json",

        "--model-size", f"{TRANSFORMER_SIZE[1]}",
        "--max-length", "128",
        "--batch-size", "32",
        "--epochs", f"{EPOCH}",
        "--learning-rate", "1e-4",
        "--seed", f"{SEED}",
        "--output-dir", "pipeline/runs/transformer_char_"+f"{TRANSFORMER_SIZE[1]}",
        "--num-workers", "0", 
        "--tokenizer-type", "char",
    ],
}, {
    "name": "train_transformer_bpe_tiny",
    "script": "transformer.py",
    "args": [
        "--train-path", "pipeline/data/splits/tokenized_train_bpe.csv",
        "--val-path",   "pipeline/data/splits/tokenized_val_bpe.csv",
        "--test-path",  "pipeline/data/splits/tokenized_test_bpe.csv",

        "--ids-column",  "full_equation_token_ids",
        "--text-column", "full_equation",

        "--vocab-json", "pipeline/data/vocabulary/vocab_bpe.json",

        "--model-size", f"{TRANSFORMER_SIZE[1]}",
        "--max-length", "128",
        "--batch-size", "32",
        "--epochs", f"{EPOCH}",
        "--learning-rate", "1e-4",
        "--seed", f"{SEED}",
        "--output-dir", "pipeline/runs/transformer_bpe_"+f"{TRANSFORMER_SIZE[1]}",
        "--num-workers", "0", 
        "--tokenizer-type", "bpe",
    ],
},
{
    "name": "train_transformer_hierarchical_tiny",
    "script": "transformer.py",
    "args": [
        "--train-path", "pipeline/data/splits/tokenized_train_hierarchical.csv",
        "--val-path",   "pipeline/data/splits/tokenized_val_hierarchical.csv",
        "--test-path",  "pipeline/data/splits/tokenized_test_hierarchical.csv",

        "--ids-column",  "full_equation_token_ids",
        "--text-column", "full_equation",

        "--vocab-json", "pipeline/data/vocabulary/vocab_hierarchical.json",

        "--model-size", f"{TRANSFORMER_SIZE[1]}",
        "--max-length", "128",
        "--batch-size", "32",
        "--epochs", f"{EPOCH}",
        "--learning-rate", "1e-4",
        "--seed", f"{SEED}",
        "--output-dir", "pipeline/runs/transformer_hierarchical_"+f"{TRANSFORMER_SIZE[1]}",
        "--num-workers", "0", 
        "--tokenizer-type", "hierarchical",
    ],
},

]


TRAIN_LSTM_STEP = [
    {
        "name": "train_lstm_char_medium",
        "script": "lstm.py",
        "args": [
            "--train-path", "pipeline/data/splits/tokenized_train_char.csv",
            "--val-path",   "pipeline/data/splits/tokenized_val_char.csv",
            "--test-path",  "pipeline/data/splits/tokenized_test_char.csv",

            "--ids-column",  "full_equation_token_ids",
            "--text-column", "full_equation",

            "--vocab-json", "pipeline/data/vocabulary/vocab_char.json",

            "--model-size", f"{LSTM_SIZE[0]}",
            "--max-length", "128",
            "--batch-size", "32",
            "--epochs", f"{EPOCH}",
            "--learning-rate", "1e-3",
            "--num-workers", "0", 
            "--seed", f"{SEED}",
            "--output-dir", "pipeline/runs/lstm_char_"+f"{LSTM_SIZE[0]}",
        ],
    },
    {
        "name": "train_lstm_bpe_medium",
        "script": "lstm.py",
        "args": [
            "--train-path", "pipeline/data/splits/tokenized_train_bpe.csv",
            "--val-path",   "pipeline/data/splits/tokenized_val_bpe.csv",
            "--test-path",  "pipeline/data/splits/tokenized_test_bpe.csv",

            "--ids-column",  "full_equation_token_ids",
            "--text-column", "full_equation",

            "--vocab-json", "pipeline/data/vocabulary/vocab_bpe.json",

            "--model-size", f"{LSTM_SIZE[0]}",
            "--max-length", "128",
            "--batch-size", "32",
            "--epochs", f"{EPOCH}",
            "--learning-rate", "1e-3",
            "--num-workers", "0", 
            "--seed", f"{SEED}",
            "--output-dir", "pipeline/runs/lstm_bpe_"+f"{LSTM_SIZE[0]}",
        ],
    },
    {
        "name": "train_lstm_hierarchical_medium",
        "script": "lstm.py",
        "args": [
            "--train-path", "pipeline/data/splits/tokenized_train_hierarchical.csv",
            "--val-path",   "pipeline/data/splits/tokenized_val_hierarchical.csv",
            "--test-path",  "pipeline/data/splits/tokenized_test_hierarchical.csv",

            "--ids-column",  "full_equation_token_ids",
            "--text-column", "full_equation",

            "--vocab-json", "pipeline/data/vocabulary/vocab_hierarchical.json",

            "--model-size", f"{LSTM_SIZE[0]}",
            "--max-length", "128",
            "--batch-size", "32",
            "--epochs", f"{EPOCH}",
            "--learning-rate", "1e-3",
            "--num-workers", "0", 
            "--seed", f"{SEED}",
            "--output-dir", "pipeline/runs/lstm_hierarchical_"+f"{LSTM_SIZE[0]}",
        ],
    },
    {
        "name": "train_lstm_char_tiny",
        "script": "lstm.py",
        "args": [
            "--train-path", "pipeline/data/splits/tokenized_train_char.csv",
            "--val-path",   "pipeline/data/splits/tokenized_val_char.csv",
            "--test-path",  "pipeline/data/splits/tokenized_test_char.csv",

            "--ids-column",  "full_equation_token_ids",
            "--text-column", "full_equation",

            "--vocab-json", "pipeline/data/vocabulary/vocab_char.json",

            "--model-size", f"{LSTM_SIZE[1]}",
            "--max-length", "128",
            "--batch-size", "32",
            "--epochs", f"{EPOCH}",
            "--learning-rate", "1e-3",
            "--num-workers", "0", 
            "--seed", f"{SEED}",
            "--output-dir", "pipeline/runs/lstm_char_"+f"{LSTM_SIZE[1]}",
        ],
    },
    {
        "name": "train_lstm_bpe_tiny",
        "script": "lstm.py",
        "args": [
            "--train-path", "pipeline/data/splits/tokenized_train_bpe.csv",
            "--val-path",   "pipeline/data/splits/tokenized_val_bpe.csv",
            "--test-path",  "pipeline/data/splits/tokenized_test_bpe.csv",

            "--ids-column",  "full_equation_token_ids",
            "--text-column", "full_equation",

            "--vocab-json", "pipeline/data/vocabulary/vocab_bpe.json",

            "--model-size", f"{LSTM_SIZE[1]}",
            "--max-length", "128",
            "--batch-size", "32",
            "--epochs", f"{EPOCH}",
            "--learning-rate", "1e-3",
            "--num-workers", "0", 
            "--seed", f"{SEED}",
            "--output-dir", "pipeline/runs/lstm_bpe_"+f"{LSTM_SIZE[1]}",
        ],
    },
    {
        "name": "train_lstm_hierarchical_tiny",
        "script": "lstm.py",
        "args": [
            "--train-path", "pipeline/data/splits/tokenized_train_hierarchical.csv",
            "--val-path",   "pipeline/data/splits/tokenized_val_hierarchical.csv",
            "--test-path",  "pipeline/data/splits/tokenized_test_hierarchical.csv",

            "--ids-column",  "full_equation_token_ids",
            "--text-column", "full_equation",

            "--vocab-json", "pipeline/data/vocabulary/vocab_hierarchical.json",

            "--model-size", f"{LSTM_SIZE[1]}",
            "--max-length", "128",
            "--batch-size", "32",
            "--epochs", f"{EPOCH}",
            "--learning-rate", "1e-3",
            "--num-workers", "0", 
            "--seed", f"{SEED}",
            "--output-dir", "pipeline/runs/lstm_hierarchical_"+f"{LSTM_SIZE[1]}",
        ],
    },
]

EVALUATION_STEP = [
    {
        "name":  "eval_runs_all_models",
        "script": "evaluate_model.py",
        "args": [
            "--physics-csv",        "pipeline/evaluation_equations.csv",
            "--checkpoint-base-dir","pipeline/runs",
            "--vocab-base-dir",     "pipeline/data/vocabulary",
            "--output-csv",         "pipeline/model_eval_results_runs.csv",
            "--device",             "cuda", #use mps if on mac, cuda if on nvidia gpu, else cpu  
        ],
    },
]







PIPELINE_STEPS = [
    *DATA,
    *TOKENIZATION_STEP,
    *TOKEN_STATS_STEP,
    *TRAIN_TRANSFORMER_STEP,
    *TRAIN_LSTM_STEP,
    *EVALUATION_STEP,
]





def run_step(step: Dict, dry_run: bool = False) -> int:
    """
    Run a single pipeline step.

    Returns:
        returncode from the process (0 means success).
    """
    name = step["name"]
    # Always resolve script path relative to this pipeline file
    script_path = ROOT_DIR / step["script"]
    args = step.get("args", [])

    cmd = [sys.executable, str(script_path)] + args

    print("=" * 70)
    print(f"STEP: {name}")
    print(f"Running: {' '.join(cmd)}")
    print("=" * 70)

    if dry_run:
        print("[DRY RUN] Skipping execution.")
        return 0

    # Ensure script exists
    if not script_path.exists():
        print(f"[ERROR] Script not found: {script_path}")
        return 1

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"[OK] Step '{name}' completed successfully.")
    else:
        print(f"[FAILED] Step '{name}' exited with code {result.returncode}.")

    return result.returncode


def list_steps():
    print("Available pipeline steps (in order):")
    for i, step in enumerate(PIPELINE_STEPS, start=1):
        print(f"  {i}. {step['name']} -> {step['script']}")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple Python pipeline runner.")
    parser.add_argument(
        "--only",
        type=str,
        help="Run only a single step by name (e.g. --only generate_equations).",
    )
    parser.add_argument(
        "--from-step",
        type=str,
        help="Start from this step name and run to the end.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Do not stop the pipeline on first error.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands but do not execute them.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available steps and exit.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list:
        list_steps()
        return

    steps = PIPELINE_STEPS

    # Filter: --only
    if args.only:
        steps = [s for s in steps if s["name"] == args.only]
        if not steps:
            print(f"[ERROR] No step named '{args.only}' found.")
            list_steps()
            sys.exit(1)

    # Filter: --from-step
    elif args.from_step:
        found = False
        new_steps = []
        for s in steps:
            if s["name"] == args.from_step:
                found = True
            if found:
                new_steps.append(s)
        if not found:
            print(f"[ERROR] No step named '{args.from_step}' found.")
            list_steps()
            sys.exit(1)
        steps = new_steps

    if not steps:
        print("[ERROR] No steps selected to run.")
        sys.exit(1)

    print("=" * 70)
    print("Pipeline starting...")
    print(f"Selected steps: {[s['name'] for s in steps]}")
    print("=" * 70)

    overall_status = 0

    for step in steps:
        rc = run_step(step, dry_run=args.dry_run)
        if rc != 0:
            overall_status = rc
            if not args.continue_on_error:
                print("[PIPELINE] Stopping due to error.")
                sys.exit(overall_status)
            else:
                print("[PIPELINE] Continuing despite error (--continue-on-error).")

    print("=" * 70)
    print("Pipeline finished.")
    print(f"Exit code: {overall_status}")
    print("=" * 70)
    sys.exit(overall_status)


if __name__ == "__main__":
    main()
