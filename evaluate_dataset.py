#!/usr/bin/env python3
import os
import pandas as pd
from collections import Counter
import re
import matplotlib.pyplot as plt
import seaborn as sns

#CSV_PATH = "HPC pipeline/data/equations.csv"
CSV_PATH = "pipeline/data/equations.csv"
#OUTPUT_DIR = "HPC pipeline/graphs"       #Plots
OUTPUT_DIR = "pipeline/graphs"

def calculate_depth(equation: str) -> int:
    """Calculate the depth of nested braces in an equation string."""
    max_depth = 0
    current_depth = 0
    for char in equation:
        if char == '{':
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif char == '}':
            current_depth -= 1
    return max_depth

def count_operators(equation: str):
    """Count the frequency of mathematical operators in an equation string.

    Note: we intentionally ignore '=' here, since every equation usually has at
    least one '=' and we usually care about arithmetic operators.
    """
    # Only match +, -, *, / (no equals sign)
    return re.findall(r"[+\-*/]", equation)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    required_cols = {
        "id",
        "full_equation",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available: {list(df.columns)}")

    equation_col = "full_equation"

    # Length of each equation (excluding spaces)
    equation_lengths = df[equation_col].apply(
        lambda s: len(s.replace(" ", "")) if isinstance(s, str) else 0
    )

    # Depth distribution (based on braces in the string)
    depths = df[equation_col].apply(
        lambda s: calculate_depth(s) if isinstance(s, str) else 0
    )

    # Operator frequency (excluding '=')
    operator_counter = Counter()
    df[equation_col].apply(
        lambda s: operator_counter.update(count_operators(s)) if isinstance(s, str) else None
    )

    # --------- Numeric stats (equations only) ----------
    num_eqs = len(df)
    avg_length = equation_lengths.mean()
    min_length = equation_lengths.min()
    max_length = equation_lengths.max()
    depth_series = pd.Series(depths)
    depth_distribution = depth_series.value_counts().sort_index()  # depth -> count
    operator_frequency = dict(operator_counter)  # operator -> count

    print(f"Number of equations: {num_eqs}\n")
    print("Equation length stats (characters, no spaces):")
    print(f"  avg_length: {avg_length:.2f}")
    print(f"  min_length: {min_length}")
    print(f"  max_length: {max_length}\n")
    print("Depth distribution (depth -> count):")

    for depth, count in depth_distribution.items():
        print(f"  {depth}: {count}")
    print()

    print("Operator frequency (operator -> count, excluding '='):")

    for op, count in operator_frequency.items():
        print(f"  '{op}': {count}")
    print()

    # --------- Plots (equations only) ----------
    TICK_SIZE = 16
    # Plot 1: Equation Length Distribution (characters, excluding spaces)
    plt.figure(figsize=(10, 6))
    sns.histplot(equation_lengths, bins=30, kde=True)
    #plt.title("Equation Length Distribution (characters, no spaces)")
    plt.xlabel("Length (characters, no spaces)", fontsize = 25)
    plt.ylabel("Frequency", fontsize = 25)
    plt.tick_params(axis='both', labelsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "2equation_length_distribution.png"))
    plt.close()

    # Plot 2: Depth Distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x=depths)
    plt.title("Depth Distribution (brace nesting)")
    plt.xlabel("Depth")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    #plt.savefig(os.path.join(OUTPUT_DIR, "depth_distribution.png"))
    plt.close()

    # Plot 3: Operator Frequency (only +, -, *, /)
    top_operators = operator_counter.most_common(10)
    if top_operators:
        operators, frequencies = zip(*top_operators)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(frequencies), y=list(operators),palette='viridis')
        #plt.title("Top Operator Frequencies (excluding '=')")
        plt.xlabel("Frequency", fontsize = 25)
        plt.ylabel("Operator", fontsize = 25)
        plt.tick_params(axis='both', labelsize=20)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "2operator_frequency.png"))
        plt.close()
    else:
        print("No operators found to plot.")


if __name__ == "__main__":
    main()