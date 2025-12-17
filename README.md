# LLM Tokenization Strategies for Mathematical Equations

A systematic comparison of tokenization methods (character-level, BPE, and hierarchical) for training sequence models on synthetic mathematical equations. 
This project evaluates how tokenization choice impacts model performance across LSTM and Transformer architectures.

## 📋 Overview

This project implements an end-to-end pipeline to:
1. **Generate** synthetic mathematical equations with controlled complexity
2. **Tokenize** equations using three different strategies
3. **Train** LSTM and Transformer models on tokenized data
4. **Evaluate** model performance across tokenization-architecture combinations

### Key Findings
- **Best overall performance**: Character-level tokenization with Transformers (BPC: 5.12-5.36)
- **Architecture differences**: Transformers significantly outperform LSTMs across all tokenizers
- **Tokenization trade-offs**: Character-level achieves best compression despite longest sequences
