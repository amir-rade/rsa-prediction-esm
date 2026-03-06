# SASA-pred

This repository contains a small, reproducible pipeline to predict **per-residue relative solvent accessibility (RSA)** from protein sequences using **protein language model embeddings (ESM2)** and lightweight neural network heads (CNN, BiLSTM, Transformer, CNN-BiLSTM).

The workflow is designed to be practical and fast:

1. Build a clean chain-level dataset with RSA labels and valid-position masks  
2. Compute and cache ESM2 residue embeddings once  
3. Train and evaluate multiple model heads directly on cached embeddings

---

## Scientific motivation

Solvent accessibility is an important structural property related to **protein folding, stability, and interaction interfaces**.  

Recent protein language models capture structural information directly from sequence representations. This project explores how well **residue-level solvent accessibility can be predicted from sequence embeddings alone** using lightweight neural architectures.

---

## Acknowledgments

This project uses the **ESM protein language model** developed by Meta AI:  
https://github.com/facebookresearch/esm