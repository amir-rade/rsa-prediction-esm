HEAD
# RSA Prediction with Protein Language Models

This project explores prediction of per-residue solvent accessibility (RSA)
from protein sequences using embeddings from protein language models such as
ESM-2 and deep learning architectures including CNNs and Transformers.

The goal is to investigate sequence–structure relationships in proteins and
develop machine learning approaches relevant to computational protein design.


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

## Notes on Paths and Reproducibility

The notebooks in this repository were developed in a local environment where the project folders (`data`, `cache`, and notebooks) were located inside the same working directory. Because of this, some paths inside the notebooks and Python modules may not directly match the directory structure on another machine.

Users replicating this workflow may need to adjust file paths depending on their local setup (for example where `data/`, `cache/`, or downloaded structures are stored).

Typical adjustments may include:

- modifying relative paths to `data/raw/`, `data/processed/`, or `cache/`
- updating dataset locations when loading cached embeddings
- adjusting paths when importing modules from `src/`

The repository is intended primarily as a **reference pipeline and code base**, and users are encouraged to adapt the paths to their own project structure.


---

# Pipeline Overview

This repository implements a small pipeline for predicting **per-residue relative solvent accessibility (RSA)** from protein sequences using **embeddings from protein language models (ESM2)** and lightweight neural architectures.

The workflow follows these main steps.

---

### 1. Extract PDB identifiers from culled FASTA files

The starting point is a culled FASTA dataset containing protein chains derived from the Protein Data Bank.  
From this file, PDB identifiers and chain IDs are parsed in order to construct a list of structures that will later be downloaded and processed.

This step enables the creation of a non-redundant structural dataset.

Notebook:  
`01_Downloading_mmCIF.ipynb`

---

### 2. Download mmCIF structure files

Using the extracted identifiers, the pipeline downloads thousands of **mmCIF files** from the Protein Data Bank.

This step builds the local structural dataset that will later be used to extract residue-level features.

Notebook:  
`01_Downloading_mmCIF.ipynb`

---

### 3. Extract residue-level structural features

Each downloaded mmCIF structure is processed to extract **per-residue structural features**, including **relative solvent accessibility (RSA)**.

The output of this stage is a residue-level table where each row corresponds to a residue belonging to a specific protein chain.

Notebook:  
`02_Extracting_RSA_from_mmCIF.ipynb`

---

### 4. Build a clean dataset for machine learning

The extracted residue data is reorganized into a dataset that can be used for neural network training.

This step includes:

- grouping residues by protein chain
- masking residues with missing RSA values
- structuring sequences and labels
- exporting the processed dataset (e.g. parquet format)

Notebook:  
`03_Creating_a_data_table.ipynb`

---

### 5. Compute protein language model embeddings

Protein sequences are embedded using **ESM2**, producing **per-residue embedding vectors**.

Because generating embeddings is computationally expensive, the embeddings are **cached to disk** so they can be reused during model training.

This avoids recomputing embeddings every time a model is trained.

Notebook:  
`04_Embedding.ipynb`

---

### 6. Train neural networks on cached embeddings

The cached embeddings are used as input features for several neural network architectures:

- CNN
- BiLSTM
- CNN-BiLSTM
- Transformer

Each model predicts **per-residue RSA values**.

Training notebooks:

- `train_CNN.ipynb`
- `train_BiLSTM.ipynb`
- `train_CNN_BiLSTM.ipynb`
- `train_transformer.ipynb`

During training, models are evaluated using:

- masked mean squared error (training loss)
- **Pearson correlation (R)** between predicted and ground-truth RSA values

Pearson correlation is used as the main evaluation metric to assess how well predicted accessibility correlates with structural accessibility derived from experimental structures.

---

# What This Repository Is Most Useful For

The primary purpose of this repository is to provide reusable code that facilitates several common tasks in structural bioinformatics and sequence-based modeling.

In particular, the code can help with:

1. **Extracting PDB identifiers from culled FASTA files**

2. **Downloading large numbers of mmCIF structure files from the Protein Data Bank**

3. **Extracting residue-level structural features from protein structures**

4. **Framing residue-level structural data in a format suitable for neural network training**

5. **Embedding protein sequences using protein language models and caching the embeddings**

6. **Training neural network models on cached embeddings while evaluating predictions using Pearson correlation**

Users may adapt individual parts of the pipeline depending on the needs of their own projects.

# Acknowledgments

## This project uses the **ESM protein language model** developed by Meta AI:  
https://github.com/facebookresearch/esm
=======

>>>>>>> 2685bb63aa318fdf55655ecaae6c5ea4d4f095e0
