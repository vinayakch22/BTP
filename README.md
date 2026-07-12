[![Python](https://img.shields.io/badge/Python-3.8%20%7C%203.9-blue.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.0.1-orange.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# H2GnnDTI — Drug-Target Interaction Prediction

A graph neural network model for predicting drug-target interactions (DTI) using heterogeneous graph learning with dynamic fusion of an AutoEncoder (AE) and an Improved Graph AutoEncoder (IGAE).

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Environment Setup](#environment-setup)
4. [Dataset Download](#dataset-download)
5. [Path Configuration](#path-configuration)
6. [Running the Project](#running-the-project)
7. [File-by-File Description](#file-by-file-description)
8. [Troubleshooting](#troubleshooting)

---

## Project Overview

H2GnnDTI predicts whether a given drug-protein pair interacts. It works in two stages:

**Stage 1 — Node Representation (`NodeRepresentation.py`):**
Drugs are encoded as molecular graphs (atoms as nodes, bonds as edges) using Graph Attention Networks (GAT). Proteins are encoded as contact-map graphs using GAT. Both are pooled into fixed-size vectors and concatenated to form a heterogeneous node feature matrix.

**Stage 2 — Interaction Prediction (`model.py`):**
The node feature matrix is fed into H2GNN, which uses a dynamic-gated fusion of an AE (capturing chemical structure) and an IGAE (capturing network topology), followed by GAT refinement and graph propagation, to reconstruct the drug-protein adjacency matrix and predict interactions.

---

## Project Structure

```
H2GnnDTI/
│
├── main.py               # Entry point — trains and evaluates the model
├── model.py              # H2GNN model definition (AE + IGAE + GAT fusion)
├── NodeRepresentation.py # GNNNet: GAT encoder for drugs and proteins
├── setting.py            # Data processing and adjacency matrix construction
├── data_load.py          # Loads raw dataset text file
├── data_process.py       # Alternative data processing (reference/unused in main)
├── graphset.py           # PyTorch Geometric dataset classes for drugs and proteins
├── pro_graph.py          # Protein graph construction (PSSM + contact map features)
├── opt.py                # All hyperparameters and argument definitions
├── utils.py              # Metrics (AUC, AUPR, F1, ACC) and seed utilities
├── scripts.py            # Pre-processing scripts (MSA generation, pconsc4 prediction)
│
└── data/
    ├── davis/
    │   ├── davis.txt         # Raw interaction pairs file
    │   ├── aln/              # Protein .aln alignment files (one per protein)
    │   └── pconsc4/          # Protein contact maps as .npy files (one per protein)
    └── kiba/
        ├── kiba.txt
        ├── aln/
        └── pconsc4/
```

---

## Environment Setup

### Prerequisites

- Python 3.8 or 3.9 (recommended)
- CUDA-capable GPU (strongly recommended; CPU will be very slow)
- Conda (recommended) or pip

### Step 1 — Create a Conda Environment

```bash
conda create -n h2gnndti python=3.8
conda activate h2gnndti
```

### Step 2 — Install PyTorch (with CUDA)

Check your CUDA version first:
```bash
nvidia-smi
```

Then install the matching PyTorch build. Example for **CUDA 11.8**:
```bash
pip install torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For **CUDA 12.1**:
```bash
pip install torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For **CPU only** (slow, not recommended for full training):
```bash
pip install torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 3 — Install PyTorch Geometric

```bash
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
```

> Replace `cu118` with your CUDA version (e.g. `cu121`, `cpu`).

### Step 4 — Install Remaining Dependencies

```bash
pip install rdkit
pip install networkx
pip install scikit-learn
pip install scipy
pip install tqdm
pip install numpy
pip install matplotlib
```

### Full requirements summary (for pip freeze / requirements.txt)

```
torch>=2.0.1
torch-geometric>=2.3.0
torch-scatter
torch-sparse
rdkit
networkx
scikit-learn
scipy
tqdm
numpy
matplotlib
```

---

## Dataset Download

The preprocessed datasets (Davis and KIBA) including raw interaction text files, protein `.aln` alignment files, and `.npy` contact map files are available via the link below.

> **Download link:** *https://drive.google.com/file/d/1I3-CQB2dfJ7xh0SDjGgziTP2XRkM71WD/view?usp=sharing*

The downloaded folder should contain:

```
data/
├── davis/
│   ├── davis.txt
│   ├── aln/          ← ~442 .aln files (one per protein)
│   └── pconsc4/      ← ~442 .npy files (one per protein)
└── kiba/
    ├── kiba.txt
    ├── aln/          ← ~229 .aln files
    └── pconsc4/      ← ~229 .npy files
```

After downloading and unzipping, place the `data/` folder inside the project root directory:

```
H2GnnDTI/
├── data/         ← place here
├── main.py
├── model.py
└── ...
```

---

## Path Configuration

There are **two files** where you must update the dataset paths to match your local machine.

### 1. `setting.py` — Lines 72–73

Find this block:

```python
msa_path = 'C:/Users/Rithika/OneDrive/Desktop/H2GnnDTI/data/' + dataset + '/aln'
contac_path = 'C:/Users/Rithika/OneDrive/Desktop/H2GnnDTI/data/' + dataset + '/pconsc4'
```

Replace with your own path. Examples:

**Windows:**
```python
msa_path = 'C:/Users/YourName/Projects/H2GnnDTI/data/' + dataset + '/aln'
contac_path = 'C:/Users/YourName/Projects/H2GnnDTI/data/' + dataset + '/pconsc4'
```

**Mac / Linux:**
```python
msa_path = '/home/yourname/H2GnnDTI/data/' + dataset + '/aln'
contac_path = '/home/yourname/H2GnnDTI/data/' + dataset + '/pconsc4'
```

> **Tip:** The cleanest approach is to use a relative path if `data/` is inside the project folder:
> ```python
> msa_path = 'data/' + dataset + '/aln'
> contac_path = 'data/' + dataset + '/pconsc4'
> ```

### 2. `main.py` — Dataset Selection (Line ~22)

Choose which dataset to train on by commenting/uncommenting:

```python
# DATASET = "davis"
DATASET = "kiba"      # ← currently set to kiba
# DATASET = "DrugBank"
```

Change to `"davis"` if you want to train on the Davis dataset:

```python
DATASET = "davis"
# DATASET = "kiba"
```


---

## Running the Project

### Step 1 — Activate the environment

```bash
conda activate h2gnndti
```

### Step 2 — Run training

```bash
python main.py
```

The script will:
1. Load the raw dataset from the `.txt` file
2. Build drug molecular graphs and protein contact-map graphs
3. Construct the heterogeneous drug-protein adjacency matrix
4. Initialise the `GNNNet` node encoder and generate node feature embeddings
5. Train the `H2GNN` model for the number of epochs set in `opt.py` (default: 250)
6. Print test metrics (AUC, AUPR, F1, ACC) every epoch
7. Print the best result at the end

### Expected output

```
Train in kiba
load data
load finished
data shuffle
All XXXXX pairs across XX drugs and XXX proteins.
------------setting2 new-target-----------------
...
epoch: 0
test loss:  0.6931
test auc: 0.XXXX  test aupr: 0.XXXX  test f1: 0.XXXX  test acc: 0.XXXX
...
---------------------------------------
Train in kiba
Elapsed time:  XXXX
best_epoch: XX
Final_AUC: 0.XXXX  Final_AUPR: 0.XXXX  Final_F1: 0.XXXX  Final_ACC: 0.XXXX
---------------------------------------
```

### Changing hyperparameters

All hyperparameters are defined in `opt.py`. Key ones to know:

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 250 | Number of training epochs |
| `--lr` | 1e-4 | Learning rate |
| `--n_z` | 20 | Latent embedding dimension |
| `--n_input` | 160 | Input feature dimension (must match GNNNet output) |
| `--seed` | 3 | Random seed |

To override from the command line:
```bash
python main.py --epochs 300 --lr 5e-4
```

---

## Explainability Pipelines

The project contains two explainability scripts to analyze the model's representations and predictions using Captum's Integrated Gradients:

### 1. Low-Level GAT Explainability (`explainability_lowlevel.py`)

This pipeline computes feature attributions for the structural encoders of drug molecules and protein residue sequences.

* **Command to Run**:
  ```bash
  C:\Users\vinay\AppData\Local\Programs\Python\Python38\python.exe explainability_lowlevel.py
  ```
* **What it does**:
  * Attributes GNNNet embedding strength back to input atom features (for drug SMILES) and residue features (for protein sequences).
  * Selects 100 interaction pairs (50 positive, 50 negative) from the dataset.
* **Output files (saved to `results/explainability_lowlevel/`)**:
  * 100 individual drug atom attribution bar charts and RDKit molecule SVGs.
  * 100 protein residue attribution contact map charts.
  * Global summary charts: `global_attribution_distributions.png` and `global_predictions.png`.
  * Raw attribution results: `all_results.pt`.

### 2. High-Level / Prediction-Level Explainability (`explainability_highlevel.py`)

This pipeline explains the final interaction predictions across five different levels, bypassing the gradient dilution from the self-attention layer.

* **Command to Run**:
  ```bash
  C:\Users\vinay\AppData\Local\Programs\Python\Python38\python.exe explainability_highlevel.py
  ```
* **What it does**:
  * **Level 1**: Attention Analysis (alpha gate values, self-attention scores, GAT refinement weights).
  * **Level 2**: Decoder Attribution (analytical gradient of predicted probability w.r.t the decoder output `z_hat`).
  * **Level 3**: Fused Latent Attribution (Integrated Gradients on `z_tilde` through the decoder path).
  * **Level 4**: AE Feature Attribution (Vectorized IG on input features through the Autoencoder branch).
  * **Level 5**: IGAE Pair Attribution (IG on input features through the Graph Autoencoder branch).
* **Output files (saved to `results/explainability_highlevel/`)**:
  * 100 multi-panel images showing the attention gates, analytical attributions, layer IG, and branch attributions side-by-side (e.g. `sample_0_prediction_explainability.png`).
  * Global summary: `global_prediction_explainability.png` (aggregates prediction distribution, gate dynamics, and branch attribution comparisons).
  * Raw attribution results: `all_results.pt`.

### 3. Extract Explainability Results (`extract_results.py`)

This script loads the raw Integrated Gradients attributions and gate weights saved by the low-level and high-level explainability pipelines, computes statistics, and outputs them to the console for thesis analysis.

* **Command to Run**:
  ```bash
  C:\Users\vinay\AppData\Local\Programs\Python\Python38\python.exe extract_results.py
  ```
* **What it does**:
  * Loads `results/explainability_lowlevel/all_results.pt` and displays stats for drug atom and protein residue attributions.
  * Loads `results/explainability_highlevel/all_results.pt` and outputs alpha fusion gate statistics, prediction scores, and branch-level importance comparisons.

---

## File-by-File Description

### `main.py`
The training script. Loads data, initialises models, runs the training loop, evaluates on the test set each epoch, and prints final best results. Also contains a `visualize_alpha()` function to plot the dynamic gate weights after training.

### `model.py`
Defines the full `H2GNN` model. Contains:
- `AE_encoder` / `AE_decoder` — standard MLP autoencoder for capturing node feature structure.
- `IGAE_encoder` / `IGAE_decoder` — graph-convolution-based autoencoder for capturing topology.
- `H2GNN` — the main model that fuses AE and IGAE outputs through a learned dynamic gate (`fusion_gate`), then applies GAT refinement and graph propagation to produce the final adjacency reconstruction.

### `NodeRepresentation.py`
Defines `GNNNet`, a dual-branch GAT encoder. The drug branch processes molecular graphs; the protein branch processes contact-map graphs. Both output 160-dimensional vectors that are concatenated row-wise into the node feature matrix used by `H2GNN`.

### `setting.py`
Core data processing pipeline. Builds drug molecular graphs (via RDKit), loads protein graphs (via `pro_graph.py`), constructs the heterogeneous adjacency matrix, applies common-neighbour similarity augmentation, and splits data into train/test sets under three settings: new-drug (setting 1), new-target (setting 2), and new-drug-target pair (setting 3).

### `data_load.py`
Reads the raw `.txt` dataset file. Each line contains: drug ID, protein ID, SMILES string, protein sequence, and interaction label (0 or 1). Returns a list of tuples.

### `graphset.py`
PyTorch Geometric `InMemoryDataset` subclasses for wrapping drug graphs (`GraphDataset`) and protein graphs (`DTADataset`) with their respective `collate` functions for batching.

### `pro_graph.py`
Builds protein graph features. Each residue is represented by a 54-dimensional feature vector (PSSM from `.aln` file + one-hot encoding + biochemical properties). Edges are derived from the predicted contact map (`.npy` file from PConsC4) at a threshold of 0.5.

### `opt.py`
All hyperparameters defined via `argparse`. Includes model architecture sizes, learning rate, latent dimension, and epochs. Import `args` from here in any file that needs these settings.

### `utils.py`
Metric computation (`metrics_graph`) returning AUC, AUPR, F1, and accuracy using threshold sweeping. Also contains `setup_seed` for reproducibility.

### `scripts.py`
Offline pre-processing utilities for generating the protein `.aln` and `.npy` files from scratch using HHblits (MSA generation), HHfilter, reformat, CCMpred/PConsC4. **You do not need to run this** if you use the provided dataset download; it was used to generate the pre-processed files already included in the download.

### `data_process.py`
An earlier version of the data processing pipeline (retained for reference). The active pipeline used in `main.py` is `setting.py`.

### `explainability_lowlevel.py`
Low-level GNN explainability script. Attributes `GNNNet` embeddings back to individual drug atoms/bonds and protein residues/contacts using Captum's Integrated Gradients.

### `explainability_highlevel.py`
High-level/prediction-level explainability script. Explains final DTI predictions across five levels (attention gates, decoder representations, and branch inputs) using analytical gradients and Integrated Gradients.

### `extract_results.py`
Statistical post-processing analysis script that aggregates, summarizes, and prints statistical metrics from the saved raw explainability files (`all_results.pt`).

---

## Validation Pipelines

These scripts scientifically validate the reliability of the explainability findings. They use the existing trained model and saved results — no model retraining is required.

### 1. Perturbation Faithfulness Test (`validation_perturbation.py`)

Tests whether the atoms/residues identified as important by Integrated Gradients actually matter for the model's computation.

* **Command to Run**:
  ```bash
  python validation_perturbation.py
  ```
* **What it does**:
  * For each of the 100 samples, zeros out the top-K important atoms/residues and recomputes the GNNNet embedding.
  * Compares the embedding L2-norm change against zeroing random K atoms/residues (averaged over 10 trials).
  * Drug K values: 3, 5, 10. Protein K values: 5, 10, 20.
* **Statistical test**: Paired t-test (top-K drop vs random-K drop).
* **Output files (saved to `results/validation/perturbation/`)**:
  * `drug_perturbation_faithfulness.png` — Bar chart with significance markers.
  * `protein_perturbation_faithfulness.png` — Bar chart with significance markers.
  * `perturbation_scatter.png` — Per-sample top-K vs random-K scatter.
  * `perturbation_statistics.txt` — Full statistical summary table.
  * `perturbation_results.pt` — Raw results.

### 2. Attribution Stability & Distribution Test (`validation_stability.py`)

Validates that IG attributions are deterministic and tests whether positive/negative interaction pairs have different attribution patterns.

* **Command to Run**:
  ```bash
  python validation_stability.py
  ```
* **What it does**:
  * **LL-3 (Stability)**: Re-runs IG 5 times on 10 samples. Computes pairwise Spearman rank correlation and Jaccard similarity of top-15 atoms/residues across runs.
  * **LL-5 (Distribution)**: Compares attribution magnitude, max importance, and entropy between positive (interacting) and negative (non-interacting) pairs using Mann-Whitney U test and Cohen's d effect size.
* **Output files (saved to `results/validation/stability/`)**:
  * `stability_test.png` — Box plot of Spearman and Jaccard across samples.
  * `pos_vs_neg_distributions.png` — 6-panel comparison of attribution statistics.
  * `stability_distribution_report.txt` — Full statistical summary.
  * `stability_distribution_results.pt` — Raw results.

---

## Troubleshooting

**`FileNotFoundError` for `.aln` or `.npy` files**
The paths in `setting.py` are incorrect. Double-check `msa_path` and `contac_path` and make sure they point to the unzipped `data/` folder.

**`FileNotFoundError` for the dataset `.txt` file**
The folder name in `data_load.py` may not match. Change `./dataset/` to `./data/` (or wherever you placed the files).

**`CRITICAL ERROR: Feature dimension does not match model input`**
The `GNNNet` output dimension does not match `args.n_input`. Both default to 160. If you change `output_dim` in `GNNNet`, update `--n_input` in `opt.py` to match.

**CUDA out of memory**
Reduce batch size or switch to a smaller dataset. The adjacency matrix `pos_transform_adj` step can be memory-intensive for large graphs.

**`torch_scatter` / `torch_sparse` import errors**
Reinstall with the correct CUDA suffix matching your installed PyTorch version:
```bash
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
```

**Slow `pos_transform_adj` step (common neighbour computation)**
This O(n²) loop runs over all node pairs and can take several minutes. This is expected behaviour on first run.
