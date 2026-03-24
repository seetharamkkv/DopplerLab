# DopplerLab: Modular ML System

A modular, reproducible machine learning system for multi-task Doppler audio analysis. Predicts vehicle trajectory path, source speed, and source distance from simulated Doppler-shifted audio clips using CNN and Transformer-based architectures.

> **Looking for the dataset simulator?**  
> Audio clips are generated using **DopplerNet**, a Flask-based Doppler audio simulator with physically accurate wave modelling, multi-path trajectory support, and a full web UI.  
> → [github.com/rohitharumugams/DopplerNet](https://github.com/rohitharumugams/DopplerNet)

---

## Overview

DopplerLab is a modularised port of the DopplerLab notebook benchmarks (CNN and Attention) into a clean, CLI-driven Python package. It supports two dataset versions (`v1`: 5–120 m, `v2`: 5–1000 m) and four model architectures across two benchmark families.

### Tasks

| Task | Type | Output |
|---|---|---|
| Trajectory path classification | 3-class | `straight` / `parabola` / `bezier` |
| Source speed estimation | Regression | m/s (range: 10–50) |
| Source distance estimation | Regression | m (range: 5–120 or 5–1000) |

### Dataset Versions

| Version | Distance Range | `MAX_DIST_M` | Folder |
|---|---|---|---|
| `v1` | 5 – 120 m | 120.0 | `Datasets/neurips_v1/audio_clips` |
| `v2` | 5 – 1000 m | 1000.0 | `Datasets/neurips_v2/audio_clips` |

Both versions use the same label format: `{vehicle}_{path}_{speed}mps_{dist}m_{id}.wav`  
7 vehicle types · 3 trajectory classes · 1,000 clips · 800 / 100 / 100 train/val/test split (class-balanced).

---

## Models

### CNN Benchmark (`*-CQT`)

| Model | Input | Architecture |
|---|---|---|
| `DopplerNet1D` (`1D-CQT`) | `(B, 7, 432)` - 7-channel CQT ridge trajectory | Conv1d encoder (7→64→128→256→128) + additive temporal attention |
| `DopplerNet2D` (`2D-CQT`) | `(B, 1, 84, 432)` - log-CQT spectrogram | Conv2d backbone + learned freq compression + additive temporal attention |

The 1D model operates on seven hand-crafted Doppler features (frequency derivatives, dominant bin, RMS energy, relative time). The 2D model uses the full log-CQT spectrogram with per-bin z-score normalisation.

### Attention Benchmark (`*-Attn`)

| Model | Input | Architecture |
|---|---|---|
| `DopplerTransformer1D` (`1D-Attn`) | `(B, 7, 432)` - 7-channel CQT ridge trajectory | Linear projection + sinusoidal PE + Transformer encoder (3 layers, nhead=4) |
| `DopplerCNNTransformer2D` (`2D-Attn`) | `(B, 1, 84, 432)` - log-CQT spectrogram | Conv2d backbone + learned freq compression + Transformer encoder (2 layers, nhead=4) |

The Attention benchmark is a direct successor to the CNN benchmark. Feature extraction, loss, training infrastructure, and evaluation protocol are intentionally identical for fair comparison. The key upgrade is replacing additive attention with full Transformer encoders and adding explicit sinusoidal positional encoding.

All four models share the same multi-task head structure:

```
→ Shared head: LayerNorm → Linear → ReLU → Dropout(0.2)
→ Path head:   Linear(128→64) → ReLU → Linear(64→3)    [cross-entropy]
→ Speed head:  Linear(128→64) → ReLU → Linear(64→1)    [Huber]
→ Dist head:   Linear(128→64) → ReLU → Linear(64→1)    [Huber, log-normalised]
```

---

## Project Structure

```
DopplerLab/
├── src/
│   ├── data/            # Dataset loading and train/val/test split logic
│   ├── features/        # CQT extraction and 7-channel 1D trajectory features
│   ├── models/          # CNN and Attention architectures + model registry
│   ├── training/        # Multi-task loss, checkpoint logic, training loop
│   └── utils/           # Path management, config, plotting, inference helpers
├── configs/             # YAML configurations per benchmark and dataset version
│   ├── v1_cnn.yaml
│   ├── v2_cnn.yaml
│   ├── v1_attention.yaml
│   └── v2_attention.yaml
├── results/             # Output CSVs, figures, and summary comparisons
├── run.py               # Central CLI entry point
└── requirements.txt
```

---

## Setup

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Configure data paths**

Copy the example env file and set `DATA_ROOT` to the directory containing your `neurips_v1` or `neurips_v2` folder:

```bash
cp .env.example .env
```

```dotenv
# .env
DATA_ROOT=/path/to/your/Datasets
```

**3. Install the local package** *(optional but recommended for imports)*

```bash
pip install -e .
```

---

## Usage

All training and evaluation runs through `run.py` with a config file:

### Training

```bash
# CNN benchmark, v2 dataset, 2D model
python run.py --mode train --config configs/v2_cnn.yaml --model-name cqt_2d --dataset-v v2

# Attention benchmark, v2 dataset, 1D model
python run.py --mode train --config configs/v2_attention.yaml --model-name attn_1d --dataset-v v2
```

### Evaluation

```bash
# Evaluate and generate all figures
python run.py --mode evaluate --config configs/v2_attention.yaml --model-name attn_2d --dataset-v v2
```

Evaluation outputs for each model:

- `results/{benchmark}/{model}/` - per-sample results CSV + summary CSV
- `results/{benchmark}/{model}/figures/` - confusion matrices, speed/distance scatter plots, MAE by distance range, error histograms, per-path MAE, and R² scores (Attention only)

---

## Training Configuration

All models share the same training setup:

| Hyperparameter | Value |
|---|---|
| Optimiser | AdamW, weight decay 1×10⁻⁴ |
| Learning rate | 3×10⁻⁴ |
| Scheduler | CosineAnnealingWarmRestarts (T₀=10, η_min=10⁻⁶) |
| Batch size | 16 |
| Epochs | 500 |
| Gradient clipping | max norm 2.0 |

**Multi-task loss:**

$$\mathcal{L} = \mathcal{L}_{CE}(\hat{p}, p) + 2.5 \cdot \mathcal{L}_{Huber}\!\left(\hat{s},\ \frac{s}{s_{\max}}\right) + 1.5 \cdot \mathcal{L}_{Huber}\!\left(\hat{d},\ \frac{\log(1+d)}{\log(1+d_{\max})}\right)$$

Distance is regressed in log-space to compress the long tail, especially important for v2's 5–1000 m range.

**Best checkpoint selection metric:**

$$\mathcal{M} = \frac{\text{Speed MAE}}{s_{\max}} + \frac{\text{Dist MAE}}{d_{\max}} + \left(1 - \frac{\text{Path Acc}}{100}\right)$$

Checkpoints overwrite a single file per model; no storage bloat.

---

## Reported Metrics

| Metric | Description | Benchmarks |
|---|---|---|
| Path accuracy (%) | 3-class classification accuracy on test set | Both |
| Speed MAE (m/s) | Mean absolute error of speed regression | Both |
| Speed median error | Median absolute error, robust to outliers | Both |
| Speed bias | Mean signed error; positive = overestimate | Both |
| Dist MAE (m) | Mean absolute error of distance regression | Both |
| Dist median error | Median absolute error of distance regression | Both |
| Dist bias | Mean signed error; positive = overestimate | Both |
| Speed R² | Coefficient of determination for speed | Attention only |
| Dist R² | Coefficient of determination for distance | Attention only |

R² is included in the Attention benchmark to give a normalised measure of explained variance, which is useful when comparing v1 (120 m) and v2 (1000 m) results on the same scale.

---

## Configs

Each YAML config file controls model selection, dataset version, loss weights, and output paths. Example (`configs/v2_attention.yaml`):

```yaml
dataset_version: v2
max_dist_m: 1000.0
model: attn_2d
batch_size: 16
epochs: 500
lr: 3e-4
loss_weights:
  path: 1.0
  speed: 2.5
  dist: 1.5
```

---

## Relationship to DopplerNet Simulator

The datasets used by this project are generated with **DopplerNet**, a Flask-based Doppler audio simulator that produces physically accurate vehicle pass-by audio clips across configurable trajectories (straight-line, parabolic, Bezier), speeds, and distances.

→ **Simulator repo:** [github.com/rohitharumugams/DopplerNet](https://github.com/rohitharumugams/DopplerNet)

Clip filenames encode all ground-truth labels: `{vehicle}_{path}_{speed}mps_{dist}m_{id}.wav`, which DopplerLab parses directly during dataset loading; no separate annotation files needed.

---

## Authors

**Seetharam Killivalavan & Rohith Arumugam Suresh**  
Computer Science and Engineering  
Sri Sivasubramaniya Nadar College of Engineering  
*Research Interns, Carnegie Mellon University*

---

## Acknowledgments

We acknowledge Carnegie Mellon University, the Language Technologies Institute, Bradley Warren, and Professor Bhiksha Raj for their research guidance and support.
