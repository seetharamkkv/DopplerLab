# Vehicle length estimation (VS13)

Estimate **overall vehicle length** (metres) from a **single-microphone pass-by recording**, using the **VS13 dataset only**. This package is not a general-purpose length estimator — models, splits, and ground-truth lengths are all tied to VS13’s 13 vehicles and 400 annotated clips.

**Question:** Can a CPA-centred log-mel spectrogram (+ speed) predict length for a car the model has never seen?

**Approach:** Two phases — hand-crafted physics features (Phase A) and a PyTorch CNN (Phase B).

---

## VS13 dataset (required)

| Item | Value |
|------|-------|
| Clips | 400 pass-by recordings |
| Vehicles | 13 (see `data/vehicle_specs.csv`) |
| Length range | 3.96–5.15 m |
| Speed | 30–105 km/h (annotated per clip) |
| Layout | `{VehicleName}/{VehicleName}_{speed}.wav` + `.txt` (`speed_kmh cpa_time_s`) |
| Default path | `length_estimation/data/vs13/` |

Ground-truth lengths and metadata live in [`data/vehicle_specs.csv`](data/vehicle_specs.csv). Folder names must match the `short_name` column (e.g. `Mazda3`, `MercedesAMG550`).

---

## Folder layout

```
length_estimation/
├── README.md                 ← this file
├── requirements.txt          ← dependencies + editable package install
│
├── src/                      # Python package (installed via pip -e src)
│   ├── pyproject.toml
│   └── length_estimation/
│       ├── config.py         # paths, spectrogram + training defaults
│       ├── run.py            # Phase A CLI
│       ├── train.py          # Phase B training
│       ├── eval.py           # Phase B evaluation
│       ├── infer.py          # Phase B inference
│       └── src/              # library modules (preprocess, phase_b, …)
│
├── data/
│   ├── vehicle_specs.csv     # VS13 vehicle catalog
│   └── vs13/                 # VS13 audio (not in git)
├── checkpoints/length_cnn/   # saved models
├── outputs/                  # reports and metrics
├── ref_docs/                 # design notes and roadmap
└── notebooks/
```

### Entry-point modules (what to run)

| Module | Purpose |
|--------|---------|
| `length_estimation.run` | **Phase A** — feature extraction, physics baselines, Ridge/RF LOVO |
| `length_estimation.train` | **Phase B** — train CNN (`split` or `lovo` mode) |
| `length_estimation.eval` | **Phase B** — evaluate checkpoints (split or LOVO) |
| `length_estimation.infer` | **Phase B** — predict length for one or many wav clips |

Run from anywhere after install:

```powershell
python -m length_estimation.train ...
```

---

## Setup

```powershell
cd d:\Antigravity\DopplerLab\length_estimation
python -m pip install -r requirements.txt
```

This installs dependencies and registers the `length_estimation` package in editable mode (`-e src`), so `python -m length_estimation.train` works from any working directory. Use the same Python interpreter you will run training with (e.g. your conda env).

Requires **PyTorch** for Phase B. Place VS13 under `data/vs13/` (or pass `--data-dir`).

---

## Evaluation methods

Three protocols are used. **LOVO is the only one that tests generalisation to an unseen vehicle.**

### 1. LOVO — leave-one-vehicle-out (Phase A & Phase B)

**Protocol:** For each of the 13 vehicles, hold that car out entirely. Train on the other 12 (~370 clips). Test only on the held-out car’s clips.

- Phase A: Ridge / RF / physics affine models per fold.
- Phase B: one CNN per fold → `fold_{VehicleName}.pt`.
- **At eval time:** Mazda3 clips use `fold_Mazda3.pt` (never trained on Mazda3). Kia clips use `fold_KiaSportage.pt`, etc.
- Models receive **spectrogram + speed only** — no vehicle ID.

This answers: *“Can we predict length for a car whose clips were not in training?”*

Pooled LOVO MAE = average error across all 400 clips, each scored by its correct held-out fold.

### 2. Train / valid split (Phase B only)

**Protocol:** Per vehicle, ~80% of clips → train, ~20% → valid (`Train_valid_split.txt`). One model (`best.pt`) sees **all 13 vehicles** in training.

Useful for checking whether a CNN can fit the data, but **not** a strict unseen-vehicle test — the same car types appear in both train and valid.

### 3. In-sample vehicle ID (diagnostic)

Nearest catalog length to the prediction → guess vehicle name. High ID accuracy with weak LOVO length MAE usually means the model learned *which engine* more than *how long*.

---

## Walkthrough

### Phase A — physics features and baselines

```powershell
# 1. Optional: write clip manifest
python -m length_estimation.run index

# 2. Extract hand-crafted features (~envelope width, Doppler width, sub-band lags, …)
python -m length_estimation.run features

# 3. LOVO regression + report
python -m length_estimation.run phase-a
```

**Reports:** `length_estimation/outputs/phase_a/`

| File | Contents |
|------|----------|
| `result_summary.txt` | Human-readable summary (start here) |
| `phase_a_summary_length_m.json` | LOVO metrics for length |
| `clip_predictions.csv` | Per-clip predictions |
| `vehicle_summary.csv` | Per-vehicle breakdown |

---

### Phase B — CNN training

#### Train / valid split (single model, all VS13 cars in training)

```powershell
python -m length_estimation.train
```

Checkpoint: `length_estimation/checkpoints/length_cnn/<run_name>/best.pt`

#### LOVO (13 models — recommended for generalisation)

```powershell
# Full 13-fold run (resumes automatically — skips finished folds)
python -m length_estimation.train --mode lovo --run-name lovo_mel_v1 --preempt

# Retrain one interrupted fold only
python -m length_estimation.train --mode lovo --run-name lovo_mel_v1 --preempt --retrain-folds Mazda3
```

Checkpoints: `length_estimation/checkpoints/length_cnn/lovo_mel_v1/fold_*.pt`

Training options: `--spec-type mel|ssq`, `--epochs`, `--device cuda|cpu`, `--no-speed`, `--skip-eval`.

---

### Phase B — evaluation

```powershell
# LOVO pooled eval (unseen-vehicle protocol) — primary metric
python -m length_estimation.eval --mode lovo --run-name lovo_mel_v1

# Train/valid split eval (in-sample cars)
python -m length_estimation.eval --mode split --run-name mel_length_20260616_004732 --split valid
```

**Reports:** `length_estimation/outputs/phase_b/<run_name>/`

| File | Contents |
|------|----------|
| `eval_summary.txt` | Human-readable summary (start here) |
| `eval_predictions.csv` | Per-clip true/pred length, vehicle, errors |
| `eval_vehicle_summary.csv` | Per-vehicle MAE |
| `eval_metrics.json` | Full metrics + diagnostics |

---

### Phase B — inference on new clips

Requires a `.txt` sidecar with `speed_kmh` and CPA time (same format as VS13).

```powershell
# Single clip
python -m length_estimation.infer `
  --checkpoint length_estimation/checkpoints/length_cnn/lovo_mel_v1/fold_Mazda3.pt `
  --wav length_estimation/data/vs13/Mazda3/Mazda3_50.wav

# All clips with one split model + CSV output
python -m length_estimation.infer `
  --checkpoint length_estimation/checkpoints/length_cnn/mel_length_20260616_004732/best.pt `
  --all --split valid
```

For a **VS13 car under LOVO**, use the fold where that car was held out (e.g. `fold_Mazda3.pt` for Mazda3). For deployment on arbitrary unknown cars, prefer the all-data `best.pt` model — with the caveat that it trained on all VS13 vehicles.

---

## Results (VS13 only)

Metrics below are from completed runs on VS13. Do not extrapolate to other fleets or recording setups.

### Phase A — length (LOVO)

| Metric | Value | Note |
|--------|-------|------|
| LOVO MAE (best model) | **0.235 m** | Same as predicting the training-set mean |
| Best physics affine | 0.236 m | Reassigned Doppler width × speed |
| Verdict | At baseline | Hand-crafted features do not generalise length |

Phase A **wheelbase** LOVO (0.096 m) beats its baseline — envelope duration tracks axle spacing better than overall length. Length prediction targets `length_m` only.

### Phase B — CNN valid split (`mel_length_20260616_004732`)

| Metric | Value | Note |
|--------|-------|------|
| Valid MAE | **0.194 m** | 81 valid clips; all 13 cars seen in training |
| Vehicle ID accuracy | 11% | Predictions bunch near ~4.4 m |
| Pred std ratio | 37% | Under-spreads vs ground truth |

Encouraging on the valid split, but **not** proof of unseen-car generalisation.

### Phase B — CNN LOVO (`lovo_mel_v1`) — decisive metric

| Metric | Value | Note |
|--------|-------|------|
| Pooled LOVO MAE | **0.097 m** | 400 clips, strict held-out-vehicle protocol |
| vs Phase A baseline | 0.235 m → **beats baseline** | |
| Vehicle ID accuracy | 34% | Diagnostic only |
| Pred std ratio | 74% | Less bunching than valid-split model |
| corr(speed, pred) | 0.08 | Low speed leakage |

**Per-vehicle LOVO MAE (highlights):**

| Vehicle | Length (m) | LOVO MAE (m) |
|---------|------------|--------------|
| Mercedes GLA | 4.42 | 0.024 |
| Renault Captur | 4.12 | 0.025 |
| Citroën C4 Picasso | 4.43 | 0.032 |
| Mercedes S550 | 5.15 | **0.328** |
| VW Passat | 4.77 | **0.295** |

Mid-size fleet (4.1–4.5 m) generalises well. Long outliers (S550, Passat) remain hard.

### How to read the metrics

| Metric | What it means |
|--------|---------------|
| **LOVO MAE** | Mean absolute length error (metres) under held-out-vehicle protocol — **primary metric** |
| **Pred std ratio** | Std(predictions) / Std(truth). Much below 100% → model predicts near the fleet mean |
| **Vehicle ID accuracy** | % of clips where nearest catalog length matches true vehicle — identity diagnostic |
| **R²** | Explained variance; useful but secondary to MAE on this narrow 1.18 m label range |

---

## Model architecture (Phase B)

- **Input:** CPA-centred log-mel spectrogram (128 × 704) + normalised speed (km/h)
- **Network:** `PassByLengthCNN` — 4-layer Conv2D, speed fusion head
- **Loss:** Huber on `length_m`
- **Checkpoint:** best validation MAE epoch (`best.pt` or `fold_*.pt`)

Config defaults: `src/length_estimation/config.py` → `PhaseBConfig`.

---

## Checkpoints (current runs)

| Run | Mode | Location |
|-----|------|----------|
| `lovo_mel_v1` | LOVO, mel | `checkpoints/length_cnn/lovo_mel_v1/` |
| `mel_length_20260616_004732` | Train/valid split | `checkpoints/length_cnn/mel_length_20260616_004732/best.pt` |

---

## Further reading

- [`ref_docs/length_estimation_progress_and_roadmap.md`](ref_docs/length_estimation_progress_and_roadmap.md) — presentation notes and next steps (SSQ input, physics channels, etc.)
- [`notebooks/length_from_spectrogram.ipynb`](notebooks/length_from_spectrogram.ipynb) — exploratory notebook

---

## Quick reference

```powershell
# Phase A (VS13 physics baselines)
python -m length_estimation.run features
python -m length_estimation.run phase-a

# Phase B train
python -m length_estimation.train --mode lovo --run-name lovo_mel_v1 --preempt

# Phase B eval (LOVO — unseen car)
python -m length_estimation.eval --mode lovo --run-name lovo_mel_v1

# Phase B infer
python -m length_estimation.infer --checkpoint length_estimation/checkpoints/length_cnn/lovo_mel_v1/fold_Mazda3.pt --wav path/to/clip.wav
```

All commands assume `pip install -r requirements.txt` was run from this folder and VS13 data is at `data/vs13/`.
