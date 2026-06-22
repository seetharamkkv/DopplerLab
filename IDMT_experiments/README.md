# IDMT_experiments

Direction-of-travel experiments on **IDMT-Traffic** (NeurIPS Benchmark 2 — lateral pass-by).

## Setup

```bash
cd IDMT_experiments
pip install -r requirements.txt
```

Place the dataset under `IDMT_Traffic/` (audio + annotation). This repo layout matches the Zenodo release.

## Quick start (one day)

```bash
# Manifest + leakage audit
python -m idmt_experiments.run index
python -m idmt_experiments.run audit

# Physics figure (mel + stereo CC, L2R vs R2L)
python -m idmt_experiments.run plot-physics

# Classical baseline (~minutes, no GPU)
python -m idmt_experiments.run classical --feature-type cc

# CNN training (EUSIPCO split, auto-eval on official test)
python -m idmt_experiments.train --feature-type mel --epochs 20 --preempt --run-name day1_mel

# Evaluate a saved run
python -m idmt_experiments.eval --run-name day1_mel

# Single-file inference
python -m idmt_experiments.infer --checkpoint checkpoints/direction/day1_mel/best.pt --wav IDMT_Traffic/audio/<clip>.wav
```

## Split modes

| Mode | CLI `--mode` | Leakage control |
|------|--------------|-----------------|
| **EUSIPCO** | `eusipco` (default) | Official train/test lists; validation = 10% of **train events** (not clips); splits by `event_id` so ME/SE/channel variants never straddle splits |
| **Location LOO** | `location_loo` | Hold out one recording site; 4 folds; resume per fold |

Default mic/channel filters: **SE / CH34** (EUSIPCO paper setting).

## Checkpoint / resume (matches `length_estimation`)

- **EUSIPCO:** `checkpoints/direction/<run_name>/best.pt` + `best_history.json`, `best.summary.json`, `train_summary.json`, `run_config.json`, `split_meta.json`
- **Location LOO:** `fold_<Location>.pt` per site + `loo_train_summary.json`
- **Resume (LOO):** skips folds with `fold_complete: true` in summary
- **Interrupted fold:** `--retrain-folds Schleusinger-Allee` (or sanitized `Schleusinger-Allee` → `Schleusinger-Allee` in filename as `fold_Schleusinger-Allee.pt`)
- **Force full retrain:** `--force-retrain`
- **Disable resume:** `--no-resume`
- **Early stop:** `--preempt` (still saves best val checkpoint)

Train normalization stats are fit on **train clips only** — no val/test leakage.

## Outputs

```
outputs/
  manifest.csv
  splits/eusipco.json
  baselines/classical_cc_3class.json
  figures/cc_direction_proof.png
  direction/<run_name>/eval_*.csv/json/txt
checkpoints/direction/<run_name>/
```

## Tasks

- **3-class (default):** L2R / R2L / no_vehicle
- **2-class:** `--n-classes 2` (vehicle patches only)

## Feature types

- `mel` — log-mel spectrogram (mono average), EUSIPCO vehicle-type settings
- `cc` — stereo cross-correlation stack (EUSIPCO direction experiment)
- `stereo_mel` — two-channel mel (ablation)

Eval includes **channel-swap causality** (L/R flip should flip L2R↔R2L prediction).

## Reference

- Plan: `../ref_docs/idmt_traffic_direction_benchmark_plan.md`
- Dataset: [Zenodo 7551553](https://zenodo.org/records/7551553)
