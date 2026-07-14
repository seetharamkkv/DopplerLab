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
python -m idmt_experiments.infer --checkpoint checkpoints/cnn/direction/day1_mel/best.pt --wav IDMT_Traffic/audio/<clip>.wav
```

## Layout (by model family)

| Track | Source | Checkpoints | Eval outputs |
|-------|--------|-------------|--------------|
| **CNN** (mel / CC) | `src/idmt_experiments/cnn/` | `checkpoints/cnn/<task>/<run>/` | `outputs/cnn/<task>/<run>/` |
| **Physics** (planned) | `src/idmt_experiments/physics/` | `checkpoints/physics/<task>/<run>/` | `outputs/physics/<task>/<run>/` |
| **Shared** | `src/idmt_experiments/src/` | — | `outputs/shared/` (manifest, splits, baselines) |

See [`outputs/README.md`](outputs/README.md) and [`../ref_docs/checkpoints.md`](../ref_docs/checkpoints.md).

## Split modes

| Mode | CLI `--mode` | Leakage control |
|------|--------------|-----------------|
| **EUSIPCO** | `eusipco` (default) | Official train/test lists; validation = 10% of **train events** (not clips); splits by `event_id` so ME/SE/channel variants never straddle splits |
| **Location LOO** | `location_loo` | Hold out one recording site; 4 folds; resume per fold |

Default mic/channel filters: **SE / CH34** (EUSIPCO paper setting).

## Checkpoint / resume (matches `length_estimation`)

- **EUSIPCO:** `checkpoints/cnn/direction/<run_name>/best.pt` + `best_history.json`, `best.summary.json`, `train_summary.json`, `run_config.json`, `split_meta.json`
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
  shared/
    manifest.csv
    splits/eusipco.json
    baselines/classical_cc_3class.json
    figures/cc_direction_proof.png
  cnn/
    direction/<run_name>/eval_*.csv/json/txt
checkpoints/cnn/direction/<run_name>/
../ref_docs/checkpoints.md   # physics + CNN milestone checklist
```

## Tasks

- **3-class (default):** L2R / R2L / no_vehicle
- **2-class:** `--n-classes 2` (vehicle patches only)
- **Weather (dry/wet):** `--task weather --n-classes 2` — classify road surface from pass-by audio (IDMT filename codes `D` / `W`).

**Important:** In IDMT-Traffic, wet recordings exist **only at Schleusinger-Allee**. The other two sites are dry-only, so a pooled split lets a model cheat via location (~89% with a location oracle). The default **`weather_site`** split trains and tests on Schleusinger-Allee only (same mic geometry, speed, road layout). Run `python -m idmt_experiments.run audit-weather` before reporting numbers.

```bash
# Confound audit + naive baselines
python -m idmt_experiments.run audit-weather

# Train dry vs wet (honest site-controlled split)
python -m idmt_experiments.train --task weather --feature-type mel --run-name weather_mel_site --preempt

# Pooled ablation only (location-confounded — do not use as primary result)
python -m idmt_experiments.train --task weather --mode weather_pooled --run-name weather_mel_pooled --preempt

# Evaluate
python -m idmt_experiments.eval --task weather --run-name weather_mel_site

# Single clip (stereo IDMT wav)
python -m idmt_experiments.infer --checkpoint checkpoints/cnn/weather/weather_mel_site/best.pt --wav IDMT_Traffic/audio/<clip>.wav
```

Eval reports **balanced accuracy**, per-class recall/precision, naive baselines, and subgroup metrics. Checkpoints live under `checkpoints/cnn/weather/<run_name>/`.

## Feature types

- `mel` — log-mel spectrogram (mono average), EUSIPCO vehicle-type settings
- `cc` — stereo cross-correlation stack (EUSIPCO direction experiment)
- `stereo_mel` — two-channel mel (ablation)

Eval includes **channel-swap causality** (L/R flip should flip L2R↔R2L prediction).

## Reference

- Plan: `../ref_docs/idmt_traffic_direction_benchmark_plan.md`
- Dataset: [Zenodo 7551553](https://zenodo.org/records/7551553)

## Transfer / complex STFT (add-on)

- **Deep mel / deep complex-STFT:** `python -m idmt_experiments.transfer --run-name <name> --feature-type mel|complex_stft --mono-source mean|left|right ...`
- Feature type `complex_stft` is also supported on the shallow CNN train path (`--feature-type complex_stft`).
- Sequential skip-if-done queue: `python run_complex_stft_queue.py` (respects `IDMT_DEVICE`, `IDMT_CHECKPOINT_DIR`, `IDMT_OUTPUT_DIR`).
- Colab notebooks: `notebooks/phase_b_colab.ipynb`, `notebooks/deep_cpx_right_colab.ipynb`.

## VM + Google Drive (weights stay out of git)

`*.pt` weights are **gitignored** (`checkpoints/**/*.pt`). Push **code** to GitHub; sync **checkpoints/outputs** to Drive from the VM.

```bash
# On the VM (after git pull)
cd IDMT_experiments
export IDMT_DRIVE_ROOT="/mnt/gdrive/Shareddrives/Spectral Transformers - Doppler/DopplerLab/cpx"
export IDMT_DATA_DIR="/path/to/IDMT_Traffic"
export IDMT_DEVICE=cuda   # or cpu
bash run_vm.sh
```

Or train first, then copy artifacts only:

```bash
python sync_artifacts_to_drive.py --drive-root "$IDMT_DRIVE_ROOT" --skip-last-pt
```

Optional env overrides (read by `idmt_experiments.config`):

| Env | Effect |
|-----|--------|
| `IDMT_DATA_DIR` | Dataset root (`audio/` + `annotation/`) |
| `IDMT_CHECKPOINT_DIR` | Where `best.pt` / `last.pt` are written |
| `IDMT_OUTPUT_DIR` | Eval metrics / predictions |
| `IDMT_DEVICE` | Queue device (`cuda` / `cpu`) |
| `IDMT_DRIVE_ROOT` | Destination for `sync_artifacts_to_drive.py` / `run_vm.sh` |

On your laptop or Colab, open the same Shared Drive `cpx/` folder to pull `best.pt` and `eval_metrics.json` (no git needed for weights).
