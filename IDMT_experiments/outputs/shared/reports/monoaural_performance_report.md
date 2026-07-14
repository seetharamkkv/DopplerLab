# Monoaural (mel) runs — consolidated performance report

All runs below use **log-mel spectrograms** from a single audio channel (`feature_type=mel`), not stereo cross-correlation (`cc`). Mic/channel filter: **SE / CH34**, EUSIPCO official test split unless noted.

Generated from `outputs/cnn/**/eval_*.json` and `eval_predictions.csv` in this repo.

---

## Why headline accuracy is inflated

The direction task is **3-class** (L2R / R2L / `no_vehicle`). On the EUSIPCO test set:

| Subset | Clips | Share |
|--------|------:|------:|
| **no_vehicle** (background) | 1,412 | **51.2%** |
| **Vehicle** (L2R + R2L) | 1,346 | 48.8% |
| **Total** | 2,758 | 100% |

`no_vehicle` is trivially easy (~99.9% recall on every run), so **overall 3-class accuracy overstates direction skill by ~10–13 points**. The meaningful numbers are the **vehicle-only** metrics below.

**Accuracy inflation (headline − vehicle-only):**

| Run | Headline acc | Vehicle-only acc | Inflation |
|-----|-------------:|-----------------:|----------:|
| `mel_3class` (mean) | 90.8% | **81.2%** | +9.6 pp |
| `mel_3class_left` | 90.1% | **79.8%** | +10.3 pp |
| `mel_3class_right` | 87.7% | **75.0%** | +12.8 pp |

---

## Direction — EUSIPCO test (primary runs)

> **Baseline checkpoint status (2026-07-02):** Run metadata and eval reports for
> `mel_3class`, `mel_3class_left`, and `mel_3class_right` are intact under
> `checkpoints/cnn/direction/<run_name>/` (JSON only). **`best.pt` weights are
> gitignored and are not present on a fresh clone** — you cannot re-run inference/eval
> until weights are restored locally or the run is retrained with the same config.
> `mel_3class_left_ep60` is a **separate** run folder; it did **not** overwrite the
> original left baseline. Do not edit core CNN code (`cnn/model.py`, `cnn/train.py`,
> `cnn/dataset.py`) when extending the physics track.

### Summary table

| Run | Mono source | Best val acc | **Headline** (3-class) | **Vehicle-only** acc | **Bal. acc** (L2R/R2L) | Macro F1 (all) | Macro F1 (vehicle 2-class) |
|-----|-------------|-------------:|-----------------------:|---------------------:|------------------------:|---------------:|---------------------------:|
| `mel_3class` | mean (L+R)/2 | 95.1% | acc **90.8%** | **81.2%** | **81.5%** | 87.5% | 81.3% |
| `mel_3class_left` | left channel | 93.2% | acc **90.1%**, bal **86.2%** | **79.8%** | **79.3%** | 86.3% | 79.5% |
| `mel_3class_right` | right channel | 95.8% | acc **87.7%**, bal **82.4%** | **75.0%** | **73.6%** | 82.0% | 73.2% |

*Vehicle-only acc = accuracy restricted to clips with true label L2R or R2L (predicting `no_vehicle` on a vehicle clip counts as wrong).*

### Per-class breakdown (full test set)

| Run | L2R recall | R2L recall | `no_vehicle` recall | False `no_vehicle` on vehicle clips |
|-----|----------:|-----------:|--------------------:|------------------------------------:|
| `mel_3class` (mean) | 86.6% | 76.5% | 99.9% | 6 |
| `mel_3class_left` | 71.2% | 87.4% | 99.9% | 2 |
| `mel_3class_right` | 51.6% | 95.7% | 99.9% | 1 |

### Direction confusion on vehicle clips only

| Run | L2R→R2L errors | R2L→L2R errors | Majority baseline (always R2L) |
|-----|---------------:|----------------:|-------------------------------:|
| `mel_3class` (mean) | 85 | 168 | 53.1% |
| `mel_3class_left` | 182 | 90 | 53.1% |
| `mel_3class_right` | 306 | 31 | 53.1% |

**Takeaways for direction:**

- Mean-mono (`mel_3class`) is the best vehicle-direction model: **~81% on the hard half** of the test set vs 53% majority baseline.
- Left/right single-channel runs show **strong ear asymmetry**: left favors R2L (87% recall), right favors R2L even more (96%) but crushes L2R (52%). This is expected — monoaural direction is inherently ambiguous without stereo cues.
- `no_vehicle` detection is essentially solved (99.9% recall, ≤6 false positives on 1,346 vehicle clips). It does not discriminate between runs.
- Channel-swap diagnostic (left/right runs only): flip consistency **6.8%** (left) / **11.1%** (right) — confirms monoaural models do not respect stereo flip symmetry (unlike `cc` models).

### Smoke / dev run (`smoke_mel`)

Early/ablation run — not a production checkpoint:

| Metric | Value |
|--------|------:|
| Headline 3-class acc | 81.6% |
| Vehicle-only acc | **62.6%** |
| `no_vehicle` recall | 99.9% |
| Inflation | +19.1 pp |

---

## Weather — mel (monoaural)

| Run | Split | n | Acc | Bal. acc | Dry recall | Wet recall | Notes |
|-----|-------|--:|----:|---------:|-----------:|-----------:|-------|
| `weather_mel` | `weather_stratified` (pooled) | 894 | **92.3%** | — | 98.2% | 66.3% | **Confounded** — wet only at Schleusinger-Allee; location oracle ≈ 89% |
| `weather_mel_site` | `weather_site` (honest) | 270 | **86.3%** | **86.5%** | 87.5% | 85.5% | Primary weather result |

**`weather_mel_site` naive baselines** (same 270-clip test set):

- Always dry: 38.5%
- Majority class / location oracle: 61.5%
- Model minus location oracle: **+24.8 pp** (real signal beyond site confound)

No `no_vehicle` class in weather — bloating does not apply here. The pooled split (`weather_mel`) is inflated by location, not background class.

---

## External inference (no held-out labels)

| Dataset | Model | Clips | Mode | Result |
|---------|-------|------:|------|--------|
| VS13 pass-by | `mel_3class` | 400 | direction-only (`no_vehicle` disabled) | 87.3% R2L, 12.8% L2R — no ground truth in outputs |

---

## What to report (de-bloated)

For **direction**, prefer these over headline accuracy:

1. **Vehicle-only accuracy** (~75–81%) — strips the easy 51% background class
2. **Balanced accuracy on L2R/R2L** — equal weight per direction (~74–82%)
3. **Per-direction recall** — shows left/right channel asymmetry
4. **Macro F1 on vehicle 2-class** (~73–81%)

For **weather**, use **`weather_mel_site`** with balanced accuracy (86.5%), not the pooled `weather_mel` run (92.3% is location-inflated).

---

## Run configuration reference

| Run | Task | `mono_source` | Split | Checkpoint epoch |
|-----|------|---------------|-------|-----------------:|
| `mel_3class` | direction | `mean` | eusipco | 10 |
| `mel_3class_left` | direction | `left` | eusipco | 26 |
| `mel_3class_right` | direction | `right` | eusipco | 16 |
| `weather_mel` | weather | `mean` | weather_stratified | 12 |
| `weather_mel_site` | weather | `mean` | weather_site | 27 |

---

## Source artifacts

| Run | Metrics | Predictions | Summary |
|-----|---------|-------------|---------|
| `mel_3class` | `outputs/cnn/direction/mel_3class/eval_metrics.json` | `eval_predictions.csv` | `eval_summary.txt` |
| `mel_3class_left` | `outputs/cnn/direction/mel_3class_left/eval_metrics.json` | `eval_predictions.csv` | `eval_summary.txt` |
| `mel_3class_right` | `outputs/cnn/direction/mel_3class_right/eval_metrics.json` | `eval_predictions.csv` | `eval_summary.txt` |
| `smoke_mel` | `outputs/cnn/direction/smoke_mel/eval_metrics.json` | `eval_predictions.csv` | `eval_summary.txt` |
| `weather_mel` | `outputs/cnn/weather/weather_mel/eval_metrics.json` | `eval_predictions.csv` | `eval_summary.txt` |
| `weather_mel_site` | `outputs/cnn/weather/weather_mel_site/eval_metrics.json` | `eval_predictions.csv` | `eval_summary.txt` |
| VS13 inference | `outputs/cnn/direction/vs13_direction/predictions_summary.json` | `predictions.csv` | — |

Checkpoints: `checkpoints/cnn/direction/<run_name>/` and `checkpoints/cnn/weather/<run_name>/`.
