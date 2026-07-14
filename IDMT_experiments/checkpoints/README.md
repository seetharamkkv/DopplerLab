# Checkpoints layout

Weights (`.pt` / `.joblib`) are **gitignored**; JSON configs, histories, and summaries under `checkpoints/` are tracked.

## Retained direction models (2-class benchmark)

Primary CNN baselines (mono / mean downmix):

| Run | Path |
|-----|------|
| `mel_3class_left` | `cnn/direction/mel_3class_left/best.pt` |
| `mel_3class_right` | `cnn/direction/mel_3class_right/best.pt` |
| `mel_3class` | `cnn/direction/mel_3class/best.pt` |

Phase B — deep residual mel CNN (100 ep):

| Run | Path |
|-----|------|
| `deep_mel_2class_left_100ep` | `transfer/direction/deep_mel_2class_left_100ep/best.pt` |
| `deep_mel_2class_right_100ep` | `transfer/direction/deep_mel_2class_right_100ep/best.pt` |

Phase C (`fusion_2class_100ep`) has **no separate weights** — it late-fuses the two Phase B checkpoints. Fusion weights are stored in `outputs/fusion/direction/fusion_2class_100ep/eval_metrics.json` (`fusion.w_left`, `fusion.w_right`).

Phase D (`film_2class_left_100ep`) weights were **removed** (underperformed baseline). Archived test metrics remain under `outputs/hybrid/direction/film_2class_left_100ep/eval_metrics.json` and in `outputs/phases_bcd_comparison.md`.

## Other tracks

- **Physics:** `physics/direction/<run>/model.joblib` (+ JSON sidecars)
- **Hybrid (archived):** `hybrid/direction/hybrid_mel_left_v3_ep60/` — metrics only in comparison table
- **CNN (other tasks):** `cnn/weather/`, `cnn/vehicle/`, ablation runs under `cnn/direction/`

## Comparison table

```bash
cd src
python -m idmt_experiments.scripts.compare_phases_bcd
```

Use `--refresh` only when checkpoint weights exist and you want to re-run eval.
