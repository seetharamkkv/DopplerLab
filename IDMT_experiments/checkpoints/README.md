# Checkpoints

Weights (`.pt` / `.joblib`) are gitignored. JSON configs, histories, and summaries under `checkpoints/` are tracked.

Only **final** `best.pt` / `model.joblib` files are kept on disk. Resume files (`last.pt`) and intermediate runs were removed.

## Direction (manuscript)

| Run | Path |
|-----|------|
| `mel_3class` | `cnn/direction/mel_3class/best.pt` |
| `mel_3class_left` | `cnn/direction/mel_3class_left/best.pt` |
| `mel_3class_right` | `cnn/direction/mel_3class_right/best.pt` |
| `cc_2class` | `cnn/direction/cc_2class/best.pt` |
| `deep_mel_2class_mean_100ep` | `transfer/direction/deep_mel_2class_mean_100ep/best.pt` |
| `deep_mel_2class_left_100ep` | `transfer/direction/deep_mel_2class_left_100ep/best.pt` |
| `deep_mel_2class_right_100ep` | `transfer/direction/deep_mel_2class_right_100ep/best.pt` |

Late fusion (`fusion_2class_100ep`) has no separate weights. It combines the left/right deep mel checkpoints. Fusion scalars live in `outputs/fusion/direction/fusion_2class_100ep/eval_metrics.json`.

## Direction (complex-STFT extension)

| Run | Path |
|-----|------|
| `cpx_3class` | `cnn/direction/cpx_3class/best.pt` |
| `cpx_3class_left` | `cnn/direction/cpx_3class_left/best.pt` |
| `cpx_3class_right` | `cnn/direction/cpx_3class_right/best.pt` |
| `deep_cpx_2class_mean` | `transfer/direction/deep_cpx_2class_mean/best.pt` |
| `deep_cpx_2class_left_100ep` | `transfer/direction/deep_cpx_2class_left_100ep/best.pt` |

## Physics direction (interventions)

| Run | Path |
|-----|------|
| `physics_lr_2class_left_v3_std` | `physics/direction/physics_lr_2class_left_v3_std/model.joblib` |
| `physics_lr_2class_right_v3_std` | `physics/direction/physics_lr_2class_right_v3_std/model.joblib` |

## Re-eval

```bash
cd src
python -m idmt_experiments.scripts.compare_phases_bcd
```

Use `--refresh` only when the checkpoint weights above are present.
