# IDMT experiment outputs

Outputs are grouped by **model family**, then **task**.

```
outputs/
├── shared/                 # Dataset index, splits, classical baselines, figures
│   ├── manifest.csv
│   ├── splits/
│   ├── baselines/
│   ├── figures/
│   └── reports/
├── cnn/                    # Mel / CC / stereo_mel CNN runs
│   ├── direction/<run_name>/
│   ├── vehicle/<run_name>/
│   └── weather/<run_name>/
├── transfer/               # Phase B — deep mel 2-class CNN
│   └── direction/<run_name>/eval_metrics.json
├── fusion/                 # Phase C — late fusion (no new backbone weights)
│   └── direction/<run_name>/eval_metrics.json
├── hybrid/                 # Archived hybrid / Phase D metrics
│   └── direction/<run_name>/
├── physics/                # Physics-informed direction models
│   └── direction/<run_name>/
└── phases_bcd_comparison.md   # Baselines + Phases B–D table (vehicle-only 2-class)
```

Checkpoints mirror the same layout under `checkpoints/<family>/`. See [`checkpoints/README.md`](../checkpoints/README.md) for which `.pt` files to keep.

Source code: `src/idmt_experiments/cnn/`, `transfer/`, `fusion/`, `hybrid/`, `physics/`.
