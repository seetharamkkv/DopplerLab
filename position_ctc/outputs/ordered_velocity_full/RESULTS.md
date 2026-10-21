# Ordered velocity tracker results

**Test ADE: 8.08 m across 150 held-out clips** (median ADE 6.02 m).

Checkpoint: `best.pt` at **epoch 150** of 300. Decoder: Viterbi L1 penalty **3.0**. Later epochs lowered train loss but did not beat the validation selection score.

## Setup

| Item | Value |
|---|---|
| Dataset | `vs13_path2d_1000_varied/audio_clips` (1000 clips) |
| Split | random clip holdout, seed 42, train 700 / val 150 / test 150 |
| Model | CNN-GRU ordered-bin vx, vy |
| Bins | 1.0 m/s from -60 to 60 |
| Hidden size | 128 |
| Batch / lr | 4 / 3e-4, cosine to 5% of lr |
| Corner weight | 3.0 |
| Selection | `0.5 * (vx_mae + vy_mae)` on val, eval every 5 epochs |
| Integration | trapezoid from labeled p0 |
| Device | MPS (Mac) |
| Run dir | `outputs/ordered_velocity_full` |

`split.json` and `normalization.npz` are fixed for this run. Resume from `last.pt` (epoch 240) finished epochs 241-300. `best.pt` was never replaced after epoch 150.

## Held-out test (best.pt, penalty 3.0)

| Metric | Mean | Median |
|---|---:|---:|
| ADE | 8.08 m | 6.02 m |
| FDE / endpoint | 14.25 m | 10.16 m |
| Path RMSE | 9.14 m | n/a |
| vx MAE | 0.95 m/s | n/a |
| vy MAE | 1.94 m/s | n/a |

n = 150 clips. See `inference/decoder.json` and `test/metrics.json`.

## Validation at the saved checkpoint

Training-time val (epoch 150, penalty 0.0, used for selection):

| Metric | Value |
|---|---:|
| Selection `0.5*(vx+vy)` | 1.488 |
| Path RMSE | 9.16 m |
| vx MAE | 0.869 m/s |
| vy MAE | 2.108 m/s |
| Endpoint | 14.43 m |

After decoder search (penalty 3.0), same checkpoint:

| Metric | Value |
|---|---:|
| Path RMSE | 9.12 m |
| vx MAE | 0.867 m/s |
| vy MAE | 2.080 m/s |
| Endpoint | 14.45 m |

`validation/metrics.json` matches the penalty-3.0 row.

## Decoder penalty search (val path RMSE)

| Penalty | Path RMSE (m) | vx MAE | vy MAE | Endpoint (m) |
|---:|---:|---:|---:|---:|
| 0.0 | 9.158 | 0.869 | 2.108 | 14.43 |
| 0.01 | 9.161 | 0.869 | 2.107 | 14.44 |
| 0.03 | 9.162 | 0.869 | 2.107 | 14.44 |
| 0.1 | 9.172 | 0.867 | 2.105 | 14.44 |
| 0.3 | 9.167 | 0.867 | 2.102 | 14.42 |
| 1.0 | 9.176 | 0.868 | 2.101 | 14.47 |
| **3.0** | **9.122** | **0.867** | **2.080** | **14.45** |

Chosen: **3.0**. Full table: `decoder.json`.

## Training trajectory

Train loss fell for all 300 epochs. Validation peaked at epoch 150, then drifted.

| Epoch | Train loss | Sel. | Path RMSE (m) | vx MAE | vy MAE | Endpoint (m) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8.454 | 3.511 | 25.76 | 4.574 | 2.449 | 41.37 |
| 5 | 4.896 | 2.545 | 16.68 | 1.895 | 3.194 | 28.85 |
| 50 | 2.932 | 1.907 | 12.07 | 1.013 | 2.801 | 20.12 |
| 100 | 1.429 | 1.638 | 10.17 | 0.935 | 2.340 | 16.61 |
| **150** | **0.948** | **1.488** | **9.16** | **0.869** | **2.108** | **14.43** |
| 200 | 0.710 | 1.512 | 9.47 | 0.887 | 2.138 | 15.65 |
| 240 | 0.621 | 1.522 | 9.60 | 0.895 | 2.150 | 15.57 |
| 300 | 0.570 | 1.524 | 9.63 | 0.896 | 2.153 | 15.59 |

Val columns during training use penalty 0.0. Epoch-level log: `history.jsonl`.

### Best five epochs by selection

| Rank | Epoch | Sel. | Path RMSE (m) | vx MAE | vy MAE | Endpoint (m) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 150 | 1.488 | 9.158 | 0.869 | 2.108 | 14.43 |
| 2 | 225 | 1.490 | 9.317 | 0.875 | 2.104 | 15.15 |
| 3 | 185 | 1.506 | 9.548 | 0.905 | 2.108 | 15.63 |
| 4 | 265 | 1.508 | 9.479 | 0.884 | 2.133 | 15.35 |
| 5 | 255 | 1.508 | 9.567 | 0.888 | 2.129 | 15.44 |

Best val path RMSE is also epoch 150 (9.158 m).

## Example clip: sample_0000844 (test)

| Metric | Value |
|---|---:|
| Path RMSE | 3.9 m |
| Endpoint | 9.5 m |

x tracks the label closely. Most residual error is in y / vy.

- `inference/sample_0000844_path.png`
- `inference/sample_0000844_xy_vxvy.png` (x and y vs time)

## Figures on GitHub

| File | Contents |
|---|---|
| `inference/gallery_paths.png` | Six test paths |
| `inference/detail_best_path.png` | Best clip `sample_0000817` |
| `inference/detail_turning_path.png` | Turning clip `sample_0000606` |
| `inference/sample_0000844_path.png` | Sample 844 path |
| `inference/sample_0000844_xy_vxvy.png` | Sample 844 x, y vs time |

Gallery clips: `sample_0000817`, `sample_0000801`, `sample_0000743`, `sample_0000006`, `sample_0000606`, `sample_0000844`.

Hashed plots under `test/` and `validation/` stay local.

## Local only (not in git)

| File | Role |
|---|---|
| `best.pt` | Epoch 150 weights used for reported test numbers |
| `last.pt` | Epoch 300 (optimizer / scheduler / RNG for resume) |
| `feature_cache/` | STFT cache |
| `inference/test_decoded.pkl` | Cached decoded test set |
| `normalization.npz` | Feature mean / std |
| `test/*.png`, `validation/*.png` | Hashed end-of-run plots |

## Notes

vx is much more accurate than vy. Extra training after epoch 150 overfits train loss and does not improve held-out path error.
