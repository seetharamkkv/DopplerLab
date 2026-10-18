# Continuous velocity tracker

No CTC, no bins. CNN–GRU emits continuous `vx, vy`. Position is a trapezoid integral from the labeled first position. Loss is velocity Huber (x and y separate) + integrated-position Huber (x and y separate) + multi-duration displacement. Corners are upweighted. There is no smooth-turn penalty.

## Ground-truth integration audit

On 130 clips in `$DOPPLER_PATH2D_CLIPS` (`vs13_path2d_10_speed_bins`), labeled velocity on the actual frame times reconstructs labeled `x, y` to **0.041 m** RMSE with the trapezoid rule and **0.080 m** with forward Euler. 128/130 clips have a heading jump of at least 12°. The frame clock and trapezoid convention match the labels, including corners.

## Overfit (200 epochs, no early stop)

Four moving clips with the most heading jumps: `sample_0000087`, `sample_0000075`, `sample_0000009`, `sample_0000124`. Dropout off, full sequences, true `p0`.

| | Epoch 1 | Epoch 200 |
|---|---:|---:|
| Path RMSE | 80.3 m | 11.0 m |
| vx MAE | 12.6 m/s | 3.8 m/s |
| vy MAE | 9.3 m/s | 7.6 m/s |

A constant majority `vy` is still cheaper than the bends. The same network, trained on **one** of those clips with a `vy`-only Huber, reached 0.55 m/s MAE in 120 epochs, so the encoder can represent the turns. The joint objective still does not, so the full vehicle-holdout run was not started.

Cumulative signed-error plots for the four clips are `overfit_sample_*.png` in this folder.
