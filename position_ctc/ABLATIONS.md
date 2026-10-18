# Position ablations

The next tracing baseline is a range-first geometric decoder, not another velocity-CTC run. Exact ground-truth \(R(t)\) into that decoder is [`outputs/oracle_range/report.md`](outputs/oracle_range/report.md). Track B (`MotionNet`) is paused until that ceiling is understood. The earlier nine path heads, including range + direction at 2.39 m orbit, are in [`history.md`](history.md).

The last trained motion checkpoint is still [`outputs/motion/motion.pt`](outputs/motion/motion.pt). Held-out figures from that run are in [`outputs/ablate/inference.md`](outputs/ablate/inference.md). The machine-readable motion record is [`outputs/motion/metrics.json`](outputs/motion/metrics.json).

## Why the previous curve was flat

On a pass-by, about three quarters of the frames have `|vy| ≤ 2` m/s, and `vx` stays within 2 m/s of its median on about 96% of frames. Cross-entropy is minimized by emitting that majority bin on every frame: about `+14` m/s in `x` and `0` in `y`. Viterbi with a one-bin step cost of 1 nat then stays there, because a two-bin jump costs 4 nats and the corner is only a few percent of the frames. The position head was trained to copy that expected velocity, so its slope was flat as well.

`x` spans about 40 m and `y` only about 3 m, so a shared position loss spends its gradient on the straight pass. A single microphone hears range and range rate. Cartesian `vy` is the small lateral leftover, and it was not in the linear readout.

## What changed

The continuous head is now a bearing and a range capped at 91 m. `vx` and `vy` are the derivative of that path, so a change in `y` is the same thing as a nonzero `vy`. Frames that leave the opening velocity are upweighted, and `y` has its own loss. The bin classifiers no longer send their gradient into the encoder, so the majority-bin loss cannot cancel the turn. Each figure adds the spatial path: ground truth, the predicted path, and the integral of the velocity head from the true start.

## Result

Same vehicle holdout. On 200 held-out clips, epoch 27:

| | Value |
|---|---:|
| `vx` MAE | 5.62 m/s |
| `vy` MAE | 2.39 m/s |
| `Δvx` MAE | 0.290 m/s per frame |
| `Δvy` MAE | 0.097 m/s per frame |
| `Δx` MAE | 0.065 m per frame |
| `Δy` MAE | 0.028 m per frame |
| Integrated position RMSE | 16.76 m |
| Integrated orbit | 16.31 m |
| Integrated range MAE | 9.74 m |
| Integrated closest point | 3.66 m |
| Predicted path RMSE | 35.73 m |
| Viterbi integrated RMSE | 7.77 m |
| Direct position RMSE | 36.49 m |

`vx` and `vy` are the derivative of the predicted path. The integrated position starts at the true first position. The predicted path does not: it is the audio-only bearing and range. Viterbi is still the bin decoder. It remains close to a constant `+14` m/s and `0`, and that constant still fits the majority of frames better than the mistimed bend, which is why its integrated error (7.77 m) is lower than the velocity head's (16.76 m). The previous checkpoint, before this fine-tune, had Viterbi `vx` MAE 0.98 m/s, `vy` MAE 1.48 m/s, and integrated RMSE 6.74 m.

The head is no longer a flat line. On validation after 10 epochs, predicted `y` already had the right spread (3.2 m against 3.0 m) while the correlation with the label was only 0.14, and `vy` correlation was about zero. The held-out figures show the same thing: `x(t)` follows the pass, then the path stalls, and the lateral bend is in the wrong place. Epoch 38 of the first run is saved at [`outputs/motion/motion_epoch38.pt`](outputs/motion/motion_epoch38.pt).

## Training

Started from that epoch-38 encoder. AdamW, weight decay 0.0001 on the trunk, none on the bearing head, gradient clip 25. Learning rate warmed up over 3 epochs to 0.0003 (0.0009 on the bearing head), then cosine down. Bin and CTC losses were detached from the encoder. Device was Apple MPS.

```bash
position_ctc/.venv/bin/python position_ctc/scripts/train_motion.py \
    --data "$DOPPLER_AUDIO_CLIPS" \
    --out position_ctc/outputs/motion \
    --checkpoint position_ctc/outputs/motion/motion_epoch38.pt \
    --epochs 30 --patience 10 --lr 0.0003 \
    --transition-lambda 0.25 --lambda-ctc 0.25 \
    --detach-bins
```

## Setup

Dataset: `$DOPPLER_AUDIO_CLIPS` (1300 simulated free-path clips). Targets are `metadata/canonical_state_frames.npy`, columns `x, vx, y, vy`. Stationary pads are removed. Vehicle holdout, seed 0: 900 train, 200 validation (Nissan Qashqai, Peugeot 307), 200 test (Kia Sportage, Renault Captur).
