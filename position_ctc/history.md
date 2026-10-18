# Ablation history

This is the earlier comparison of nine audio-only path heads, plus a fine-tune of the winner, on one vehicle holdout. The row that was selected from this study is **range + direction, fine-tuned**. The current summary of that model is [`ABLATIONS.md`](ABLATIONS.md). Its held-out figures are [`outputs/ablate/inference.md`](outputs/ablate/inference.md).

The machine-readable copy, including every validation epoch, is [`outputs/ablate/ablations.json`](outputs/ablate/ablations.json). Per-model training time was not recorded.

## Result

On 200 held-out clips the selected model reaches **2.39 m** orbit error, **0.85 m** range error, and **0.56 m** closest-point error. A mean path that ignores the audio sits at 7.58 m, 4.63 m, and 6.08 m.

Orbit error is the root-mean-square distance after the best rotation and reflection about the microphone. Range error is the average absolute gap between predicted and true distance from the microphone. Closest-point error is the absolute gap between the two minimum distances. Canonical RMS is the same position error with no rotation.

| Model | Audio only | Orbit | Range MAE | Closest point | Checkpoint |
|---|---|---:|---:|---:|---|
| Mean path | no | 7.58 m | 4.63 m | 6.08 m | none |
| **Range + direction, fine-tuned** | yes | **2.39 m** | **0.85 m** | **0.56 m** | `range_polar_ft.pt` |
| Range + direction | yes | 2.38 m | 0.88 m | 0.69 m | `range_polar.pt` |
| Residual on the mean path | yes | 2.53 m | 1.21 m | 0.97 m | `residual.pt` |
| Cartesian path + range and velocity | yes | 2.85 m | 1.23 m | 1.65 m | `multitask.pt` |
| Cartesian path | yes | 2.88 m | 1.47 m | 1.29 m | `xy.pt` |
| Direction × range, light range loss | yes | 3.02 m | 1.50 m | 1.93 m | `polar.pt` |
| Straight constant-speed pass | yes | 4.64 m | 2.78 m | 2.31 m | `straight.pt` |
| Integrated velocity | yes | 6.12 m | 4.14 m | 1.08 m | `integrate.pt` |
| Integrated velocity, true start | no | 7.60 m | 4.86 m | 3.50 m | `integrate_gt0.pt` |
| Summed frame steps | yes | 8.06 m | 5.76 m | 3.46 m | `delta.pt` |

`integrate_gt0` is given the true first position at training and at test. It is not an audio-only system. The fine-tune was selected because validation orbit fell from 2.73 m to 2.52 m. Its test orbit is 0.01 m higher than the pre-finetune checkpoint, and its range and closest-point errors are lower.

## Shared setup

Dataset: `$DOPPLER_AUDIO_CLIPS` (1300 simulated free-path clips, 13 vehicles, 100 clips each). Audio is mono 16-bit PCM at 44.1 kHz. Targets are `metadata/canonical_state_frames.npy`, columns `x, vx, y, vy`. The microphone is the origin. Canonical labels already place closest approach on the positive y-axis.

Stationary pads are removed. A frame stays only when it lies between the first and last frame whose speed exceeds 0.5 m/s.

Vehicle holdout, seed 0, Python 3.14.7:

| Split | Clips | Vehicles |
|---|---:|---|
| Train | 900 | CitroenC4Picasso, Mazda3, MercedesAMG550, MercedesGLA, OpelInsignia, Peugeot208, Peugeot3008, RenaultScenic, VWPassat |
| Validation | 200 | NissanQashqai, Peugeot307 |
| Test | 200 | KiaSportage, RenaultCaptur |

Features are computed from the wav, not from `spectrograms/stft.npy`. The STFT uses a 2048-point Hann window, hop 512, and reflect padding. Each frame has 144 channels: 48 log-mel bands from 40 Hz to 4 kHz, then power-weighted sine and cosine of phase in those bands. Log-mel is standardized per clip. Training adds a spectral tilt, a frequency mask on 70% of clips, noise of standard deviation 0.01, and a 320-frame crop on 25% of clips. Evaluation uses the full clip and no augmentation.

The encoder is shared. Two stride-2 convolutions, 144 to 64 to 96, kernel size 5, then a bidirectional GRU with hidden size 128 per direction. Each output frame is 256-dimensional on a grid four times coarser than the audio. A linear interpolation returns the head output to every audio frame. Dropout is 0.1. The encoder is bidirectional, so this is an offline reconstruction, not a streaming model.

Optimization is Adam, learning rate 0.001, weight decay 0.0001, batch size 8, global gradient clip 5. The loss is masked smooth L1 on position with beta 2. Padded frames are excluded. Training stops after 30 epochs, or after 6 epochs without a 0.02 m drop in validation orbit error. The saved weights are the best validation epoch, and the test numbers above are that checkpoint. Device was Apple MPS. Python 3.14.7, PyTorch 2.14.1, NumPy 2.5.3.

```bash
position_ctc/.venv/bin/python position_ctc/scripts/ablate_position.py --epochs 30 --patience 6 --batch-size 8
```

The nine-mode process, including data loading and evaluation, took 3254 s. The fine-tune took 372 s. Those are wall-clock times, not per-model training times.

## Ablations

### Range + direction (`range_polar`, then `range_polar_ft`)

The head emits a 2-D direction and one range channel. The direction is normalized. Range is `softplus(z) + 0.5` meters. The Cartesian point is upsampled, then the upsampled radius replaces the radius of the interpolated point, so angle comes from interpolation and distance comes from the range channel. An auxiliary smooth L1 on range has weight 2. The lighter `polar` ablation is the same head with that weight set to 0.5.

`range_polar` ran 25 epochs and kept epoch 19. Test orbit 2.38 m, range 0.88 m, closest point 0.69 m. The fine-tune reloads that checkpoint, resets Adam at learning rate 0.0002, and runs up to 20 epochs with patience 8. It kept fine-tune epoch 17. That is the selected model.

```bash
position_ctc/.venv/bin/python position_ctc/scripts/ablate_position.py \
  --modes range_polar \
  --resume position_ctc/outputs/ablate/range_polar.pt \
  --epochs 20 --patience 8 --lr 0.0002 --batch-size 8
```

Inference:

```bash
position_ctc/.venv/bin/python position_ctc/scripts/infer_position.py
```

### Residual on the mean path (`residual`)

Every training path is resampled to 200 points and averaged. That average is a frozen buffer. The network predicts a residual and adds it to the upsampled mean. No auxiliary loss. 22 epochs, best epoch 16. Test orbit 2.53 m. It beats the mean path and loses to range + direction.

### Cartesian path plus range and velocity (`multitask`)

Same 2-D position head as `xy`. Added losses: weight 0.5 on range, and weight 0.15 on the finite-difference velocity of the predicted path. Ran the full 30 epochs. Best epoch 28. Test orbit 2.85 m.

### Cartesian path (`xy`)

A linear map from each encoder frame to `(x, y)`, then upsampling. No auxiliary loss. 29 epochs, best epoch 23. Test orbit 2.88 m. This is the direct regression baseline for the other heads.

### Direction times range, light range loss (`polar`)

Identical parameterization to range + direction. The range auxiliary weight is 0.5 instead of 2. 22 epochs, best epoch 16. Test orbit 3.02 m and closest-point error 1.93 m. The stronger range penalty is what separates the selected model from this one.

### Straight constant-speed pass (`straight`)

The encoder is averaged over time and mapped to three numbers: speed `softplus(s) + 4` m/s, closest-approach distance `softplus(c) + 1` m, and the fraction of the clip at which closest approach occurs. The path is `x = speed * (t - t_cpa)`, `y = cpa`. It cannot represent a turn. 19 epochs, best epoch 13. Test orbit 4.64 m. Closest-approach distance is the part it gets partly right. Along-track shape is not.

### Integrated velocity (`integrate`)

The head emits `(vx, vy)` in m/s. Position is a learned origin plus a forward Euler sum with `dt = 512/44100` s. An auxiliary loss of weight 0.15 matches that velocity to the finite difference of the true path. 18 epochs, best epoch 12. Test orbit 6.12 m. A small velocity bias accumulates. Direct position prediction does not have that accumulation.

### Integrated velocity with the true start (`integrate_gt0`)

The same integrator, except the origin is the ground-truth position at the first supervised frame, including at test time. 11 epochs, best epoch 5. Test orbit 7.60 m. Supplying the start does not repair the drift, and this number must not be quoted as an audio-only result.

### Summed frame steps (`delta`)

The head emits a displacement per frame. Position is a learned origin plus the cumulative sum. There is no velocity and no turn-rate limit. 11 epochs, best epoch 5. Test orbit 8.06 m, worse than predicting nothing but the mean path.

## Turn limit

This is not one of the nine trained ablations. `position_ctc.kinematics` caps yaw rate by the tighter of a 3.86 m steering radius and 4.90 m/s² lateral acceleration. A fine-tune that added this penalty did not reduce violations, and the kept weights are still `range_polar_ft.pt`. At inference, `project_path` clips the selected model's path onto that cap. Violations fall to zero and test orbit rises from 2.39 m to 4.86 m. The ground-truth polylines themselves violate the cap on 12% of tenth-second chords.

## What these runs are not

The CTC token trainer in `scripts/train.py` is a separate smoke test. It matched the majority initial velocity pair and is not in the table above. These ablations do not emit acceleration tokens, do not take oracle displacement as an encoder input, and do not use autoregressive feedback, 3D, or language conditioning.
