# Core motion model

Selected checkpoint: `best_path.pt`, epoch 10, saved for the lowest validation trajectory RMSE (8.065 m). The trajectory is the integral of graded-Viterbi velocities from the labeled $(x_0, y_0)$.

Epoch 6 reached validation turning $v_y$ MAE 3.948, below epoch 38's 4.183. That checkpoint is saved separately. A lower path error alone is not turn recovery.

## Model

Magnitude and phase features go through a 1D convolution along time. Frequency bins are the input channels. Two stride-2 pools shorten the sequence, a bidirectional LSTM reads it, and the states are linearly upsampled back onto the spectrogram timestamps. Separate dense heads emit $v_x$ and $v_y$ bin logits. Separate CTC heads add a blank symbol. There is no position, acceleration, bearing, or initial-position head.

Neighborhood targets use $\sigma = 2.0$ m/s, one bin width. CTC weight $\lambda = 0.25$. Both axes are averaged over supervised frames only. The CNN and the LSTM are trained from the explicit initialization above, learning rate 1.0e-03.

## Data and decoder

Observer-time labels are unchanged. The trailing audio extends through the propagation delay of the supervised motion and is encoder context only.

Mean tail: 18.0 frames. Delayed lateral frames covered: 257007 / 257007.

Dense $P_x$ and $P_y$ are decoded by independent Viterbi. The graded cost allows a 6 m/s jump per 11.61 ms frame, which is 517 m/s². That cutoff is the audited decoder setting, not a vehicle acceleration limit. Bin centers run from −30 to 30 m/s in steps of 2.

## Validation

| Metric | Epoch 38 val | Best path, train | Best path, val |
| --- | ---: | ---: | ---: |
| $v_x$ MAE | 1.036 | 1.105 | 1.153 |
| $v_y$ MAE | 1.566 | 1.592 | 1.571 |
| Turning $v_y$ MAE | 4.183 | 4.330 | 4.141 |
| Zero-bin false motion | 0.000 | 0.015 | 0.011 |
| Trajectory RMSE (m) | 7.087 | 7.807 | 8.065 |
| Max lateral error (m) | 9.449 | 9.788 | 9.470 |
| Anchor error (m) | 0.000 | 0.000 | 0.000 |
| Forbidden steps | 0.000 | 0.000 | 0.000 |

Forbidden steps are the total count on the split. The other rows are means over clips. Maximum lateral error is the mean, over clips, of each clip's largest $|\hat y - y|$.

Train trajectory RMSE at this checkpoint is 7.807 m and validation is 8.065 m. Train turning $v_y$ MAE is 4.330; validation is 4.141.
