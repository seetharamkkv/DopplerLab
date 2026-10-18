# Full motion model

Selected checkpoint: `best_path.pt`, epoch 30, saved for the lowest validation trajectory RMSE (7.196 m). This is the integrated Viterbi trajectory. It is not, by itself, evidence that lateral turns were recovered.

Epoch 16 reached validation turning $v_y$ MAE 3.992, below epoch 38's 4.183.

## Data

Observer-time labels are unchanged. The leading stationary pad is still removed. The trailing audio now extends through the propagation delay of the supervised motion (`range / 343`), so the bidirectional LSTM can hear the sound of the last lateral frames. Those extra frames are not supervised and are not included in the trajectory metrics.

Mean tail: 18.0 frames. Delayed lateral frames covered: 257007 / 257007.

Input mean and standard deviation, and the continuous loss scales, were computed on the training split only.

## Decoder

Dense $P_x$ and $P_y$ are decoded by independent Viterbi. CTC emissions are training targets only. The graded cost uses a 6 m/s jump per 11.61 ms frame, which is 517 m/s². That cutoff is the audited decoder setting. It is not a claim about how fast the vehicle can accelerate. Bin centers run from −30 to 30 m/s in steps of 2. The integral uses each clip's frame timestamps and starts at the labeled $(x_0, y_0)$.

## Initialization

Epoch 38 supplied the LSTM recurrent weights, the velocity and CTC heads, and the position and acceleration matrices. Its convolution was not loaded: that network pooled time and mixed the mel bins, and this one downsamples frequency only. The new CNN, feature projection, and initial-position MLP were initialized explicitly. The extra LSTM input columns that carry $p_0$ were drawn from $N(0, 0.02)$.

Pretrained LSTM and velocity/CTC parameters use learning rate 1.0e-04. The new CNN, projection, position MLP, and auxiliary position/acceleration heads use 1.0e-03. Auxiliary loss weights ramp from 0 over the first 452 of 4520 updates. The first backward pass reached the CNN and the LSTM.

## Validation

| Metric | Epoch 38 val | Best path, train | Best path, val |
| --- | ---: | ---: | ---: |
| $v_x$ MAE | 1.036 | 0.727 | 1.185 |
| $v_y$ MAE | 1.566 | 1.414 | 1.429 |
| Turning $v_y$ MAE | 4.183 | 4.155 | 4.038 |
| Zero-bin false motion | 0.000 | 0.072 | 0.096 |
| Trajectory RMSE (m) | 7.087 | 4.654 | 7.196 |
| Max lateral error (m) | 9.449 | 7.478 | 7.611 |
| Anchor error (m) | 0.000 | 0.000 | 0.000 |
| Forbidden steps | 0.000 | 0.000 | 0.000 |

Forbidden steps are the total count on the split. The other rows are means over clips. Maximum lateral error is the mean, over clips, of each clip's largest $|\hat y - y|$.

Train trajectory RMSE at this checkpoint is 4.654 m and validation is 7.196 m. Train turning $v_y$ MAE is 4.155; validation is 4.038.
