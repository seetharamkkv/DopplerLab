"""Minimal audio -> frequency Conv1d -> BiLSTM -> velocity -> trajectory.

Dependencies: torch; matplotlib only for plot_result(). One unpadded clip per call.
Inputs are float tensors on the model's device. No audio resampling is implicit.
Frame labels MUST match the returned STFT times, not just the number of frames.
Keep trailing audio context; target_slice selects the supervised interval.
CTC uses auxiliary blank-inclusive heads; dense heads supply timed velocities.
"""
import torch
from torch import nn
from torch.nn import functional as F


class VehicleTrajectory(nn.Module):
    def __init__(self, sample_rate=44100, n_fft=2048, hop=512, hidden=128):
        super().__init__()
        self.sample_rate, self.n_fft, self.hop = sample_rate, n_fft, hop
        self.register_buffer('window', torch.hann_window(n_fft))
        self.register_buffer('centers', torch.arange(-30., 32., 2.))
        # Each STFT column is [1, F]. Convolution/strides act on FREQUENCY.
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(16, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(16), nn.Flatten(),
            nn.Linear(32 * 16, hidden), nn.ReLU())
        self.initial_position = nn.Linear(2, 16)
        self.lstm = nn.LSTM(hidden + 16, hidden, batch_first=True,
                            bidirectional=True)
        self.velocity_heads = nn.ModuleList([nn.Linear(2 * hidden, 31)
                                             for _ in range(2)])
        self.ctc_heads = nn.ModuleList([nn.Linear(2 * hidden, 32)
                                        for _ in range(2)])

    def forward(self, waveform, p0):
        """waveform [N] mono, p0 [2] meters; returns full-audio frame outputs."""
        if waveform.ndim != 1 or waveform.numel() < 2:
            raise ValueError('Provide a mono waveform [N] with at least 2 samples.')
        if p0.shape != (2,):
            raise ValueError('p0 must have shape [2].')
        stft = torch.stft(waveform, self.n_fft, hop_length=self.hop,
                         window=self.window, center=True, pad_mode='constant',
                         return_complex=True)
        # Centered STFT: frame t is centered on sample t * hop.
        times = torch.arange(stft.shape[-1], device=waveform.device,
                             dtype=torch.float64) * self.hop / self.sample_rate
        # Exclude a possible center exactly beyond the last recorded sample.
        valid = times <= (waveform.numel() - 1) / self.sample_rate
        times, stft = times[valid], stft[:, valid]
        spectrum = torch.log1p(stft.abs()).T   # [T, F]; preserves absolute level
        features = self.cnn(spectrum.unsqueeze(1))  # [T, hidden]
        # Fixed unit scaling, not fitted from validation/test labels.
        anchor = torch.tanh(self.initial_position(p0 / 100.))
        features = torch.cat([features, anchor.expand(len(times), -1)], dim=-1)
        h, _ = self.lstm(features.unsqueeze(0))
        h = h.squeeze(0)
        dense = torch.stack([head(h) for head in self.velocity_heads], dim=1)
        ctc = torch.stack([head(h) for head in self.ctc_heads], dim=1)
        return {'logits': dense, 'ctc_logits': ctc, 'times': times}


def motion_loss(outputs, velocity, target_times, centers, target_slice=None,
                sigma=1.0, ctc_weight=0.1):
    """velocity [T,2] m/s, times [T] on the ORIGINAL audio clock.

    Gaussian neighborhood cross-entropy + standard CTC per axis.
    sigma is in m/s; sigma=1 is half the default bin spacing.
    target_slice can retain extra trailing encoder context without labeling it.
    """
    region = slice(None) if target_slice is None else target_slice
    logits = outputs['logits'][region]
    ctc_logits = outputs['ctc_logits'][region]
    times = outputs['times'][region]
    if velocity.shape != logits.shape[:2] or len(velocity) == 0:
        raise ValueError('Velocity labels must be [T,2] for the selected frames.')
    if target_times.shape != times.shape or not torch.allclose(
            target_times.to(times), times, atol=1e-6, rtol=0):
        raise ValueError('Labels must be aligned to the actual STFT frame times.')
    if not torch.isfinite(velocity).all() or not torch.isfinite(target_times).all():
        raise ValueError('Labels/times must be finite.')
    if (velocity < centers.min()).any() or (velocity > centers.max()).any():
        raise ValueError('Velocity is outside the bin range; extend the centers.')
    if sigma <= 0:
        raise ValueError('sigma must be positive.')
    distance = velocity[..., None] - centers
    soft_targets = torch.softmax(-0.5 * (distance / sigma).square(), dim=-1)
    # Mean over time, sum over axes and classes.
    neighborhood = -(soft_targets * logits.log_softmax(-1)).sum(-1).mean(0).sum()
    indices = distance.abs().argmin(-1)
    ctc_loss = logits.new_zeros(())
    T = len(velocity)
    for axis in range(2):
        # Blank = 0; physical bins have CTC indices 1..K.
        target = torch.unique_consecutive(indices[:, axis]) + 1
        log_probs = ctc_logits[:, axis].log_softmax(-1).unsqueeze(1)  # [T,1,K+1]
        ctc_loss = ctc_loss + F.ctc_loss(
            log_probs, target, torch.tensor([T], dtype=torch.long),
            torch.tensor([len(target)], dtype=torch.long),
            blank=0, reduction='mean', zero_infinity=False)
    total = neighborhood + ctc_weight * ctc_loss
    if not torch.isfinite(total):
        raise ValueError('Non-finite loss; inspect logits and CTC alignment.')
    return total, {'neighborhood': neighborhood.detach(), 'ctc': ctc_loss.detach()}


@torch.no_grad()
def viterbi(log_probs, centers, penalty=0.25, max_jump=6.0):
    """[T,K] -> [T] bin indices. Quadratic cost in bin steps.

    max_jump is m/s PER FRAME, a discretization constraint, not a realistic
    acceleration guarantee. Defaults reproduce the earlier three-bin mask.
    """
    if log_probs.ndim != 2 or len(log_probs) == 0:
        raise ValueError('Expected nonempty [T,K] log probabilities.')
    if penalty < 0 or max_jump < 0:
        raise ValueError('Transition parameters must be nonnegative.')
    delta = (centers[:, None] - centers[None, :]).abs()  # previous, current
    width = centers[1] - centers[0]
    transition = -penalty * (delta / width).square()
    transition = transition.masked_fill(delta > max_jump, -torch.inf)
    score = log_probs[0]
    back = []
    for emission in log_probs[1:]:
        best_score, predecessor = (score[:, None] + transition).max(dim=0)
        score = best_score + emission
        back.append(predecessor)
    index = score.argmax()
    path = [index]
    for predecessor in reversed(back):
        index = predecessor[index]
        path.append(index)
    return torch.stack(path[::-1])


def integrate(velocity, times, p0):
    """[T,2] m/s -> [T,2] meters. Exact anchor; no step before sample zero."""
    if velocity.shape != (len(times), 2) or len(times) == 0:
        raise ValueError('Expected velocity [T,2] and nonempty times [T].')
    dt = times.diff().to(velocity)
    if not (dt > 0).all():
        raise ValueError('Times must increase strictly.')
    increments = 0.5 * (velocity[:-1] + velocity[1:]) * dt[:, None]
    displacement = torch.cat([velocity.new_zeros(1, 2), increments.cumsum(0)])
    return p0[None] + displacement


@torch.no_grad()
def predict(model, waveform, p0, target_slice=None, decoder='viterbi',
            penalty=0.25, max_jump=6.0):
    model.eval()
    output = model(waveform, p0)
    region = slice(None) if target_slice is None else target_slice
    logits, times = output['logits'][region], output['times'][region]
    log_probs = logits.log_softmax(-1)
    if decoder == 'greedy':
        indices = log_probs.argmax(-1)
    elif decoder == 'viterbi':
        indices = torch.stack([viterbi(log_probs[:, a], model.centers,
                                       penalty, max_jump) for a in range(2)], dim=1)
    else:
        raise ValueError('decoder must be greedy or viterbi')
    velocity = model.centers[indices]
    return {'times': times, 'probabilities': log_probs.exp(), 'velocity': velocity,
            'position': integrate(velocity, times, p0)}


def train_step(model, optimizer, waveform, p0, velocity, target_times,
               target_slice=None, sigma=1.0, ctc_weight=0.1):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = model(waveform, p0)
    loss, parts = motion_loss(outputs, velocity, target_times, model.centers,
                              target_slice, sigma, ctc_weight)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    return {'loss': loss.item(), **{k: v.item() for k, v in parts.items()}}


def plot_result(result, true_velocity, true_position, filename='trajectory.png'):
    """Optional four-panel diagnostic; ground truth must match selected frames."""
    import matplotlib.pyplot as plt
    t = result['times'].detach().cpu().numpy()
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    for axis, (ax_v, ax_p) in enumerate(zip(axes[0], axes[1])):
        for ax, predicted, actual, label, unit in [
            (ax_v, result['velocity'], true_velocity, ('vx', 'vy')[axis], 'm/s'),
            (ax_p, result['position'], true_position, ('x', 'y')[axis], 'm')]:
            ax.plot(t, actual[:, axis].detach().cpu(), 'k', label='Ground truth')
            ax.plot(t, predicted[:, axis].detach().cpu(), label='Prediction')
            ax.set(title=label, xlabel='Time (s)', ylabel=unit)
            ax.grid(alpha=0.3)
            ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def main():
    # Synthetic SMOKE TEST only: this random waveform does not encode this path.
    # Replace waveform and labels with real paired simulator examples to train.
    torch.manual_seed(0)
    model = VehicleTrajectory(sample_rate=8000, n_fft=256, hop=80, hidden=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    waveform = torch.randn(8000) * 0.01
    p0 = torch.tensor([-20., 10.])
    with torch.no_grad():
        times = model(waveform, p0)['times']
    velocity = torch.zeros(len(times), 2)
    velocity[:, 0] = 14.
    velocity[len(times)//2:, 1] = 4.
    stats = train_step(model, optimizer, waveform, p0, velocity, times)
    result = predict(model, waveform, p0)
    assert torch.equal(result['position'][0], p0)
    print('Smoke test (not a trained trajectory model):', stats)
    print('Velocity:', result['velocity'].shape, 'Position:', result['position'].shape)


if __name__ == '__main__':
    main()
