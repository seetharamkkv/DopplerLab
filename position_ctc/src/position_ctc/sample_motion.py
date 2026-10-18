import torch
import torch.nn as nn
import torch.nn.functional as F


class VehicleTrajectory(nn.Module):
    def __init__(self):
        super().__init__()

        self.register_buffer("window", torch.hann_window(2048))
        self.register_buffer("centers", torch.arange(-30., 32., 2.))

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(16, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(16), nn.Flatten(),
            nn.Linear(32 * 16, 128), nn.ReLU()
        )

        self.lstm = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        self.output = nn.Linear(256, 62)  # 31 bins each for vx and vy

    def forward(self, audio):
        # Audio -> spectrogram
        spec = torch.stft(
            audio, 2048, hop_length=512,
            window=self.window, return_complex=True
        ).abs().log1p()

        # CNN extracts features from each time frame
        batch, freq, time = spec.shape
        frames = spec.transpose(1, 2).reshape(batch * time, 1, freq)
        features = self.cnn(frames).reshape(batch, time, 128)

        # BiLSTM learns temporal patterns
        features, _ = self.lstm(features)

        # Probability distributions for vx and vy
        logits = self.output(features).reshape(batch, time, 2, 31)
        return F.softmax(logits, dim=-1)


def predict(model, audio, p0):
    probabilities = model(audio)

    # Choose most likely velocity bin at each time step
    indices = probabilities.argmax(dim=-1)
    velocity = model.centers[indices]

    # Integrate velocities to get positions
    dt = 512 / 44100
    increments = (velocity[:, 1:] + velocity[:, :-1]) * 0.5 * dt
    position = torch.cat([
        torch.zeros_like(velocity[:, :1]),
        increments.cumsum(dim=1)
    ], dim=1) + p0

    return velocity, position


if __name__ == "__main__":
    model = VehicleTrajectory()

    # Dummy audio for testing
    audio = torch.randn(1, 44100)
    p0 = torch.tensor([-20., 10.])

    velocity, position = predict(model, audio, p0)

    print("Velocity:", velocity.shape)
    print("Position:", position.shape)
    print("Initial position:", position[0, 0])