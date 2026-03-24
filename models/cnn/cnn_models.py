import torch
import torch.nn as nn
from models.base_blocks import task_head

class DopplerNet1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(7,   64,  9, padding=4), nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Dropout1d(0.1),
            nn.Conv1d(64,  128, 7, padding=3), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout1d(0.1),
            nn.Conv1d(128, 256, 5, padding=2), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout1d(0.15),
            nn.Conv1d(256, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.attn = nn.Sequential(nn.Conv1d(128, 64, 1), nn.Tanh(), nn.Conv1d(64, 1, 1))
        self.shared = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.2))
        self.path_head  = task_head(128, 3)
        self.speed_head = task_head(128, 1)
        self.dist_head  = task_head(128, 1)

    def forward(self, x):
        h = self.encoder(x)
        w = torch.softmax(self.attn(h), dim=-1)
        g = (h * w).sum(dim=-1)
        z = self.shared(g)
        return self.path_head(z), self.speed_head(z).squeeze(1), self.dist_head(z).squeeze(1)

class DopplerNet2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1,   32, 3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(),
            nn.Conv2d(32,  64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64,  128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.freq_compress = nn.Sequential(nn.Conv2d(128, 64, kernel_size=(21, 1), padding=0), nn.BatchNorm2d(64), nn.ReLU())
        self.proj = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU())
        self.attn = nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))
        self.shared = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 128), nn.ReLU())
        self.path_head  = task_head(128, 3)
        self.speed_head = task_head(128, 1)
        self.dist_head  = task_head(128, 1)

    def forward(self, x):
        h = self.cnn(x); h = self.freq_compress(h).squeeze(2); h = self.proj(h)
        h_t = h.permute(0, 2, 1); w = torch.softmax(self.attn(h_t), dim=1)
        g = (h_t * w).sum(dim=1); z = self.shared(g)
        return self.path_head(z), self.speed_head(z).squeeze(1), self.dist_head(z).squeeze(1)
