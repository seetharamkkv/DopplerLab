import torch
import torch.nn as nn
from models.base_blocks import PositionalEncoding, task_head
from Shared.features.extraction import MAX_T

class DopplerTransformer1D(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=3, dim_ff=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(7, d_model)
        self.pos_enc    = PositionalEncoding(d_model, max_len=MAX_T, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.shared = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(0.2))
        self.path_head  = task_head(d_model, 3)
        self.speed_head = task_head(d_model, 1)
        self.dist_head  = task_head(d_model, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1); x = self.input_proj(x); x = self.pos_enc(x); x = self.encoder(x)
        g = x.mean(dim=1); z = self.shared(g)
        return (self.path_head(z), self.speed_head(z).squeeze(1), self.dist_head(z).squeeze(1))

class DopplerCNNTransformer2D(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=2, dim_ff=256, dropout=0.1):
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
        self.proj = nn.Sequential(nn.Conv1d(64, d_model, 1), nn.BatchNorm1d(d_model), nn.ReLU())
        self.pos_enc = PositionalEncoding(d_model, max_len=200, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.shared = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, d_model), nn.ReLU())
        self.path_head  = task_head(d_model, 3)
        self.speed_head = task_head(d_model, 1)
        self.dist_head  = task_head(d_model, 1)

    def forward(self, x):
        h = self.cnn(x); h = self.freq_compress(h).squeeze(2); h = self.proj(h); h = h.permute(0, 2, 1); h = self.pos_enc(h); h = self.encoder(h)
        g = h.mean(dim=1); z = self.shared(g)
        return (self.path_head(z), self.speed_head(z).squeeze(1), self.dist_head(z).squeeze(1))
