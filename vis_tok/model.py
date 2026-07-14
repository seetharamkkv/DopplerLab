import torch
import torch.nn as nn

class AudioTransformerModel(nn.Module):
    def __init__(self, embed_dim=384, num_heads=6, num_layers=4, feedforward_dim=1024, 
                 num_classes=3, dropout=0.3):
        """
        Compact Transformer for Doppler audio — sized to avoid overfitting on small datasets.
        
        v4 changes vs v3:
        - embed_dim: 768 → 384 (4x fewer parameters)
        - num_layers: 6 → 4
        - feedforward_dim: 2048 → 1024
        - dropout: 0.15 → 0.3
        - Simpler 1-hidden-layer MLP heads
        
        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer encoder layers.
            feedforward_dim (int): Feedforward MLP dimension.
            num_classes (int): Number of trajectory classes.
            dropout (float): Dropout rate.
        """
        super().__init__()
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            enable_nested_tensor=False
        )
        
        # LayerNorm before prediction heads
        self.norm = nn.LayerNorm(embed_dim)
        
        # Prediction Heads (simpler to reduce overfitting)
        self.trajectory_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        self.speed_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )
        
        self.distance_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )
        
        # Learnable multi-task loss weights (Kendall et al., 2018)
        self.log_sigma_traj = nn.Parameter(torch.zeros(1))
        self.log_sigma_speed = nn.Parameter(torch.zeros(1))
        self.log_sigma_dist = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Tokenized sequence (B, N+1, D)
        Returns:
            dict: Predictions for trajectory, speed, and distance.
        """
        x = self.transformer(x)
        cls_token = self.norm(x[:, 0, :])
        
        trajectory_logits = self.trajectory_head(cls_token)
        speed_pred = self.speed_head(cls_token)
        distance_pred = self.distance_head(cls_token)
        
        return {
            "trajectory": trajectory_logits,
            "speed": speed_pred,
            "distance": distance_pred
        }
    
    def compute_weighted_loss(self, l_traj, l_speed, l_dist):
        """
        Uncertainty-weighted multi-task loss (Kendall et al., 2018).
        """
        w_traj = torch.exp(-self.log_sigma_traj)
        w_speed = torch.exp(-self.log_sigma_speed)
        w_dist = torch.exp(-self.log_sigma_dist)
        
        total = (w_traj * l_traj + 0.5 * self.log_sigma_traj +
                 w_speed * l_speed + 0.5 * self.log_sigma_speed +
                 w_dist * l_dist + 0.5 * self.log_sigma_dist)
        
        return total

if __name__ == "__main__":
    model = AudioTransformerModel()
    dummy_tokens = torch.randn(2, 129, 384)
    out = model(dummy_tokens)
    print(f"Trajectory shape: {out['trajectory'].shape}")
    print(f"Speed shape: {out['speed'].shape}")
    print(f"Distance shape: {out['distance'].shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
