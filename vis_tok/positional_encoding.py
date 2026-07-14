import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        """
        Learnable positional encoding for the tokenized sequence.
        Args:
            num_patches (int): Total number of patches in the spectrogram.
            embed_dim (int): Dimension of the output tokens.
        """
        super().__init__()
        # +1 for the [CLS] token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Initialization
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Token sequence of shape (B, N+1, D)
        """
        # Element-wise addition of positional embeddings
        return x + self.pos_embed

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        """
        Sinusoidal positional encoding for cases where we want fixed position information.
        """
        super().__init__()
        pe = torch.zeros(num_patches + 1, embed_dim)
        position = torch.arange(0, num_patches + 1, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Token sequence of shape (B, N+1, D)
        """
        # x shape: [B, N+1, D]
        # pe shape: [1, N+1, D]
        return x + self.pe[:, :x.size(1)]

if __name__ == "__main__":
    # Test learnable PE
    pe = PositionalEncoding(num_patches=128, embed_dim=764)
    dummy_tokens = torch.randn(2, 129, 764) # [B, N+1, D]
    output = pe(dummy_tokens)
    print(f"Token shape after PE: {output.shape}")
