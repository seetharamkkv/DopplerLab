import torch
import torch.nn as nn

class PatchTokenizer(nn.Module):
    def __init__(self, patch_h=128, patch_w=4, in_channels=3, embed_dim=384):
        """
        Rectangular patch tokenizer for Doppler spectrograms.
        
        Uses full-height × narrow-width patches so each token captures the ENTIRE
        frequency spectrum at a specific time slice. This makes it natural for the
        Transformer to track frequency shifts (Doppler curves) across time.
        
        Args:
            patch_h (int): Patch height (should equal n_mels for full-frequency tokens).
            patch_w (int): Patch width (narrow time slice, e.g., 4 frames).
            in_channels (int): Number of input channels (3 for Mel+Delta+DeltaDelta).
            embed_dim (int): Dimension of the output tokens.
        """
        super().__init__()
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Rectangular Conv2d: kernel covers full frequency, narrow time
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=(patch_h, patch_w),
            stride=(patch_h, patch_w)
        )
        
        # [CLS] token for global aggregation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Initialization
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input spectrogram of shape (B, C, H, W)
                              e.g., (B, 3, 128, 512)
        Returns:
            torch.Tensor: Sequence of tokens (B, N+1, D)
                          e.g., (B, 129, 768) for 512/4 = 128 patches + 1 CLS
        """
        B, C, H, W = x.shape
        
        if H % self.patch_h != 0 or W % self.patch_w != 0:
            raise ValueError(
                f"Input ({H}, {W}) must be divisible by patch size ({self.patch_h}, {self.patch_w})"
            )
        
        # Extract and project patches: (B, C, H, W) -> (B, D, 1, W/pw)
        x = self.proj(x)
        
        # Flatten spatial dims: (B, D, 1, N) -> (B, D, N)
        x = x.flatten(2)
        
        # Transpose to sequence: (B, N, D)
        x = x.transpose(1, 2)
        
        # Prepend [CLS] token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        return x

if __name__ == "__main__":
    tokenizer = PatchTokenizer()
    # 3-channel spectrogram: (B, 3, 128, 512)
    dummy_input = torch.randn(2, 3, 128, 512)
    tokens = tokenizer(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Tokenized sequence shape: {tokens.shape}")
    # Expect: (2, 129, 768) -> 512/4 = 128 patches + 1 CLS
