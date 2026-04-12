# stage5_baseline.py
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, patch_len=6, stride=3, in_channels=14, d_model=64):
        super().__init__()
        self.patch_len = patch_len
        self.stride    = stride
        self.proj = nn.Linear(patch_len * in_channels, d_model)

    def forward(self, x):
        # x: (B, W, F)
        B, W, F = x.shape
        patches = x.unfold(1, self.patch_len, self.stride)  # (B, N, F, P)
        N = patches.shape[1]
        patches = patches.reshape(B, N, -1)                  # (B, N, F*P)
        return self.proj(patches)                             # (B, N, d_model)


class BaselineTransformer(nn.Module):
    def __init__(self, in_channels=14, d_model=64, nhead=4,
                 num_layers=2, patch_len=6, stride=3):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_len, stride, in_channels, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (B, W, F)
        z = self.patch_embed(x)     # (B, N, d_model)
        z = self.encoder(z)         # (B, N, d_model)
        z = z.mean(dim=1)           # (B, d_model)  — mean pooling
        return self.head(z).squeeze(-1)  # (B,)