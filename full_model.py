# stage6_full_model.py
import torch
import torch.nn as nn
from per_pyr import PeriodicPyramid
from baseline import PatchEmbedding

#normal transformer scaled w drift
class DriftConditionedAttention(nn.Module):
    """Drift-gated Transformer encoder layer."""
    def __init__(self, d_model=64, nhead=4, F=14):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        # Gate: maps δ_d → scalar gate per token
        self.drift_gate = nn.Sequential(
            nn.Linear(F, d_model),
            nn.Sigmoid()
        )

    def forward(self, z, delta_d):
        """
        z       : (B, N, d_model)
        delta_d : (B, F)
        """
        # transformer learns features and we convert drift to gate
        gate = self.drift_gate(delta_d).unsqueeze(1)  # (B, 1, d_model)
        z = self.encoder(z) * gate                     # element-wise gate
        return z

#final pred uses both drift signals
class JointForecastHead(nn.Module):
    """
    ŷ = Linear(z) · σ(W_g [δ_d ; δ_p])
    """
    def __init__(self, d_model=64, F=14):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)
        self.gate   = nn.Linear(2 * F, 1)

    def forward(self, z, delta_d, delta_p):
        """
        z       : (B, d_model)
        delta_d : (B, F)  — distribution drift
        delta_p : (B, F)  — periodic shift
        """
        base = self.linear(z)                          # (B, 1)
        g    = torch.sigmoid(self.gate(
                   torch.cat([delta_d, delta_p], dim=-1)))  # (B, 1)
        return (base * g).squeeze(-1)                  # (B,)

#full model 
class DriftAwarePatchTST(nn.Module):
    def __init__(self, in_channels=14, d_model=64, nhead=4,
                 patch_len=6, stride=3, top_k=5, lam=1.0, window=30):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_len, stride, in_channels, d_model)
        self.drift_attn  = DriftConditionedAttention(d_model, nhead, in_channels)
        self.pyramid     = PeriodicPyramid(window, in_channels, top_k, lam)
        self.head        = JointForecastHead(d_model, in_channels)
        self.F = in_channels
# stage6_full_model.py — add to DriftAwarePatchTST.forward()

    def forward(self, x, delta_d, ablation=None):
        """
        ablation: None | 'no_D' | 'no_dP' | 'periodic_only'
        """
        # Periodic pyramid always runs
        trusted_embed, delta_p = self.pyramid(x, delta_d)

        # Ablation overrides
        #no dist drift - ignre drift - model dsnt see drift 
        if ablation == 'no_D':
            delta_d_eff = torch.zeros_like(delta_d)
        else:
            delta_d_eff = delta_d

        #no periodic drift
        if ablation == 'no_dP':
            delta_p_eff = torch.zeros_like(delta_p)
        
        #pattern used but affected by drift 
        elif ablation == 'periodic_only':
            delta_d_eff = torch.zeros_like(delta_d)   # zero out D
            delta_p_eff = delta_p                       # keep only ΔP
        else:
            delta_p_eff = delta_p

        z = self.patch_embed(x)
        z = self.drift_attn(z, delta_d_eff)
        z = z.mean(dim=1)
        y_hat = self.head(z, delta_d_eff, delta_p_eff)
        return y_hat, delta_p