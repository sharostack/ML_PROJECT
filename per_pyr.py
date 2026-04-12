# stage4_periodic_pyramid.py  (FIXED)
import torch
import torch.nn as nn

class PeriodicPyramid(nn.Module):
    """
    FFT → top-k frequencies → amplitude trust modulated by drift.
    ã_k = a_k · σ(w_k − λ · δ̄_d)
    ΔP  = ||a_t − a_{t-1}||  ← actual periodic drift
    """
    def __init__(self, seq_len: int, in_channels: int, top_k: int = 5, lam: float = 1.0):
        super().__init__()
        self.top_k = top_k
        self.lam   = lam
        self.F     = in_channels
        self.w_k   = nn.Parameter(torch.ones(top_k))
        self.freq_proj = nn.Linear(top_k * in_channels, in_channels)

        # Stores previous batch amplitudes for ΔP computation
        self.register_buffer('prev_amp', torch.zeros(1, top_k, in_channels))
        self._prev_initialized = False

    def forward(self, x: torch.Tensor, delta_d: torch.Tensor):
        """
        x       : (B, W, F)
        delta_d : (B, F)
        returns : trusted_embed (B, F), delta_p (B, F)
        """
        B, W, F = x.shape

        # --- FFT + top-k amplitudes ---
        xf      = torch.fft.rfft(x, dim=1)               # (B, W//2+1, F)
        amp     = xf.abs()                                 # (B, W//2+1, F)
        top_vals, _ = amp.topk(self.top_k, dim=1)         # (B, K, F)

        # --- ΔP: actual periodic drift = L2 change in amplitude spectrum ---
        if not self._prev_initialized:
            self.prev_amp = top_vals.detach().mean(dim=0, keepdim=True)  # (1, K, F)
            self._prev_initialized = True

        # delta_p per sample per sensor: (B, F)
        delta_p = (top_vals - self.prev_amp).abs().mean(dim=1)  # (B, F)

        # Update reference: exponential moving average of batch
        self.prev_amp = (0.9 * self.prev_amp +
                         0.1 * top_vals.detach().mean(dim=0, keepdim=True))

        # --- Amplitude trust: ã_k = a_k · σ(w_k − λ·δ̄_d) ---
        delta_bar = delta_d.mean(dim=-1, keepdim=True).unsqueeze(1)  # (B,1,1)
        wk        = self.w_k.unsqueeze(0).unsqueeze(-1)               # (1,K,1)
        trust     = torch.sigmoid(wk - self.lam * delta_bar)          # (B,K,1)
        trusted   = top_vals * trust                                   # (B,K,F)

        out = self.freq_proj(trusted.reshape(B, -1))                   # (B,F)
        return out, delta_p   # ← now returns BOTH