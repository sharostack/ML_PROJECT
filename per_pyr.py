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

    def reset_state(self):
        """Clear EMA memory (call before each eval forward to avoid cross-batch leakage)."""
        self._prev_initialized = False
        self.prev_amp.zero_()

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

        # Batch mean amplitude (1, K, F) — used for EMA (train only) or local ref (eval)
        curr_amp = top_vals.detach().mean(dim=0, keepdim=True)

        # --- ΔP + EMA: only persist / update during training ---
        if self.training:
            if not self._prev_initialized:
                self.prev_amp = curr_amp.clone()
                self._prev_initialized = True
            delta_p = (top_vals - self.prev_amp).abs().mean(dim=1)  # (B, F)
            self.prev_amp = (0.9 * self.prev_amp + 0.1 * curr_amp)
        else:
            # Eval/test: no EMA carry-over; reference is batch-local (no cross-batch state)
            delta_p = (top_vals - curr_amp).abs().mean(dim=1)

        # --- Amplitude trust: ã_k = a_k · σ(w_k − λ·δ̄_d) ---
        delta_bar = delta_d.mean(dim=-1, keepdim=True).unsqueeze(1)  # (B,1,1)
        wk        = self.w_k.unsqueeze(0).unsqueeze(-1)               # (1,K,1)
        trust     = torch.sigmoid(wk - self.lam * delta_bar)          # (B,K,1)
        trusted   = top_vals * trust                                   # (B,K,F)

        out = self.freq_proj(trusted.reshape(B, -1))                   # (B,F)
        return out, delta_p   # ← now returns BOTH