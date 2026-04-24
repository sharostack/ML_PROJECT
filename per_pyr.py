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

        # Stores previous per-engine amplitudes for true temporal ΔP.
        self.prev_amp = None
        self.prev_amp_by_engine = {}

    def reset_state(self, engine_ids=None):
        """Clear temporal memory globally or for specific engine IDs."""
        if engine_ids is None:
            self.prev_amp = None
            self.prev_amp_by_engine.clear()
            return
        for eid in engine_ids:
            self.prev_amp_by_engine.pop(int(eid), None)

    def forward(self, x: torch.Tensor, delta_d: torch.Tensor, engine_ids=None):
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

        # Current periodic signature per sample: (B, F)
        curr_amp = top_vals.mean(dim=1)

        # --- Temporal ΔP: compare current vs previous for same engine/sample stream ---
        delta_p = torch.zeros_like(curr_amp)
        if engine_ids is None:
            if self.prev_amp is None:
                delta_p = torch.zeros_like(curr_amp)
            else:
                delta_p = (curr_amp - self.prev_amp).abs()
            self.prev_amp = curr_amp.detach()
        else:
            if torch.is_tensor(engine_ids):
                engine_ids = engine_ids.detach().cpu().tolist()
            for b, eid in enumerate(engine_ids):
                eid = int(eid)
                prev = self.prev_amp_by_engine.get(eid)
                if prev is None:
                    delta_p[b] = 0.0
                else:
                    delta_p[b] = (curr_amp[b] - prev).abs()
                self.prev_amp_by_engine[eid] = curr_amp[b].detach()

        # --- Amplitude trust: ã_k = a_k · σ(w_k − λ·δ̄_d) ---
        delta_bar = delta_d.mean(dim=-1, keepdim=True).unsqueeze(1)  # (B,1,1)
        wk        = self.w_k.unsqueeze(0).unsqueeze(-1)               # (1,K,1)
        trust     = torch.sigmoid(wk - self.lam * delta_bar)          # (B,K,1)
        trusted   = top_vals * trust                                   # (B,K,F)

        out = self.freq_proj(trusted.reshape(B, -1))                   # (B,F)
        return out, delta_p   # ← now returns BOTH