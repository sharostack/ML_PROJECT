# stage3_drift.py
import torch
import numpy as np
from scipy.stats import ks_2samp

class KSDriftDetector:
    """
    Sliding-window KS drift detector.
    Returns δ_d ∈ R^F — per-sensor KS statistic vector.
    """
    def __init__(self, ref_window: torch.Tensor):
        # ref_window: (W, F) reference distribution
        #store ref data and no. of sensors
        self.ref = ref_window.cpu().numpy()  # (W, F)
        self.F = self.ref.shape[1]

    def compute(self, current_window: torch.Tensor) -> torch.Tensor:
        """current_window: (W, F) → returns δ_d: (F,)"""
        #current data
        cur = current_window.cpu().numpy()
        stats = np.zeros(self.F)
        for f in range(self.F):
            #past vs current
            stat, _ = ks_2samp(self.ref[:, f], cur[:, f])
            stats[f] = stat
        return torch.tensor(stats, dtype=torch.float32)

    def update_ref(self, new_window: torch.Tensor):
        self.ref = new_window.cpu().numpy()

#model trains in batches so need to drift for all 64 samples at once
def batch_ks_drift(x_batch: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Vectorised KS over a batch.
    x_batch: (B, W, F), ref: (W, F) → δ_d: (B, F)
    """
    B, W, F = x_batch.shape
    deltas = torch.zeros(B, F)
    ref_np = ref.cpu().numpy()
    for b in range(B):
        for f in range(F):
            s, _ = ks_2samp(ref_np[:, f], x_batch[b, :, f].cpu().numpy())
            deltas[b, f] = s
    return deltas  # (B, F)