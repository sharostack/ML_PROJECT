# stage8_plots.py
import torch, numpy as np, matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import CMAPSSDataset
from drift_det import batch_ks_drift
from baseline import BaselineTransformer
from full_model import DriftAwarePatchTST

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
WINDOW, CLIP = 30, 125

val_ds  = CMAPSSDataset('CMAPSSData/train_FD001.txt', window=WINDOW, split='val')
val_dl  = DataLoader(val_ds, batch_size=256, shuffle=False)
ref_x,_ = next(iter(val_dl)); ref = ref_x[0]

def get_preds(model, loader, use_drift=False):
    model.eval(); preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            d = batch_ks_drift(x, ref).to(DEVICE) if use_drift else None
            yh = model(x, d)[0] if use_drift else model(x)
            preds.extend((yh.cpu()*CLIP).numpy())
            trues.extend((y*CLIP).numpy())
    return np.array(preds), np.array(trues)

# Load models
baseline = BaselineTransformer().to(DEVICE)
baseline.load_state_dict(torch.load('best_baseline.pt', map_location=DEVICE))
full     = DriftAwarePatchTST(window=WINDOW).to(DEVICE)
full.load_state_dict(torch.load('best_full.pt', map_location=DEVICE))

p_base, t = get_preds(baseline, val_dl, use_drift=False)
p_full, _ = get_preds(full,     val_dl, use_drift=True)

# --- Plot 1: Predicted vs True RUL ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, preds, title in zip(axes, [p_base, p_full], ['Baseline PatchTST', 'Drift-Aware Model']):
    ax.scatter(t, preds, alpha=0.3, s=5, c='steelblue')
    ax.plot([0,CLIP],[0,CLIP],'r--', label='Perfect')
    mae  = np.mean(np.abs(preds-t))
    rmse = np.sqrt(np.mean((preds-t)**2))
    ax.set_title(f'{title}\nMAE={mae:.1f}  RMSE={rmse:.1f}')
    ax.set_xlabel('True RUL'); ax.set_ylabel('Predicted RUL')
    ax.legend()
plt.tight_layout(); plt.savefig('rul_scatter.png', dpi=150); plt.show()

# --- Plot 2: Error distribution ---
fig, ax = plt.subplots(figsize=(8,4))
ax.hist(p_base - t, bins=50, alpha=0.6, label='Baseline', color='steelblue')
ax.hist(p_full - t, bins=50, alpha=0.6, label='Drift-Aware', color='tomato')
ax.axvline(0, color='k', ls='--')
ax.set_xlabel('Prediction Error (cycles)'); ax.set_ylabel('Count')
ax.set_title('Error Distribution'); ax.legend()
plt.tight_layout(); plt.savefig('error_dist.png', dpi=150); plt.show()

# --- Plot 3: KS drift over time for one engine ---
from stage1_load_and_explore import load_cmapss
from scipy.stats import ks_2samp
SENSOR_COLS_USE = [f'sensor{i}' for i in [2,3,4,7,8,9,11,12,13,14,15,17,20,21]]
df = load_cmapss()
eng = df[df['engine_id']==1][SENSOR_COLS_USE].values
ref_w = eng[:WINDOW]
ks_over_time = []
for t_idx in range(WINDOW, len(eng)):
    win = eng[t_idx-WINDOW:t_idx]
    stat = np.mean([ks_2samp(ref_w[:,f], win[:,f])[0] for f in range(len(SENSOR_COLS_USE))])
    ks_over_time.append(stat)

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(ks_over_time, color='darkorange')
ax.set_xlabel('Time step'); ax.set_ylabel('Mean KS statistic')
ax.set_title('Distribution Drift (δ̄_d) over Engine Lifecycle — Engine 1')
ax.axhline(0.1, ls='--', c='red', label='Drift threshold')
ax.legend(); plt.tight_layout()
plt.savefig('ks_drift_over_time.png', dpi=150); plt.show()