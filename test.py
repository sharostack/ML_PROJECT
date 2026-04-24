"""
Evaluates trained models on CMAPSS test_FD001 dataset

MATCHES TRAIN PIPELINE:
- streaming-style reference (same logic as training)
- NO double CLIP_RUL
- identical drift computation path
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from dataset import CMAPSSDataset
from drift_det import batch_ks_drift
from baseline import BaselineTransformer
from full_model import DriftAwarePatchTST

# ─────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
WINDOW = 30
BATCH  = 64

DATA_DIR  = Path.home() / 'Downloads' / 'projects' / 'archive'
TRAIN_PATH = DATA_DIR / 'train_FD001.txt'
TEST_PATH  = DATA_DIR / 'test_FD001.txt'
CKPT_DIR   = Path('checkpoints')
CLIP_RUL   = 125


# ─────────────────────────────
def nasa_score(preds, trues):
    diff = preds - trues
    score = np.where(diff < 0,
                     np.exp(-diff / 13) - 1,
                     np.exp(diff / 10) - 1)
    return float(score.sum())


# ─────────────────────────────
# SAME ref logic as train (IMPORTANT FIX)
# ─────────────────────────────
def compute_ref(train_loader):
    running_sum = None
    n = 0

    with torch.no_grad():
        for batch in train_loader:
            x = batch[0]
            if running_sum is None:
                running_sum = x.sum(dim=0)
            else:
                running_sum += x.sum(dim=0)
            n += x.size(0)

    return running_sum / n


# ─────────────────────────────
def _reset_periodic_eval_state(model):
    mod = getattr(model, 'periodic_pyramid', None) or getattr(model, 'pyramid', None)
    if mod is not None and hasattr(mod, 'reset_state'):
        mod.reset_state()


# ─────────────────────────────
def get_loaders():
    train_ds = CMAPSSDataset(str(TRAIN_PATH), window=WINDOW, split='train')
    test_ds = CMAPSSDataset(
    str(TEST_PATH),
    window=WINDOW,
    split='test',
    scaler=train_ds.scaler
)

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=False)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False)

    ref_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=False)
    ref    = compute_ref(ref_dl)

    return test_dl, ref


# ─────────────────────────────
def build_model(name):
    if name == 'baseline':
        return BaselineTransformer().to(DEVICE)
    return DriftAwarePatchTST(window=WINDOW).to(DEVICE)


# ─────────────────────────────
@torch.no_grad()
def evaluate(model, loader, ref, use_drift=False, ablation=None):
    model.eval()
    preds, trues = [], []
    _reset_periodic_eval_state(model)

    for batch in loader:
        x, y = batch[0], batch[1]
        engine_ids = batch[2] if len(batch) > 2 else None
        x = x.to(DEVICE)
        if engine_ids is not None:
            engine_ids = engine_ids.to(DEVICE)

        if use_drift:
            delta_d = batch_ks_drift(x, ref).to(DEVICE)
            y_hat, _ = model(x, delta_d, ablation=ablation, engine_ids=engine_ids)
        else:
            y_hat = model(x)

        preds.extend(y_hat.cpu().numpy())
        trues.extend(y.numpy())

    preds = np.array(preds)
    trues = np.array(trues)

    preds_real = preds * CLIP_RUL
    trues_real = trues * CLIP_RUL

    mae  = float(np.mean(np.abs(preds_real - trues_real)))
    rmse = float(np.sqrt(np.mean((preds_real - trues_real) ** 2)))
    nasa = nasa_score(preds_real, trues_real)
    return mae, rmse, nasa


# ─────────────────────────────
VARIANTS = {
    'baseline':      ('baseline', False, None),
    'full':          ('full', True, None),
    'no_D':          ('full', True, 'no_D'),
    'no_dP':         ('full', True, 'no_dP'),
    'periodic_only': ('full', True, 'periodic_only'),
}


# ─────────────────────────────
if __name__ == '__main__':

    test_dl, ref = get_loaders()

    print("\n" + "="*60)
    print(" FINAL TEST RESULTS (aligned with training)")
    print("="*60)

    results = {}

    for name, (model_type, use_drift, ablation) in VARIANTS.items():

        print(f"\nEvaluating: {name}")

        model = build_model(model_type)

        ckpt_path = CKPT_DIR / f'best_{name}.pt'
        ckpt = torch.load(ckpt_path, map_location=DEVICE)

        model.load_state_dict(ckpt['state_dict'])

        mae, rmse, nasa = evaluate(
            model, test_dl, ref,
            use_drift=use_drift,
            ablation=ablation
        )

        results[name] = (mae, rmse, nasa)

        print(f"MAE : {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"NASA: {nasa:.1f}")

    print("\n" + "="*60)
    print(f"{'Model':<20} {'MAE':>8} {'RMSE':>8} {'NASA':>10}")
    print("-"*60)

    for name, (mae, rmse, nasa) in results.items():
        print(f"{name:<20} {mae:>8.2f} {rmse:>8.2f} {nasa:>10.1f}")

    print("="*60)