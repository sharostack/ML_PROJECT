"""
stage7_train.py  —  fixes applied
  1. Reference window: streaming mean (no full RAM load)
  2. Val split: engine-ID-based from train_FD001.txt (correct — test file
     is NEVER touched here; that belongs in test.py only)
  3. MLflow: log_artifact + set_tag added
  4. CLIP_RUL applied once, in one place, documented clearly
"""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import mlflow
import mlflow.pytorch

from dataset    import CMAPSSDataset
from drift_det  import batch_ks_drift
from baseline   import BaselineTransformer
from full_model import DriftAwarePatchTST

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'
WINDOW   = 30
EPOCHS   = 40
BATCH    = 64
SEED     = 42
PATIENCE = 10

# CLIP_RUL: applied ONCE — inside CMAPSSDataset when building labels.
# eval_epoch reads raw model output (already in RUL space) — no re-scaling here.
# If your dataset returns normalised [0,1] labels, set CLIP_RUL in dataset only.
CLIP_RUL = 125

BASE_DIR   = Path.home() / 'Downloads' / 'projects' / 'archive'
TRAIN_PATH = BASE_DIR / 'train_FD001.txt'
# test_FD001.txt is intentionally NOT used here — only in test.py
CKPT_DIR   = Path('checkpoints')
CKPT_DIR.mkdir(exist_ok=True)

VAL_ENGINE_FRACTION = 0.2   # last 20% of engine IDs → val, rest → train


# ─────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ─────────────────────────────────────────────
# NASA ASYMMETRIC SCORE  (primary CMAPSS metric)
# Late predictions penalised ~10x more than early
# ─────────────────────────────────────────────
def nasa_score(preds: np.ndarray, trues: np.ndarray) -> float:
    diff = preds - trues
    score = np.where(diff < 0,
                     np.exp(-diff / 13) - 1,
                     np.exp( diff / 10) - 1)
    return float(score.sum())


# ─────────────────────────────────────────────
# REFERENCE WINDOW  (streaming — no full RAM load)
# Fix 1: iterate loader in one pass, accumulate sum, divide by N
# Memory cost: O(W x F) instead of O(N x W x F)
# ─────────────────────────────────────────────
def compute_ref(train_dl) -> torch.Tensor:
    """Streaming mean over all train windows. Shape: (W, F)."""
    running_sum = None
    n_total = 0
    with torch.no_grad():
        for batch in train_dl:
            x = batch[0]
            if running_sum is None:
                running_sum = x.sum(dim=0)
            else:
                running_sum += x.sum(dim=0)
            n_total += x.size(0)
    return running_sum / n_total


# ─────────────────────────────────────────────
# DATA LOADERS
# Val = last VAL_ENGINE_FRACTION of engine IDs from train file.
# This is the correct split — never use test_FD001.txt here.
# Scaler fitted on train engines only (no leakage into val).
# ─────────────────────────────────────────────
def get_loaders():
    train_ds = CMAPSSDataset(
        str(TRAIN_PATH),
        window=WINDOW,
        split='train',
        val_engine_fraction=VAL_ENGINE_FRACTION,
    )
    val_ds = CMAPSSDataset(
        str(TRAIN_PATH),
        window=WINDOW,
        split='val',
        val_engine_fraction=VAL_ENGINE_FRACTION,
        scaler=train_ds.scaler,
    )

    # Keep temporal order so ΔP uses true t -> t-1 progression per engine.
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=False, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)

    ref_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=False, num_workers=0)
    ref    = compute_ref(ref_dl)

    return train_dl, val_dl, ref


# ─────────────────────────────────────────────
# TRAIN EPOCH
# ─────────────────────────────────────────────
def train_epoch(model, loader, opt, criterion, ref,
                use_drift=False, ablation=None):
    model.train()
    total_loss = 0.0

    for batch in loader:
        x, y = batch[0], batch[1]
        engine_ids = batch[2] if len(batch) > 2 else None
        x, y = x.to(DEVICE), y.to(DEVICE)
        if engine_ids is not None:
            engine_ids = engine_ids.to(DEVICE)
        opt.zero_grad()

        if use_drift:
            delta_d = batch_ks_drift(x, ref).to(DEVICE)
            y_hat, _ = model(x, delta_d, ablation=ablation, engine_ids=engine_ids)
        else:
            y_hat = model(x)

        loss = criterion(y_hat, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        total_loss += loss.item()

    return total_loss / len(loader)


# ─────────────────────────────────────────────
# EVAL EPOCH
# NOTE: model outputs are already in RUL space — dataset applies CLIP_RUL
# when building labels. We do NOT multiply again here.
# Pick ONE place for CLIP_RUL: either dataset (recommended) or here, never both.
# ─────────────────────────────────────────────
def _reset_periodic_eval_state(model):
    """Avoid EMA / prev_amp carrying across batches during eval (full model only)."""
    mod = getattr(model, 'periodic_pyramid', None) or getattr(model, 'pyramid', None)
    if mod is not None and hasattr(mod, 'reset_state'):
        mod.reset_state()


# ─────────────────────────────────────────────
@torch.no_grad()
def eval_epoch(model, loader, ref, use_drift=False, ablation=None):
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


# ─────────────────────────────────────────────
# VARIANT CONFIG
# ─────────────────────────────────────────────
VARIANT_CFG = {
    'baseline':      ('baseline', False, None),
    'full':          ('full',     True,  None),
    'no_D':          ('full',     True,  'no_D'),
    'no_dP':         ('full',     True,  'no_dP'),
    'periodic_only': ('full',     True,  'periodic_only'),
}

def build_model(model_type):
    if model_type == 'baseline':
        return BaselineTransformer().to(DEVICE)
    return DriftAwarePatchTST(window=WINDOW).to(DEVICE)


# ─────────────────────────────────────────────
# RUN — single variant
# ─────────────────────────────────────────────
def run(variant_name):
    model_type, use_drift, ablation = VARIANT_CFG[variant_name]
    set_seed()

    train_dl, val_dl, ref = get_loaders()
    model     = build_model(model_type)
    criterion = nn.MSELoss()
    opt       = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_mae   = float('inf')
    best_ckpt  = CKPT_DIR / f'best_{variant_name}.pt'
    no_improve = 0

    with mlflow.start_run(run_name=variant_name):
        mlflow.set_tag('model_type', variant_name)          # Fix 3
        mlflow.log_params({
            'variant': variant_name, 'ablation': str(ablation),
            'use_drift': use_drift, 'epochs': EPOCHS,
            'batch': BATCH, 'window': WINDOW, 'seed': SEED,
            'val_engine_fraction': VAL_ENGINE_FRACTION,
            'clip_rul': CLIP_RUL,
        })

        for ep in range(1, EPOCHS + 1):
            loss = train_epoch(model, train_dl, opt, criterion, ref,
                               use_drift=use_drift, ablation=ablation)
            sched.step()
            mlflow.log_metric('train_loss', loss, step=ep)

            if ep % 2 == 0:
                mae, rmse, nasa = eval_epoch(model, val_dl, ref,
                                             use_drift=use_drift, ablation=ablation)
                mlflow.log_metrics(
                    {'val_mae': mae, 'val_rmse': rmse, 'val_nasa': nasa}, step=ep
                )

                if mae < best_mae:
                    best_mae   = mae
                    no_improve = 0
                    torch.save({'epoch': ep, 'state_dict': model.state_dict(),
                                'best_mae': best_mae}, best_ckpt)
                else:
                    no_improve += 1

                print(f"Ep {ep} | Loss {loss:.4f} | MAE {mae:.2f} | RMSE {rmse:.2f} | NASA {nasa:.1f}")

                if no_improve >= PATIENCE:
                    break

        ckpt = torch.load(best_ckpt, map_location=DEVICE)
        model.load_state_dict(ckpt['state_dict'])
        mae, rmse, nasa = eval_epoch(model, val_dl, ref,
                                     use_drift=use_drift, ablation=ablation)

        mlflow.log_metrics({'best_val_mae': mae, 'best_val_rmse': rmse,
                            'best_val_nasa': nasa})
        mlflow.log_artifact(str(best_ckpt))                 # Fix 3
        mlflow.pytorch.log_model(model, f'model_{variant_name}')

    return mae, rmse, nasa


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    mlflow.set_experiment('RULPRED_CMAPSS_FD001')
    results = {}

    for variant in VARIANT_CFG:
        results[variant] = run(variant)

    print(f"\n{'='*62}")
    print(f"  {'Model':<22} {'MAE':>8} {'RMSE':>8} {'NASA':>10}")
    print(f"  {'-'*52}")
    for name, (mae, rmse, nasa) in results.items():
        print(f"  {name:<22} {mae:>8.2f} {rmse:>8.2f} {nasa:>10.1f}")
    print(f"{'='*62}")