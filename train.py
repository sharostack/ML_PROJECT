# stage7_train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from stage2_dataset import CMAPSSDataset
from stage3_drift import batch_ks_drift
from stage5_baseline import BaselineTransformer
from stage6_full_model import DriftAwarePatchTST
import numpy as np

DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
WINDOW     = 30
EPOCHS     = 40
BATCH      = 64
CLIP_RUL   = 125

# ─────────────────────────────────────────────
# 1. DATA LOADERS  (scaler fitted on train only)
# ─────────────────────────────────────────────
def get_loaders():
    train_ds = CMAPSSDataset(
        'CMAPSSData/train_FD001.txt', window=WINDOW, split='train'
    )
    val_ds = CMAPSSDataset(
        'CMAPSSData/train_FD001.txt', window=WINDOW, split='val',
        scaler=train_ds.scaler          # ← pass fitted scaler, no leakage
    )
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)

    # Reference window = mean over first batch (stable, not a single noisy sample)
    ref_x, _ = next(iter(train_dl))
    ref = ref_x.mean(dim=0)             # (W, F)
    return train_dl, val_dl, ref


# ─────────────────────────────────────────────
# 2. TRAIN EPOCH
# ─────────────────────────────────────────────
def train_epoch(model, loader, opt, criterion, ref,
                use_drift=False, ablation=None):
    model.train()
    total_loss = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()

        if use_drift:
            delta_d = batch_ks_drift(x, ref).to(DEVICE)   # (B, F)
            y_hat, _ = model(x, delta_d, ablation=ablation)
        else:
            y_hat = model(x)                               # baseline

        loss = criterion(y_hat, y)
        loss.backward()
        opt.step()
        total_loss += loss.item()

    return total_loss / len(loader)


# ─────────────────────────────────────────────
# 3. EVAL EPOCH
# ─────────────────────────────────────────────
@torch.no_grad()
def eval_epoch(model, loader, ref,
               use_drift=False, ablation=None, clip_rul=125):
    model.eval()
    preds, trues = [], []

    for x, y in loader:
        x = x.to(DEVICE)

        if use_drift:
            delta_d = batch_ks_drift(x, ref).to(DEVICE)
            y_hat, _ = model(x, delta_d, ablation=ablation)
        else:
            y_hat = model(x)

        preds.extend((y_hat.cpu() * clip_rul).numpy())
        trues.extend((y           * clip_rul).numpy())

    preds = np.array(preds)
    trues = np.array(trues)
    mae   = np.mean(np.abs(preds - trues))
    rmse  = np.sqrt(np.mean((preds - trues) ** 2))
    return mae, rmse


# ─────────────────────────────────────────────
# 4. RUN — single variant
# ─────────────────────────────────────────────
VARIANT_CFG = {
    # name           model_type   use_drift  ablation
    'baseline':      ('baseline', False,     None),
    'full':          ('full',     True,      None),
    'no_D':          ('full',     True,      'no_D'),
    'no_dP':         ('full',     True,      'no_dP'),
    'periodic_only': ('full',     True,      'periodic_only'),
}

def build_model(model_type):
    if model_type == 'baseline':
        return BaselineTransformer().to(DEVICE)
    return DriftAwarePatchTST(window=WINDOW).to(DEVICE)


def run(variant_name):
    model_type, use_drift, ablation = VARIANT_CFG[variant_name]

    train_dl, val_dl, ref = get_loaders()
    model     = build_model(model_type)
    criterion = nn.MSELoss()
    opt       = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    print(f"\n{'='*55}")
    print(f"  Training : {variant_name}  |  ablation={ablation}")
    print(f"{'='*55}")

    best_mae   = float('inf')
    best_ckpt  = f'best_{variant_name}.pt'

    for ep in range(1, EPOCHS + 1):
        loss = train_epoch(
            model, train_dl, opt, criterion, ref,
            use_drift=use_drift, ablation=ablation
        )
        sched.step()

        if ep % 5 == 0:
            mae, rmse = eval_epoch(
                model, val_dl, ref,
                use_drift=use_drift, ablation=ablation
            )
            flag = ''
            if mae < best_mae:
                best_mae = mae
                torch.save(model.state_dict(), best_ckpt)
                flag = '  ← best'
            print(f"  Ep {ep:3d} | Loss {loss:.4f} | "
                  f"MAE {mae:.2f} | RMSE {rmse:.2f}{flag}")

    # Final eval on best checkpoint
    model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
    mae, rmse = eval_epoch(
        model, val_dl, ref,
        use_drift=use_drift, ablation=ablation
    )
    print(f"\n  [{variant_name}]  Best MAE: {mae:.2f}  |  RMSE: {rmse:.2f}")
    return mae, rmse


# ─────────────────────────────────────────────
# 5. MAIN — all 5 variants + summary table
# ─────────────────────────────────────────────
if __name__ == '__main__':
    results = {}
    for variant in VARIANT_CFG:
        results[variant] = run(variant)

    print(f"\n{'='*55}")
    print(f"  {'Model':<22} {'MAE':>8} {'RMSE':>8}")
    print(f"  {'-'*38}")
    for name, (mae, rmse) in results.items():
        print(f"  {name:<22} {mae:>8.2f} {rmse:>8.2f}")
    print(f"{'='*55}")