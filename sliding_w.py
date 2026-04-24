# stage2_dataset.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

#only keep useful sensors
SENSOR_COLS = [f'sensor{i}' for i in range(1, 22)]
# Drop flat/useless sensors (FD001 known)
DROP = ['sensor1','sensor5','sensor6','sensor10','sensor16','sensor18','sensor19']
USE_SENSORS = [s for s in SENSOR_COLS if s not in DROP]  # 14 sensors

#how many cycles left b4 failure
def add_rul(df):
    mc = df.groupby('engine_id')['cycle'].max().reset_index()
    mc.columns = ['engine_id','max_cycle']
    df = df.merge(mc, on='engine_id')
    df['RUL'] = df['max_cycle'] - df['cycle']
    return df.drop(columns='max_cycle')
#koad dataset
class CMAPSSDataset(Dataset):
    def __init__(self, path, window=30, clip_rul=125, split='train', val_frac=0.15,
                 val_engine_fraction=None, scaler=None):
        # Backward-compatible alias: some training scripts pass `val_engine_fraction`.
        if val_engine_fraction is not None:
            val_frac = val_engine_fraction

        COLS = ['engine_id','cycle','setting1','setting2','setting3'] + \
               [f'sensor{i}' for i in range(1,22)]
        df = pd.read_csv(path, sep=r'\s+', header=None, names=COLS)
        if split == 'test':
            rul_file = str(path).replace('test_FD001', 'RUL_FD001')
            true_rul = pd.read_csv(rul_file, header=None).values.flatten()

            self.windows, self.labels, self.engine_ids = [], [], []

            engine_ids = sorted(df['engine_id'].unique())
            assert len(engine_ids) == len(true_rul), \
            "Mismatch between test engines and RUL labels"

            for i, eid in enumerate(engine_ids):
                grp = df[df['engine_id'] == eid].sort_values('cycle')
                vals = grp[USE_SENSORS].values
                last_window = vals[-window:]
                self.windows.append(last_window)
                self.engine_ids.append(int(eid))

                rul_val = min(true_rul[i], clip_rul)
                self.labels.append(rul_val / clip_rul)

            self.windows = np.array(self.windows, dtype=np.float32)
            self.labels  = np.array(self.labels, dtype=np.float32)

            assert scaler is not None
            self.scaler = scaler
        else:
            # Engine-level split (prevents the same engine appearing in both sets).
            all_engines = df['engine_id'].unique()
            n_val = int(len(all_engines) * val_frac)

            val_engines = set(all_engines[-n_val:]) if n_val > 0 else set()
            train_engines = set(all_engines[:-n_val]) if n_val > 0 else set(all_engines)

            # Safety: enforce no overlap of engine IDs across splits.
            assert len(train_engines & val_engines) == 0, "Engine overlap between train and val"

            split_engines = train_engines if split == 'train' else val_engines
            if split == 'val' and len(split_engines) == 0:
                raise ValueError(f"No validation engines created (val_frac={val_frac}).")

            df_split = df[df['engine_id'].isin(split_engines)].copy()
            df_split = add_rul(df_split)
            df_split['RUL'] = df_split['RUL'].clip(upper=clip_rul)

            # Build windows/labels AFTER splitting.
            self.windows, self.labels, self.engine_ids = [], [], []
            for eid, grp in df_split.groupby('engine_id'):
                grp = grp.sort_values('cycle')
                vals = grp[USE_SENSORS].values
                rul  = grp['RUL'].values

                # Create sliding windows for this engine.
                for t in range(window, len(vals)):
                    self.windows.append(vals[t-window:t])
                    self.labels.append(rul[t-1])
                    self.engine_ids.append(int(eid))

            self.windows = np.array(self.windows, dtype=np.float32)

            # Normalize labels to [0, 1] using CLIP_RUL (only once).
            self.labels  = np.array(self.labels, dtype=np.float32) / clip_rul

            # Fit scaler ONLY on train-window data.
            if split == 'train':
                flat = self.windows.reshape(-1, self.windows.shape[-1])
                self.scaler = MinMaxScaler().fit(flat)
            else:
                assert scaler is not None, "Pass train scaler to val dataset"
                self.scaler = scaler

        # Transform
        B, W, F = self.windows.shape
        flat = self.windows.reshape(-1, F)
        self.windows = self.scaler.transform(flat).reshape(B, W, F).astype(np.float32)

    def __len__(self): return len(self.windows)
    def __getitem__(self, i):
        return (
            torch.tensor(self.windows[i]),
            torch.tensor(self.labels[i]),
            torch.tensor(self.engine_ids[i], dtype=torch.long),
        )
def get_loaders(window=30, batch_size=64):
    train_ds = CMAPSSDataset('/Users/sharo2/Downloads/projects/archive/train_FD001.txt', window=window, split='train')
    val_ds   = CMAPSSDataset('/Users/sharo2/Downloads/projects/archive/train_FD001.txt', window=window, split='val',
                            scaler=train_ds.scaler)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size)

    return train_loader, val_loader
    
# Quick sanity check
if __name__ == '__main__':
    ds = CMAPSSDataset('/Users/sharo2/Downloads/projects/archive/train_FD001.txt', window=30)
    print(f"Train samples: {len(ds)}, window shape: {ds[0][0].shape}")
    dl = DataLoader(ds, batch_size=64, shuffle=True)
    batch = next(iter(dl))
    x, y = batch[0], batch[1]
    print(f"Batch X: {x.shape}, Y: {y.shape}")