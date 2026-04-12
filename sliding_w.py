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
                 scaler=None):
        COLS = ['engine_id','cycle','setting1','setting2','setting3'] + \
               [f'sensor{i}' for i in range(1,22)]
        df = pd.read_csv(path, sep='\s+', header=None, names=COLS)
        df = add_rul(df)
        df['RUL'] = df['RUL'].clip(upper=clip_rul)

        self.windows, self.labels = [], []
        for eid, grp in df.groupby('engine_id'):
            vals = grp[USE_SENSORS].values
            rul  = grp['RUL'].values
            #creating sliidng windows
            for t in range(window, len(vals)):
                self.windows.append(vals[t-window:t])
                self.labels.append(rul[t])

        self.windows = np.array(self.windows, dtype=np.float32)
        #normalize labels - rul b/w 0 and 1
        self.labels  = np.array(self.labels,  dtype=np.float32) / clip_rul

        # Split FIRST, then fit scaler only on train
        n = len(self.windows)
        #train validation split
        split_idx = int(n * (1 - val_frac))

        if split == 'train':
            #train
            self.windows = self.windows[:split_idx]
            self.labels  = self.labels[:split_idx]
            # Fit scaler on train windows only
            flat = self.windows.reshape(-1, self.windows.shape[-1])
            self.scaler = MinMaxScaler().fit(flat)
        else:
            #validation
            self.windows = self.windows[split_idx:]
            self.labels  = self.labels[split_idx:]
            assert scaler is not None, "Pass train scaler to val dataset"
            self.scaler = scaler

        # Transform
        B, W, F = self.windows.shape
        flat = self.windows.reshape(-1, F)
        self.windows = self.scaler.transform(flat).reshape(B, W, F).astype(np.float32)

    def __len__(self): return len(self.windows)
    def __getitem__(self, i):
        return torch.tensor(self.windows[i]), torch.tensor(self.labels[i])
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
    x, y = next(iter(dl))
    print(f"Batch X: {x.shape}, Y: {y.shape}")