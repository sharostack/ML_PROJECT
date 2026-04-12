# stage1_load_and_explore.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

#to label data
COLS = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + \
       [f'sensor{i}' for i in range(1, 22)]

#load dataset
def load_cmapss(path='/Users/sharo2/Downloads/projects/archive/train_FD001.txt'):
    df = pd.read_csv(path, sep='\s+', header=None, names=COLS)
    # Compute RUL - where each engine fails - last cycle
    max_cycle = df.groupby('engine_id')['cycle'].max().reset_index()
    max_cycle.columns = ['engine_id', 'max_cycle']
    df = df.merge(max_cycle, on='engine_id')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df.drop(columns='max_cycle', inplace=True)
    return df

df = load_cmapss()
#basic info
print(df.shape) 
print(df.head())

# --- Verify drift exists via KS test across early vs late cycles ---
SENSOR_COLS = [f'sensor{i}' for i in range(1, 22)]
#comparing early life vs late life
early = df[df['cycle'] <= 30][SENSOR_COLS]
late  = df[df['cycle'] >= df['cycle'].max() - 30][SENSOR_COLS]

print("\nKS test (early vs late cycles):")
for col in SENSOR_COLS:
    #KS TEST - aare dist diff?
    stat, p = ks_2samp(early[col], late[col])
    print(f"  {col}: stat={stat:.3f}, p={p:.4f} {'*** DRIFT' if p < 0.05 else ''}")

# --- Plot a few sensors for one engine ---
eng1 = df[df['engine_id'] == 1]
fig, axes = plt.subplots(3, 3, figsize=(14, 8))
for ax, col in zip(axes.flat, ['sensor2','sensor3','sensor4','sensor7',
                                'sensor8','sensor9','sensor11','sensor12','sensor15']):
    ax.plot(eng1['cycle'], eng1[col])
    ax.set_title(col); ax.set_xlabel('cycle')
plt.suptitle('Sensor Readings — Engine 1 (FD001)', fontsize=13)
plt.tight_layout(); plt.savefig('sensor_drift.png', dpi=150)
plt.show()