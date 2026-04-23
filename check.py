import pandas as pd

DATA_DIR = "/Users/sharo2/Downloads/projects/archive"

# Load test data
COLS = ['engine_id','cycle','setting1','setting2','setting3'] + \
       [f'sensor{i}' for i in range(1,22)]

test_df = pd.read_csv(f"{DATA_DIR}/test_FD001.txt",
                      sep=r'\s+', header=None, names=COLS)

# Load RUL file
rul = pd.read_csv(f"{DATA_DIR}/RUL_FD001.txt", header=None).values.flatten()

# Extract engine IDs
engine_ids = sorted(test_df['engine_id'].unique())

print("Num engines (test):", len(engine_ids))
print("Num RUL entries   :", len(rul))

# Check alignment
assert len(engine_ids) == len(rul), "❌ Mismatch in engine count!"

print("\nFirst 5 engines vs RUL:")
for i in range(5):
    print(f"Engine {engine_ids[i]} → RUL {rul[i]}")

print("\nLast 5 engines vs RUL:")
for i in range(-5, 0):
    print(f"Engine {engine_ids[i]} → RUL {rul[i]}")