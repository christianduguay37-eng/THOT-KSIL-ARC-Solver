import json
import numpy as np
from pathlib import Path

train_dir = Path('Atelier_ARC_Prize/training')

def examine(tid):
    with open(train_dir / f'{tid}.json') as f:
        data = json.load(f)
    print(f"\n=================== TASK {tid} ===================")
    for i, p in enumerate(data['train']):
        inp = np.array(p['input'])
        out = np.array(p['output'])
        print(f"--- Train {i} (In: {inp.shape}, Out: {out.shape}) ---")
        print(f"In uq: {np.unique(inp)}, Out:\n{out}")

for tid in ['f8b3ba0a', 'f8ff0b80', '780d0b14', '9ecd008a']:
    examine(tid)
