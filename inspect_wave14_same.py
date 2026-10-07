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
        diff = (inp != out)
        print(f"Diff count: {np.sum(diff)}")
        print(f"Diff vals in: {inp[diff]}, out: {out[diff]}")

for tid in ['d23f8c26', 'd43fd935', 'd89b689b', 'ea786f4a']:
    examine(tid)
