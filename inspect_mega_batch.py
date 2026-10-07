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
        if inp.size <= 50 and out.size <= 50:
            print(f"In:\n{inp}\nOut:\n{out}")
        elif out.size <= 30:
            print(f"In uq: {np.unique(inp)}, Out:\n{out}")
        else:
            diff = (inp != out) if inp.shape == out.shape else None
            d_str = f"Diff: {np.sum(diff)}" if diff is not None else ""
            print(f"In uq: {np.unique(inp)}, Out uq: {np.unique(out)} {d_str}")

targets = [
    'a740d043', 'd10ecb37', 'ff28f65a', 'cf98881b', 'a699fb00',
    'a9f96cdd', 'f25ffba3', 'd406998b', 'a3df8b1e', 'a85d4709'
]

for tid in targets:
    examine(tid)
