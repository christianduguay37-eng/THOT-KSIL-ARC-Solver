import json
import numpy as np

targets = ['2dd70a9a', '3631a71a', '36d67576', '36fdfd69', '3e980e27']

for tid in targets:
    with open(f'training/{tid}.json') as f:
        d = json.load(f)
    print(f'=== TASK {tid} ===')
    print(f'Train: {len(d["train"])}, Test: {len(d["test"])}')
    for i, p in enumerate(d['train']):
        inp = np.array(p['input'])
        out = np.array(p['output'])
        print(f'  Train {i}: in {inp.shape} -> out {out.shape} | in {np.unique(inp).tolist()} -> out {np.unique(out).tolist()}')
    for i, p in enumerate(d['test']):
        inp = np.array(p['input'])
        out = np.array(p['output']) if 'output' in p else None
        print(f'  Test {i}: in {inp.shape} -> out {out.shape if out is not None else "?"} | in {np.unique(inp).tolist()}')
    print()
