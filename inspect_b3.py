import json
import numpy as np

tasks = ['97a05b5b', '98cf29f8', '9aec4887', '9d9215db', '9edfc990']
for tid in tasks:
    with open(f'training/{tid}.json') as f:
        d = json.load(f)
    print(f'=== TASK {tid} ===')
    print(f"Train pairs: {len(d['train'])}, Test pairs: {len(d['test'])}")
    for i, p in enumerate(d['train']):
        inp = np.array(p['input'])
        out = np.array(p['output'])
        print(f"Train {i}: in shape {inp.shape}, out shape {out.shape}, in cols {np.unique(inp).tolist()}, out cols {np.unique(out).tolist()}")
    for i, p in enumerate(d['test']):
        inp = np.array(p['input'])
        out = np.array(p['output'])
        print(f"Test {i}: in shape {inp.shape}, out shape {out.shape}, in cols {np.unique(inp).tolist()}, out cols {np.unique(out).tolist()}")
