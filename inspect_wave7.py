import json, numpy as np

tasks = [
    '8d5021e8', '8e5a5113', '60b61512', '694f12f3', '88a10436',
    '8f2ea7aa', '8403a5d5', '868de0fa', '539a4f51', '234bbc79'
]

for tid in tasks:
    with open(f'Atelier_ARC_Prize/training/{tid}.json') as f:
        d = json.load(f)
    print(f"================ {tid} ================")
    tr = d['train']
    te = d['test']
    print(f"Train count: {len(tr)}, Test count: {len(te)}")
    for i, p in enumerate(tr[:2]):
        inp = np.array(p['input'])
        out = np.array(p['output'])
        print(f"  Train {i}: in {inp.shape} -> out {out.shape}, uniq in {np.unique(inp)}, uniq out {np.unique(out)}")
        print("  In:\n", inp)
        print("  Out:\n", out)
    for i, p in enumerate(te[:1]):
        inp = np.array(p['input'])
        out = np.array(p.get('output', []))
        print(f"  Test {i}: in {inp.shape} -> out {out.shape if len(out) else 'None'}, uniq in {np.unique(inp)}")
        print("  In:\n", inp)
        if len(out):
            print("  Out:\n", out)
