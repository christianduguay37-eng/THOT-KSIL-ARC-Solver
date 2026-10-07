import json
import re
import numpy as np
from pathlib import Path
from scipy.ndimage import label

train_dir = Path('Atelier_ARC_Prize/training')
with open('Atelier_ARC_Prize/palier_20_darwin.py', 'r', encoding='utf-8') as f:
    text = f.read()

canonical = set(re.findall(r'"([0-9a-f]{8})"', text))
all_tasks = sorted([p.stem for p in train_dir.glob('*.json')])
unsolved = [t for t in all_tasks if t not in canonical]
print(f'Unsolved count: {len(unsolved)}')

discovered_solutions = {}

def check_task(tid, fn):
    with open(train_dir / f'{tid}.json') as f:
        data = json.load(f)
    for p in data['train']:
        inp = np.array(p['input'])
        out = np.array(p['output'])
        pred = fn(inp)
        if pred is None or not np.array_equal(pred, out):
            return False
    for p in data['test']:
        inp = np.array(p['input'])
        out = np.array(p['output'])
        pred = fn(inp)
        if pred is None or not np.array_equal(pred, out):
            return False
    return True

# Let's inspect the shapes of all unsolved tasks to categorize them
categories = {
    'same_shape': [],
    'scale_integer': [],
    'kron_square': [],
    'subgrid_crop': [],
    '1d_or_small': [],
    'other': []
}

for tid in unsolved:
    with open(train_dir / f'{tid}.json') as f:
        data = json.load(f)
    train_shapes = [(len(p['input']), len(p['input'][0]), len(p['output']), len(p['output'][0])) for p in data['train']]
    test_shapes = [(len(p['input']), len(p['input'][0]), len(p['output']), len(p['output'][0])) for p in data['test']]
    all_sh = train_shapes + test_shapes
    
    same = all(hi == ho and wi == wo for hi, wi, ho, wo in all_sh)
    if same:
        categories['same_shape'].append(tid)
        continue
    
    # check integer upscale
    is_up = all(ho % hi == 0 and wo % wi == 0 and (ho // hi == wo // wi) and (ho // hi > 1) for hi, wi, ho, wo in all_sh)
    if is_up:
        categories['scale_integer'].append(tid)
        continue
        
    # check crop / fixed small output
    fixed_out = all(ho <= 5 and wo <= 5 for hi, wi, ho, wo in all_sh)
    if fixed_out:
        categories['subgrid_crop'].append(tid)
        continue
        
    categories['other'].append(tid)

for k, v in categories.items():
    print(f'Category {k}: {len(v)} tasks')
