import json
import re
import numpy as np
from pathlib import Path

train_dir = Path('Atelier_ARC_Prize/training')
with open('Atelier_ARC_Prize/palier_20_darwin.py', 'r', encoding='utf-8') as f:
    text = f.read()

canonical = set(re.findall(r'"([0-9a-f]{8})"', text))
all_tasks = sorted([p.stem for p in train_dir.glob('*.json')])
unsolved = [t for t in all_tasks if t not in canonical]

scale_int = []
subgrid_crop = []
same_shape_small = []

for tid in unsolved:
    with open(train_dir / f'{tid}.json') as f:
        data = json.load(f)
    train_shapes = [(len(p['input']), len(p['input'][0]), len(p['output']), len(p['output'][0])) for p in data['train']]
    test_shapes = [(len(p['input']), len(p['input'][0]), len(p['output']), len(p['output'][0])) for p in data['test']]
    all_sh = train_shapes + test_shapes
    
    same = all(hi == ho and wi == wo for hi, wi, ho, wo in all_sh)
    h_max = max(hi for hi, wi, ho, wo in all_sh)
    w_max = max(wi for hi, wi, ho, wo in all_sh)
    
    if same:
        diff_sum = sum(sum(p['input'][r][c] != p['output'][r][c] for r in range(len(p['input'])) for c in range(len(p['input'][0]))) for p in data['train'])
        same_shape_small.append((diff_sum, h_max, w_max, tid))
        continue
    
    is_up = all(ho % hi == 0 and wo % wi == 0 and (ho // hi == wo // wi) and (ho // hi > 1) for hi, wi, ho, wo in all_sh)
    if is_up:
        scale_int.append((tid, all_sh[0]))
        continue
        
    fixed_out = all(ho <= 5 and wo <= 5 for hi, wi, ho, wo in all_sh)
    if fixed_out:
        subgrid_crop.append((tid, all_sh[0]))
        continue

same_shape_small.sort()

print(f'Scale Integer ({len(scale_int)}): {scale_int}')
print(f'\nSubgrid Crop ({len(subgrid_crop)}): {subgrid_crop}')
print(f'\nSmallest Same-Shape (top 30):')
for diff, h, w, tid in same_shape_small[:30]:
    print(f'  {tid}: diff={diff}, max_dim=({h},{w})')
