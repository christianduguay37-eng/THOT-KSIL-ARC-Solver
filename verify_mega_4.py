import json
import numpy as np
from pathlib import Path

train_dir = Path('Atelier_ARC_Prize/training')

def p_a740d043(grid):
    mask = (grid != 1)
    if not np.any(mask):
        return grid
    rows, cols = np.where(mask)
    rmin, rmax = rows.min(), rows.max()
    cmin, cmax = cols.min(), cols.max()
    sub = grid[rmin:rmax+1, cmin:cmax+1].copy()
    sub[sub == 1] = 0
    return sub

def p_f25ffba3(grid):
    H = grid.shape[0]
    out = grid.copy()
    h2 = H // 2
    out[:h2] = np.flipud(grid[h2:])
    return out

def p_a3df8b1e(grid):
    H, W = grid.shape
    out = np.zeros((H, W), dtype=int)
    r = H - 1
    c = 0
    dc = 1
    while r >= 0:
        out[r, c] = 1
        r -= 1
        if W > 1:
            if c + dc >= W:
                dc = -1
            elif c + dc < 0:
                dc = 1
            c += dc
    return out

def p_a85d4709(grid):
    MAP = {0: 2, 1: 4, 2: 3}
    out = np.zeros((3, 3), dtype=int)
    for r in range(3):
        cols = np.where(grid[r] == 5)[0]
        if len(cols) > 0:
            c = cols[0]
            out[r, :] = MAP[c]
    return out

tests = [
    ('a740d043', p_a740d043),
    ('f25ffba3', p_f25ffba3),
    ('a3df8b1e', p_a3df8b1e),
    ('a85d4709', p_a85d4709),
]

for tid, fn in tests:
    with open(train_dir / f'{tid}.json') as f:
        data = json.load(f)
    tr_ok = all(np.array_equal(fn(np.array(p['input'])), np.array(p['output'])) for p in data['train'])
    te_ok = all(np.array_equal(fn(np.array(p['input'])), np.array(p['output'])) for p in data['test'])
    print(f'Task {tid}: Train={tr_ok}, Test={te_ok}')
