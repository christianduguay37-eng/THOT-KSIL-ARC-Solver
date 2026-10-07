import json
import numpy as np
from collections import defaultdict

def solve_6cdd2623(inp):
    H, W = inp.shape
    out = np.zeros_like(inp)
    
    # Collect matching lines by color
    color_lines = defaultdict(list)
    
    for r in range(H):
        if inp[r, 0] != 0 and inp[r, 0] == inp[r, W - 1]:
            color_lines[inp[r, 0]].append(('row', r))
            
    for c in range(W):
        if inp[0, c] != 0 and inp[0, c] == inp[H - 1, c]:
            color_lines[inp[0, c]].append(('col', c))
            
    # Find the color with the most matching lines (or count == 2)
    best_color = max(color_lines.keys(), key=lambda c: len(color_lines[c]))
    
    for line_type, idx in color_lines[best_color]:
        if line_type == 'row':
            out[idx, :] = best_color
        else:
            out[:, idx] = best_color
            
    return out

with open("training/6cdd2623.json") as f:
    d = json.load(f)

for i, p in enumerate(d["train"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_6cdd2623(inp)
    assert np.array_equal(pred, expected), f"Train {i} failed!"
for i, p in enumerate(d["test"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_6cdd2623(inp)
    assert np.array_equal(pred, expected), f"Test {i} failed!"
print("6cdd2623: 100% PASS ON ALL TRAIN AND TEST!")
