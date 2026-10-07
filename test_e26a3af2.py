import json
import numpy as np
from collections import Counter

def solve_e26a3af2(inp):
    H, W = inp.shape
    row_mode_counts = []
    col_mode_counts = []
    row_modes = []
    col_modes = []
    
    for r in range(H):
        mc = Counter(inp[r, :]).most_common(1)[0]
        row_modes.append(mc[0])
        row_mode_counts.append(mc[1])
        
    for c in range(W):
        mc = Counter(inp[:, c]).most_common(1)[0]
        col_modes.append(mc[0])
        col_mode_counts.append(mc[1])
        
    row_score = np.mean(row_mode_counts) / W
    col_score = np.mean(col_mode_counts) / H
    
    out = np.zeros_like(inp)
    if row_score > col_score:
        for r in range(H):
            out[r, :] = row_modes[r]
    else:
        for c in range(W):
            out[:, c] = col_modes[c]
    return out

with open("training/e26a3af2.json") as f:
    d = json.load(f)

for i, p in enumerate(d["train"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_e26a3af2(inp)
    assert np.array_equal(pred, expected), f"Train {i} failed!"
for i, p in enumerate(d["test"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_e26a3af2(inp)
    assert np.array_equal(pred, expected), f"Test {i} failed!"
print("e26a3af2: 100% PASS ON ALL TRAIN AND TEST!")
