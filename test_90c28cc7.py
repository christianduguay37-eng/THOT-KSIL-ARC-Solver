import json
import numpy as np

def solve_90c28cc7(inp):
    pts = np.argwhere(inp != 0)
    r_min, c_min = pts.min(axis=0)
    r_max, c_max = pts.max(axis=0)
    
    sub = inp[r_min:r_max+1, c_min:c_max+1]
    H, W = sub.shape
    
    row_cuts = [0]
    for r in range(H - 1):
        if not np.array_equal(sub[r, :], sub[r+1, :]):
            row_cuts.append(r + 1)
    row_cuts.append(H)
    
    col_cuts = [0]
    for c in range(W - 1):
        if np.any(sub[:, c] != sub[:, c+1]):
            col_cuts.append(c + 1)
    col_cuts.append(W)
    
    num_rows = len(row_cuts) - 1
    num_cols = len(col_cuts) - 1
    out = np.zeros((num_rows, num_cols), dtype=int)
    
    for i in range(num_rows):
        r_mid = (row_cuts[i] + row_cuts[i+1]) // 2
        for j in range(num_cols):
            c_mid = (col_cuts[j] + col_cuts[j+1]) // 2
            out[i, j] = sub[r_mid, c_mid]
            
    return out

with open("training/90c28cc7.json") as f:
    d = json.load(f)

for i, p in enumerate(d["train"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_90c28cc7(inp)
    assert np.array_equal(pred, expected), f"Train {i} failed!"
for i, p in enumerate(d["test"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_90c28cc7(inp)
    assert np.array_equal(pred, expected), f"Test {i} failed!"
print("90c28cc7: 100% PASS ON ALL TRAIN AND TEST!")
