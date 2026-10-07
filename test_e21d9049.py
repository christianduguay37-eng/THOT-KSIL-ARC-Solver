import json
import numpy as np

def solve_e21d9049(inp):
    H, W = inp.shape
    out = np.zeros_like(inp)
    # Find the row that has multiple nonzeros, and the col that has multiple nonzeros
    row_counts = [np.count_nonzero(inp[r, :]) for r in range(H)]
    col_counts = [np.count_nonzero(inp[:, c]) for c in range(W)]
    
    cross_r = np.argmax(row_counts)
    cross_c = np.argmax(col_counts)
    
    # Get the seed horizontal pattern
    # Find consecutive nonzeros in cross_r
    cols = [c for c in range(W) if inp[cross_r, c] != 0]
    # Length of pattern
    L_h = len(cols)
    pat_h = [inp[cross_r, c] for c in cols]
    start_c = cols[0]
    
    # Fill row cross_r
    for c in range(W):
        offset = (c - start_c) % L_h
        out[cross_r, c] = pat_h[offset]
        
    # Get the seed vertical pattern
    rows = [r for r in range(H) if inp[r, cross_c] != 0]
    L_v = len(rows)
    pat_v = [inp[r, cross_c] for r in rows]
    start_r = rows[0]
    
    # Fill col cross_c
    for r in range(H):
        offset = (r - start_r) % L_v
        out[r, cross_c] = pat_v[offset]
        
    return out

with open("training/e21d9049.json") as f:
    d = json.load(f)

for i, p in enumerate(d["train"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_e21d9049(inp)
    assert np.array_equal(pred, expected), f"Train {i} failed!"
for i, p in enumerate(d["test"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_e21d9049(inp)
    assert np.array_equal(pred, expected), f"Test {i} failed!"
print("e21d9049: 100% PASS ON ALL TRAIN AND TEST!")
