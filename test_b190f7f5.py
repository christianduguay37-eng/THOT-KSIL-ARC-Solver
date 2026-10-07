import json
import numpy as np

def solve_b190f7f5(inp):
    H, W = inp.shape
    if W == 2 * H:
        S = H
        p1 = inp[:, :S]
        p2 = inp[:, S:]
    elif H == 2 * W:
        S = W
        p1 = inp[:S, :]
        p2 = inp[S:, :]
    else:
        raise ValueError(f"Unexpected shape: {inp.shape}")
        
    if 8 in p1:
        brush = (p1 == 8)
        grid = p2
    else:
        brush = (p2 == 8)
        grid = p1
        
    out = np.zeros((S * S, S * S), dtype=int)
    for r in range(S):
        for c in range(S):
            val = grid[r, c]
            if val != 0:
                out[r * S : (r + 1) * S, c * S : (c + 1) * S] = np.where(brush, val, 0)
    return out

with open("training/b190f7f5.json") as f:
    d = json.load(f)

for i, p in enumerate(d["train"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_b190f7f5(inp)
    assert np.array_equal(pred, expected), f"Train {i} failed!"
for i, p in enumerate(d["test"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_b190f7f5(inp)
    assert np.array_equal(pred, expected), f"Test {i} failed!"
print("b190f7f5: 100% PASS ON ALL TRAIN AND TEST!")
