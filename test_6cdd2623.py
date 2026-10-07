import json
import numpy as np

def solve_6cdd2623(inp):
    H, W = inp.shape
    out = np.zeros_like(inp)
    # Check matching endpoints on left and right border
    for r in range(H):
        if inp[r, 0] != 0 and inp[r, 0] == inp[r, W - 1]:
            out[r, :] = inp[r, 0]
    # Check matching endpoints on top and bottom border
    for c in range(W):
        if inp[0, c] != 0 and inp[0, c] == inp[H - 1, c]:
            out[:, c] = inp[0, c]
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
