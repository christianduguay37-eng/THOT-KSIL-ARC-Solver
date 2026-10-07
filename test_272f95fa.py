import json
import numpy as np

def solve_272f95fa(inp):
    H, W = inp.shape
    # Find the rows and cols of 8s
    # Full rows of 8
    h_lines = [r for r in range(H) if np.all(inp[r, :] == 8)]
    # Full cols of 8
    v_lines = [c for c in range(W) if np.all(inp[:, c] == 8)]
    
    # We have 2 h_lines and 2 v_lines
    r0, r1 = h_lines[0], h_lines[1]
    c0, c1 = v_lines[0], v_lines[1]
    
    out = inp.copy()
    
    # Top-mid (0..r0, c0+1..c1) -> 2
    out[0:r0, c0+1:c1] = 2
    # Bot-mid (r1+1..H, c0+1..c1) -> 1
    out[r1+1:H, c0+1:c1] = 1
    # Mid-left (r0+1..r1, 0:c0) -> 4
    out[r0+1:r1, 0:c0] = 4
    # Mid-center (r0+1..r1, c0+1..c1) -> 6
    out[r0+1:r1, c0+1:c1] = 6
    # Mid-right (r0+1..r1, c1+1:W) -> 3
    out[r0+1:r1, c1+1:W] = 3
    
    return out

with open("training/272f95fa.json") as f:
    d = json.load(f)

for i, p in enumerate(d["train"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_272f95fa(inp)
    assert np.array_equal(pred, expected), f"Train {i} failed!"
for i, p in enumerate(d["test"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_272f95fa(inp)
    assert np.array_equal(pred, expected), f"Test {i} failed!"
print("272f95fa: 100% PASS ON ALL TRAIN AND TEST!")
