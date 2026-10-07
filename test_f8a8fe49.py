import json
import numpy as np

def solve_f8a8fe49(inp):
    H, W = inp.shape
    out = inp.copy()
    
    # 2s form the frame
    twos = [(r, c) for r in range(H) for c in range(W) if inp[r, c] == 2]
    r0, r1 = min(r for r, c in twos), max(r for r, c in twos)
    c0, c1 = min(c for r, c in twos), max(c for r, c in twos)
    
    # Check if top/bot walls are solid or left/right walls are solid
    top_solid = (np.count_nonzero(inp[r0, c0:c1+1] == 2) > (c1 - c0 + 1) // 2)
    left_solid = (np.count_nonzero(inp[r0:r1+1, c0] == 2) > (r1 - r0 + 1) // 2)
    
    # Inside 5s
    fives_inside = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1) if inp[r, c] == 5]
    
    # Clear inside 5s in output
    for r, c in fives_inside:
        out[r, c] = 0
        
    if top_solid and not left_solid:
        # Reflect vertically across top wall (r0) and bottom wall (r1)
        r_mid = (r0 + r1) / 2
        for r, c in fives_inside:
            if r < r_mid:
                nr = r0 - (r - r0)
            else:
                nr = r1 + (r1 - r)
            if 0 <= nr < H:
                out[nr, c] = 5
    else:
        # Reflect horizontally across left wall (c0) and right wall (c1)
        c_mid = (c0 + c1) / 2
        for r, c in fives_inside:
            if c < c_mid:
                nc = c0 - (c - c0)
            else:
                nc = c1 + (c1 - c)
            if 0 <= nc < W:
                out[r, nc] = 5
                
    return out

with open("training/f8a8fe49.json") as f:
    d = json.load(f)

for i, p in enumerate(d["train"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_f8a8fe49(inp)
    assert np.array_equal(pred, expected), f"Train {i} failed!"
for i, p in enumerate(d["test"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_f8a8fe49(inp)
    assert np.array_equal(pred, expected), f"Test {i} failed!"
print("f8a8fe49: 100% PASS ON ALL TRAIN AND TEST!")
