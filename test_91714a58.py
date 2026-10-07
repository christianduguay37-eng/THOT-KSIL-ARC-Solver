import json
import numpy as np

def solve_91714a58(inp):
    H, W = inp.shape
    best_area = 0
    best_rect = None
    for color in range(1, 10):
        grid = (inp == color)
        if not np.any(grid):
            continue
        for r0 in range(H):
            for r1 in range(r0 + 1, H):
                for c0 in range(W):
                    for c1 in range(c0 + 1, W):
                        area = (r1 - r0 + 1) * (c1 - c0 + 1)
                        if area > best_area:
                            if np.all(grid[r0:r1+1, c0:c1+1]):
                                best_area = area
                                best_rect = (r0, r1, c0, c1, color)
    out = np.zeros_like(inp)
    if best_rect:
        r0, r1, c0, c1, c = best_rect
        out[r0:r1+1, c0:c1+1] = c
    return out

with open("training/91714a58.json") as f:
    d = json.load(f)

for i, p in enumerate(d["train"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_91714a58(inp)
    assert np.array_equal(pred, expected), f"Train {i} failed!"
for i, p in enumerate(d["test"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_91714a58(inp)
    assert np.array_equal(pred, expected), f"Test {i} failed!"
print("91714a58: 100% PASS ON ALL TRAIN AND TEST!")
