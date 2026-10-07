import json
import numpy as np
from scipy.ndimage import label

def solve_7b6016b9(inp):
    H, W = inp.shape
    out = inp.copy()
    lbl, num = label(inp == 0)
    for c in range(1, num + 1):
        mask = (lbl == c)
        pts = np.argwhere(mask)
        touches = (pts[:, 0].min() == 0 or pts[:, 0].max() == H - 1 or
                   pts[:, 1].min() == 0 or pts[:, 1].max() == W - 1)
        if touches:
            out[mask] = 3
        else:
            out[mask] = 2
    return out

with open("training/7b6016b9.json") as f:
    d = json.load(f)

for i, p in enumerate(d["train"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_7b6016b9(inp)
    assert np.array_equal(pred, expected), f"Train {i} failed!"
for i, p in enumerate(d["test"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_7b6016b9(inp)
    assert np.array_equal(pred, expected), f"Test {i} failed!"
print("7b6016b9: 100% PASS ON ALL TRAIN AND TEST!")
