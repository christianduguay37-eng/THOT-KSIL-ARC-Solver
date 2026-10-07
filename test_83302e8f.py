import json
from collections import Counter
import numpy as np
from scipy.ndimage import label

def solve_83302e8f(inp):
    lbl, num = label(inp == 0)
    sizes = [(lbl == c).sum() for c in range(1, num + 1)]
    modal_size = Counter(sizes).most_common(1)[0][0]
    out = inp.copy()
    for c in range(1, num + 1):
        mask = (lbl == c)
        if mask.sum() == modal_size:
            out[mask] = 3
        else:
            out[mask] = 4
    return out

with open("training/83302e8f.json") as f:
    d = json.load(f)

for i, p in enumerate(d["train"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_83302e8f(inp)
    assert np.array_equal(pred, expected), f"Train {i} failed!"
for i, p in enumerate(d["test"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_83302e8f(inp)
    assert np.array_equal(pred, expected), f"Test {i} failed!"
print("83302e8f: 100% PASS ON ALL TRAIN AND TEST!")
