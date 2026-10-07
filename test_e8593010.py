import json
import numpy as np
from scipy.ndimage import label

def solve_e8593010(inp):
    out = inp.copy()
    # 4-connected components of 0
    structure = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    labeled, num_features = label(inp == 0, structure=structure)
    
    # Map component size to color:
    # size 1 -> 3
    # size 2 -> 2
    # size 3 -> 1
    size_to_color = {1: 3, 2: 2, 3: 1}
    
    for feat_id in range(1, num_features + 1):
        mask = (labeled == feat_id)
        sz = np.sum(mask)
        color = size_to_color.get(sz, 0)
        out[mask] = color
        
    return out

with open("training/e8593010.json") as f:
    d = json.load(f)

for i, p in enumerate(d["train"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_e8593010(inp)
    assert np.array_equal(pred, expected), f"Train {i} failed!"
for i, p in enumerate(d["test"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_e8593010(inp)
    assert np.array_equal(pred, expected), f"Test {i} failed!"
print("e8593010: 100% PASS ON ALL TRAIN AND TEST!")
