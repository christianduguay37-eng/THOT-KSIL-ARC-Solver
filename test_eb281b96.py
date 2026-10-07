import json
import numpy as np

def solve_eb281b96(inp):
    H, W = inp.shape
    # Sequence of indices for 2 full oscillations (down-up-down-up):
    # e.g. H=3: 0,1,2, 1,0, 1,2, 1,0 -> length 9
    # e.g. H=4: 0,1,2,3, 2,1,0, 1,2,3, 2,1,0 -> length 13
    # e.g. H=5: 0,1,2,3,4, 3,2,1,0, 1,2,3,4, 3,2,1,0 -> length 17
    cycle = list(range(H)) + list(range(H - 2, 0, -1))
    # Two full cycles plus the first element
    indices = cycle + cycle + [0]
    out = inp[indices, :]
    return out

with open("training/eb281b96.json") as f:
    d = json.load(f)

for i, p in enumerate(d["train"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_eb281b96(inp)
    assert np.array_equal(pred, expected), f"Train {i} failed!"
for i, p in enumerate(d["test"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_eb281b96(inp)
    assert np.array_equal(pred, expected), f"Test {i} failed!"
print("eb281b96: 100% PASS ON ALL TRAIN AND TEST!")
