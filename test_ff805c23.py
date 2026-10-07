import json
import numpy as np

def solve_ff805c23(inp):
    H, W = inp.shape
    found = None
    for r0 in range(H - 4):
        for c0 in range(W - 4):
            sub = inp[r0:r0+5, c0:c0+5]
            val = sub[0, 0]
            if val != 0 and np.all(sub == val):
                if (inp == val).sum() == 25:
                    found = (r0, c0, val)
                    break
        if found:
            break
            
    r0, c0, val = found
    out = np.rot90(inp, 2)[r0:r0+5, c0:c0+5]
    return out

with open("training/ff805c23.json") as f:
    d = json.load(f)

for i, p in enumerate(d["train"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_ff805c23(inp)
    assert np.array_equal(pred, expected), f"Train {i} failed!"
for i, p in enumerate(d["test"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_ff805c23(inp)
    assert np.array_equal(pred, expected), f"Test {i} failed!"
print("ff805c23: 100% PASS ON ALL TRAIN AND TEST!")
