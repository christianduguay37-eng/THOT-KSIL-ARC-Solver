import json
import numpy as np

def solve_d687bc17(inp):
    H, W = inp.shape
    out = np.zeros_like(inp)
    # copy borders
    out[0, :] = inp[0, :]
    out[-1, :] = inp[-1, :]
    out[:, 0] = inp[:, 0]
    out[:, -1] = inp[:, -1]
    
    # Border colors
    top_c = inp[0, 1]
    bot_c = inp[-1, 1]
    left_c = inp[1, 0]
    right_c = inp[1, -1]
    
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            val = inp[r, c]
            if val == 0:
                continue
            if val == top_c:
                out[1, c] = val
            elif val == bot_c:
                out[H - 2, c] = val
            elif val == left_c:
                out[r, 1] = val
            elif val == right_c:
                out[r, W - 2] = val
    return out

with open("training/d687bc17.json") as f:
    d = json.load(f)

for i, p in enumerate(d["train"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_d687bc17(inp)
    assert np.array_equal(pred, expected), f"Train {i} failed!"
for i, p in enumerate(d["test"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_d687bc17(inp)
    assert np.array_equal(pred, expected), f"Test {i} failed!"
print("d687bc17: 100% PASS ON ALL TRAIN AND TEST!")
