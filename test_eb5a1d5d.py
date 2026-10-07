import json
import numpy as np

def solve_eb5a1d5d(inp):
    H, W = inp.shape
    chain = []
    r0, r1, c0, c1 = 0, H - 1, 0, W - 1
    
    while r0 <= r1 and c0 <= c1:
        perimeter = list(inp[r0, c0:c1+1]) + list(inp[r1, c0:c1+1]) + list(inp[r0:r1+1, c0]) + list(inp[r0:r1+1, c1])
        c = max(set(perimeter), key=perimeter.count)
        chain.append(c)
        sub = inp[r0:r1+1, c0:c1+1]
        interior = np.argwhere(sub != c)
        if len(interior) == 0:
            break
        ir0, ic0 = interior.min(axis=0)
        ir1, ic1 = interior.max(axis=0)
        r0, r1 = r0 + ir0, r0 + ir1
        c0, c1 = c0 + ic0, c0 + ic1
        
    N = len(chain)
    S = 2 * N - 1
    out = np.zeros((S, S), dtype=int)
    for k, col in enumerate(chain):
        out[k:S-k, k:S-k] = col
    return out

with open("training/eb5a1d5d.json") as f:
    d = json.load(f)

for i, p in enumerate(d["train"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_eb5a1d5d(inp)
    assert np.array_equal(pred, expected), f"Train {i} failed!"
for i, p in enumerate(d["test"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_eb5a1d5d(inp)
    assert np.array_equal(pred, expected), f"Test {i} failed!"
print("eb5a1d5d: 100% PASS ON ALL TRAIN AND TEST!")
